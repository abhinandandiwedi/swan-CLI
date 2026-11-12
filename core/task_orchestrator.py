"""Task Orchestrator for SWAN-HIVE.

Provides persistent SQLite-backed task queuing, DAG dependency resolution,
strict task lifecycle state machines, and write-scope lock coordination.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.workspace_manager import validate_project_id
from core.model_router import SUPPORTED_ROLES


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TaskOrchestratorError(Exception):
    """Base exception for all Task Orchestrator errors."""
    pass


class TaskValidationError(TaskOrchestratorError):
    """Raised when a task specification or batch fails validation."""
    pass


class DuplicateTaskError(TaskValidationError):
    """Raised when a task ID is duplicated within a batch or database."""
    pass


class MissingDependencyError(TaskValidationError):
    """Raised when a task depends on a non-existent task ID."""
    pass


class DependencyCycleError(TaskValidationError):
    """Raised when cyclic dependencies are detected in the task graph."""
    pass


class InvalidStateTransitionError(TaskOrchestratorError):
    """Raised when an illegal state transition is attempted."""
    pass


class ScopeLockConflictError(TaskOrchestratorError):
    """Raised when a task's write scope conflicts with currently held locks."""
    pass


class BatchLimitExceededError(TaskValidationError):
    """Raised when submitted batch exceeds the maximum allowed task count."""
    pass


# ---------------------------------------------------------------------------
# Task State Machine
# ---------------------------------------------------------------------------

class TaskState(str, Enum):
    """Task lifecycle states in SWAN-HIVE."""
    SUBMITTED = "SUBMITTED"
    VALIDATING_SPEC = "VALIDATING_SPEC"
    QUEUED = "QUEUED"
    WAITING_DEPENDENCY = "WAITING_DEPENDENCY"
    READY = "READY"
    WAITING_RESOURCE = "WAITING_RESOURCE"
    RUNNING = "RUNNING"
    WORKER_COMPLETED = "WORKER_COMPLETED"
    WORKER_FAILED = "WORKER_FAILED"
    RETRY_PENDING = "RETRY_PENDING"
    VALIDATION_PENDING = "VALIDATION_PENDING"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


# Enforce valid transitions strictly.
VALID_TRANSITIONS: Dict[TaskState, Set[TaskState]] = {
    TaskState.SUBMITTED: {
        TaskState.VALIDATING_SPEC,
        TaskState.CANCELLED,
    },
    TaskState.VALIDATING_SPEC: {
        TaskState.QUEUED,
        TaskState.WAITING_DEPENDENCY,
        TaskState.READY,
        TaskState.BLOCKED,
        TaskState.CANCELLED,
    },
    TaskState.QUEUED: {
        TaskState.WAITING_DEPENDENCY,
        TaskState.READY,
        TaskState.BLOCKED,
        TaskState.CANCELLED,
    },
    TaskState.WAITING_DEPENDENCY: {
        TaskState.READY,
        TaskState.BLOCKED,
        TaskState.CANCELLED,
    },
    TaskState.READY: {
        TaskState.WAITING_RESOURCE,
        TaskState.RUNNING,
        TaskState.BLOCKED,
        TaskState.CANCELLED,
    },
    TaskState.WAITING_RESOURCE: {
        TaskState.READY,
        TaskState.RUNNING,
        TaskState.BLOCKED,
        TaskState.CANCELLED,
    },
    TaskState.RUNNING: {
        TaskState.WORKER_COMPLETED,
        TaskState.WORKER_FAILED,
        TaskState.WAITING_RESOURCE,
        TaskState.CANCELLED,
    },
    TaskState.WORKER_COMPLETED: {
        TaskState.VALIDATION_PENDING,
        TaskState.CANCELLED,
    },
    TaskState.WORKER_FAILED: {
        TaskState.RETRY_PENDING,
        TaskState.BLOCKED,
        TaskState.CANCELLED,
    },
    TaskState.RETRY_PENDING: {
        TaskState.READY,
        TaskState.QUEUED,
        TaskState.BLOCKED,
        TaskState.CANCELLED,
    },
    TaskState.VALIDATION_PENDING: {
        TaskState.RETRY_PENDING,
        TaskState.BLOCKED,
        TaskState.CANCELLED,
    },
    TaskState.BLOCKED: {
        TaskState.READY,
        TaskState.CANCELLED,
    },
    TaskState.CANCELLED: set(),
}

TERMINAL_STATES: Set[TaskState] = {
    TaskState.CANCELLED,
    TaskState.VALIDATION_PENDING,
}


# ---------------------------------------------------------------------------
# Scope & Path Helpers
# ---------------------------------------------------------------------------

IDENTIFIER_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{1,64}$")
DEFAULT_MAX_BATCH_SIZE: int = 50
DEFAULT_TASK_TIMEOUT: int = 60
DEFAULT_RETRY_LIMIT: int = 2
MAX_RETRY_LIMIT: int = 10


def validate_scope_path(path_str: str) -> str:
    """Validate and normalize a file or directory path for read/write scopes.

    Rejects:
    - Empty or whitespace strings.
    - Path traversal ('..').
    - Absolute paths ('/' or Windows drives).
    - Null bytes.
    """
    if not isinstance(path_str, str) or not path_str.strip():
        raise TaskValidationError("Scope path cannot be empty")
    
    clean = path_str.strip()
    if "\x00" in clean:
        raise TaskValidationError(f"Scope path contains null bytes: {path_str!r}")
    
    # Must be relative
    if clean.startswith("/") or clean.startswith("\\"):
        raise TaskValidationError(f"Scope path must be relative to workspace: {path_str!r}")
    if len(clean) >= 2 and clean[1] == ":" and clean[0].isalpha():
        raise TaskValidationError(f"Scope path must not contain drive letters: {path_str!r}")
    
    # Split parts and check for path traversal
    norm = PurePosixPath(clean)
    for part in norm.parts:
        if part == "..":
            raise TaskValidationError(f"Scope path must not contain directory traversal ('..'): {path_str!r}")
        if part in ("", "."):
            continue

    normalized_str = norm.as_posix()
    if normalized_str in ("", "."):
        raise TaskValidationError(f"Scope path resolves to empty root: {path_str!r}")
    return normalized_str


def is_path_in_scope(file_path: str, scope: List[str]) -> bool:
    """Check if file_path falls inside or matches any path/directory in scope.

    Supports exact file matches and directory prefix matches.
    """
    if not scope:
        return False
    
    try:
        norm_file = PurePosixPath(validate_scope_path(file_path))
    except TaskValidationError:
        return False
    
    for scope_item in scope:
        try:
            norm_scope = PurePosixPath(validate_scope_path(scope_item))
        except TaskValidationError:
            continue
        if norm_file == norm_scope or norm_scope in norm_file.parents:
            return True
    return False


def paths_overlap(path1: str, path2: str) -> bool:
    """Determine if two relative paths overlap (exact match or parent/child directory)."""
    try:
        p1 = PurePosixPath(validate_scope_path(path1))
        p2 = PurePosixPath(validate_scope_path(path2))
    except TaskValidationError:
        return False
    return p1 == p2 or p1 in p2.parents or p2 in p1.parents


def detect_dependency_cycles(graph: Dict[str, List[str]]) -> List[str]:
    """Detect cycles in a directed graph using 3-color DFS.

    Args:
        graph: Mapping of task_id -> list of dependent task_ids it depends on.

    Returns:
        List representing the cycle path (e.g. ['A', 'B', 'A']), or empty list if acyclic.
    """
    visited: Dict[str, int] = {node: 0 for node in graph}
    stack: List[str] = []

    def dfs(node: str) -> List[str]:
        visited[node] = 1  # visiting (GRAY)
        stack.append(node)
        for dep in graph.get(node, []):
            if dep not in visited:
                continue
            if visited[dep] == 1:
                idx = stack.index(dep)
                return stack[idx:] + [dep]
            elif visited[dep] == 0:
                cycle = dfs(dep)
                if cycle:
                    return cycle
        stack.pop()
        visited[node] = 2  # visited (BLACK)
        return []

    for node in graph:
        if visited[node] == 0:
            cycle = dfs(node)
            if cycle:
                return cycle
    return []


# ---------------------------------------------------------------------------
# Task Specification Schema
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    """Structured specification of a task in SWAN-HIVE."""
    task_id: str
    batch_id: str
    project_id: str
    title: str
    description: str
    role: str
    schema_version: str = "1.0"
    model_preference: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    read_scope: List[str] = field(default_factory=list)
    write_scope: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    validation_requirements: List[str] = field(default_factory=list)
    timeout: int = DEFAULT_TASK_TIMEOUT
    retry_limits: int = DEFAULT_RETRY_LIMIT
    creation_timestamp: float = field(default_factory=time.time)
    base_commit: Optional[str] = None

    def validate(self) -> None:
        """Validate all fields according to schema requirements."""
        if not self.schema_version:
            raise TaskValidationError("schema_version must be provided")

        if not isinstance(self.task_id, str) or not IDENTIFIER_REGEX.match(self.task_id):
            raise TaskValidationError(
                f"task_id must match regex '^[a-zA-Z0-9_\\-\\.]+$' (1-64 chars), got: {self.task_id!r}"
            )

        if not isinstance(self.batch_id, str) or not IDENTIFIER_REGEX.match(self.batch_id):
            raise TaskValidationError(
                f"batch_id must match regex '^[a-zA-Z0-9_\\-\\.]+$' (1-64 chars), got: {self.batch_id!r}"
            )

        # Validate project_id using Workspace Manager validation
        try:
            validate_project_id(self.project_id)
        except Exception as e:
            raise TaskValidationError(f"Invalid project_id '{self.project_id}': {e}") from e

        if not isinstance(self.title, str) or not self.title.strip():
            raise TaskValidationError("title must be a non-empty string")
        if len(self.title) > 256:
            raise TaskValidationError(f"title length ({len(self.title)}) exceeds maximum of 256 characters")

        if not isinstance(self.description, str):
            raise TaskValidationError("description must be a string")
        if len(self.description) > 4096:
            raise TaskValidationError(f"description length ({len(self.description)}) exceeds maximum of 4096 characters")

        if not isinstance(self.role, str) or self.role.lower() not in SUPPORTED_ROLES:
            raise TaskValidationError(
                f"Unsupported role '{self.role}'. Supported roles are: {sorted(SUPPORTED_ROLES)}"
            )

        if self.model_preference is not None:
            if not isinstance(self.model_preference, str) or not self.model_preference.strip():
                raise TaskValidationError("model_preference must be a non-empty string when provided")

        if not isinstance(self.depends_on, list):
            raise TaskValidationError("depends_on must be a list of task_id strings")
        for dep in self.depends_on:
            if not isinstance(dep, str) or not dep.strip():
                raise TaskValidationError(f"Invalid dependency '{dep}': must be non-empty string")
            if dep == self.task_id:
                raise TaskValidationError(f"Task '{self.task_id}' cannot depend on itself")

        # Validate scopes
        if not isinstance(self.read_scope, list):
            raise TaskValidationError("read_scope must be a list of paths")
        self.read_scope = [validate_scope_path(p) for p in self.read_scope]

        if not isinstance(self.write_scope, list):
            raise TaskValidationError("write_scope must be a list of paths")
        self.write_scope = [validate_scope_path(p) for p in self.write_scope]

        # Validate timeout
        if not isinstance(self.timeout, int) or self.timeout < 1 or self.timeout > 3600:
            raise TaskValidationError(f"timeout must be an integer between 1 and 3600 seconds, got: {self.timeout}")

        # Validate retry limits
        if not isinstance(self.retry_limits, int) or self.retry_limits < 0 or self.retry_limits > MAX_RETRY_LIMIT:
            raise TaskValidationError(
                f"retry_limits must be an integer between 0 and {MAX_RETRY_LIMIT}, got: {self.retry_limits}"
            )

        if not isinstance(self.creation_timestamp, (int, float)) or self.creation_timestamp <= 0:
            raise TaskValidationError("creation_timestamp must be a positive timestamp")

        if self.base_commit is not None:
            if not isinstance(self.base_commit, str) or not self.base_commit.strip():
                raise TaskValidationError("base_commit must be a non-empty string when provided")

    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskSpec:
        """Construct and validate a TaskSpec from a dictionary."""
        spec = cls(
            schema_version=data.get("schema_version", "1.0"),
            batch_id=data.get("batch_id", ""),
            task_id=data.get("task_id", ""),
            project_id=data.get("project_id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            role=data.get("role", ""),
            model_preference=data.get("model_preference"),
            depends_on=list(data.get("depends_on", [])),
            read_scope=list(data.get("read_scope", [])),
            write_scope=list(data.get("write_scope", [])),
            acceptance_criteria=list(data.get("acceptance_criteria", [])),
            validation_requirements=list(data.get("validation_requirements", [])),
            timeout=int(data.get("timeout", DEFAULT_TASK_TIMEOUT)),
            retry_limits=int(data.get("retry_limits", DEFAULT_RETRY_LIMIT)),
            creation_timestamp=float(data.get("creation_timestamp", time.time())),
            base_commit=data.get("base_commit"),
        )
        spec.validate()
        return spec


@dataclass
class TaskRecord:
    """Complete persistent task representation including lifecycle state."""
    task_id: str
    batch_id: str
    project_id: str
    state: TaskState
    spec: TaskSpec
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Task Orchestrator (SQLite Persistent Queue)
# ---------------------------------------------------------------------------

class TaskOrchestrator:
    """Manages persistent task queue, DAG dependency tracking, state machine, and locks."""

    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        db_path: Union[str, Path] = ":memory:",
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    ):
        self.db_path = str(db_path)
        self.max_batch_size = max_batch_size
        self._init_connection()
        self._init_schema()

    def _init_connection(self) -> None:
        """Establish SQLite connection with foreign keys and row factory."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        if self.db_path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL;")

    def _init_schema(self) -> None:
        """Initialize database schema tables and indices."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_info (
                    version TEXT PRIMARY KEY
                );
            """)
            self.conn.execute("""
                INSERT OR IGNORE INTO schema_info (version) VALUES (?);
            """, (self.SCHEMA_VERSION,))

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
                );
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS task_dependencies (
                    task_id TEXT NOT NULL,
                    depends_on_task_id TEXT NOT NULL,
                    PRIMARY KEY (task_id, depends_on_task_id),
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
                );
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    old_state TEXT,
                    new_state TEXT,
                    message TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS worker_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    worker_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
                );
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS scope_locks (
                    task_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    PRIMARY KEY (task_id, path)
                );
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_reports (
                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
                );
            """)

            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_batch_state ON tasks(batch_id, state);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_worker_results_task ON worker_results(task_id);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_validation_reports_task ON validation_reports(task_id);")

    def submit_batch(
        self,
        batch_id: str,
        project_id: str,
        tasks: List[TaskSpec],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TaskSpec]:
        """Submit a batch of structured tasks.

        Validates all tasks, enforces batch limits, checks for missing dependencies,
        detects DAG cycles, and atomically records tasks into SQLite.
        """
        if not isinstance(tasks, list) or len(tasks) == 0:
            raise TaskValidationError("Task batch cannot be empty")
        if len(tasks) > self.max_batch_size:
            raise BatchLimitExceededError(
                f"Batch size {len(tasks)} exceeds limit of {self.max_batch_size}"
            )

        validate_project_id(project_id)

        # 1. Check for duplicate task IDs within the batch
        seen_ids: Set[str] = set()
        for task in tasks:
            if not isinstance(task, TaskSpec):
                raise TaskValidationError(f"Expected TaskSpec instance, got {type(task)}")
            task.validate()

            if task.batch_id != batch_id:
                raise TaskValidationError(
                    f"Task {task.task_id} batch_id '{task.batch_id}' does not match batch '{batch_id}'"
                )
            if task.project_id != project_id:
                raise TaskValidationError(
                    f"Task {task.task_id} project_id '{task.project_id}' does not match batch project '{project_id}'"
                )

            if task.task_id in seen_ids:
                raise DuplicateTaskError(f"Duplicate task ID in batch: '{task.task_id}'")
            seen_ids.add(task.task_id)

        # 2. Check for duplicate task IDs against database
        placeholders = ",".join(["?"] * len(tasks))
        task_id_list = [t.task_id for t in tasks]
        rows = self.conn.execute(
            f"SELECT task_id FROM tasks WHERE task_id IN ({placeholders})",
            task_id_list,
        ).fetchall()
        if rows:
            existing_id = rows[0]["task_id"]
            raise DuplicateTaskError(f"Task ID already exists in database: '{existing_id}'")

        # 3. Check for missing dependencies
        # Dependencies must either be in this batch or already recorded in the database
        batch_ids = {t.task_id for t in tasks}
        all_external_deps: Set[str] = set()
        for task in tasks:
            for dep in task.depends_on:
                if dep not in batch_ids:
                    all_external_deps.add(dep)

        if all_external_deps:
            placeholders_deps = ",".join(["?"] * len(all_external_deps))
            db_deps = {
                row["task_id"]
                for row in self.conn.execute(
                    f"SELECT task_id FROM tasks WHERE task_id IN ({placeholders_deps})",
                    list(all_external_deps),
                ).fetchall()
            }
            missing = all_external_deps - db_deps
            if missing:
                missing_id = sorted(missing)[0]
                raise MissingDependencyError(f"Dependency not found in batch or database: '{missing_id}'")

        # 4. Cycle detection
        dep_graph: Dict[str, List[str]] = {}
        for task in tasks:
            dep_graph[task.task_id] = list(task.depends_on)
        # Add external dependencies into graph to trace full paths if needed
        for dep in all_external_deps:
            dep_graph[dep] = []

        cycle = detect_dependency_cycles(dep_graph)
        if cycle:
            cycle_str = " -> ".join(cycle)
            raise DependencyCycleError(f"Cyclic dependency detected: {cycle_str}")

        # 5. Atomic persistence
        now = time.time()
        metadata_str = json.dumps(metadata or {})

        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO batches (batch_id, project_id, created_at, status, metadata_json)
                VALUES (?, ?, ?, ?, ?)
            """, (batch_id, project_id, now, "ACTIVE", metadata_str))

            for task in tasks:
                spec_json = json.dumps(task.to_dict())
                # Initial state determination:
                # If dependencies are required, state is WAITING_DEPENDENCY; else READY
                initial_state = TaskState.WAITING_DEPENDENCY if task.depends_on else TaskState.READY

                self.conn.execute("""
                    INSERT INTO tasks (task_id, batch_id, project_id, state, spec_json, retry_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                """, (task.task_id, batch_id, project_id, initial_state.value, spec_json, now, now))

                for dep in task.depends_on:
                    self.conn.execute("""
                        INSERT INTO task_dependencies (task_id, depends_on_task_id)
                        VALUES (?, ?)
                    """, (task.task_id, dep))

                self.conn.execute("""
                    INSERT INTO events (batch_id, task_id, timestamp, event_type, old_state, new_state, message, metadata_json)
                    VALUES (?, ?, ?, 'SUBMITTED', NULL, ?, 'Task submitted to batch', '{}')
                """, (batch_id, task.task_id, now, initial_state.value))

        return tasks

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Fetch full task record by task ID."""
        row = self.conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if not row:
            return None
        spec_data = json.loads(row["spec_json"])
        spec = TaskSpec.from_dict(spec_data)
        return TaskRecord(
            task_id=row["task_id"],
            batch_id=row["batch_id"],
            project_id=row["project_id"],
            state=TaskState(row["state"]),
            spec=spec,
            retry_count=row["retry_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_task_spec(self, task_id: str) -> Optional[TaskSpec]:
        """Fetch task specification by task ID."""
        rec = self.get_task(task_id)
        return rec.spec if rec else None

    def get_task_state(self, task_id: str) -> Optional[TaskState]:
        """Fetch current state of a task."""
        row = self.conn.execute(
            "SELECT state FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        return TaskState(row["state"]) if row else None

    def list_tasks(
        self,
        batch_id: Optional[str] = None,
        state: Optional[TaskState] = None,
    ) -> List[TaskRecord]:
        """List tasks optionally filtered by batch_id and/or state."""
        query = "SELECT * FROM tasks"
        params: List[Any] = []
        clauses: List[str] = []

        if batch_id:
            clauses.append("batch_id = ?")
            params.append(batch_id)
        if state:
            clauses.append("state = ?")
            params.append(state.value)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at ASC"

        rows = self.conn.execute(query, params).fetchall()
        records: List[TaskRecord] = []
        for row in rows:
            spec_data = json.loads(row["spec_json"])
            spec = TaskSpec.from_dict(spec_data)
            records.append(TaskRecord(
                task_id=row["task_id"],
                batch_id=row["batch_id"],
                project_id=row["project_id"],
                state=TaskState(row["state"]),
                spec=spec,
                retry_count=row["retry_count"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            ))
        return records

    def transition_task(
        self,
        task_id: str,
        target_state: TaskState,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskState:
        """Enforce strict state machine transitions and log transition events."""
        current_state = self.get_task_state(task_id)
        if current_state is None:
            raise TaskOrchestratorError(f"Task '{task_id}' not found")

        if current_state == target_state:
            return current_state

        allowed = VALID_TRANSITIONS.get(current_state, set())
        if target_state not in allowed:
            raise InvalidStateTransitionError(
                f"Illegal state transition for task '{task_id}': cannot move from {current_state.value} to {target_state.value}"
            )

        now = time.time()
        metadata_str = json.dumps(metadata or {})

        with self.conn:
            # If moving to RETRY_PENDING, increment retry_count
            if target_state == TaskState.RETRY_PENDING:
                self.conn.execute("""
                    UPDATE tasks
                    SET state = ?, retry_count = retry_count + 1, updated_at = ?
                    WHERE task_id = ?
                """, (target_state.value, now, task_id))
            else:
                self.conn.execute("""
                    UPDATE tasks
                    SET state = ?, updated_at = ?
                    WHERE task_id = ?
                """, (target_state.value, now, task_id))

            # Fetch batch_id for event log
            task_row = self.conn.execute("SELECT batch_id FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            batch_id = task_row["batch_id"] if task_row else ""

            self.conn.execute("""
                INSERT INTO events (batch_id, task_id, timestamp, event_type, old_state, new_state, message, metadata_json)
                VALUES (?, ?, ?, 'STATE_CHANGE', ?, ?, ?, ?)
            """, (batch_id, task_id, now, current_state.value, target_state.value, reason, metadata_str))

        return target_state

    def get_ready_tasks(self, batch_id: Optional[str] = None) -> List[TaskSpec]:
        """Fetch all tasks in READY state."""
        records = self.list_tasks(batch_id=batch_id, state=TaskState.READY)
        return [r.spec for r in records]

    # -----------------------------------------------------------------------
    # Scope Lock Coordination
    # -----------------------------------------------------------------------

    def acquire_task_locks(self, task_id: str) -> bool:
        """Attempt to acquire write-scope locks for all paths in task.write_scope.

        Returns True if all locks acquired, False if any path conflicts with an active lock.
        """
        spec = self.get_task_spec(task_id)
        if not spec:
            raise TaskOrchestratorError(f"Task '{task_id}' not found")

        if not spec.write_scope:
            return True  # No write scope to lock

        with self.conn:
            # Query all active locks held by other tasks
            active_locks = self.conn.execute(
                "SELECT task_id, path FROM scope_locks WHERE task_id != ?",
                (task_id,),
            ).fetchall()

            for requested_path in spec.write_scope:
                for lock in active_locks:
                    locked_path = lock["path"]
                    if paths_overlap(requested_path, locked_path):
                        return False  # Lock conflict detected!

            # No conflicts, acquire locks atomically
            now = time.time()
            for requested_path in spec.write_scope:
                self.conn.execute("""
                    INSERT OR REPLACE INTO scope_locks (task_id, path, acquired_at)
                    VALUES (?, ?, ?)
                """, (task_id, requested_path, now))

        return True

    def release_task_locks(self, task_id: str) -> None:
        """Release all write-scope locks held by task_id."""
        with self.conn:
            self.conn.execute("DELETE FROM scope_locks WHERE task_id = ?", (task_id,))

    def get_held_locks(self) -> List[Dict[str, Any]]:
        """List all currently held write-scope locks."""
        rows = self.conn.execute("SELECT * FROM scope_locks ORDER BY acquired_at ASC").fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Dependency Resolution & Propagation
    # -----------------------------------------------------------------------

    def evaluate_dependencies_and_advance(self, batch_id: Optional[str] = None) -> List[str]:
        """Evaluate dependencies for tasks in WAITING_DEPENDENCY.

        - If all dependencies are WORKER_COMPLETED or VALIDATION_PENDING -> transition to READY.
        - If any dependency is BLOCKED, CANCELLED, or WORKER_FAILED (with retries exhausted) -> transition to BLOCKED.

        Returns list of task IDs that transitioned to READY.
        """
        waiting_tasks = self.list_tasks(batch_id=batch_id, state=TaskState.WAITING_DEPENDENCY)
        advanced_to_ready: List[str] = []

        for record in waiting_tasks:
            deps = record.spec.depends_on
            if not deps:
                self.transition_task(record.task_id, TaskState.READY, reason="No dependencies")
                advanced_to_ready.append(record.task_id)
                continue

            all_satisfied = True
            blocked = False
            blocked_reason = ""

            for dep_id in deps:
                dep_record = self.get_task(dep_id)
                if not dep_record:
                    blocked = True
                    blocked_reason = f"Upstream dependency '{dep_id}' is missing"
                    break

                if dep_record.state in (TaskState.WORKER_COMPLETED, TaskState.VALIDATION_PENDING):
                    continue
                elif dep_record.state in (TaskState.BLOCKED, TaskState.CANCELLED):
                    blocked = True
                    blocked_reason = f"Upstream dependency '{dep_id}' is in {dep_record.state.value}"
                    break
                elif dep_record.state == TaskState.WORKER_FAILED:
                    if dep_record.retry_count >= dep_record.spec.retry_limits:
                        blocked = True
                        blocked_reason = f"Upstream dependency '{dep_id}' failed permanently"
                        break
                    else:
                        all_satisfied = False
                else:
                    all_satisfied = False

            if blocked:
                self.transition_task(
                    record.task_id,
                    TaskState.BLOCKED,
                    reason=f"Dependency failure: {blocked_reason}",
                )
            elif all_satisfied:
                self.transition_task(
                    record.task_id,
                    TaskState.READY,
                    reason="All dependencies satisfied",
                )
                advanced_to_ready.append(record.task_id)

        return advanced_to_ready

    # -----------------------------------------------------------------------
    # Worker Results & Events Persistence
    # -----------------------------------------------------------------------

    def record_worker_result(self, result: Any) -> int:
        """Persist worker result record to database.

        Accepts WorkerResult dataclass or dictionary.
        """
        now = time.time()
        if hasattr(result, "to_dict"):
            data = result.to_dict()
        elif isinstance(result, dict):
            data = dict(result)
        else:
            raise TaskOrchestratorError("WorkerResult must be a dataclass or dict")

        task_id = str(data.get("task_id", ""))
        batch_id = str(data.get("batch_id", ""))
        worker_id = str(data.get("worker_id", ""))
        status = str(data.get("status", ""))
        model = str(data.get("model", ""))
        result_json = json.dumps(data)

        with self.conn:
            cur = self.conn.execute("""
                INSERT INTO worker_results (task_id, batch_id, worker_id, status, model, result_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (task_id, batch_id, worker_id, status, model, result_json, now))
            return cur.lastrowid

    def record_validation_report(self, report: Any) -> int:
        """Persist validation report to database."""
        now = time.time()
        if hasattr(report, "to_dict"):
            data = report.to_dict()
        elif isinstance(report, dict):
            data = dict(report)
        else:
            raise TaskOrchestratorError("ValidationReport must be a dataclass or dict")

        task_id = str(data.get("task_id", ""))
        batch_id = str(data.get("batch_id", ""))
        status = str(data.get("validation_status", data.get("status", "")))
        report_json = json.dumps(data)

        with self.conn:
            cur = self.conn.execute("""
                INSERT INTO validation_reports (task_id, batch_id, status, report_json, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (task_id, batch_id, status, report_json, now))
            return cur.lastrowid

    def get_validation_reports(self, task_id: str) -> List[Dict[str, Any]]:
        """Fetch all validation reports recorded for a task."""
        rows = self.conn.execute(
            "SELECT * FROM validation_reports WHERE task_id = ? ORDER BY created_at ASC",
            (task_id,),
        ).fetchall()
        reports = []
        for row in rows:
            rep_dict = dict(row)
            rep_dict["report"] = json.loads(row["report_json"])
            reports.append(rep_dict)
        return reports

    def get_latest_validation_report(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the most recent validation report for a task."""
        row = self.conn.execute(
            "SELECT * FROM validation_reports WHERE task_id = ? ORDER BY created_at DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        if not row:
            return None
        rep_dict = dict(row)
        rep_dict["report"] = json.loads(row["report_json"])
        return rep_dict

    def get_worker_results(self, task_id: str) -> List[Dict[str, Any]]:
        """Fetch all recorded results for a specific task."""
        rows = self.conn.execute(
            "SELECT * FROM worker_results WHERE task_id = ? ORDER BY created_at ASC",
            (task_id,),
        ).fetchall()
        results = []
        for row in rows:
            res_dict = dict(row)
            res_dict["result"] = json.loads(row["result_json"])
            results.append(res_dict)
        return results

    def get_events(
        self,
        task_id: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch audit log events."""
        query = "SELECT * FROM events"
        params: List[Any] = []
        clauses: List[str] = []

        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        if batch_id:
            clauses.append("batch_id = ?")
            params.append(batch_id)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY event_id ASC"

        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Cancellation & Cleanup
    # -----------------------------------------------------------------------

    def cancel_task(self, task_id: str, reason: str = "User cancelled") -> None:
        """Cancel a single task and release its write-scope locks."""
        current_state = self.get_task_state(task_id)
        if not current_state:
            raise TaskOrchestratorError(f"Task '{task_id}' not found")
        if current_state in TERMINAL_STATES:
            return  # Already terminal

        self.transition_task(task_id, TaskState.CANCELLED, reason=reason)
        self.release_task_locks(task_id)

    def cancel_batch(self, batch_id: str, reason: str = "Batch cancelled") -> None:
        """Cancel all non-terminal tasks in a batch."""
        tasks = self.list_tasks(batch_id=batch_id)
        for t in tasks:
            if t.state not in TERMINAL_STATES:
                try:
                    self.cancel_task(t.task_id, reason=reason)
                except Exception:
                    pass
        with self.conn:
            self.conn.execute(
                "UPDATE batches SET status = 'CANCELLED' WHERE batch_id = ?",
                (batch_id,),
            )

    def close(self) -> None:
        """Close SQLite database connection."""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()
