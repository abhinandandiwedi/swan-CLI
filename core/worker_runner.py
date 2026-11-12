"""Worker Runner for SWAN-HIVE.

Implements a logical worker runtime that executes bounded coding tasks using
Ollama models, strict write-scope containment, Secure Execution Engine,
Context Manager, and Hardware Resource Monitor.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Set, Union

from core.workspace_manager import WorkspaceManager, detect_git_root
from core.execution_engine import (
    ExecutionEngine,
    classify_command,
    validate_command_string,
    PERMISSION_READ_ONLY,
    PERMISSION_TEST_BUILD,
    PERMISSION_MUTATING,
    PERMISSION_DESTRUCTIVE,
    PERMISSION_PRIVILEGED,
    PERMISSION_UNKNOWN,
)
from core.context_manager import (
    ContextManager,
    ContextSession,
    SessionRole,
    ObservationType,
)
from core.model_router import (
    ModelRouter,
    ModelInfo,
    ModelNotFoundError,
    OllamaConnectionError,
)
from core.resource_monitor import (
    ResourceMonitor,
    AdmissionDecision,
    AdmissionStatus,
)
from core.task_orchestrator import (
    TaskOrchestrator,
    TaskSpec,
    TaskState,
    is_path_in_scope,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class WorkerRunnerError(Exception):
    """Base exception for Worker Runner errors."""
    pass


class ModelUnavailableError(WorkerRunnerError):
    """Raised when the requested or required model is not available."""
    pass


class ResourceWaitError(WorkerRunnerError):
    """Raised when hardware resource monitor requests inference queuing."""
    pass


class ScopeViolationError(WorkerRunnerError):
    """Raised when worker attempts to access or modify outside authorized scope."""
    pass


class WorkerExecutionError(WorkerRunnerError):
    """Raised when an unrecoverable worker execution error occurs."""
    pass


# ---------------------------------------------------------------------------
# Worker Result Contract
# ---------------------------------------------------------------------------

@dataclass
class WorkerResult:
    """Typed, validated contract for worker execution results."""
    schema_version: str = "1.0"
    batch_id: str = ""
    task_id: str = ""
    worker_id: str = ""
    model: str = ""
    status: str = "completed"  # "completed", "failed", "resource_wait"
    summary: str = ""
    changed_files: List[str] = field(default_factory=list)
    commands_requested: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    error: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    worktree_id: Optional[str] = None

    def validate(self) -> None:
        """Validate WorkerResult fields."""
        if not self.schema_version:
            raise WorkerRunnerError("WorkerResult must include schema_version")
        if not self.task_id:
            raise WorkerRunnerError("WorkerResult must include task_id")
        if self.status not in ("completed", "failed", "resource_wait"):
            raise WorkerRunnerError(f"Invalid WorkerResult status: {self.status}")
        if not isinstance(self.changed_files, list):
            raise WorkerRunnerError("changed_files must be a list")
        if not isinstance(self.commands_requested, list):
            raise WorkerRunnerError("commands_requested must be a list")
        if not isinstance(self.observations, list):
            raise WorkerRunnerError("observations must be a list")
        if not isinstance(self.uncertainties, list):
            raise WorkerRunnerError("uncertainties must be a list")

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkerResult:
        """Construct and validate WorkerResult from a dictionary."""
        res = cls(
            schema_version=data.get("schema_version", "1.0"),
            batch_id=str(data.get("batch_id", "")),
            task_id=str(data.get("task_id", "")),
            worker_id=str(data.get("worker_id", "")),
            model=str(data.get("model", "")),
            status=str(data.get("status", "completed")),
            summary=str(data.get("summary", "")),
            changed_files=list(data.get("changed_files", [])),
            commands_requested=list(data.get("commands_requested", [])),
            observations=list(data.get("observations", [])),
            uncertainties=list(data.get("uncertainties", [])),
            error=data.get("error"),
            usage=dict(data.get("usage", {})),
            duration_ms=float(data.get("duration_ms", 0.0)),
            worktree_id=data.get("worktree_id"),
        )
        res.validate()
        return res


# ---------------------------------------------------------------------------
# Retry Policy Helper
# ---------------------------------------------------------------------------

NON_RETRYABLE_PATTERNS = [
    "scope violation",
    "write_scope",
    "forbidden command",
    "privileged",
    "destructive",
    "permission",
    "model not found",
    "model unavailable",
    "validation error",
    "invalid project",
    "lock conflict",
    "resource rejected",
]


def is_retryable_error(error_message: Optional[str]) -> bool:
    """Determine whether an error qualifies for automatic bounded retry."""
    if not error_message:
        return True
    lower = error_message.lower()
    for pattern in NON_RETRYABLE_PATTERNS:
        if pattern in lower:
            return False
    return True


# ---------------------------------------------------------------------------
# Worker Runner
# ---------------------------------------------------------------------------

class WorkerRunner:
    """Logical worker runtime executing tasks using Ollama and execution engine."""

    def __init__(
        self,
        workspace_manager: Optional[WorkspaceManager] = None,
        execution_engine: Optional[ExecutionEngine] = None,
        context_manager: Optional[ContextManager] = None,
        model_router: Optional[ModelRouter] = None,
        resource_monitor: Optional[ResourceMonitor] = None,
        ollama_endpoint: str = "http://localhost:11434",
        ollama_client_fn: Optional[Callable[[Dict[str, Any], int], Dict[str, Any]]] = None,
    ):
        self.workspace_manager = workspace_manager
        self.execution_engine = execution_engine
        self.context_manager = context_manager or ContextManager()
        self.model_router = model_router or ModelRouter(endpoint=ollama_endpoint)
        self.resource_monitor = resource_monitor
        self.ollama_endpoint = ollama_endpoint.rstrip("/")
        self.ollama_client_fn = ollama_client_fn

    def _call_ollama(self, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """Send inference request to Ollama /api/generate."""
        if self.ollama_client_fn is not None:
            return self.ollama_client_fn(payload, timeout)

        url = f"{self.ollama_endpoint}/api/generate"
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_bytes = resp.read()
            return json.loads(resp_bytes.decode("utf-8"))

    def _resolve_base_commit(self, workspace_path: Path, task: TaskSpec) -> Optional[str]:
        """Find base commit hash if workspace is inside a git repo."""
        if task.base_commit:
            return task.base_commit
        try:
            head_path = workspace_path / ".git" / "HEAD"
            if head_path.is_file():
                content = head_path.read_text(encoding="utf-8").strip()
                if content.startswith("ref: "):
                    ref_rel = content[5:]
                    ref_file = workspace_path / ".git" / ref_rel
                    if ref_file.is_file():
                        return ref_file.read_text(encoding="utf-8").strip()
                else:
                    return content
        except Exception:
            pass
        return None

    def execute_task(
        self,
        task: TaskSpec,
        workspace_path: Path,
        orchestrator: Optional[TaskOrchestrator] = None,
        worker_id: Optional[str] = None,
        use_worktree: bool = False,
    ) -> WorkerResult:
        """Execute a single bounded task.

        Pipeline:
        1. Optionally provision an isolated Git worktree if use_worktree=True.
        2. Acquire write-scope locks (via orchestrator).
        3. Query Model Router for role/preferred model.
        4. Query Resource Monitor for inference admission.
        5. Build bounded context with Context Manager.
        6. Invoke Ollama model.
        7. Validate model response JSON schema.
        8. Enforce read/write scope containment on all file writes.
        9. Execute allowed test/build commands via Secure Execution Engine.
        10. Persist results and release locks safely.
        """
        start_time = time.time()
        effective_worker_id = worker_id or f"worker-{task.role}-{task.task_id}"
        workspace_path = Path(workspace_path).resolve()
        base_commit = self._resolve_base_commit(workspace_path, task)

        effective_workspace = workspace_path
        assigned_worktree_id = base_commit

        if use_worktree:
            from core.validator import WorktreeManager
            wt_mgr = WorktreeManager(workspace=workspace_path)
            if wt_mgr.is_git_repo(workspace_path):
                wt_dir = workspace_path / ".swan" / "worktrees" / task.task_id
                branch_name = f"swan/{task.batch_id}/{task.task_id}"
                ok, res_path = wt_mgr.create_worktree(wt_dir, branch_name=branch_name, base_commit=task.base_commit)
                if ok:
                    effective_workspace = wt_dir
                    assigned_worktree_id = str(wt_dir)

        # 1. Acquire write locks if orchestrator is provided
        if orchestrator is not None:
            lock_ok = orchestrator.acquire_task_locks(task.task_id)
            if not lock_ok:
                duration_ms = (time.time() - start_time) * 1000
                return WorkerResult(
                    schema_version="1.0",
                    batch_id=task.batch_id,
                    task_id=task.task_id,
                    worker_id=effective_worker_id,
                    status="failed",
                    summary="Lock acquisition failed due to write-scope conflict",
                    error="Lock conflict: Another task holds an overlapping write-scope lock",
                    duration_ms=duration_ms,
                    worktree_id=assigned_worktree_id,
                )

        # 2. Select Model
        model_info: Optional[ModelInfo] = None
        try:
            if task.model_preference:
                model_info = self.model_router.select_explicit_model(task.model_preference)
            else:
                model_info = self.model_router.select_model_for_role(task.role)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            if orchestrator is not None:
                orchestrator.release_task_locks(task.task_id)
            return WorkerResult(
                schema_version="1.0",
                batch_id=task.batch_id,
                task_id=task.task_id,
                worker_id=effective_worker_id,
                status="failed",
                summary="Model selection failed",
                error=f"Model unavailable: {e}",
                duration_ms=duration_ms,
                worktree_id=base_commit,
            )

        model_name = model_info.name if model_info else "unknown"

        # 3. Resource Admission
        inference_acquired = False
        if self.resource_monitor is not None:
            try:
                decision: AdmissionDecision = self.resource_monitor.evaluate_admission(
                    model_name=model_info.name,
                    context_tokens=4096,
                )
                if decision.status == AdmissionStatus.WAIT:
                    duration_ms = (time.time() - start_time) * 1000
                    if orchestrator is not None:
                        orchestrator.release_task_locks(task.task_id)
                    return WorkerResult(
                        schema_version="1.0",
                        batch_id=task.batch_id,
                        task_id=task.task_id,
                        worker_id=effective_worker_id,
                        model=model_info.name,
                        status="resource_wait",
                        summary="Resource admission deferred",
                        error=f"Resource wait: {decision.reason}",
                        duration_ms=duration_ms,
                        worktree_id=base_commit,
                    )
                elif decision.status == AdmissionStatus.REJECT:
                    duration_ms = (time.time() - start_time) * 1000
                    if orchestrator is not None:
                        orchestrator.release_task_locks(task.task_id)
                    return WorkerResult(
                        schema_version="1.0",
                        batch_id=task.batch_id,
                        task_id=task.task_id,
                        worker_id=effective_worker_id,
                        model=model_info.name,
                        status="failed",
                        summary="Resource admission rejected",
                        error=f"Resource rejected: {decision.reason}",
                        duration_ms=duration_ms,
                        worktree_id=base_commit,
                    )
                self.resource_monitor.acquire_inference()
                inference_acquired = True
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                if orchestrator is not None:
                    orchestrator.release_task_locks(task.task_id)
                return WorkerResult(
                    schema_version="1.0",
                    batch_id=task.batch_id,
                    task_id=task.task_id,
                    worker_id=effective_worker_id,
                    model=model_info.name,
                    status="failed",
                    summary="Resource monitor error",
                    error=f"Resource monitor evaluation failed: {e}",
                    duration_ms=duration_ms,
                    worktree_id=base_commit,
                )

        try:
            # 4. Prepare Execution Engine
            engine = self.execution_engine
            if engine is None:
                engine = ExecutionEngine(
                    workspace=effective_workspace,
                    approval_mode="prompt",
                    timeout=task.timeout,
                )

            # 5. Build Context with ContextManager
            role_enum = SessionRole.WORKER

            session = self.context_manager.create_session(
                session_id=f"worker-{task.task_id}",
                project_id=task.project_id,
                role=role_enum,
                context_budget=min(model_info.context_window or 8192, 16384),
            )
            session.set_system_instructions(
                "You are an autonomous SWAN worker. Output valid JSON only with keys: "
                "summary (str), files_to_modify (list of {path, content}), "
                "commands_to_run (list of str), observations (list of str), uncertainties (list of str)."
            )
            session.set_objective(f"{task.title}\n{task.description}")
            session.set_task_specification(
                f"Task ID: {task.task_id}\n"
                f"Read Scope: {task.read_scope}\n"
                f"Write Scope: {task.write_scope}\n"
                f"Acceptance Criteria: {task.acceptance_criteria}\n"
                f"Validation Requirements: {task.validation_requirements}"
            )

            # Read files in read_scope and add as observations
            for read_path in task.read_scope:
                file_full = effective_workspace / read_path
                if file_full.is_file():
                    read_ok, read_val = engine.read_file(read_path)
                    if read_ok:
                        session.add_observation(
                            ObservationType.FILE,
                            f"=== File: {read_path} ===\n{read_val}",
                            priority=2,
                        )

            prompt = session.build_prompt()

            # 6. Call Ollama
            ollama_payload = {
                "model": model_info.name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.2,
                    "num_ctx": model_info.context_window or 8192,
                },
            }

            try:
                raw_resp = self._call_ollama(ollama_payload, timeout=task.timeout)
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                return WorkerResult(
                    schema_version="1.0",
                    batch_id=task.batch_id,
                    task_id=task.task_id,
                    worker_id=effective_worker_id,
                    model=model_info.name,
                    status="failed",
                    summary="Ollama inference request failed",
                    error=f"Ollama failure: {e}",
                    duration_ms=duration_ms,
                    worktree_id=assigned_worktree_id,
                )

            # 7. Parse Model Response
            model_text = raw_resp.get("response", "")
            usage = {
                "total_duration": raw_resp.get("total_duration", 0),
                "load_duration": raw_resp.get("load_duration", 0),
                "prompt_eval_count": raw_resp.get("prompt_eval_count", 0),
                "eval_count": raw_resp.get("eval_count", 0),
            }

            try:
                parsed_output = json.loads(model_text)
                if isinstance(parsed_output, str):
                    parsed_output = json.loads(parsed_output)
                if not isinstance(parsed_output, dict):
                    raise ValueError("Model output root must be a JSON object")
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                return WorkerResult(
                    schema_version="1.0",
                    batch_id=task.batch_id,
                    task_id=task.task_id,
                    worker_id=effective_worker_id,
                    model=model_info.name,
                    status="failed",
                    summary="Failed to parse model response as valid JSON",
                    error=f"Malformed model response JSON: {e}",
                    usage=usage,
                    duration_ms=duration_ms,
                    worktree_id=base_commit,
                )

            summary = str(parsed_output.get("summary", "")).strip()
            files_to_modify = parsed_output.get("files_to_modify", [])
            commands_to_run = parsed_output.get("commands_to_run", [])
            observations = list(parsed_output.get("observations", []))
            uncertainties = list(parsed_output.get("uncertainties", []))

            if not isinstance(files_to_modify, list):
                files_to_modify = []
            if not isinstance(commands_to_run, list):
                commands_to_run = []

            # 8. Apply file modifications with strict write-scope validation
            changed_files: List[str] = []
            for item in files_to_modify:
                if not isinstance(item, dict):
                    continue
                file_rel = item.get("path")
                content = item.get("content", "")
                if not file_rel or not isinstance(file_rel, str):
                    continue

                # Check write scope containment
                if not is_path_in_scope(file_rel, task.write_scope):
                    duration_ms = (time.time() - start_time) * 1000
                    return WorkerResult(
                        schema_version="1.0",
                        batch_id=task.batch_id,
                        task_id=task.task_id,
                        worker_id=effective_worker_id,
                        model=model_info.name,
                        status="failed",
                        summary="Task write-scope violation",
                        changed_files=changed_files,
                        observations=observations,
                        uncertainties=uncertainties,
                        error=f"Scope violation: File '{file_rel}' is not in authorized write_scope: {task.write_scope}",
                        usage=usage,
                        duration_ms=duration_ms,
                        worktree_id=assigned_worktree_id,
                    )

                # Write through ExecutionEngine
                write_ok, write_val = engine.write_file(file_rel, content, overwrite_approved=True)
                if not write_ok:
                    duration_ms = (time.time() - start_time) * 1000
                    return WorkerResult(
                        schema_version="1.0",
                        batch_id=task.batch_id,
                        task_id=task.task_id,
                        worker_id=effective_worker_id,
                        model=model_info.name,
                        status="failed",
                        summary=f"Failed to write file '{file_rel}'",
                        changed_files=changed_files,
                        error=f"File write failed: {write_val}",
                        usage=usage,
                        duration_ms=duration_ms,
                        worktree_id=assigned_worktree_id,
                    )
                changed_files.append(file_rel)

            # 9. Execute requested commands with permission classification
            commands_requested: List[Dict[str, Any]] = []
            for cmd in commands_to_run:
                if not isinstance(cmd, str) or not cmd.strip():
                    continue

                try:
                    cmd_args = validate_command_string(cmd)
                    permission = classify_command(cmd_args)
                except Exception:
                    permission = PERMISSION_UNKNOWN

                # Workers may ONLY execute test/build and read-only commands
                if permission in (PERMISSION_READ_ONLY, PERMISSION_TEST_BUILD):
                    exec_res = engine.execute_command(cmd, timeout=task.timeout, require_approval=False, approved=True)
                    commands_requested.append({
                        "command": cmd,
                        "permission": permission,
                        "executed": True,
                        "returncode": exec_res.get("returncode"),
                        "stdout": exec_res.get("stdout"),
                        "stderr": exec_res.get("stderr"),
                    })
                    if exec_res.get("stdout"):
                        observations.append(f"Command '{cmd}' stdout:\n{exec_res.get('stdout')}")
                    if exec_res.get("stderr"):
                        observations.append(f"Command '{cmd}' stderr:\n{exec_res.get('stderr')}")
                else:
                    # Forbidden command requested by worker
                    commands_requested.append({
                        "command": cmd,
                        "permission": permission,
                        "executed": False,
                        "blocked": True,
                        "reason": f"Workers are not permitted to execute {permission} commands",
                    })

                    # If destructive or privileged, fail the task immediately
                    if permission in (PERMISSION_DESTRUCTIVE, PERMISSION_PRIVILEGED):
                        duration_ms = (time.time() - start_time) * 1000
                        return WorkerResult(
                            schema_version="1.0",
                            batch_id=task.batch_id,
                            task_id=task.task_id,
                            worker_id=effective_worker_id,
                            model=model_info.name,
                            status="failed",
                            summary="Forbidden command requested by worker",
                            changed_files=changed_files,
                            commands_requested=commands_requested,
                            observations=observations,
                            uncertainties=uncertainties,
                            error=f"Forbidden command requested: '{cmd}' ({permission})",
                            usage=usage,
                            duration_ms=duration_ms,
                            worktree_id=assigned_worktree_id,
                        )

            # 10. Worker Success
            duration_ms = (time.time() - start_time) * 1000
            return WorkerResult(
                schema_version="1.0",
                batch_id=task.batch_id,
                task_id=task.task_id,
                worker_id=effective_worker_id,
                model=model_info.name,
                status="completed",
                summary=summary,
                changed_files=changed_files,
                commands_requested=commands_requested,
                observations=observations,
                uncertainties=uncertainties,
                error=None,
                usage=usage,
                duration_ms=duration_ms,
                worktree_id=assigned_worktree_id,
            )

        finally:
            if inference_acquired and self.resource_monitor is not None:
                self.resource_monitor.release_inference()
            if orchestrator is not None:
                orchestrator.release_task_locks(task.task_id)

    def execute_batch(
        self,
        batch_id: str,
        orchestrator: TaskOrchestrator,
        workspace_path: Path,
        use_worktrees: bool = False,
    ) -> Dict[str, WorkerResult]:
        """Execute a batch of tasks sequentially respecting DAG dependencies and locks.

        Concurrency is strictly ONE active inference at a time.
        """
        results: Dict[str, WorkerResult] = {}
        workspace_path = Path(workspace_path).resolve()

        while True:
            # 1. Advance dependency states
            orchestrator.evaluate_dependencies_and_advance(batch_id=batch_id)

            # 2. Get next ready task
            ready_tasks = orchestrator.get_ready_tasks(batch_id=batch_id)
            if not ready_tasks:
                # No more ready tasks. Check if any are still running or waiting
                all_tasks = orchestrator.list_tasks(batch_id=batch_id)
                active_states = {TaskState.READY, TaskState.RUNNING, TaskState.RETRY_PENDING}
                has_active = any(t.state in active_states for t in all_tasks)
                if not has_active:
                    break
                else:
                    # Tasks might be waiting on dependencies or resources
                    break

            # Pick next ready task (sequential execution)
            current_task = ready_tasks[0]

            # Transition task to RUNNING
            orchestrator.transition_task(
                current_task.task_id,
                TaskState.RUNNING,
                reason="Starting worker execution",
            )

            # Execute worker
            result = self.execute_task(
                task=current_task,
                workspace_path=workspace_path,
                orchestrator=orchestrator,
                use_worktree=use_worktrees,
            )
            results[current_task.task_id] = result
            orchestrator.record_worker_result(result)

            # Evaluate worker outcome
            if result.status == "completed":
                orchestrator.transition_task(
                    current_task.task_id,
                    TaskState.WORKER_COMPLETED,
                    reason=result.summary or "Worker completed actions",
                )
                orchestrator.transition_task(
                    current_task.task_id,
                    TaskState.VALIDATION_PENDING,
                    reason="Forwarded for future validation",
                )
            elif result.status == "resource_wait":
                orchestrator.transition_task(
                    current_task.task_id,
                    TaskState.WAITING_RESOURCE,
                    reason=result.error or "Resource monitor deferral",
                )
                # Break batch execution loop when resource admission defers
                break
            else:  # "failed"
                orchestrator.transition_task(
                    current_task.task_id,
                    TaskState.WORKER_FAILED,
                    reason=result.error or "Worker execution failed",
                )
                # Check retryability
                retryable = is_retryable_error(result.error)
                task_rec = orchestrator.get_task(current_task.task_id)
                retry_count = task_rec.retry_count if task_rec else 0

                if retryable and retry_count < current_task.retry_limits:
                    orchestrator.transition_task(
                        current_task.task_id,
                        TaskState.RETRY_PENDING,
                        reason=f"Scheduling retry ({retry_count + 1}/{current_task.retry_limits})",
                    )
                    orchestrator.transition_task(
                        current_task.task_id,
                        TaskState.READY,
                        reason="Ready for retry attempt",
                    )
                else:
                    orchestrator.transition_task(
                        current_task.task_id,
                        TaskState.BLOCKED,
                        reason="Retries exhausted or non-retryable failure",
                    )
                    # Propagate block to downstream tasks
                    orchestrator.evaluate_dependencies_and_advance(batch_id=batch_id)

        return results
