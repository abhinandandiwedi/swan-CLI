"""Manager Verdict Protocol for SWAN-HIVE.

Implements the formal verdict protocol (ACCEPT, REQUEST_REVISION, RETRY, REJECT)
allowing External Managers (Claude, Gemini, etc.) to review and steer task batches.
Enforces security guardrails ensuring Manager verdicts can NEVER override
security validations or scope violations.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Union

from core.task_orchestrator import (
    TaskOrchestrator,
    TaskRecord,
    TaskState,
    TaskOrchestratorError,
)
from core.validator import (
    ValidationReport,
    ValidationStatus,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class VerdictError(Exception):
    """Base exception for Manager Verdict protocol errors."""
    pass


class VerdictValidationError(VerdictError):
    """Raised when verdict schema or task state is invalid."""
    pass


class SecurityOverrideError(VerdictError):
    """Raised when a Manager verdict attempts to override security or scope violations."""
    pass


class RetryLimitExceededError(VerdictError):
    """Raised when revision or retry is requested on a task that exhausted its retry limit."""
    pass


# ---------------------------------------------------------------------------
# Verdict Types & Contract
# ---------------------------------------------------------------------------

class VerdictType(str, Enum):
    """Allowed verdict decisions from an External Manager."""
    ACCEPT = "ACCEPT"
    REQUEST_REVISION = "REQUEST_REVISION"
    RETRY = "RETRY"
    REJECT = "REJECT"


@dataclass
class ManagerVerdict:
    """Structured contract for an External Manager review verdict."""
    batch_id: str
    task_id: str
    verdict: VerdictType
    reason: str
    schema_version: str = "1.0"
    verdict_id: str = field(default_factory=lambda: f"vrd-{uuid.uuid4().hex[:12]}")
    feedback: Optional[str] = None
    manager_id: str = "external-manager"
    timestamp: float = field(default_factory=time.time)

    def validate(self) -> None:
        """Validate verdict fields and types."""
        if not self.schema_version:
            raise VerdictValidationError("schema_version is required")
        if not self.batch_id or not isinstance(self.batch_id, str):
            raise VerdictValidationError("batch_id must be a non-empty string")
        if not self.task_id or not isinstance(self.task_id, str):
            raise VerdictValidationError("task_id must be a non-empty string")
        if not isinstance(self.verdict, VerdictType):
            try:
                self.verdict = VerdictType(str(self.verdict).upper())
            except ValueError:
                valid = [v.value for v in VerdictType]
                raise VerdictValidationError(
                    f"Invalid verdict '{self.verdict}'. Must be one of: {valid}"
                )
        if not self.reason or not isinstance(self.reason, str) or not self.reason.strip():
            raise VerdictValidationError("reason must be a non-empty explanation for the verdict")

    def to_dict(self) -> Dict[str, Any]:
        """Convert verdict to plain dictionary."""
        d = dataclasses.asdict(self)
        d["verdict"] = self.verdict.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ManagerVerdict:
        """Construct and validate ManagerVerdict from dictionary."""
        raw_verdict = data.get("verdict", "")
        if isinstance(raw_verdict, str):
            try:
                verdict_enum = VerdictType(raw_verdict.upper())
            except ValueError:
                valid = [v.value for v in VerdictType]
                raise VerdictValidationError(f"Invalid verdict '{raw_verdict}'. Must be one of: {valid}")
        elif isinstance(raw_verdict, VerdictType):
            verdict_enum = raw_verdict
        else:
            raise VerdictValidationError("verdict must be specified")

        v = cls(
            schema_version=data.get("schema_version", "1.0"),
            verdict_id=data.get("verdict_id") or f"vrd-{uuid.uuid4().hex[:12]}",
            batch_id=str(data.get("batch_id", "")),
            task_id=str(data.get("task_id", "")),
            verdict=verdict_enum,
            reason=str(data.get("reason", "")),
            feedback=data.get("feedback"),
            manager_id=str(data.get("manager_id", "external-manager")),
            timestamp=float(data.get("timestamp", time.time())),
        )
        v.validate()
        return v


# ---------------------------------------------------------------------------
# Verdict Service & Security Enforcer
# ---------------------------------------------------------------------------

class ManagerVerdictService:
    """Enforces verdict processing and guarantees security validation supremacy."""

    @staticmethod
    def apply_verdict(
        verdict: ManagerVerdict,
        orchestrator: TaskOrchestrator,
    ) -> TaskRecord:
        """Apply a Manager verdict to a task in the Orchestrator.

        Security Constraints:
        - Manager verdict CANNOT override security or scope violations.
        - If task had out-of-scope modifications or security failures, ACCEPT is rejected.
        - Retries are strictly bounded by task.retry_limits.
        """
        verdict.validate()

        task_record = orchestrator.get_task(verdict.task_id)
        if not task_record:
            raise VerdictValidationError(f"Task '{verdict.task_id}' not found in orchestrator")

        if task_record.batch_id != verdict.batch_id:
            raise VerdictValidationError(
                f"Task '{verdict.task_id}' belongs to batch '{task_record.batch_id}', not '{verdict.batch_id}'"
            )

        # Verdicts are only allowed on completed or validation_pending tasks
        allowed_states = {
            TaskState.VALIDATION_PENDING,
            TaskState.WORKER_COMPLETED,
            TaskState.WORKER_FAILED,
        }
        if task_record.state not in allowed_states:
            raise VerdictValidationError(
                f"Cannot apply verdict to task in state '{task_record.state.value}'. "
                f"Allowed states: {[s.value for s in allowed_states]}"
            )

        # Fetch latest validation report if any
        latest_report_dict = orchestrator.get_latest_validation_report(verdict.task_id)
        validation_report = (
            ValidationReport.from_dict(latest_report_dict["report"])
            if latest_report_dict
            else None
        )

        # -------------------------------------------------------------------
        # Security Guardrail: Verdict Supremacy Check
        # -------------------------------------------------------------------
        if verdict.verdict == VerdictType.ACCEPT:
            if validation_report is not None:
                if validation_report.out_of_scope_changes:
                    raise SecurityOverrideError(
                        f"SECURITY VIOLATION: Manager cannot ACCEPT task '{verdict.task_id}' because "
                        f"Validator detected out-of-scope changes: {validation_report.out_of_scope_changes}"
                    )
                if validation_report.validation_status != ValidationStatus.PASSED.value:
                    raise SecurityOverrideError(
                        f"SECURITY VIOLATION: Manager cannot ACCEPT task '{verdict.task_id}' with "
                        f"validation status '{validation_report.validation_status}' (checks failed: "
                        f"{[c.name for c in validation_report.checks if not c.passed]})"
                    )

            # Record ACCEPT verdict event
            orchestrator.conn.execute("""
                INSERT INTO events (batch_id, task_id, timestamp, event_type, old_state, new_state, message, metadata_json)
                VALUES (?, ?, ?, 'VERDICT_ACCEPT', ?, ?, ?, ?)
            """, (
                verdict.batch_id,
                verdict.task_id,
                time.time(),
                task_record.state.value,
                task_record.state.value,
                f"Manager ACCEPT: {verdict.reason}",
                json.dumps(verdict.to_dict()),
            ))

        # -------------------------------------------------------------------
        # REQUEST_REVISION / RETRY Handling
        # -------------------------------------------------------------------
        elif verdict.verdict in (VerdictType.REQUEST_REVISION, VerdictType.RETRY):
            # Enforce retry bounds
            if task_record.retry_count >= task_record.spec.retry_limits:
                orchestrator.transition_task(
                    verdict.task_id,
                    TaskState.BLOCKED,
                    reason=f"Manager requested revision, but retry limit ({task_record.spec.retry_limits}) is exhausted",
                    metadata={"verdict": verdict.to_dict()},
                )
                raise RetryLimitExceededError(
                    f"Retry limit ({task_record.spec.retry_limits}) exhausted for task '{verdict.task_id}'"
                )

            # Transition: VALIDATION_PENDING (or WORKER_FAILED) -> RETRY_PENDING -> READY
            transition_reason = f"Manager {verdict.verdict.value}: {verdict.reason}"
            if verdict.feedback:
                transition_reason += f" (Feedback: {verdict.feedback})"

            orchestrator.transition_task(
                verdict.task_id,
                TaskState.RETRY_PENDING,
                reason=transition_reason,
                metadata={"verdict": verdict.to_dict()},
            )
            orchestrator.transition_task(
                verdict.task_id,
                TaskState.READY,
                reason="Task queued for retry execution",
            )

        # -------------------------------------------------------------------
        # REJECT Handling
        # -------------------------------------------------------------------
        elif verdict.verdict == VerdictType.REJECT:
            orchestrator.transition_task(
                verdict.task_id,
                TaskState.BLOCKED,
                reason=f"Manager REJECT: {verdict.reason}",
                metadata={"verdict": verdict.to_dict()},
            )
            # Propagate block down the DAG
            orchestrator.evaluate_dependencies_and_advance(batch_id=verdict.batch_id)

        # Persist verdict to database if table exists
        try:
            with orchestrator.conn:
                orchestrator.conn.execute("""
                    CREATE TABLE IF NOT EXISTS manager_verdicts (
                        verdict_id TEXT PRIMARY KEY,
                        batch_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        verdict TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        feedback TEXT NOT NULL DEFAULT '',
                        manager_id TEXT NOT NULL DEFAULT '',
                        created_at REAL NOT NULL,
                        FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
                    );
                """)
                orchestrator.conn.execute("""
                    INSERT OR REPLACE INTO manager_verdicts (verdict_id, batch_id, task_id, verdict, reason, feedback, manager_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    verdict.verdict_id,
                    verdict.batch_id,
                    verdict.task_id,
                    verdict.verdict.value,
                    verdict.reason,
                    verdict.feedback or "",
                    verdict.manager_id,
                    verdict.timestamp,
                ))
        except Exception:
            pass

        updated_task = orchestrator.get_task(verdict.task_id)
        return updated_task
