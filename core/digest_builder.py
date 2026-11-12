"""Compact Manager Digest Builder for SWAN-HIVE.

Aggregates batch execution status, independent validation reports, diff statistics,
and actionable manager review items into high-density, compact digests without
leaking lengthy logs or full source diffs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Union

from core.task_orchestrator import (
    TaskOrchestrator,
    TaskRecord,
    TaskState,
)
from core.validator import (
    ValidationReport,
    Validator,
)


@dataclass
class BatchDigest:
    """Compact summary of a batch execution for External Manager review."""
    batch_id: str
    project_id: str
    base_commit: Optional[str] = None
    total_tasks: int = 0
    passed_count: int = 0
    failed_count: int = 0
    blocked_count: int = 0
    retried_count: int = 0
    validation_status: str = "passed"  # "passed", "failed", "blocked", "partial"
    short_task_summaries: List[Dict[str, Any]] = field(default_factory=list)
    diff_statistics: Dict[str, Any] = field(default_factory=dict)
    important_error_lines: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    manager_actions_required: List[str] = field(default_factory=list)
    evidence_references: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert digest to plain dictionary."""
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize digest to JSON formatted string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Format a compact, high-density Markdown summary."""
        lines = [
            f"# SWAN Batch Execution Digest: `{self.batch_id}`",
            f"**Project:** `{self.project_id}` | **Status:** `{self.validation_status.upper()}` | **Base Commit:** `{self.base_commit or 'N/A'}`",
            "",
            "## 1. Metrics",
            f"- **Total Tasks:** {self.total_tasks}",
            f"- **Passed / Validated:** {self.passed_count}",
            f"- **Failed:** {self.failed_count}",
            f"- **Blocked:** {self.blocked_count}",
            f"- **Retried:** {self.retried_count}",
            "",
            "## 2. Diff Summary",
            f"- **Files Changed:** {self.diff_statistics.get('total_files_changed', 0)}",
            f"- **Insertions:** +{self.diff_statistics.get('total_insertions', 0)}",
            f"- **Deletions:** -{self.diff_statistics.get('total_deletions', 0)}",
            f"- **Whitespace Warnings:** {self.diff_statistics.get('total_whitespace_errors', 0)}",
            "",
            "## 3. Tasks Overview",
            "| Task ID | Role | Title | State | Validation | Changed Files |",
            "|---|---|---|---|---|---|",
        ]

        for s in self.short_task_summaries:
            t_id = s.get("task_id", "")
            role = s.get("role", "")
            title = s.get("title", "")[:30]
            state = s.get("state", "")
            val = s.get("validation", "N/A")
            files = ", ".join(s.get("changed_files", [])) or "None"
            lines.append(f"| `{t_id}` | {role} | {title} | `{state}` | `{val}` | {files} |")

        if self.important_error_lines:
            lines.extend([
                "",
                "## 4. Key Error Log Snippets",
            ])
            for err in self.important_error_lines:
                lines.append(f"- ⚠️ {err}")

        if self.warnings:
            lines.extend([
                "",
                "## 5. Warnings",
            ])
            for w in self.warnings:
                lines.append(f"- ℹ️ {w}")

        lines.extend([
            "",
            "## 6. Actions Required from Manager",
        ])
        if self.manager_actions_required:
            for act in self.manager_actions_required:
                lines.append(f"- [ ] **ACTION:** {act}")
        else:
            lines.append("- None. Ready for subsequent pipeline stage.")

        return "\n".join(lines)


class DigestBuilder:
    """Builds compact batch digests and detailed task records for Manager review."""

    def __init__(
        self,
        orchestrator: TaskOrchestrator,
        validator: Optional[Validator] = None,
        workspace_path: Optional[Path] = None,
    ):
        self.orchestrator = orchestrator
        self.validator = validator or Validator()
        self.workspace_path = Path(workspace_path).resolve() if workspace_path else None

    def build_digest(
        self,
        batch_id: str,
        validation_reports: Optional[Dict[str, ValidationReport]] = None,
        base_commit: Optional[str] = None,
    ) -> BatchDigest:
        """Construct compact batch-level summary digest.

        Aggregates tasks, results, validation reports, and diff stats.
        """
        tasks = self.orchestrator.list_tasks(batch_id=batch_id)
        project_id = tasks[0].project_id if tasks else "unknown"

        total_tasks = len(tasks)
        passed_count = 0
        failed_count = 0
        blocked_count = 0
        retried_count = 0

        short_summaries: List[Dict[str, Any]] = []
        error_lines: List[str] = []
        all_warnings: List[str] = []
        manager_actions: List[str] = []

        total_files = 0
        total_ins = 0
        total_dels = 0
        total_ws_errors = 0

        for t in tasks:
            # Check retries
            if t.retry_count > 0:
                retried_count += t.retry_count

            # Task status accounting
            if t.state in (TaskState.VALIDATION_PENDING, TaskState.WORKER_COMPLETED):
                passed_count += 1
            elif t.state == TaskState.WORKER_FAILED:
                failed_count += 1
            elif t.state == TaskState.BLOCKED:
                blocked_count += 1

            # Get validation report
            val_report = None
            if validation_reports and t.task_id in validation_reports:
                val_report = validation_reports[t.task_id]
            else:
                db_val = self.orchestrator.get_latest_validation_report(t.task_id)
                if db_val:
                    val_report = ValidationReport.from_dict(db_val["report"])

            # Get worker result
            worker_results = self.orchestrator.get_worker_results(t.task_id)
            latest_worker = worker_results[-1]["result"] if worker_results else {}

            changed_files = []
            if val_report:
                changed_files = val_report.diff_stats.changed_files
                total_files += val_report.diff_stats.files_changed
                total_ins += val_report.diff_stats.insertions
                total_dels += val_report.diff_stats.deletions
                total_ws_errors += val_report.diff_stats.whitespace_errors
                all_warnings.extend(val_report.warnings)

                for check in val_report.checks:
                    for err in check.errors:
                        if err not in error_lines:
                            error_lines.append(f"[{t.task_id}] {err}")

                if val_report.out_of_scope_changes:
                    manager_actions.append(
                        f"CRITICAL: Task '{t.task_id}' modified out-of-scope files {val_report.out_of_scope_changes}. Investigate security violation."
                    )
            elif latest_worker:
                changed_files = latest_worker.get("changed_files", [])
                if latest_worker.get("error"):
                    error_lines.append(f"[{t.task_id}] {latest_worker.get('error')}")

            # Base commit fallback
            if not base_commit and t.spec.base_commit:
                base_commit = t.spec.base_commit

            short_summaries.append({
                "task_id": t.task_id,
                "role": t.spec.role,
                "title": t.spec.title,
                "state": t.state.value,
                "validation": val_report.validation_status if val_report else "not_run",
                "summary": latest_worker.get("summary", "")[:120],
                "changed_files": changed_files,
            })

        # Determine overall validation status
        if blocked_count > 0:
            overall_status = "blocked"
        elif failed_count > 0:
            overall_status = "failed"
        elif passed_count == total_tasks and total_tasks > 0:
            overall_status = "passed"
        else:
            overall_status = "partial"

        # Manager actions required
        if blocked_count > 0:
            manager_actions.append(f"Review {blocked_count} blocked task(s) and resolve dependencies or spec violations.")
        if failed_count > 0:
            manager_actions.append(f"Review {failed_count} failed task(s) that exhausted retries.")
        if overall_status == "passed" and total_tasks > 0:
            manager_actions.append("Review aggregated git diff and grant human approval before merge.")

        # Deduplicate warnings
        unique_warnings = list(dict.fromkeys(all_warnings))[:10]
        # Deduplicate and cap error lines
        unique_errors = list(dict.fromkeys(error_lines))[:10]

        diff_stats = {
            "total_files_changed": total_files,
            "total_insertions": total_ins,
            "total_deletions": total_dels,
            "total_whitespace_errors": total_ws_errors,
        }

        evidence_refs = {
            "batch_id": batch_id,
            "task_count": total_tasks,
            "tasks": [t.task_id for t in tasks],
        }

        return BatchDigest(
            batch_id=batch_id,
            project_id=project_id,
            base_commit=base_commit,
            total_tasks=total_tasks,
            passed_count=passed_count,
            failed_count=failed_count,
            blocked_count=blocked_count,
            retried_count=retried_count,
            validation_status=overall_status,
            short_task_summaries=short_summaries,
            diff_statistics=diff_stats,
            important_error_lines=unique_errors,
            warnings=unique_warnings,
            manager_actions_required=manager_actions,
            evidence_references=evidence_refs,
            timestamp=time.time(),
        )

    def get_task_detail(self, task_id: str) -> Dict[str, Any]:
        """Obtain detailed view of a specific task on demand."""
        task_rec = self.orchestrator.get_task(task_id)
        if not task_rec:
            return {"error": f"Task '{task_id}' not found"}

        worker_results = self.orchestrator.get_worker_results(task_id)
        validation_reports = self.orchestrator.get_validation_reports(task_id)
        events = self.orchestrator.get_events(task_id=task_id)

        return {
            "task_id": task_rec.task_id,
            "batch_id": task_rec.batch_id,
            "project_id": task_rec.project_id,
            "state": task_rec.state.value,
            "spec": task_rec.spec.to_dict(),
            "retry_count": task_rec.retry_count,
            "created_at": task_rec.created_at,
            "updated_at": task_rec.updated_at,
            "worker_results": [r["result"] for r in worker_results],
            "validation_reports": [r["report"] for r in validation_reports],
            "events": events,
        }
