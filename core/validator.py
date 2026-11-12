"""Validation Engine and Git Worktree Safety for SWAN-HIVE.

Independently verifies worker modifications against Git state and declared task scopes,
executes tests within Secure Execution boundaries, enforces worktree isolation safety,
and produces typed ValidationReports.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.execution_engine import (
    ExecutionEngine,
    classify_command,
    validate_command_string,
    PERMISSION_READ_ONLY,
    PERMISSION_TEST_BUILD,
)
from core.task_orchestrator import (
    TaskSpec,
    is_path_in_scope,
    validate_scope_path,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ValidatorError(Exception):
    """Base exception for Validator errors."""
    pass


class WorktreeError(ValidatorError):
    """Raised when an unsafe Git worktree operation is attempted."""
    pass


class ScopeIntegrityError(ValidatorError):
    """Raised when filesystem modifications escape authorized task boundaries."""
    pass


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class ValidationStatus(str, Enum):
    """Status outcomes for validation evaluation."""
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    NOT_RUN = "not_run"


@dataclass
class ValidationCheckResult:
    """Individual validation check outcome."""
    name: str
    passed: bool
    details: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class DiffStats:
    """Statistics extracted from Git diff."""
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0
    whitespace_errors: int = 0
    changed_files: List[str] = field(default_factory=list)
    raw_diff: str = ""
    diff_truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ValidationReport:
    """Typed, validated report summarizing independent validation findings."""
    batch_id: str
    task_id: str
    validation_status: str  # "passed", "failed", "blocked", "not_run"
    schema_version: str = "1.0"
    checks: List[ValidationCheckResult] = field(default_factory=list)
    diff_stats: DiffStats = field(default_factory=DiffStats)
    out_of_scope_changes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    retry_recommended: bool = False
    retry_feedback: Optional[str] = None
    evidence_references: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        data["checks"] = [c.to_dict() if hasattr(c, "to_dict") else c for c in self.checks]
        data["diff_stats"] = self.diff_stats.to_dict() if hasattr(self.diff_stats, "to_dict") else self.diff_stats
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidationReport:
        checks_data = data.get("checks", [])
        checks = [
            ValidationCheckResult(**c) if isinstance(c, dict) else c
            for c in checks_data
        ]
        diff_data = data.get("diff_stats", {})
        diff_stats = DiffStats(**diff_data) if isinstance(diff_data, dict) else DiffStats()

        return cls(
            schema_version=data.get("schema_version", "1.0"),
            batch_id=str(data.get("batch_id", "")),
            task_id=str(data.get("task_id", "")),
            validation_status=str(data.get("validation_status", "not_run")),
            checks=checks,
            diff_stats=diff_stats,
            out_of_scope_changes=list(data.get("out_of_scope_changes", [])),
            warnings=list(data.get("warnings", [])),
            retry_recommended=bool(data.get("retry_recommended", False)),
            retry_feedback=data.get("retry_feedback"),
            evidence_references=dict(data.get("evidence_references", {})),
            timestamp=float(data.get("timestamp", time.time())),
        )


# ---------------------------------------------------------------------------
# Git Worktree Manager
# ---------------------------------------------------------------------------

class WorktreeManager:
    """Manages Git worktree safety, uncommitted change detection, and merge boundaries."""

    def __init__(self, workspace: Path, execution_engine: Optional[ExecutionEngine] = None):
        self.workspace = Path(workspace).resolve()
        self.engine = execution_engine or ExecutionEngine(workspace=self.workspace, approval_mode="prompt")

    def _run_git(self, args: List[str], cwd: Optional[Path] = None, timeout: int = 30) -> subprocess.CompletedProcess:
        """Run git with shell=False."""
        target_cwd = str(cwd.resolve()) if cwd else str(self.workspace)
        return subprocess.run(
            ["git"] + args,
            cwd=target_cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def is_git_repo(self, path: Optional[Path] = None) -> bool:
        """Check if path is inside a Git repository."""
        target = path or self.workspace
        proc = self._run_git(["rev-parse", "--is-inside-work-tree"], cwd=target)
        return proc.returncode == 0 and proc.stdout.strip() == "true"

    def has_uncommitted_changes(self, path: Optional[Path] = None) -> bool:
        """Detect whether workspace has uncommitted (staged, unstaged, or untracked) changes."""
        target = path or self.workspace
        if not self.is_git_repo(target):
            return False
        proc = self._run_git(["status", "--porcelain=v1"], cwd=target)
        if proc.returncode != 0:
            return False
        return bool(proc.stdout.strip())

    def get_base_commit(self, path: Optional[Path] = None) -> Optional[str]:
        """Obtain current HEAD commit hash."""
        target = path or self.workspace
        if not self.is_git_repo(target):
            return None
        proc = self._run_git(["rev-parse", "HEAD"], cwd=target)
        return proc.stdout.strip() if proc.returncode == 0 else None

    def create_worktree(
        self,
        worktree_path: Path,
        branch_name: str,
        base_commit: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Safely create an isolated Git worktree.

        Fails if target path exists or if git fails.
        """
        worktree_path = Path(worktree_path).resolve()
        if worktree_path.exists():
            return False, f"Target worktree path already exists: {worktree_path}"

        if not self.is_git_repo(self.workspace):
            return False, "Main workspace is not a Git repository"

        cmd = ["worktree", "add", "-b", branch_name, str(worktree_path)]
        if base_commit:
            cmd.append(base_commit)

        proc = self._run_git(cmd)
        if proc.returncode != 0:
            return False, f"Failed to create worktree: {proc.stderr.strip()}"

        return True, str(worktree_path)

    def is_worktree_clean(self, worktree_path: Path) -> bool:
        """Check if worktree has no uncommitted changes."""
        return not self.has_uncommitted_changes(worktree_path)

    def detect_merge_conflicts(
        self,
        branch_or_commit: str,
        target_branch: str = "HEAD",
    ) -> Tuple[bool, List[str]]:
        """Safely detect merge conflicts without altering the active working directory.

        Returns (has_conflicts, conflict_files).
        """
        if not self.is_git_repo(self.workspace):
            return False, []

        # Use git merge-tree to test 3-way merge cleanly
        proc = self._run_git(["merge-tree", target_branch, branch_or_commit])
        if proc.returncode != 0:
            return True, [f"Merge conflict detection failed: {proc.stderr.strip()}"]

        output = proc.stdout
        conflicts: List[str] = []
        if "+<<<<<<<" in output or "changed in both" in output:
            for line in output.splitlines():
                if "changed in both" in line or line.startswith("CONFLICT"):
                    conflicts.append(line.strip())
            return True, conflicts or ["Merge conflicts detected in merge-tree"]

        return False, []

    def remove_worktree(self, worktree_path: Path, force: bool = False) -> Tuple[bool, str]:
        """Remove a worktree safely.

        Refuses to delete if unreviewed changes exist unless force=True.
        """
        worktree_path = Path(worktree_path).resolve()
        if not worktree_path.exists():
            return True, "Worktree path does not exist"

        if not force and self.has_uncommitted_changes(worktree_path):
            return False, "Worktree contains unreviewed modifications; refusing to delete without force approval"

        cmd = ["worktree", "remove"]
        if force:
            cmd.append("--force")
        cmd.append(str(worktree_path))

        proc = self._run_git(cmd)
        if proc.returncode != 0:
            return False, f"Failed to remove worktree: {proc.stderr.strip()}"

        return True, "Worktree removed successfully"

    def requires_human_approval_to_merge(self) -> bool:
        """Human approval is strictly required before any Git merge."""
        return True


# ---------------------------------------------------------------------------
# Validation Engine
# ---------------------------------------------------------------------------

SHORTSTAT_REGEX = re.compile(
    r"(\d+)\s+file[s]?\s+changed(?:,\s+(\d+)\s+insertion[s]?\(\+\))?(?:,\s+(\d+)\s+deletion[s]?\(-\))?"
)


class Validator:
    """Independent validator verifying worker outputs, scope compliance, and diffs."""

    def __init__(
        self,
        execution_engine: Optional[ExecutionEngine] = None,
        max_diff_lines: int = 500,
        max_diff_bytes: int = 50000,
        strict_whitespace: bool = False,
    ):
        self.engine = execution_engine
        self.max_diff_lines = max_diff_lines
        self.max_diff_bytes = max_diff_bytes
        self.strict_whitespace = strict_whitespace

    def _run_git(self, args: List[str], cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess:
        """Execute git command safely without shell=True."""
        return subprocess.run(
            ["git"] + args,
            cwd=str(cwd.resolve()),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def get_actual_changed_files(
        self,
        workspace_path: Path,
        base_commit: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Collect actual changed files from Git.

        Returns list of dicts: {'status': 'M'|'A'|'D'|'R'|'??', 'path': '...', 'old_path': Optional[str]}.
        """
        workspace_path = Path(workspace_path).resolve()
        changed: List[Dict[str, str]] = []

        if not (workspace_path / ".git").exists():
            return changed

        # 1. Check working directory status (both tracked changes and untracked files)
        proc_status = self._run_git(["status", "--porcelain=v1"], cwd=workspace_path)
        if proc_status.returncode == 0:
            for line in proc_status.stdout.splitlines():
                if len(line) < 4:
                    continue
                code = line[:2].strip()
                rest = line[3:].strip().strip('"')
                old_p = None
                new_p = rest
                if " -> " in rest:
                    old_p, new_p = rest.split(" -> ", 1)
                    old_p = old_p.strip('"')
                    new_p = new_p.strip('"')
                changed.append({"status": code or "M", "path": new_p, "old_path": old_p})

        # 2. If base_commit is specified and distinct from HEAD, also include committed changes
        if base_commit:
            proc_diff = self._run_git(
                ["diff", "--name-status", f"{base_commit}..HEAD"],
                cwd=workspace_path,
            )
            if proc_diff.returncode == 0:
                seen_paths = {c["path"] for c in changed}
                for line in proc_diff.stdout.splitlines():
                    parts = line.strip().split("\t")
                    if not parts:
                        continue
                    code = parts[0]
                    path = parts[-1].strip('"')
                    if path not in seen_paths:
                        old_p = parts[1].strip('"') if len(parts) > 2 else None
                        changed.append({"status": code, "path": path, "old_path": old_p})

        return changed

    def check_task_scope(
        self,
        task: TaskSpec,
        workspace_path: Path,
        actual_changes: List[Dict[str, str]],
    ) -> Tuple[ValidationCheckResult, List[str]]:
        """Validate that all actual changes fall strictly within task.write_scope.

        Detects:
        - Out-of-scope modifications.
        - Deletions outside write scope.
        - Symlink escapes pointing outside workspace.
        """
        workspace_path = Path(workspace_path).resolve()
        out_of_scope: List[str] = []
        errors: List[str] = []
        warnings: List[str] = []

        for item in actual_changes:
            rel_path = item["path"]

            # Symlink escape verification
            target_path = workspace_path / rel_path
            if target_path.is_symlink():
                try:
                    resolved = target_path.resolve()
                    if workspace_path != resolved and workspace_path not in resolved.parents:
                        err = f"Symlink escape: '{rel_path}' points outside workspace to '{resolved}'"
                        errors.append(err)
                        out_of_scope.append(rel_path)
                        continue
                except Exception as e:
                    err = f"Symlink resolution failed for '{rel_path}': {e}"
                    errors.append(err)
                    out_of_scope.append(rel_path)
                    continue

            # Check write scope containment
            if not is_path_in_scope(rel_path, task.write_scope):
                out_of_scope.append(rel_path)
                errors.append(f"Out-of-scope file modification: '{rel_path}' is not in declared write_scope: {task.write_scope}")

            # Also check old_path if rename occurred
            if item.get("old_path"):
                old_p = item["old_path"]
                if not is_path_in_scope(old_p, task.write_scope):
                    out_of_scope.append(old_p)
                    errors.append(f"Out-of-scope file rename: original '{old_p}' is not in declared write_scope: {task.write_scope}")

        passed = (len(errors) == 0 and len(out_of_scope) == 0)
        details = (
            "All changed files are within declared write scope."
            if passed
            else f"Found {len(out_of_scope)} out-of-scope modifications or security violations."
        )

        check_res = ValidationCheckResult(
            name="task_scope_validation",
            passed=passed,
            details=details,
            errors=errors,
            warnings=warnings,
            evidence={"actual_changes": actual_changes, "out_of_scope": out_of_scope},
        )
        return check_res, out_of_scope

    def check_git_diff(
        self,
        workspace_path: Path,
        base_commit: Optional[str] = None,
    ) -> Tuple[ValidationCheckResult, DiffStats]:
        """Collect Git diff statistics, parse insertions/deletions, and check whitespace."""
        workspace_path = Path(workspace_path).resolve()
        stats = DiffStats()
        errors: List[str] = []
        warnings: List[str] = []

        if not (workspace_path / ".git").exists():
            return ValidationCheckResult(
                name="git_diff_validation",
                passed=True,
                details="Not a Git repository; skipping diff validation",
            ), stats

        # 1. Collect shortstat
        diff_target = [base_commit] if base_commit else []
        proc_stat = self._run_git(["diff", "--shortstat"] + diff_target, cwd=workspace_path)
        if proc_stat.returncode == 0 and proc_stat.stdout.strip():
            m = SHORTSTAT_REGEX.search(proc_stat.stdout)
            if m:
                stats.files_changed = int(m.group(1))
                stats.insertions = int(m.group(2)) if m.group(2) else 0
                stats.deletions = int(m.group(3)) if m.group(3) else 0

        # Also get names of changed files
        proc_names = self._run_git(["diff", "--name-only"] + diff_target, cwd=workspace_path)
        if proc_names.returncode == 0:
            stats.changed_files = [
                f.strip().strip('"')
                for f in proc_names.stdout.splitlines()
                if f.strip()
            ]

        # 2. Whitespace check (git diff --check)
        proc_check = self._run_git(["diff", "--check"] + diff_target, cwd=workspace_path)
        if proc_check.returncode != 0 and proc_check.stdout.strip():
            ws_lines = proc_check.stdout.strip().splitlines()
            stats.whitespace_errors = len(ws_lines)
            if self.strict_whitespace:
                errors.append(f"Found {len(ws_lines)} whitespace errors in diff")
            else:
                warnings.append(f"Found {len(ws_lines)} whitespace errors in diff (non-fatal)")

        # 3. Collect raw diff with bounding
        proc_raw = self._run_git(["diff"] + diff_target, cwd=workspace_path)
        if proc_raw.returncode == 0:
            raw_text = proc_raw.stdout
            lines = raw_text.splitlines()
            if len(lines) > self.max_diff_lines or len(raw_text.encode("utf-8")) > self.max_diff_bytes:
                truncated_lines = lines[:self.max_diff_lines]
                stats.raw_diff = "\n".join(truncated_lines) + "\n... [diff truncated for size] ..."
                stats.diff_truncated = True
                warnings.append("Raw diff was truncated due to output limits")
            else:
                stats.raw_diff = raw_text

        passed = len(errors) == 0
        details = (
            f"{stats.files_changed} files changed, +{stats.insertions}/-{stats.deletions} lines."
        )

        check_res = ValidationCheckResult(
            name="git_diff_validation",
            passed=passed,
            details=details,
            errors=errors,
            warnings=warnings,
            evidence=stats.to_dict(),
        )
        return check_res, stats

    def run_validation_tests(
        self,
        task: TaskSpec,
        workspace_path: Path,
    ) -> ValidationCheckResult:
        """Run explicitly permitted validation commands using ExecutionEngine."""
        workspace_path = Path(workspace_path).resolve()
        errors: List[str] = []
        warnings: List[str] = []
        evidence: Dict[str, Any] = {}

        if not task.validation_requirements:
            return ValidationCheckResult(
                name="validation_tests",
                passed=True,
                details="No validation commands configured for this task",
            )

        engine = self.engine or ExecutionEngine(
            workspace=workspace_path,
            approval_mode="prompt",
            timeout=task.timeout,
        )

        for cmd_str in task.validation_requirements:
            try:
                cmd_args = validate_command_string(cmd_str)
                permission = classify_command(cmd_args)
            except Exception as e:
                errors.append(f"Validation command rejected: {cmd_str} ({e})")
                continue

            if permission not in (PERMISSION_READ_ONLY, PERMISSION_TEST_BUILD):
                errors.append(
                    f"Forbidden validation command: '{cmd_str}' has permission '{permission}' (only read-only and test/build permitted)"
                )
                continue

            exec_res = engine.execute_command(
                cmd_str,
                timeout=task.timeout,
                require_approval=False,
                approved=True,
            )

            cmd_key = cmd_str[:40]
            evidence[cmd_key] = {
                "returncode": exec_res.get("returncode"),
                "stdout": exec_res.get("stdout", "")[:500],
                "stderr": exec_res.get("stderr", "")[:500],
            }

            if exec_res.get("returncode") != 0:
                stderr_snippet = exec_res.get("stderr", "").strip() or exec_res.get("stdout", "").strip()
                concise_err = stderr_snippet.splitlines()[-1] if stderr_snippet else "Exit status non-zero"
                errors.append(f"Validation test failed: '{cmd_str}' (exit {exec_res.get('returncode')}): {concise_err}")

        passed = len(errors) == 0
        details = (
            f"All {len(task.validation_requirements)} validation tests passed."
            if passed
            else f"{len(errors)} validation test commands failed."
        )

        return ValidationCheckResult(
            name="validation_tests",
            passed=passed,
            details=details,
            errors=errors,
            warnings=warnings,
            evidence=evidence,
        )

    def run_static_checks(
        self,
        workspace_path: Path,
        changed_files: List[str],
        tools: Optional[List[str]] = None,
    ) -> ValidationCheckResult:
        """Run configured optional lint and static analysis tools."""
        workspace_path = Path(workspace_path).resolve()
        errors: List[str] = []
        warnings: List[str] = []
        evidence: Dict[str, Any] = {}

        configured_tools = tools or ["py_compile"]
        py_files = [f for f in changed_files if f.endswith(".py")]

        for tool in configured_tools:
            if tool == "py_compile":
                # Builtin Python compiler check
                for pf in py_files:
                    full_p = workspace_path / pf
                    if full_p.is_file():
                        proc = subprocess.run(
                            ["python3", "-m", "py_compile", str(full_p)],
                            capture_output=True,
                            text=True,
                            timeout=15,
                            check=False,
                        )
                        if proc.returncode != 0:
                            errors.append(f"Syntax error in '{pf}': {proc.stderr.strip()}")
            elif tool in ("flake8", "ruff", "mypy"):
                # Optional external tools: check availability honestly
                tool_path = shutil.which(tool)
                if not tool_path:
                    warnings.append(f"Optional static checker '{tool}' not installed; skipped check.")
                elif py_files:
                    proc = subprocess.run(
                        [tool] + [str(workspace_path / f) for f in py_files],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    evidence[tool] = {"returncode": proc.returncode, "output": proc.stdout[:500]}
                    if proc.returncode != 0:
                        errors.append(f"Static check '{tool}' reported issues:\n{proc.stdout[:300]}")
            else:
                warnings.append(f"Unknown or unconfigured static checker '{tool}'; skipped.")

        passed = len(errors) == 0
        details = (
            "Static analysis completed cleanly."
            if passed
            else f"Static analysis identified {len(errors)} errors."
        )

        return ValidationCheckResult(
            name="static_checks",
            passed=passed,
            details=details,
            errors=errors,
            warnings=warnings,
            evidence=evidence,
        )

    def validate_task(
        self,
        task: TaskSpec,
        workspace_path: Path,
        base_commit: Optional[str] = None,
        worker_result: Optional[Any] = None,
        static_tools: Optional[List[str]] = None,
    ) -> ValidationReport:
        """Execute complete independent validation on task modifications."""
        workspace_path = Path(workspace_path).resolve()
        checks: List[ValidationCheckResult] = []
        all_warnings: List[str] = []

        # 1. Identify actual changed files independently
        actual_changes = self.get_actual_changed_files(workspace_path, base_commit=base_commit)

        # 2. Scope verification
        scope_check, out_of_scope = self.check_task_scope(task, workspace_path, actual_changes)
        checks.append(scope_check)
        all_warnings.extend(scope_check.warnings)

        # 3. Git Diff verification
        diff_check, diff_stats = self.check_git_diff(workspace_path, base_commit=base_commit)
        checks.append(diff_check)
        all_warnings.extend(diff_check.warnings)

        # 4. Validation tests execution
        test_check = self.run_validation_tests(task, workspace_path)
        checks.append(test_check)
        all_warnings.extend(test_check.warnings)

        # 5. Optional static checks
        changed_paths = [c["path"] for c in actual_changes]
        static_check = self.run_static_checks(workspace_path, changed_paths, tools=static_tools)
        checks.append(static_check)
        all_warnings.extend(static_check.warnings)

        # Synthesize status and retry recommendation
        all_passed = all(c.passed for c in checks)
        retry_recommended = False
        retry_feedback = None

        if all_passed:
            validation_status = ValidationStatus.PASSED.value
        else:
            validation_status = ValidationStatus.FAILED.value
            # Check if failure is recoverable (e.g. test failure vs security violation)
            if out_of_scope:
                # Security/Scope failure is NEVER automatically retried
                retry_recommended = False
                retry_feedback = f"Validation rejected: Unauthorized changes in {out_of_scope}"
            elif not test_check.passed or not static_check.passed:
                # Test or compilation failure is retryable
                retry_recommended = True
                failed_msgs = test_check.errors + static_check.errors
                retry_feedback = "; ".join(failed_msgs[:2])
            else:
                retry_recommended = False
                retry_feedback = "Validation failed due to diff/whitespace errors"

        evidence_refs: Dict[str, Any] = {
            "checks_run": [c.name for c in checks],
            "files_modified_count": len(actual_changes),
        }
        if base_commit:
            evidence_refs["base_commit"] = base_commit

        return ValidationReport(
            batch_id=task.batch_id,
            task_id=task.task_id,
            validation_status=validation_status,
            schema_version="1.0",
            checks=checks,
            diff_stats=diff_stats,
            out_of_scope_changes=out_of_scope,
            warnings=all_warnings,
            retry_recommended=retry_recommended,
            retry_feedback=retry_feedback,
            evidence_references=evidence_refs,
            timestamp=time.time(),
        )

