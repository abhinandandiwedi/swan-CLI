"""Unit tests for Validator and WorktreeManager.

Tests scope validation, symlink escape detection, git diff statistics,
whitespace errors, test execution containment, worktree isolation,
retry recommendations, and untrusted WorkerResult handling.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import MagicMock

from core.validator import (
    Validator,
    WorktreeManager,
    ValidationReport,
    ValidationCheckResult,
    DiffStats,
)
from core.task_orchestrator import (
    TaskSpec,
)
from core.worker_runner import (
    WorkerResult,
)


class TestWorktreeManager(unittest.TestCase):
    """Test safe Git worktree management and user uncommitted change preservation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ws = Path(self.temp_dir.name)
        # Initialize Git repo
        subprocess.run(["git", "init"], cwd=self.ws, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=self.ws, check=True)
        subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=self.ws, check=True)
        (self.ws / "README.md").write_text("# Test Repo\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=self.ws, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=self.ws, capture_output=True, check=True)

        self.wt_mgr = WorktreeManager(workspace=self.ws)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_has_uncommitted_changes(self):
        # Clean workspace
        self.assertFalse(self.wt_mgr.has_uncommitted_changes())

        # Dirty workspace (modified file)
        (self.ws / "README.md").write_text("# Test Repo Modified\n", encoding="utf-8")
        self.assertTrue(self.wt_mgr.has_uncommitted_changes())

        # Restore file, add untracked file
        subprocess.run(["git", "checkout", "--", "README.md"], cwd=self.ws, check=True)
        (self.ws / "untracked.txt").write_text("untracked\n", encoding="utf-8")
        self.assertTrue(self.wt_mgr.has_uncommitted_changes())

    def test_get_base_commit(self):
        commit = self.wt_mgr.get_base_commit()
        self.assertIsNotNone(commit)
        self.assertEqual(len(commit), 40)

    def test_create_worktree_success(self):
        wt_path = self.ws / "worktrees" / "task-01"
        ok, msg = self.wt_mgr.create_worktree(wt_path, branch_name="branch-task-01")
        self.assertTrue(ok)
        self.assertTrue(wt_path.exists())
        self.assertTrue((wt_path / "README.md").exists())

    def test_create_worktree_existing_path_fails(self):
        wt_path = self.ws / "worktrees" / "task-existing"
        wt_path.mkdir(parents=True)
        ok, msg = self.wt_mgr.create_worktree(wt_path, branch_name="b-exists")
        self.assertFalse(ok)
        self.assertIn("already exists", msg)

    def test_remove_worktree_refuses_when_dirty_without_force(self):
        wt_path = self.ws / "worktrees" / "task-dirty"
        self.wt_mgr.create_worktree(wt_path, branch_name="b-dirty")

        # Make modifications inside worktree
        (wt_path / "new_file.py").write_text("# new\n", encoding="utf-8")
        self.assertFalse(self.wt_mgr.is_worktree_clean(wt_path))

        # Refuses to remove without force
        ok, msg = self.wt_mgr.remove_worktree(wt_path, force=False)
        self.assertFalse(ok)
        self.assertIn("refusing to delete without force", msg)
        self.assertTrue(wt_path.exists())

        # Removes when force=True
        ok_force, _ = self.wt_mgr.remove_worktree(wt_path, force=True)
        self.assertTrue(ok_force)

    def test_merge_requires_human_approval(self):
        self.assertTrue(self.wt_mgr.requires_human_approval_to_merge())


class TestTaskScopeValidation(unittest.TestCase):
    """Test comparing actual filesystem and Git changes against declared task write_scope."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ws = Path(self.temp_dir.name)
        subprocess.run(["git", "init"], cwd=self.ws, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=self.ws, check=True)
        subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=self.ws, check=True)
        (self.ws / "src").mkdir(parents=True)
        (self.ws / "src" / "app.py").write_text("def app(): pass\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.ws, check=True)
        subprocess.run(["git", "commit", "-m", "Init"], cwd=self.ws, capture_output=True, check=True)

        self.validator = Validator()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_correct_changes_pass_scope(self):
        # Modify src/app.py (within scope)
        (self.ws / "src" / "app.py").write_text("def app(): return 42\n", encoding="utf-8")

        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="proj",
            title="Update app",
            description="desc",
            role="coder",
            write_scope=["src/app.py"],
        )

        actual_changes = self.validator.get_actual_changed_files(self.ws)
        self.assertEqual(len(actual_changes), 1)

        check_res, out_of_scope = self.validator.check_task_scope(task, self.ws, actual_changes)
        self.assertTrue(check_res.passed)
        self.assertEqual(len(out_of_scope), 0)

    def test_out_of_scope_modification_fails(self):
        # Worker modified a file outside declared write_scope
        (self.ws / "src" / "unauthorized.py").write_text("secret = 1\n", encoding="utf-8")

        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="proj",
            title="Update app",
            description="desc",
            role="coder",
            write_scope=["src/app.py"],
        )

        actual_changes = self.validator.get_actual_changed_files(self.ws)
        check_res, out_of_scope = self.validator.check_task_scope(task, self.ws, actual_changes)
        self.assertFalse(check_res.passed)
        self.assertIn("src/unauthorized.py", out_of_scope)

    def test_symlink_escape_fails_immediately(self):
        # Create a symlink escaping workspace
        link = self.ws / "src" / "escape_link"
        link.symlink_to("/etc/passwd")

        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="proj",
            title="Symlink attack",
            description="desc",
            role="coder",
            write_scope=["src/"],
        )

        actual_changes = [{"status": "A", "path": "src/escape_link"}]
        check_res, out_of_scope = self.validator.check_task_scope(task, self.ws, actual_changes)
        self.assertFalse(check_res.passed)
        self.assertTrue(any("Symlink escape" in err for err in check_res.errors))


class TestGitDiffValidation(unittest.TestCase):
    """Test git diff statistics collection, whitespace checking, and truncation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ws = Path(self.temp_dir.name)
        subprocess.run(["git", "init"], cwd=self.ws, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=self.ws, check=True)
        subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=self.ws, check=True)
        (self.ws / "file.txt").write_text("line1\nline2\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.ws, check=True)
        subprocess.run(["git", "commit", "-m", "Init"], cwd=self.ws, capture_output=True, check=True)

        self.validator = Validator(max_diff_lines=5)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_git_diff_statistics(self):
        # Add 3 lines, delete 1 line
        (self.ws / "file.txt").write_text("line1\nline_new_1\nline_new_2\nline_new_3\n", encoding="utf-8")

        check_res, diff_stats = self.validator.check_git_diff(self.ws)
        self.assertTrue(check_res.passed)
        self.assertEqual(diff_stats.files_changed, 1)
        self.assertEqual(diff_stats.insertions, 3)
        self.assertEqual(diff_stats.deletions, 1)
        self.assertIn("file.txt", diff_stats.changed_files)

    def test_whitespace_errors_detected(self):
        # Introduce trailing whitespace
        (self.ws / "file.txt").write_text("line1   \nline2\n", encoding="utf-8")

        check_res, diff_stats = self.validator.check_git_diff(self.ws)
        self.assertTrue(diff_stats.whitespace_errors > 0)
        self.assertTrue(len(check_res.warnings) > 0)

    def test_large_diff_truncation(self):
        # Write many lines to exceed max_diff_lines=5
        lines = [f"line {i}" for i in range(20)]
        (self.ws / "file.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        check_res, diff_stats = self.validator.check_git_diff(self.ws)
        self.assertTrue(diff_stats.diff_truncated)
        self.assertIn("truncated", diff_stats.raw_diff)


class TestValidationTestExecution(unittest.TestCase):
    """Test executing validation tests through ExecutionEngine boundaries."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ws = Path(self.temp_dir.name)
        self.validator = Validator()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_successful_validation_test(self):
        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="proj",
            title="Test",
            description="Desc",
            role="coder",
            validation_requirements=["python3 -c \"exit(0)\""],
        )
        res = self.validator.run_validation_tests(task, self.ws)
        self.assertTrue(res.passed)
        self.assertEqual(len(res.errors), 0)

    def test_failed_validation_test(self):
        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="proj",
            title="Test",
            description="Desc",
            role="coder",
            validation_requirements=["python3 -c \"import sys; sys.stderr.write('AssertionError: expected 1 got 2\\n'); exit(1)\""],
        )
        res = self.validator.run_validation_tests(task, self.ws)
        self.assertFalse(res.passed)
        self.assertTrue(len(res.errors) > 0)
        self.assertIn("AssertionError", res.errors[0])

    def test_validation_command_timeout(self):
        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="proj",
            title="Timeout",
            description="Desc",
            role="coder",
            timeout=1,
            validation_requirements=["python3 -c \"import time; time.sleep(10)\""],
        )
        res = self.validator.run_validation_tests(task, self.ws)
        self.assertFalse(res.passed)
        self.assertTrue(len(res.errors) > 0)

    def test_forbidden_validation_command_denied(self):
        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="proj",
            title="Evil",
            description="Desc",
            role="coder",
            validation_requirements=["sudo rm -rf /"],
        )
        res = self.validator.run_validation_tests(task, self.ws)
        self.assertFalse(res.passed)
        self.assertIn("Forbidden validation command", res.errors[0])


class TestStaticChecks(unittest.TestCase):
    """Test syntax compilation and missing optional static analysis tools."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ws = Path(self.temp_dir.name)
        self.validator = Validator()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_py_compile_catches_syntax_error(self):
        (self.ws / "bad.py").write_text("def broken_syntax(:\n", encoding="utf-8")
        res = self.validator.run_static_checks(self.ws, ["bad.py"], tools=["py_compile"])
        self.assertFalse(res.passed)
        self.assertTrue(any("Syntax error" in e for e in res.errors))

    def test_missing_optional_tool_reported_honestly(self):
        (self.ws / "good.py").write_text("x = 1\n", encoding="utf-8")
        res = self.validator.run_static_checks(self.ws, ["good.py"], tools=["non_existent_tool_xyz"])
        self.assertTrue(res.passed)
        self.assertTrue(any("skipped" in w for w in res.warnings))


class TestFullValidationWorkflowAndTampering(unittest.TestCase):
    """Test full validation pipeline and resistance to WorkerResult tampering."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ws = Path(self.temp_dir.name)
        subprocess.run(["git", "init"], cwd=self.ws, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=self.ws, check=True)
        subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=self.ws, check=True)
        (self.ws / "calc.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.ws, check=True)
        subprocess.run(["git", "commit", "-m", "Init"], cwd=self.ws, capture_output=True, check=True)

        self.validator = Validator()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_worker_result_tampering_ignored(self):
        # Worker claims it modified calc.py, but in reality it modified evil.py!
        (self.ws / "evil.py").write_text("# evil file\n", encoding="utf-8")

        fake_worker_result = WorkerResult(
            task_id="t1",
            batch_id="b1",
            worker_id="w-tamper",
            model="swan-coder",
            status="completed",
            summary="I only modified calc.py!",
            changed_files=["calc.py"],  # Tampered: lied about what was changed!
        )

        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="proj",
            title="Calc task",
            description="desc",
            role="coder",
            write_scope=["calc.py"],
        )

        # Validator independently inspects Git state!
        report = self.validator.validate_task(
            task,
            self.ws,
            worker_result=fake_worker_result,
        )

        # Must catch evil.py and fail validation!
        self.assertEqual(report.validation_status, "failed")
        self.assertIn("evil.py", report.out_of_scope_changes)
        self.assertFalse(report.retry_recommended)  # Security violation is non-retryable


if __name__ == "__main__":
    unittest.main()
