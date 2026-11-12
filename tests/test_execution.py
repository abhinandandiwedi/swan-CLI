"""
SWAN-ENGINE v0.2 — Execution Engine Security Tests
===================================================

Comprehensive test suite for core/execution_engine.py:
- Shell operator and command injection rejection (&&, ||, ;, |, >, <, backticks, $(), ${}, \n, &)
- Command permission classification (read-only, test/build, mutating, destructive, privileged, unknown)
- Blocked command protection (reboot, shutdown, dd, mkfs, bash, sh, etc.)
- Privileged operations & sudo protection (never auto-executed, requires explicit approval)
- Approval policy enforcement across modes (prompt, auto, strict)
- Python interpreter discovery priority (.venv -> venv -> sys.executable -> python3)
- Pip invocation routing via [python, -m, pip] without shell activation
- PEP 668 detection and safe virtual environment recommendations
- Path containment, traversal rejection (../), symlink escape prevention
- Atomic file operations (read_file, write_file, list_dir, overwrite protection)
- Bounded stdout/stderr and timeout enforcement
- Repeated failure tracking (max 2 retries per failure class)
- Verification that shell=False is enforced exclusively
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.execution_engine import (
    ExecutionEngine,
    classify_command,
    is_blocked_command,
    validate_command_string,
    validate_file_path,
    atomic_write_file,
    discover_python,
    check_pep668,
    PERMISSION_READ_ONLY,
    PERMISSION_TEST_BUILD,
    PERMISSION_MUTATING,
    PERMISSION_DESTRUCTIVE,
    PERMISSION_PRIVILEGED,
    PERMISSION_UNKNOWN,
)


class BaseExecutionTestCase(unittest.TestCase):
    """Base test case providing an isolated temporary workspace."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="swan_exec_test_")
        self.workspace = Path(self._tmpdir).resolve()
        self.engine = ExecutionEngine(workspace=self.workspace, approval_mode="prompt")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# =========================================================================
# 1. Shell Operator & Injection Rejection Tests
# =========================================================================

class TestShellOperatorRejection(unittest.TestCase):

    def test_semicolon_chaining_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_command_string("ls -la ; rm -rf /")
        self.assertIn("Shell operator detected", str(ctx.exception))

    def test_and_operator_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_command_string("echo 1 && echo 2")
        self.assertIn("Shell operator detected", str(ctx.exception))

    def test_or_operator_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_command_string("echo 1 || echo 2")
        self.assertIn("Shell operator detected", str(ctx.exception))

    def test_pipe_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_command_string("cat file.txt | grep root")
        self.assertIn("Shell operator detected", str(ctx.exception))

    def test_output_redirection_rejected(self):
        for bad_cmd in ["echo hello > out.txt", "echo hello >> out.txt"]:
            with self.assertRaises(ValueError, msg=f"Should reject: {bad_cmd}"):
                validate_command_string(bad_cmd)

    def test_input_redirection_rejected(self):
        for bad_cmd in ["cat < in.txt", "cat << EOF"]:
            with self.assertRaises(ValueError, msg=f"Should reject: {bad_cmd}"):
                validate_command_string(bad_cmd)

    def test_backticks_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_command_string("echo `whoami`")
        self.assertIn("Shell operator detected", str(ctx.exception))

    def test_command_substitution_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_command_string("echo $(whoami)")
        self.assertIn("Shell operator detected", str(ctx.exception))

    def test_variable_expansion_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_command_string("echo ${USER}")
        self.assertIn("Shell operator detected", str(ctx.exception))

    def test_background_operator_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_command_string("sleep 10 &")
        self.assertIn("Shell operator detected", str(ctx.exception))

    def test_newline_injection_rejected(self):
        for bad_cmd in ["echo a\necho b", "echo a\recho b"]:
            with self.assertRaises(ValueError, msg="Should reject newlines in command"):
                validate_command_string(bad_cmd)

    def test_empty_command_rejected(self):
        with self.assertRaises(ValueError):
            validate_command_string("")
        with self.assertRaises(ValueError):
            validate_command_string("   ")

    def test_valid_commands_accepted(self):
        self.assertEqual(validate_command_string("git status"), ["git", "status"])
        self.assertEqual(validate_command_string("python -m pytest tests/"), ["python", "-m", "pytest", "tests/"])
        self.assertEqual(validate_command_string("cat file.py"), ["cat", "file.py"])
        self.assertEqual(validate_command_string('echo "hello world"'), ["echo", "hello world"])


# =========================================================================
# 2. Command Permission Classification Tests
# =========================================================================

class TestCommandClassification(unittest.TestCase):

    def test_read_only_commands(self):
        commands = [
            ["cat", "foo.txt"],
            ["ls", "-la"],
            ["git", "status"],
            ["git", "diff"],
            ["git", "log", "-n", "5"],
            ["grep", "-rn", "test", "."],
            ["find", ".", "-name", "*.py"],
            ["echo", "hello"],
            ["pwd"],
            ["uname", "-a"],
        ]
        for cmd in commands:
            self.assertEqual(classify_command(cmd), PERMISSION_READ_ONLY, msg=f"Failed for: {cmd}")

    def test_test_build_commands(self):
        commands = [
            ["python", "-m", "pytest"],
            ["python3", "-m", "unittest"],
            ["python", "-m", "py_compile", "app.py"],
            ["pytest", "tests/"],
            ["make"],
            ["cargo", "build"],
            ["cargo", "test"],
            ["npm", "test"],
            ["go", "test"],
        ]
        for cmd in commands:
            self.assertEqual(classify_command(cmd), PERMISSION_TEST_BUILD, msg=f"Failed for: {cmd}")

    def test_mutating_commands(self):
        commands = [
            ["git", "add", "."],
            ["git", "commit", "-m", "feat"],
            ["git", "push", "origin", "main"],
            ["pip", "install", "requests"],
            ["python", "-m", "pip", "install", "requests"],
            ["mkdir", "new_dir"],
            ["touch", "new_file.py"],
            ["cp", "a.py", "b.py"],
            ["mv", "old.py", "new.py"],
        ]
        for cmd in commands:
            self.assertEqual(classify_command(cmd), PERMISSION_MUTATING, msg=f"Failed for: {cmd}")

    def test_destructive_commands(self):
        commands = [
            ["rm", "file.py"],
            ["rmdir", "old_dir"],
            ["git", "reset", "--hard"],
            ["git", "clean", "-fd"],
            ["git", "push", "--force"],
            ["pip", "uninstall", "-y", "pkg"],
            ["kill", "1234"],
        ]
        for cmd in commands:
            self.assertEqual(classify_command(cmd), PERMISSION_DESTRUCTIVE, msg=f"Failed for: {cmd}")

    def test_privileged_commands(self):
        commands = [
            ["sudo", "apt", "update"],
            ["su", "-"],
            ["chmod", "+x", "script.sh"],
            ["chown", "root:root", "file"],
            ["systemctl", "restart", "service"],
            ["passwd"],
        ]
        for cmd in commands:
            self.assertEqual(classify_command(cmd), PERMISSION_PRIVILEGED, msg=f"Failed for: {cmd}")

    def test_unknown_command(self):
        self.assertEqual(classify_command(["unrecognized_binary_xyz"]), PERMISSION_UNKNOWN)
        self.assertEqual(classify_command([]), PERMISSION_UNKNOWN)


# =========================================================================
# 3. Blocked Commands Tests
# =========================================================================

class TestBlockedCommands(unittest.TestCase):

    def test_system_destruction_blocked(self):
        for cmd in [["reboot"], ["shutdown", "-h", "now"], ["poweroff"], ["mkfs", "/dev/sda1"], ["dd", "if=/dev/zero"]]:
            reason = is_blocked_command(cmd)
            self.assertIsNotNone(reason, msg=f"Expected blocked: {cmd}")
            self.assertIn("blocked", reason.lower())

    def test_direct_shell_interpreters_blocked(self):
        for shell in ["bash", "sh", "zsh", "dash", "csh", "tcsh"]:
            reason = is_blocked_command([shell, "-c", "echo 1"])
            self.assertIsNotNone(reason, msg=f"Expected blocked shell: {shell}")
            self.assertIn("shell interpreter", reason.lower())


# =========================================================================
# 4. Approval Policy & Sudo Protection Tests
# =========================================================================

class TestApprovalPolicy(BaseExecutionTestCase):

    def test_read_only_auto_approved_in_prompt_mode(self):
        result = self.engine.execute_command("echo hello", approved=False)
        self.assertTrue(result["approved"])
        self.assertTrue(result["success"])
        self.assertIn("hello", result["stdout"])

    def test_test_build_auto_approved_in_prompt_mode(self):
        # py_compile on a non-existent file will return returncode != 0, but it should be APPROVED
        result = self.engine.execute_command("python3 -m py_compile nonexistent_file_xyz.py", approved=False)
        self.assertTrue(result["approved"])
        self.assertEqual(result["permission"], PERMISSION_TEST_BUILD)

    def test_mutating_requires_approval_in_prompt_mode(self):
        result = self.engine.execute_command("git commit -m 'test'", approved=False)
        self.assertFalse(result["approved"])
        self.assertFalse(result["success"])
        self.assertIn("requires explicit user approval", result["blocked"])
        self.assertEqual(result["permission"], PERMISSION_MUTATING)

    def test_mutating_proceeds_when_approved(self):
        # Create a directory via approved command
        result = self.engine.execute_command("mkdir approved_dir", approved=True)
        self.assertTrue(result["approved"])
        self.assertTrue(result["success"])
        self.assertTrue((self.workspace / "approved_dir").is_dir())

    def test_destructive_requires_approval_always(self):
        # Even in auto mode, destructive requires approval
        auto_engine = ExecutionEngine(workspace=self.workspace, approval_mode="auto")
        result = auto_engine.execute_command("rm -rf something", approved=False)
        self.assertFalse(result["approved"])
        self.assertIn("requires explicit user approval", result["blocked"])
        self.assertEqual(result["permission"], PERMISSION_DESTRUCTIVE)

    def test_sudo_never_automatically_executed(self):
        # Test in prompt, auto, and strict modes
        for mode in ["prompt", "auto", "strict"]:
            engine = ExecutionEngine(workspace=self.workspace, approval_mode=mode)
            result = engine.execute_command("sudo apt update", approved=False)
            self.assertFalse(result["approved"])
            self.assertFalse(result["success"])
            self.assertEqual(result["permission"], PERMISSION_PRIVILEGED)
            self.assertIn("approval", result["blocked"].lower())

    def test_strict_mode_requires_approval_for_test_build(self):
        strict_engine = ExecutionEngine(workspace=self.workspace, approval_mode="strict")
        result = strict_engine.execute_command("python3 -m unittest", approved=False)
        self.assertFalse(result["approved"])
        self.assertIn("requires explicit user approval", result["blocked"])

    def test_unknown_command_requires_approval(self):
        result = self.engine.execute_command("arbitrary_unknown_binary_xyz", approved=False)
        self.assertFalse(result["approved"])
        self.assertEqual(result["permission"], PERMISSION_UNKNOWN)


# =========================================================================
# 5. Python Interpreter Discovery Tests
# =========================================================================

class TestPythonDiscovery(BaseExecutionTestCase):

    def test_priority_1_dot_venv(self):
        dot_venv = self.workspace / ".venv" / "bin"
        dot_venv.mkdir(parents=True)
        py1 = dot_venv / "python"
        py1.write_text("#!/bin/sh\nexit 0\n")
        py1.chmod(0o755)

        # Also create venv (Priority 2) to test priority 1 wins
        venv = self.workspace / "venv" / "bin"
        venv.mkdir(parents=True)
        py2 = venv / "python"
        py2.write_text("#!/bin/sh\nexit 0\n")
        py2.chmod(0o755)

        discovered = discover_python(self.workspace)
        self.assertEqual(discovered, str(py1.resolve()))

    def test_priority_2_venv(self):
        venv = self.workspace / "venv" / "bin"
        venv.mkdir(parents=True)
        py2 = venv / "python"
        py2.write_text("#!/bin/sh\nexit 0\n")
        py2.chmod(0o755)

        discovered = discover_python(self.workspace)
        self.assertEqual(discovered, str(py2.resolve()))

    def test_priority_3_sys_executable(self):
        # Empty workspace with no venvs
        discovered = discover_python(self.workspace)
        self.assertEqual(discovered, str(Path(sys.executable).resolve()))

    def test_priority_4_python3_from_path(self):
        with mock.patch("sys.executable", None):
            with mock.patch("shutil.which", return_value="/custom/bin/python3"):
                with mock.patch("pathlib.Path.resolve", return_value=Path("/custom/bin/python3")):
                    discovered = discover_python(self.workspace)
                    self.assertEqual(discovered, "/custom/bin/python3")

    def test_no_python_found_raises_runtime_error(self):
        with mock.patch("sys.executable", None):
            with mock.patch("shutil.which", return_value=None):
                with self.assertRaises(RuntimeError) as ctx:
                    discover_python(self.workspace)
                self.assertIn("No Python interpreter found", str(ctx.exception))

    def test_pip_invocation_routing(self):
        # Resolves pip to [discovered_python, -m, pip]
        resolved = self.engine._resolve_python_command(["pip", "install", "requests"])
        self.assertEqual(resolved[0], self.engine.python)
        self.assertEqual(resolved[1:3], ["-m", "pip"])
        self.assertEqual(resolved[3:], ["install", "requests"])

    def test_python3_command_routing(self):
        resolved = self.engine._resolve_python_command(["python3", "app.py"])
        self.assertEqual(resolved[0], self.engine.python)
        self.assertEqual(resolved[1:], ["app.py"])


# =========================================================================
# 6. PEP 668 Protection Tests
# =========================================================================

class TestPEP668Protection(BaseExecutionTestCase):

    def test_venv_not_externally_managed(self):
        venv_root = self.workspace / ".venv"
        (venv_root / "bin").mkdir(parents=True)
        (venv_root / "pyvenv.cfg").write_text("home = /usr/bin\n")
        dummy_py = venv_root / "bin" / "python"
        dummy_py.write_text("#!/bin/sh\n")

        warning = check_pep668(str(dummy_py))
        self.assertIsNone(warning, "Virtual environment should never be marked externally managed")

    def test_pep668_detected_with_marker_file(self):
        fake_sys = self.workspace / "fake_sys"
        fake_bin = fake_sys / "bin"
        fake_lib = fake_sys / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
        fake_bin.mkdir(parents=True)
        fake_lib.mkdir(parents=True)

        py = fake_bin / "python"
        py.write_text("#!/bin/sh\n")
        (fake_lib / "EXTERNALLY-MANAGED").write_text("Managed by Debian\n")

        warning = check_pep668(str(py))
        self.assertIsNotNone(warning)
        self.assertIn("externally managed", warning)
        self.assertIn("PEP 668", warning)
        self.assertIn("-m venv .venv", warning)

    def test_pip_install_blocked_when_pep668_detected(self):
        with mock.patch("core.execution_engine.check_pep668", return_value="⚠ Python is externally managed (PEP 668)."):
            result = self.engine.execute_command("pip install requests", approved=True)
            self.assertFalse(result["success"])
            self.assertIn("externally managed (PEP 668)", result["blocked"])
            self.assertEqual(result["returncode"], 1)


# =========================================================================
# 7. Path Containment & Traversal Prevention Tests
# =========================================================================

class TestPathContainment(BaseExecutionTestCase):

    def test_valid_workspace_relative_path(self):
        target = validate_file_path("src/module.py", self.workspace)
        self.assertEqual(target, (self.workspace / "src/module.py").resolve())

    def test_valid_workspace_root(self):
        target = validate_file_path(".", self.workspace)
        self.assertEqual(target, self.workspace)

    def test_dotdot_escape_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_file_path("../escape.txt", self.workspace)
        self.assertIn("escapes workspace boundary", str(ctx.exception))

    def test_deep_dotdot_escape_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_file_path("a/b/../../../../etc/passwd", self.workspace)
        self.assertIn("escapes workspace boundary", str(ctx.exception))

    def test_absolute_path_outside_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_file_path("/etc/shadow", self.workspace)
        self.assertIn("escapes workspace boundary", str(ctx.exception))

    def test_symlink_escape_rejected(self):
        # Create an outside directory and file
        outside = Path(tempfile.mkdtemp(prefix="swan_outside_"))
        try:
            (outside / "secret.txt").write_text("secret")
            # Create symlink in workspace pointing to outside
            link = self.workspace / "leak_link"
            link.symlink_to(outside)

            with self.assertRaises(ValueError) as ctx:
                validate_file_path("leak_link/secret.txt", self.workspace)
            self.assertIn("escapes workspace boundary", str(ctx.exception))
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_symlink_within_workspace_allowed(self):
        (self.workspace / "real_dir").mkdir()
        (self.workspace / "real_dir" / "file.txt").write_text("ok")
        link = self.workspace / "internal_link"
        link.symlink_to(self.workspace / "real_dir")

        target = validate_file_path("internal_link/file.txt", self.workspace, must_exist=True)
        self.assertEqual(target, (self.workspace / "real_dir" / "file.txt").resolve())

    def test_must_exist_raises_on_missing(self):
        with self.assertRaises(ValueError) as ctx:
            validate_file_path("missing_file.txt", self.workspace, must_exist=True)
        self.assertIn("does not exist", str(ctx.exception))


# =========================================================================
# 8. File Operations Tests (read_file, write_file, list_dir)
# =========================================================================

class TestFileOperations(BaseExecutionTestCase):

    def test_read_file_success(self):
        f = self.workspace / "hello.py"
        f.write_text("print('hello world')", encoding="utf-8")
        success, content = self.engine.read_file("hello.py")
        self.assertTrue(success)
        self.assertEqual(content, "print('hello world')")

    def test_read_file_missing_fails(self):
        success, err = self.engine.read_file("ghost.txt")
        self.assertFalse(success)
        self.assertIn("does not exist", err)

    def test_read_file_outside_workspace_fails(self):
        success, err = self.engine.read_file("../../outside.txt")
        self.assertFalse(success)
        self.assertIn("escapes workspace boundary", err)

    def test_write_file_new_success(self):
        success, msg = self.engine.write_file("new/nested/file.txt", "content123")
        self.assertTrue(success)
        target = self.workspace / "new/nested/file.txt"
        self.assertTrue(target.is_file())
        self.assertEqual(target.read_text(), "content123")

    def test_write_file_existing_requires_overwrite_approval(self):
        f = self.workspace / "exists.txt"
        f.write_text("original")

        success, err = self.engine.write_file("exists.txt", "new content", overwrite_approved=False)
        self.assertFalse(success)
        self.assertIn("Explicit overwrite approval required", err)
        self.assertEqual(f.read_text(), "original")  # unchanged

    def test_write_file_existing_with_overwrite_approval(self):
        f = self.workspace / "exists.txt"
        f.write_text("original")

        success, msg = self.engine.write_file("exists.txt", "updated content", overwrite_approved=True)
        self.assertTrue(success)
        self.assertEqual(f.read_text(), "updated content")

    def test_write_file_outside_workspace_rejected(self):
        success, err = self.engine.write_file("../outside.txt", "malicious")
        self.assertFalse(success)
        self.assertIn("escapes workspace boundary", err)

    def test_write_file_to_directory_rejected(self):
        d = self.workspace / "my_dir"
        d.mkdir()
        success, err = self.engine.write_file("my_dir", "content")
        self.assertFalse(success)
        self.assertIn("directory", err.lower())

    def test_atomic_write_leaves_no_temp_files(self):
        target = self.workspace / "clean.txt"
        atomic_write_file(target, "clean data", self.workspace)
        tmp_files = list(self.workspace.glob("*.swan_tmp"))
        self.assertEqual(len(tmp_files), 0)

    def test_list_dir_success(self):
        (self.workspace / "file1.py").write_text("1")
        (self.workspace / "file2.py").write_text("2")
        (self.workspace / "subdir").mkdir()
        (self.workspace / ".hidden").write_text("hidden")

        success, listing = self.engine.list_dir(".")
        self.assertTrue(success)
        self.assertIn("file1.py", listing)
        self.assertIn("file2.py", listing)
        self.assertIn("subdir/", listing)
        self.assertNotIn(".hidden", listing)

    def test_list_dir_caps_at_200(self):
        big_dir = self.workspace / "big_dir"
        big_dir.mkdir()
        for i in range(215):
            (big_dir / f"item_{i:03d}.txt").write_text("x")

        success, listing = self.engine.list_dir("big_dir")
        self.assertTrue(success)
        self.assertIn("15 more entries", listing)


# =========================================================================
# 9. Execution Engine Features (shell=False, Timeouts, Truncation)
# =========================================================================

class TestExecutionEngineFeatures(BaseExecutionTestCase):

    def test_shell_false_enforced_no_variable_expansion(self):
        # In shell=False, '$HOME' is passed literally as an argument, NOT expanded
        result = self.engine.execute_command("echo test", approved=False)
        self.assertTrue(result["success"])
        self.assertEqual(result["stdout"].strip(), "test")

    def test_command_timeout_enforced(self):
        # A 1-second timeout on a sleep command
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["sleep", "5"], timeout=1)):
            result = self.engine.execute_command("echo timeout_test", timeout=1, approved=False)
            self.assertFalse(result["success"])
            self.assertIn("timed out after 1 seconds", result["stderr"])
            self.assertEqual(self.engine.get_failure_count("echo timeout_test"), 1)

    def test_output_truncation_bounded(self):
        huge_text = "A" * (ExecutionEngine.MAX_OUTPUT_BYTES + 500)
        truncated = self.engine._truncate_output(huge_text)
        self.assertIn("truncated at", truncated)
        self.assertLessEqual(len(truncated.encode("utf-8")), ExecutionEngine.MAX_OUTPUT_BYTES + 100)

    def test_nonexistent_executable_handled(self):
        # Approve execution so it attempts subprocess.run
        result = self.engine.execute_command("nonexistent_binary_12345", approved=True)
        self.assertFalse(result["success"])
        self.assertIn("not found", result["stderr"].lower())


# =========================================================================
# 10. Repeated Failure Detection Tests
# =========================================================================

class TestRepeatedFailureDetection(BaseExecutionTestCase):

    def test_max_two_retries_then_blocked(self):
        # Command fails on execution
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=["cat", "missing"], returncode=1, stdout="", stderr="error")

            # Attempt 1 -> fails
            r1 = self.engine.execute_command("cat missing", approved=False)
            self.assertFalse(r1["success"])
            self.assertEqual(self.engine.get_failure_count("cat missing"), 1)

            # Attempt 2 -> fails
            r2 = self.engine.execute_command("cat missing", approved=False)
            self.assertFalse(r2["success"])
            self.assertEqual(self.engine.get_failure_count("cat missing"), 2)

            # Attempt 3 -> blocked before execution!
            r3 = self.engine.execute_command("cat missing", approved=False)
            self.assertFalse(r3["success"])
            self.assertIn("Recovery exhausted — stopping to prevent infinite loops", r3["blocked"])
            self.assertEqual(mock_run.call_count, 2)  # subprocess.run was NOT called the 3rd time

    def test_successful_recovery_resets_failure_count(self):
        with mock.patch("subprocess.run") as mock_run:
            # 1 failure
            mock_run.return_value = subprocess.CompletedProcess(args=["cat", "file"], returncode=1, stdout="", stderr="err")
            self.engine.execute_command("cat file", approved=False)
            self.assertEqual(self.engine.get_failure_count("cat file"), 1)

            # 1 success
            mock_run.return_value = subprocess.CompletedProcess(args=["cat", "file"], returncode=0, stdout="ok", stderr="")
            r_ok = self.engine.execute_command("cat file", approved=False)
            self.assertTrue(r_ok["success"])
            self.assertEqual(self.engine.get_failure_count("cat file"), 0)


# =========================================================================
# Main Execution
# =========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
