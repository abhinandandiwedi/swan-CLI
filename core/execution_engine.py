"""
SWAN-ENGINE v0.2 — Secure Execution Engine
============================================

Hardened command and file execution with:
- shell=False exclusively
- Shell operator and injection rejection (&&, ||, ;, |, >, <, backticks, $(), ${}, \n, &)
- Command permission classification:
    * read-only
    * test/build
    * mutating
    * destructive
    * privileged
- Blocked dangerous commands (system destruction, direct shell interpreters)
- Explicit user approval for sensitive/privileged operations
- Python interpreter discovery (.venv -> venv -> sys.executable -> python3)
- Correct pip invocation via [python, -m, pip] without shell activation
- PEP 668 detection and safe venv recommendations
- Workspace path containment (prevents ../ traversal, absolute paths outside root, symlink escapes)
- Atomic file writes within workspace
- Bounded stdout/stderr and explicit execution timeouts
- Repeated failure tracking (max 2 attempts per failure class)
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Shell Operator Rejection Pattern
# ---------------------------------------------------------------------------
# Reject any shell control operators, command chaining, redirections,
# backticks, variable/command substitutions, backgrounding, and newlines.
_SHELL_OPERATORS_RE = re.compile(
    r'[;&|<>\n\r`]|(\$\s*[\(\{])'
)


# ---------------------------------------------------------------------------
# Command Permission Categories
# ---------------------------------------------------------------------------
PERMISSION_READ_ONLY = "read-only"
PERMISSION_TEST_BUILD = "test/build"
PERMISSION_MUTATING = "mutating"
PERMISSION_DESTRUCTIVE = "destructive"
PERMISSION_PRIVILEGED = "privileged"
PERMISSION_UNKNOWN = "unknown"

_COMMAND_PERMISSIONS: Dict[str, Set[str]] = {
    # Read-only: safe to run without user approval
    PERMISSION_READ_ONLY: {
        "cat", "head", "tail", "less", "more", "wc", "file",
        "ls", "find", "tree", "du", "df", "stat",
        "grep", "rg", "ag", "awk", "sed",
        "git status", "git log", "git diff", "git show", "git branch",
        "git remote", "git tag", "git rev-parse", "git describe",
        "python -c", "python3 -c",
        "which", "whereis", "type",
        "echo", "printf", "date", "uname", "whoami", "pwd", "env",
        "diff", "md5sum", "sha256sum",
    },

    # Test/build: safe, produces build/test artifacts
    PERMISSION_TEST_BUILD: {
        "python -m pytest", "python3 -m pytest",
        "python -m unittest", "python3 -m unittest",
        "python -m py_compile", "python3 -m py_compile",
        "python -m mypy", "python3 -m mypy",
        "python -m flake8", "python3 -m flake8",
        "python -m black", "python3 -m black",
        "python -m ruff", "python3 -m ruff",
        "make", "cmake", "ninja",
        "cargo build", "cargo test", "cargo check",
        "npm test", "npm run", "npm run build", "npm run lint",
        "yarn test", "yarn build", "yarn lint",
        "pnpm test", "pnpm build", "pnpm lint",
        "go test", "go build", "go vet",
        "pytest", "mypy", "flake8", "black", "ruff",
        "gcc", "g++", "clang", "clang++", "rustc",
    },

    # Mutating: modifies files or git state, needs approval
    PERMISSION_MUTATING: {
        "git add", "git commit", "git checkout", "git switch",
        "git merge", "git rebase", "git stash",
        "git push", "git pull", "git fetch",
        "pip install", "pip3 install",
        "python -m pip install", "python3 -m pip install",
        "npm install", "yarn install", "pnpm install",
        "mkdir", "touch", "cp", "mv",
        "cargo add", "cargo update",
    },

    # Destructive: deletes files, discards history, kills processes
    PERMISSION_DESTRUCTIVE: {
        "rm", "rmdir",
        "git reset", "git clean", "git push --force", "git restore",
        "pip uninstall", "pip3 uninstall",
        "python -m pip uninstall", "python3 -m pip uninstall",
        "npm uninstall", "yarn remove", "pnpm remove",
        "kill", "killall", "pkill",
    },

    # Privileged: administrative, system-level, permissions
    PERMISSION_PRIVILEGED: {
        "sudo", "su", "doas",
        "chmod", "chown", "chgrp",
        "systemctl", "service",
        "visudo", "passwd", "useradd", "userdel", "usermod",
        "iptables", "ip6tables", "nft",
    },
}

# Commands that are unconditionally blocked (never executed under any circumstances)
_BLOCKED_COMMANDS: Set[str] = {
    # System destruction and control
    "reboot", "shutdown", "poweroff", "halt", "init",
    "mkfs", "fdisk", "parted", "dd",
    # Shell interpreters: blocked to enforce shell=False policy
    "bash", "sh", "zsh", "dash", "csh", "tcsh", "ksh",
}


# ---------------------------------------------------------------------------
# Command Validation & Classification
# ---------------------------------------------------------------------------

def validate_command_string(command_str: str) -> List[str]:
    """
    Validate and parse a command string.

    1. Checks for shell control operators and command injection.
    2. Parses arguments using shlex.split().

    Raises ValueError on dangerous or malformed input.
    """
    command_str = command_str.strip()
    if not command_str:
        raise ValueError("Empty command string.")

    # Reject shell operators BEFORE parsing
    match = _SHELL_OPERATORS_RE.search(command_str)
    if match:
        raise ValueError(
            f"🚫 Shell operator detected: '{match.group()}'\n"
            f"  Command: {command_str}\n"
            "  SWAN does not execute shell pipelines, redirections, "
            "backgrounding, or command chains.\n"
            "  Split into separate commands or use file tools."
        )

    try:
        args = shlex.split(command_str)
    except ValueError as e:
        raise ValueError(f"Failed to parse command: {e}\n  Input: {command_str}")

    if not args:
        raise ValueError("Command parsed to empty argument list.")

    return args


def classify_command(args: List[str]) -> str:
    """
    Classify a command by its permission level.

    Returns:
        PERMISSION_READ_ONLY
        PERMISSION_TEST_BUILD
        PERMISSION_MUTATING
        PERMISSION_DESTRUCTIVE
        PERMISSION_PRIVILEGED
        PERMISSION_UNKNOWN
    """
    if not args:
        return PERMISSION_UNKNOWN

    cmd = args[0]
    cmd_base = os.path.basename(cmd)

    # Check for sudo/su/doas specifically
    if cmd_base in ("sudo", "su", "doas"):
        return PERMISSION_PRIVILEGED

    # Build progressively longer command prefixes to match (e.g. 'git', 'git status', 'git push --force')
    prefixes = [cmd_base]
    for i in range(1, min(len(args), 4)):
        prefixes.append(" ".join([cmd_base] + args[1:i+1]))

    # Match in reverse order (most specific first)
    for prefix in reversed(prefixes):
        for level in [
            PERMISSION_PRIVILEGED,
            PERMISSION_DESTRUCTIVE,
            PERMISSION_MUTATING,
            PERMISSION_TEST_BUILD,
            PERMISSION_READ_ONLY,
        ]:
            if prefix in _COMMAND_PERMISSIONS[level]:
                return level

    # Check if base executable is known
    for level in [
        PERMISSION_PRIVILEGED,
        PERMISSION_DESTRUCTIVE,
        PERMISSION_MUTATING,
        PERMISSION_TEST_BUILD,
        PERMISSION_READ_ONLY,
    ]:
        if cmd_base in _COMMAND_PERMISSIONS[level]:
            return level

    return PERMISSION_UNKNOWN


def is_blocked_command(args: List[str]) -> Optional[str]:
    """
    Check if a command is unconditionally blocked.

    Returns a reason string if blocked, None if allowed.
    """
    if not args:
        return "Empty command"

    cmd_base = os.path.basename(args[0])

    # Prevent running shell scripts directly through shell binaries
    if cmd_base in ("bash", "sh", "zsh", "dash", "csh", "tcsh", "ksh"):
        return f"Direct shell interpreter '{cmd_base}' is blocked to enforce shell=False policy."

    if cmd_base in _BLOCKED_COMMANDS:
        return f"Command '{cmd_base}' is blocked for security reasons."

    return None


# ---------------------------------------------------------------------------
# Python Interpreter Discovery & PEP 668 Protection
# ---------------------------------------------------------------------------

def discover_python(workspace: Path) -> str:
    """
    Discover the best Python interpreter for a workspace.

    Priority:
      1. .venv/bin/python (workspace-local)
      2. venv/bin/python (workspace-local)
      3. sys.executable (current interpreter)
      4. python3 from PATH

    Returns the path to the Python interpreter.
    Raises RuntimeError if no interpreter is found.
    """
    workspace = workspace.resolve()

    # Priority 1: .venv
    venv_py = workspace / ".venv" / "bin" / "python"
    if venv_py.is_file() and os.access(str(venv_py), os.X_OK):
        return str(venv_py.resolve())

    # Priority 2: venv
    venv_py = workspace / "venv" / "bin" / "python"
    if venv_py.is_file() and os.access(str(venv_py), os.X_OK):
        return str(venv_py.resolve())

    # Priority 3: sys.executable
    if sys.executable and os.path.isfile(sys.executable):
        return str(Path(sys.executable).resolve())

    # Priority 4: python3 from PATH
    which_py = shutil.which("python3")
    if which_py:
        return str(Path(which_py).resolve())

    raise RuntimeError(
        "No Python interpreter found. Checked:\n"
        f"  1. {workspace / '.venv/bin/python'}\n"
        f"  2. {workspace / 'venv/bin/python'}\n"
        f"  3. sys.executable={sys.executable}\n"
        "  4. python3 from PATH"
    )


def check_pep668(python_path: str) -> Optional[str]:
    """
    Check if the Python installation is PEP 668 externally managed.

    Virtual environments with pyvenv.cfg are NOT externally managed.
    System installations with an EXTERNALLY-MANAGED marker are detected.

    Returns a warning recommendation if externally managed, None otherwise.
    """
    try:
        p = Path(python_path).resolve()

        # In a virtualenv, pyvenv.cfg is in the venv root
        if (p.parent.parent / "pyvenv.cfg").is_file() or (p.parent / "pyvenv.cfg").is_file():
            return None

        # Check standard system stdlib marker locations
        stdlib_candidates = [
            p.parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}",
            Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}"),
            Path("/usr/lib/python3"),
        ]
        for candidate in stdlib_candidates:
            marker = candidate / "EXTERNALLY-MANAGED"
            if marker.is_file():
                return _format_pep668_warning(str(p))

        # Query the interpreter directly
        res = subprocess.run(
            [str(p), "-c", "import sysconfig, pathlib; p = pathlib.Path(sysconfig.get_path('stdlib')) / 'EXTERNALLY-MANAGED'; print('YES' if p.exists() else 'NO')"],
            capture_output=True, text=True, timeout=5, shell=False
        )
        if res.returncode == 0 and res.stdout.strip() == "YES":
            return _format_pep668_warning(str(p))
    except Exception:
        pass

    return None


def _format_pep668_warning(python_path: str) -> str:
    return (
        f"⚠ Python at '{python_path}' is externally managed (PEP 668).\n"
        "  Do NOT install packages into system Python.\n"
        "  Recommendation: Create and use a project virtual environment:\n"
        f"    {python_path} -m venv .venv\n"
        "    .venv/bin/pip install <package>"
    )


# ---------------------------------------------------------------------------
# Path Containment & File Security
# ---------------------------------------------------------------------------

def validate_file_path(
    requested_path: str,
    workspace: Path,
    must_exist: bool = False,
) -> Path:
    """
    Validate and resolve a file path against the workspace root.

    Security checks:
    - Resolves to absolute path
    - Prevents ../ traversal outside workspace
    - Prevents symlink escapes pointing outside workspace
    - Rejects paths outside workspace boundary

    Returns the resolved absolute Path.
    Raises ValueError on security violations.
    """
    workspace = workspace.resolve()

    if os.path.isabs(requested_path):
        target = Path(requested_path)
    else:
        target = workspace / requested_path

    resolved = target.resolve()

    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise ValueError(
            f"🚫 Path escapes workspace boundary.\n"
            f"  Requested: {requested_path}\n"
            f"  Resolved:  {resolved}\n"
            f"  Workspace: {workspace}\n"
            "  This violates workspace containment."
        )

    if must_exist and not resolved.exists():
        raise ValueError(f"Path does not exist: {resolved}")

    return resolved


def atomic_write_file(
    target: Path,
    content: str,
    workspace: Path,
    encoding: str = "utf-8",
) -> Path:
    """
    Atomically write content to a file within the workspace.

    Ensures parent directories exist.
    Uses tempfile.mkstemp() + os.fsync() + os.replace() in the target directory.

    Returns the resolved Path of the written file.
    Raises ValueError if target is outside workspace.
    """
    resolved = validate_file_path(str(target), workspace)

    if resolved.is_dir():
        raise ValueError(f"Target path is an existing directory: {resolved}")

    resolved.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(resolved.parent),
        suffix=".swan_tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fp:
            fp.write(content)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp_path, str(resolved))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return resolved


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """
    Hardened Execution Engine for SWAN agents.

    Enforces:
    - shell=False execution exclusively
    - Strict path containment
    - Permission classification & human approval policy
    - Interpreter resolution & PEP 668 protection
    - Bounded execution time and output
    - Failure tracking and loop prevention
    """

    DEFAULT_TIMEOUT: int = 30
    MAX_TIMEOUT: int = 120
    MAX_OUTPUT_BYTES: int = 8192
    MAX_RETRIES_PER_FAILURE: int = 2

    def __init__(
        self,
        workspace: Path,
        approval_mode: str = "prompt",
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Parameters
        ----------
        workspace : Path
            The active project workspace root.
        approval_mode : str
            - "prompt": auto-approve read-only & test/build; require approval for mutating, destructive, privileged
            - "auto": auto-approve read-only & test/build; require approval for mutating; destructive & privileged ALWAYS require approval
            - "strict": require approval for everything except read-only
        timeout : int
            Default command timeout in seconds.
        """
        self.workspace = workspace.resolve()
        self.approval_mode = approval_mode
        self.timeout = min(timeout, self.MAX_TIMEOUT)
        self._python: Optional[str] = None
        self._failure_counts: Dict[str, int] = {}

    @property
    def python(self) -> str:
        """Lazy-discover the Python interpreter."""
        if self._python is None:
            self._python = discover_python(self.workspace)
        return self._python

    def get_failure_count(self, failure_class: str) -> int:
        """Return the number of recorded failures for a command."""
        return self._failure_counts.get(failure_class, 0)

    def reset_failures(self) -> None:
        """Reset all failure counters."""
        self._failure_counts.clear()

    def _truncate_output(self, output: str) -> str:
        """Truncate output to MAX_OUTPUT_BYTES safely without breaking UTF-8."""
        encoded = output.encode("utf-8", errors="replace")
        if len(encoded) > self.MAX_OUTPUT_BYTES:
            truncated = encoded[:self.MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
            return truncated + f"\n... [truncated at {self.MAX_OUTPUT_BYTES} bytes]"
        return output

    def _get_failure_key(self, args: List[str]) -> str:
        """Derive a failure classification key from command arguments."""
        if not args:
            return "unknown"
        base = os.path.basename(args[0])
        if len(args) > 1 and not args[1].startswith("-"):
            return f"{base} {args[1]}"
        return base

    def _record_failure(self, failure_class: str) -> bool:
        """Record a command failure. Returns True if retry limit not exceeded."""
        count = self._failure_counts.get(failure_class, 0) + 1
        self._failure_counts[failure_class] = count
        return count <= self.MAX_RETRIES_PER_FAILURE

    def _clear_failure(self, failure_class: str) -> None:
        """Clear failure count after a successful execution."""
        self._failure_counts.pop(failure_class, None)

    def _needs_approval(self, permission: str, args: List[str]) -> bool:
        """Check if an operation requires explicit user approval."""
        # Privileged commands (like sudo) and destructive operations always require approval
        if permission in (PERMISSION_DESTRUCTIVE, PERMISSION_PRIVILEGED, PERMISSION_UNKNOWN):
            return True

        if self.approval_mode == "strict":
            return permission != PERMISSION_READ_ONLY

        # In prompt or auto mode: mutating requires approval
        if permission == PERMISSION_MUTATING:
            return True

        return False

    def _resolve_python_command(self, args: List[str]) -> List[str]:
        """
        Replace 'python' or 'python3' with discovered interpreter.
        Route 'pip' or 'pip3' through [python, -m, pip].
        """
        if not args:
            return args

        cmd = os.path.basename(args[0])
        if cmd in ("python", "python3"):
            return [self.python] + args[1:]
        if cmd in ("pip", "pip3"):
            return [self.python, "-m", "pip"] + args[1:]
        return args

    def execute_command(
        self,
        command_str: str,
        timeout: Optional[int] = None,
        require_approval: bool = True,
        approved: bool = False,
    ) -> dict:
        """
        Securely parse, validate, and execute a command with shell=False.

        Parameters
        ----------
        command_str : str
            The command string to execute.
        timeout : int, optional
            Timeout override.
        require_approval : bool
            Whether to enforce approval policies.
        approved : bool
            Whether user has explicitly approved this execution.

        Returns
        -------
        dict with:
            "success": bool
            "stdout": str
            "stderr": str
            "returncode": int
            "approved": bool
            "blocked": Optional[str]
            "permission": str
            "command": List[str]
        """
        result = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "approved": True,
            "blocked": None,
            "permission": PERMISSION_UNKNOWN,
            "command": [],
        }

        # 1. Parse and validate syntax
        try:
            args = validate_command_string(command_str)
            result["command"] = args
        except ValueError as e:
            result["blocked"] = str(e)
            return result

        # 2. Check blocked list
        blocked_reason = is_blocked_command(args)
        if blocked_reason:
            result["blocked"] = blocked_reason
            return result

        # 3. Classify permission
        permission = classify_command(args)
        result["permission"] = permission

        # 4. Check approval
        if require_approval and not approved and self._needs_approval(permission, args):
            result["approved"] = False
            result["blocked"] = (
                f"Command requires explicit user approval (permission: {permission}).\n"
                f"  Command: {command_str}\n"
                "  Approve this operation to proceed."
            )
            return result

        # 5. Check repeated failure loop
        failure_key = self._get_failure_key(args)
        if self._failure_counts.get(failure_key, 0) >= self.MAX_RETRIES_PER_FAILURE:
            result["blocked"] = (
                f"Command '{failure_key}' has failed {self._failure_counts[failure_key]} times. "
                "Recovery exhausted — stopping to prevent infinite loops."
            )
            return result

        # 6. Check PEP 668 for pip install attempts
        cmd_base = os.path.basename(args[0])
        is_pip_install = (
            (cmd_base in ("pip", "pip3") and "install" in args) or
            (cmd_base in ("python", "python3") and len(args) > 2 and args[1] == "-m" and args[2] == "pip" and "install" in args)
        )
        if is_pip_install:
            pep668_msg = check_pep668(self.python)
            if pep668_msg:
                result["blocked"] = pep668_msg
                result["stderr"] = pep668_msg
                result["returncode"] = 1
                return result

        # 7. Resolve Python interpreter
        resolved_args = self._resolve_python_command(args)

        # 8. Execute via subprocess with shell=False exclusively
        effective_timeout = min(timeout or self.timeout, self.MAX_TIMEOUT)

        try:
            proc = subprocess.run(
                resolved_args,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                shell=False,  # CRITICAL: shell=False exclusively
            )
            result["stdout"] = self._truncate_output(proc.stdout)
            result["stderr"] = self._truncate_output(proc.stderr)
            result["returncode"] = proc.returncode
            result["success"] = (proc.returncode == 0)

            if proc.returncode != 0:
                self._record_failure(failure_key)
            else:
                self._clear_failure(failure_key)

        except subprocess.TimeoutExpired:
            result["stderr"] = f"⏱ Command timed out after {effective_timeout} seconds: {' '.join(resolved_args)}"
            self._record_failure(failure_key)

        except FileNotFoundError:
            result["stderr"] = f"Executable not found: {resolved_args[0]}"
            result["blocked"] = f"Executable not found: {resolved_args[0]}"

        except PermissionError:
            result["stderr"] = f"Permission denied executing: {resolved_args[0]}"
            result["blocked"] = f"Permission denied executing: {resolved_args[0]}"

        except OSError as e:
            result["stderr"] = f"OS error executing command: {e}"
            self._record_failure(failure_key)

        return result

    # ------ File Operations ------

    def read_file(self, path_str: str) -> Tuple[bool, str]:
        """
        Read a file within the workspace.

        Returns (success: bool, content_or_error: str).
        """
        try:
            resolved = validate_file_path(path_str, self.workspace, must_exist=True)
            if not resolved.is_file():
                return False, f"Not a regular file: {resolved}"
            content = resolved.read_text(encoding="utf-8", errors="replace")
            return True, self._truncate_output(content)
        except (ValueError, OSError) as e:
            return False, str(e)

    def write_file(
        self,
        path_str: str,
        content: str,
        overwrite_approved: bool = False,
    ) -> Tuple[bool, str]:
        """
        Atomically write a file within the workspace.
        Requires explicit approval if overwriting an existing file.

        Returns (success: bool, message: str).
        """
        try:
            resolved = validate_file_path(path_str, self.workspace)

            if resolved.is_dir():
                return False, f"Target path is an existing directory: {resolved}"

            if resolved.exists() and not overwrite_approved:
                return False, (
                    f"⚠ File already exists: {resolved}\n"
                    "  Explicit overwrite approval required."
                )

            written = atomic_write_file(resolved, content, self.workspace)
            return True, f"✔ Written: {written}"
        except (ValueError, OSError) as e:
            return False, str(e)

    def list_dir(self, path_str: str = ".") -> Tuple[bool, str]:
        """
        List directory contents within the workspace.
        Excludes hidden files and caps at 200 items.

        Returns (success: bool, listing_or_error: str).
        """
        try:
            resolved = validate_file_path(path_str, self.workspace, must_exist=True)
            if not resolved.is_dir():
                return False, f"Not a directory: {resolved}"

            entries = sorted(resolved.iterdir())
            visible = [
                p.name + ("/" if p.is_dir() else "")
                for p in entries
                if not p.name.startswith(".")
            ]
            if len(visible) > 200:
                visible = visible[:200] + [f"... ({len(visible) - 200} more entries)"]
            return True, "\n".join(visible) if visible else "(empty directory)"
        except (ValueError, OSError) as e:
            return False, str(e)
