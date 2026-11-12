"""
SWAN-ENGINE v0.2 — Workspace Manager
=====================================

Multi-workspace discovery, registration, and resolution.

Workspace Resolution Priority:
    1. Explicit --workspace CLI argument
    2. SWAN_WORKSPACE environment variable
    3. Registered project containing current working directory
    4. Git repository root (session-only if unregistered)
    5. Configured default project

Schema: projects.json
    {
        "version": 1,
        "projects": {
            "<project_id>": {
                "name": "Human Readable Name",
                "path": "/absolute/resolved/path",
                "created": "ISO-8601 timestamp",
                "model": "swan-coder"
            }
        },
        "default_project": "<project_id>" | null
    }

Schema: config.json
    {
        "version": 1,
        "default_model": "swan-coder",
        "context_window": 16384,
        "max_steps": 8
    }
"""

import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Platform-dependent file locking
# ---------------------------------------------------------------------------
try:
    import fcntl

    def _lock(fp):
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)

    def _unlock(fp):
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)

except ImportError:
    # Windows / platforms without fcntl — advisory no-op
    def _lock(fp):
        pass

    def _unlock(fp):
        pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path("~/.config/swan").expanduser()
_PROJECTS_FILE = _CONFIG_DIR / "projects.json"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

_EMPTY_REGISTRY = {
    "version": 1,
    "projects": {},
    "default_project": None,
}

_EMPTY_CONFIG = {
    "version": 1,
    "default_model": "swan-coder",
    "context_window": 16384,
    "max_steps": 8,
}

# Project ID rules: lowercase alphanumeric + hyphens, 1–48 chars
_PROJECT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,47}$")

# Directories that strongly indicate a project root
_PROJECT_MARKERS = {
    ".git", "pyproject.toml", "setup.py", "setup.cfg",
    "Cargo.toml", "package.json", "Makefile", "CMakeLists.txt",
    "go.mod", "pom.xml", "build.gradle",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(directory: Path) -> Path:
    """Create a directory (and parents) if it does not exist."""
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _atomic_json_write(path: Path, data: dict) -> None:
    """
    Write *data* to *path* atomically.

    Uses a temporary file on the same filesystem followed by os.replace()
    so the write is crash-safe on Linux ext4/btrfs.
    """
    parent = path.parent
    _ensure_dir(parent)

    fd, tmp_path = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp_path, str(path))
    except BaseException:
        # Clean up the temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _read_json(path: Path, default: dict) -> dict:
    """Read a JSON file, returning a deep copy of *default* if missing or corrupt."""
    if not path.is_file():
        return copy.deepcopy(default)
    try:
        with open(path, "r", encoding="utf-8") as fp:
            _lock(fp)
            try:
                data = json.load(fp)
            finally:
                _unlock(fp)
        if not isinstance(data, dict):
            return copy.deepcopy(default)
        return data
    except (json.JSONDecodeError, OSError):
        return copy.deepcopy(default)


def _locked_update(path: Path, default: dict, updater) -> dict:
    """
    Read JSON, apply *updater(data)* which should return the new data,
    and atomically write it back while holding the file lock.

    Returns the updated data dict.
    """
    _ensure_dir(path.parent)

    # Create file if it doesn't exist
    if not path.exists():
        _atomic_json_write(path, copy.deepcopy(default))

    with open(path, "r+", encoding="utf-8") as fp:
        _lock(fp)
        try:
            try:
                data = json.load(fp)
            except json.JSONDecodeError:
                data = copy.deepcopy(default)

            new_data = updater(data)
            if new_data is None:
                new_data = data

            # Write atomically (release lock after replace)
            _atomic_json_write(path, new_data)
            return new_data
        finally:
            _unlock(fp)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_project_id(project_id: str) -> str:
    """
    Validate and normalize a project ID.

    Returns the cleaned ID or raises ValueError.
    """
    pid = project_id.strip().lower()
    if not _PROJECT_ID_RE.match(pid):
        raise ValueError(
            f"Invalid project ID '{project_id}'. "
            "Must be 1-48 chars, lowercase alphanumeric and hyphens, "
            "starting with a letter or digit."
        )
    return pid


def validate_workspace_path(path_str: str) -> Path:
    """
    Validate a workspace path.

    - Expands ~ and resolves symlinks.
    - Rejects paths that don't exist or aren't directories.
    - Rejects the home directory itself (never scan entire home).

    Returns the resolved absolute Path.
    """
    p = Path(path_str).expanduser().resolve()

    if not p.exists():
        raise ValueError(f"Path does not exist: {p}")
    if not p.is_dir():
        raise ValueError(f"Path is not a directory: {p}")

    home = Path.home().resolve()
    if p == home:
        raise ValueError(
            "Refusing to use the home directory as a workspace. "
            "Please specify a project subdirectory."
        )

    return p


def is_project_root(path: Path) -> bool:
    """Check whether *path* looks like a project root directory."""
    if not path.is_dir():
        return False
    for marker in _PROJECT_MARKERS:
        if (path / marker).exists():
            return True
    return False


# ---------------------------------------------------------------------------
# Git root detection
# ---------------------------------------------------------------------------

def detect_git_root(start: Optional[Path] = None) -> Optional[Path]:
    """
    Walk up from *start* (default: cwd) to find the nearest .git root.

    Returns the resolved Path or None.
    """
    if start is None:
        start = Path.cwd()
    start = start.resolve()
    current = start
    home = Path.home().resolve()

    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            # Reached filesystem root
            return None
        if parent == home:
            # Don't climb above home
            return None
        current = parent


# ---------------------------------------------------------------------------
# WorkspaceManager
# ---------------------------------------------------------------------------

class WorkspaceManager:
    """
    Manages SWAN project workspaces: registration, discovery, resolution.
    """

    def __init__(
        self,
        projects_file: Optional[Path] = None,
        config_file: Optional[Path] = None,
    ):
        self._projects_file = projects_file or _PROJECTS_FILE
        self._config_file = config_file or _CONFIG_FILE
        self._config_dir = self._projects_file.parent

    # ------ Registry I/O ------

    def _load_registry(self) -> dict:
        return _read_json(self._projects_file, _EMPTY_REGISTRY)

    def _save_registry(self, data: dict) -> None:
        _atomic_json_write(self._projects_file, data)

    def _load_config(self) -> dict:
        return _read_json(self._config_file, _EMPTY_CONFIG)

    # ------ Project CRUD ------

    def list_projects(self) -> dict:
        """
        Return the projects dict: {project_id: {name, path, created, model}}.
        """
        reg = self._load_registry()
        return dict(reg.get("projects", {}))

    def get_default_project(self) -> Optional[str]:
        """Return the default project ID, or None."""
        reg = self._load_registry()
        return reg.get("default_project")

    def add_project(
        self,
        project_id: str,
        name: str,
        path: str,
        model: str = "swan-coder",
    ) -> dict:
        """
        Register a new project.

        Returns the project entry dict.
        Raises ValueError on invalid input or duplicate.
        """
        pid = validate_project_id(project_id)
        resolved = validate_workspace_path(path)
        name = name.strip()
        if not name:
            raise ValueError("Project name cannot be empty.")

        def updater(data):
            projects = data.setdefault("projects", {})
            # Check duplicate ID
            if pid in projects:
                raise ValueError(
                    f"Project '{pid}' is already registered at "
                    f"'{projects[pid]['path']}'."
                )
            # Check duplicate path
            for existing_id, entry in projects.items():
                existing_path = Path(entry["path"]).resolve()
                if existing_path == resolved:
                    raise ValueError(
                        f"Path '{resolved}' is already registered as "
                        f"project '{existing_id}'."
                    )

            entry = {
                "name": name,
                "path": str(resolved),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "model": model,
            }
            projects[pid] = entry
            # If this is the first project, make it default
            if data.get("default_project") is None and len(projects) == 1:
                data["default_project"] = pid
            data["projects"] = projects
            return data

        _locked_update(self._projects_file, _EMPTY_REGISTRY, updater)

        # Re-read to return the entry
        reg = self._load_registry()
        return reg["projects"][pid]

    def remove_project(self, project_id: str) -> bool:
        """
        Remove a registered project.

        Returns True if removed, raises ValueError if not found.
        """
        pid = validate_project_id(project_id)
        removed = False

        def updater(data):
            nonlocal removed
            projects = data.get("projects", {})
            if pid not in projects:
                raise ValueError(f"Project '{pid}' is not registered.")
            del projects[pid]
            data["projects"] = projects
            if data.get("default_project") == pid:
                data["default_project"] = None
            removed = True
            return data

        _locked_update(self._projects_file, _EMPTY_REGISTRY, updater)
        return removed

    def set_default_project(self, project_id: str) -> None:
        """
        Set the default project.

        Raises ValueError if the project is not registered.
        """
        pid = validate_project_id(project_id)

        def updater(data):
            projects = data.get("projects", {})
            if pid not in projects:
                raise ValueError(
                    f"Project '{pid}' is not registered. "
                    "Register it first with 'swan projects add'."
                )
            data["default_project"] = pid
            return data

        _locked_update(self._projects_file, _EMPTY_REGISTRY, updater)

    def get_project(self, project_id: str) -> Optional[dict]:
        """Return a single project entry or None."""
        pid = validate_project_id(project_id)
        projects = self.list_projects()
        return projects.get(pid)

    # ------ Workspace Resolution ------

    def find_project_by_path(self, search_path: Path) -> Optional[str]:
        """
        Find a registered project that contains *search_path*.

        Returns the project_id or None.
        """
        search_path = search_path.resolve()
        projects = self.list_projects()
        for pid, entry in projects.items():
            project_path = Path(entry["path"]).resolve()
            try:
                search_path.relative_to(project_path)
                return pid
            except ValueError:
                continue
        return None

    def resolve_workspace(
        self,
        cli_workspace: Optional[str] = None,
        env_workspace: Optional[str] = None,
        cwd: Optional[Path] = None,
    ) -> tuple:
        """
        Resolve the active workspace using the priority chain.

        Parameters
        ----------
        cli_workspace : str, optional
            Explicit --workspace argument from CLI.
        env_workspace : str, optional
            Value of SWAN_WORKSPACE environment variable.
            If None, reads from os.environ automatically.
        cwd : Path, optional
            Current working directory. Defaults to Path.cwd().

        Returns
        -------
        (workspace_path: Path, project_id: str | None, source: str)
            workspace_path: Resolved absolute path to workspace root.
            project_id: The registered project ID, or None for session-only.
            source: One of 'cli', 'env', 'registered', 'git', 'default'.

        Raises
        ------
        ValueError
            If no workspace can be resolved.
        """
        if cwd is None:
            cwd = Path.cwd()
        cwd = cwd.resolve()

        if env_workspace is None:
            env_workspace = os.environ.get("SWAN_WORKSPACE")

        # Priority 1: Explicit CLI argument
        if cli_workspace:
            ws_path = validate_workspace_path(cli_workspace)
            # Check if this path is a registered project
            pid = self.find_project_by_path(ws_path)
            return (ws_path, pid, "cli")

        # Priority 2: Environment variable
        if env_workspace:
            ws_path = validate_workspace_path(env_workspace)
            pid = self.find_project_by_path(ws_path)
            return (ws_path, pid, "env")

        # Priority 3: Registered project containing cwd
        pid = self.find_project_by_path(cwd)
        if pid is not None:
            entry = self.get_project(pid)
            if entry:
                ws_path = Path(entry["path"]).resolve()
                return (ws_path, pid, "registered")

        # Priority 4: Git repository root (session-only)
        git_root = detect_git_root(cwd)
        if git_root is not None:
            return (git_root, None, "git")

        # Priority 5: Default project
        default_pid = self.get_default_project()
        if default_pid is not None:
            entry = self.get_project(default_pid)
            if entry:
                default_path = Path(entry["path"]).resolve()
                if default_path.is_dir():
                    return (default_path, default_pid, "default")

        raise ValueError(
            "Cannot resolve workspace. Options:\n"
            "  - Use: swan --workspace /path/to/project <command>\n"
            "  - Set: SWAN_WORKSPACE=/path/to/project\n"
            "  - Register: swan projects add --name 'My Project' "
            "--path /path/to/project\n"
            "  - Run from inside a Git repository"
        )

    # ------ Project State Directories ------

    def get_project_state_dir(self, project_id: str) -> Path:
        """
        Return the state directory for a project.

        Creates it if it does not exist.
        Located at <config_dir>/state/projects/<project_id>/
        """
        state_dir = self._config_dir / "state" / "projects" / project_id
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir

    def get_global_state_dir(self) -> Path:
        """
        Return the global state directory.

        Located at <config_dir>/state/
        """
        state_dir = self._config_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir

    # ------ Display Helpers ------

    def format_project_list(self) -> str:
        """Return a formatted string listing all registered projects."""
        projects = self.list_projects()
        default_pid = self.get_default_project()

        if not projects:
            return "No projects registered.\nUse: swan projects add --name 'Name' --path /path"

        lines = []
        lines.append(f"{'ID':<20} {'Name':<25} {'Path':<40} {'Default'}")
        lines.append("-" * 90)
        for pid, entry in sorted(projects.items()):
            marker = " ★" if pid == default_pid else ""
            name = entry.get("name", "")[:24]
            path = entry.get("path", "")
            # Truncate long paths
            if len(path) > 39:
                path = "..." + path[-36:]
            lines.append(f"{pid:<20} {name:<25} {path:<40}{marker}")
        return "\n".join(lines)
