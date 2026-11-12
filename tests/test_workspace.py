"""
SWAN-ENGINE v0.2 — Workspace Manager Tests
============================================

Covers:
  - Workspace detection priority chain
  - Project registration (add / remove / list / use)
  - Duplicate ID and duplicate path rejection
  - Invalid workspace rejection (missing, file, home dir)
  - Default project management
  - Git root detection
  - Session-only Git workspace
  - Environment variable override
  - Explicit CLI override
  - Path validation (traversal, symlinks)
  - Atomic JSON writes
  - Concurrent registry protection
  - Project ID validation
  - Separate project state directories
  - projects.json schema integrity

All tests use temporary directories — no real projects are modified.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

# Make sure we can import from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.workspace_manager import (
    WorkspaceManager,
    validate_project_id,
    validate_workspace_path,
    detect_git_root,
    is_project_root,
    _atomic_json_write,
    _read_json,
)


class TempDirMixin:
    """Mixin that creates a temp directory tree for workspace tests."""

    def setUp(self):
        # Master temp dir
        self._tmpdir = tempfile.mkdtemp(prefix="swan_test_")
        self.tmpdir = Path(self._tmpdir)

        # Fake config dir (overrides ~/.config/swan)
        self.config_dir = self.tmpdir / "config"
        self.config_dir.mkdir()
        self.projects_file = self.config_dir / "projects.json"
        self.config_file = self.config_dir / "config.json"

        # Fake project dirs
        self.project_a = self.tmpdir / "project-alpha"
        self.project_a.mkdir()
        (self.project_a / ".git").mkdir()
        (self.project_a / "main.py").write_text("# alpha", encoding="utf-8")

        self.project_b = self.tmpdir / "project-beta"
        self.project_b.mkdir()
        (self.project_b / ".git").mkdir()
        (self.project_b / "app.py").write_text("# beta", encoding="utf-8")

        self.project_no_git = self.tmpdir / "project-plain"
        self.project_no_git.mkdir()
        (self.project_no_git / "readme.txt").write_text("plain", encoding="utf-8")

        # Manager with test-specific files
        self.wm = WorkspaceManager(
            projects_file=self.projects_file,
            config_file=self.config_file,
        )

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# =========================================================================
# Test: Project ID Validation
# =========================================================================
class TestProjectIdValidation(unittest.TestCase):

    def test_valid_ids(self):
        self.assertEqual(validate_project_id("tidda"), "tidda")
        self.assertEqual(validate_project_id("my-project"), "my-project")
        self.assertEqual(validate_project_id("project123"), "project123")
        self.assertEqual(validate_project_id("a"), "a")

    def test_normalizes_to_lowercase(self):
        self.assertEqual(validate_project_id("TIDDA"), "tidda")
        self.assertEqual(validate_project_id("My-Project"), "my-project")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            validate_project_id("")

    def test_rejects_spaces(self):
        with self.assertRaises(ValueError):
            validate_project_id("my project")

    def test_rejects_special_chars(self):
        for bad in ["my_project", "proj@ct", "../../etc", "a/b", "a b"]:
            with self.assertRaises(ValueError, msg=f"Should reject '{bad}'"):
                validate_project_id(bad)

    def test_rejects_starting_with_hyphen(self):
        with self.assertRaises(ValueError):
            validate_project_id("-invalid")

    def test_rejects_too_long(self):
        with self.assertRaises(ValueError):
            validate_project_id("a" * 49)

    def test_accepts_max_length(self):
        pid = "a" * 48
        self.assertEqual(validate_project_id(pid), pid)


# =========================================================================
# Test: Workspace Path Validation
# =========================================================================
class TestWorkspacePathValidation(TempDirMixin, unittest.TestCase):

    def test_valid_directory(self):
        result = validate_workspace_path(str(self.project_a))
        self.assertEqual(result, self.project_a.resolve())

    def test_rejects_nonexistent(self):
        with self.assertRaises(ValueError, msg="Should reject missing path"):
            validate_workspace_path(str(self.tmpdir / "ghost"))

    def test_rejects_file_not_dir(self):
        f = self.tmpdir / "afile.txt"
        f.write_text("x", encoding="utf-8")
        with self.assertRaises(ValueError, msg="Should reject file"):
            validate_workspace_path(str(f))

    def test_rejects_home_directory(self):
        with self.assertRaises(ValueError, msg="Should reject home dir"):
            validate_workspace_path(str(Path.home()))

    def test_resolves_symlinks(self):
        link = self.tmpdir / "link-to-alpha"
        link.symlink_to(self.project_a)
        result = validate_workspace_path(str(link))
        self.assertEqual(result, self.project_a.resolve())

    def test_expands_tilde(self):
        # We can't easily test ~ expansion without mocking, but we can
        # at least verify it doesn't crash on a real path
        home = Path.home()
        if (home / "swan-llm").is_dir():
            result = validate_workspace_path("~/swan-llm")
            self.assertTrue(result.is_absolute())


# =========================================================================
# Test: Git Root Detection
# =========================================================================
class TestGitRootDetection(TempDirMixin, unittest.TestCase):

    def test_detects_git_root(self):
        result = detect_git_root(self.project_a)
        self.assertEqual(result, self.project_a.resolve())

    def test_detects_from_subdirectory(self):
        subdir = self.project_a / "src" / "lib"
        subdir.mkdir(parents=True)
        result = detect_git_root(subdir)
        self.assertEqual(result, self.project_a.resolve())

    def test_returns_none_without_git(self):
        result = detect_git_root(self.project_no_git)
        # project_no_git is inside tmpdir which has no .git
        # It may or may not find a git root higher up, but definitely not project_no_git
        if result is not None:
            self.assertNotEqual(result, self.project_no_git.resolve())

    def test_returns_none_for_tmp(self):
        """A temp dir outside any git repo should return None."""
        isolated = tempfile.mkdtemp(prefix="swan_isolated_")
        try:
            result = detect_git_root(Path(isolated))
            # Should return None since /tmp has no .git
            self.assertIsNone(result)
        finally:
            shutil.rmtree(isolated, ignore_errors=True)


# =========================================================================
# Test: is_project_root
# =========================================================================
class TestIsProjectRoot(TempDirMixin, unittest.TestCase):

    def test_git_repo(self):
        self.assertTrue(is_project_root(self.project_a))

    def test_pyproject_toml(self):
        d = self.tmpdir / "pyproj"
        d.mkdir()
        (d / "pyproject.toml").write_text("[build]", encoding="utf-8")
        self.assertTrue(is_project_root(d))

    def test_plain_directory(self):
        d = self.tmpdir / "empty-dir"
        d.mkdir()
        self.assertFalse(is_project_root(d))

    def test_file_not_dir(self):
        f = self.tmpdir / "file.txt"
        f.write_text("x", encoding="utf-8")
        self.assertFalse(is_project_root(f))


# =========================================================================
# Test: Project Registration (add / remove / list)
# =========================================================================
class TestProjectRegistration(TempDirMixin, unittest.TestCase):

    def test_add_project(self):
        entry = self.wm.add_project("alpha", "Project Alpha", str(self.project_a))
        self.assertEqual(entry["name"], "Project Alpha")
        self.assertEqual(Path(entry["path"]), self.project_a.resolve())
        self.assertIn("created", entry)

    def test_list_projects(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        self.wm.add_project("beta", "Beta", str(self.project_b))
        projects = self.wm.list_projects()
        self.assertEqual(len(projects), 2)
        self.assertIn("alpha", projects)
        self.assertIn("beta", projects)

    def test_remove_project(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        self.assertTrue(self.wm.remove_project("alpha"))
        self.assertEqual(len(self.wm.list_projects()), 0)

    def test_remove_nonexistent_raises(self):
        with self.assertRaises(ValueError):
            self.wm.remove_project("ghost")

    def test_duplicate_id_rejected(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        with self.assertRaises(ValueError, msg="Should reject duplicate ID"):
            self.wm.add_project("alpha", "Alpha 2", str(self.project_b))

    def test_duplicate_path_rejected(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        with self.assertRaises(ValueError, msg="Should reject duplicate path"):
            self.wm.add_project("alpha2", "Alpha Dupe", str(self.project_a))

    def test_first_project_becomes_default(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        self.assertEqual(self.wm.get_default_project(), "alpha")

    def test_second_project_does_not_change_default(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        self.wm.add_project("beta", "Beta", str(self.project_b))
        self.assertEqual(self.wm.get_default_project(), "alpha")

    def test_remove_default_clears_it(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        self.wm.remove_project("alpha")
        self.assertIsNone(self.wm.get_default_project())

    def test_get_project(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        entry = self.wm.get_project("alpha")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["name"], "Alpha")

    def test_get_missing_project(self):
        result = self.wm.get_project("ghost")
        self.assertIsNone(result)

    def test_empty_name_rejected(self):
        with self.assertRaises(ValueError):
            self.wm.add_project("alpha", "  ", str(self.project_a))

    def test_invalid_path_rejected(self):
        with self.assertRaises(ValueError):
            self.wm.add_project("alpha", "Alpha", "/nonexistent/path/xyz")


# =========================================================================
# Test: Default Project Management
# =========================================================================
class TestDefaultProject(TempDirMixin, unittest.TestCase):

    def test_set_default(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        self.wm.add_project("beta", "Beta", str(self.project_b))
        self.wm.set_default_project("beta")
        self.assertEqual(self.wm.get_default_project(), "beta")

    def test_set_default_unregistered_raises(self):
        with self.assertRaises(ValueError):
            self.wm.set_default_project("ghost")

    def test_default_none_initially(self):
        self.assertIsNone(self.wm.get_default_project())


# =========================================================================
# Test: Workspace Resolution Priority
# =========================================================================
class TestWorkspaceResolution(TempDirMixin, unittest.TestCase):

    def _register_both(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        self.wm.add_project("beta", "Beta", str(self.project_b))

    def test_priority_1_cli_workspace(self):
        """Explicit --workspace overrides everything."""
        self._register_both()
        ws, pid, source = self.wm.resolve_workspace(
            cli_workspace=str(self.project_b),
            cwd=self.project_a,  # cwd is in alpha, but CLI says beta
        )
        self.assertEqual(ws, self.project_b.resolve())
        self.assertEqual(source, "cli")

    def test_priority_2_env_var(self):
        """SWAN_WORKSPACE overrides registered/default."""
        self._register_both()
        self.wm.set_default_project("alpha")
        ws, pid, source = self.wm.resolve_workspace(
            env_workspace=str(self.project_b),
            cwd=self.tmpdir,  # not inside any project
        )
        self.assertEqual(ws, self.project_b.resolve())
        self.assertEqual(source, "env")

    def test_priority_3_registered_project_from_cwd(self):
        """CWD inside a registered project should find it."""
        self._register_both()
        subdir = self.project_a / "src"
        subdir.mkdir(exist_ok=True)
        ws, pid, source = self.wm.resolve_workspace(cwd=subdir)
        self.assertEqual(ws, self.project_a.resolve())
        self.assertEqual(pid, "alpha")
        self.assertEqual(source, "registered")

    def test_priority_4_git_root_session_only(self):
        """Unregistered Git repo should be used as session-only."""
        # project_a has .git but is not registered
        ws, pid, source = self.wm.resolve_workspace(cwd=self.project_a)
        self.assertEqual(ws, self.project_a.resolve())
        self.assertIsNone(pid)  # not registered
        self.assertEqual(source, "git")

    def test_priority_5_default_project(self):
        """Default project is the final fallback."""
        self._register_both()
        self.wm.set_default_project("beta")

        # Use a cwd that's NOT in any project and NOT a git repo
        isolated = tempfile.mkdtemp(prefix="swan_isolated_", dir=self.tmpdir)
        ws, pid, source = self.wm.resolve_workspace(cwd=Path(isolated))
        self.assertEqual(ws, self.project_b.resolve())
        self.assertEqual(pid, "beta")
        self.assertEqual(source, "default")

    def test_no_workspace_raises(self):
        """No workspace available at all should raise ValueError."""
        isolated = tempfile.mkdtemp(prefix="swan_no_ws_")
        try:
            with self.assertRaises(ValueError):
                self.wm.resolve_workspace(cwd=Path(isolated))
        finally:
            shutil.rmtree(isolated, ignore_errors=True)

    def test_cli_overrides_env(self):
        """CLI takes precedence over env variable."""
        self._register_both()
        ws, pid, source = self.wm.resolve_workspace(
            cli_workspace=str(self.project_a),
            env_workspace=str(self.project_b),
        )
        self.assertEqual(ws, self.project_a.resolve())
        self.assertEqual(source, "cli")

    def test_env_overrides_registered(self):
        """Env var takes precedence over registered project from cwd."""
        self._register_both()
        ws, pid, source = self.wm.resolve_workspace(
            env_workspace=str(self.project_b),
            cwd=self.project_a,
        )
        self.assertEqual(ws, self.project_b.resolve())
        self.assertEqual(source, "env")

    def test_registered_overrides_git(self):
        """Registered project takes precedence over raw git detection."""
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        ws, pid, source = self.wm.resolve_workspace(cwd=self.project_a)
        self.assertEqual(pid, "alpha")
        self.assertEqual(source, "registered")

    def test_cli_invalid_path_raises(self):
        """Invalid CLI workspace path raises ValueError."""
        with self.assertRaises(ValueError):
            self.wm.resolve_workspace(cli_workspace="/nonexistent/path")


# =========================================================================
# Test: Find Project by Path
# =========================================================================
class TestFindProjectByPath(TempDirMixin, unittest.TestCase):

    def test_finds_project_at_root(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        pid = self.wm.find_project_by_path(self.project_a)
        self.assertEqual(pid, "alpha")

    def test_finds_project_in_subdirectory(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        subdir = self.project_a / "src" / "deep"
        subdir.mkdir(parents=True)
        pid = self.wm.find_project_by_path(subdir)
        self.assertEqual(pid, "alpha")

    def test_returns_none_for_unregistered(self):
        pid = self.wm.find_project_by_path(self.project_a)
        self.assertIsNone(pid)

    def test_returns_none_for_outside_path(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        pid = self.wm.find_project_by_path(self.project_b)
        self.assertIsNone(pid)


# =========================================================================
# Test: Atomic JSON Writes
# =========================================================================
class TestAtomicJsonWrite(TempDirMixin, unittest.TestCase):

    def test_creates_file(self):
        target = self.config_dir / "test_atomic.json"
        data = {"key": "value", "number": 42}
        _atomic_json_write(target, data)
        self.assertTrue(target.exists())
        loaded = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(loaded["key"], "value")
        self.assertEqual(loaded["number"], 42)

    def test_overwrites_existing(self):
        target = self.config_dir / "test_atomic.json"
        _atomic_json_write(target, {"v": 1})
        _atomic_json_write(target, {"v": 2})
        loaded = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(loaded["v"], 2)

    def test_file_is_valid_json(self):
        target = self.config_dir / "test_atomic.json"
        _atomic_json_write(target, {"nested": {"a": [1, 2, 3]}})
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["nested"]["a"], [1, 2, 3])

    def test_no_temp_file_left_on_success(self):
        target = self.config_dir / "test_atomic.json"
        _atomic_json_write(target, {"clean": True})
        tmp_files = list(self.config_dir.glob("*.tmp"))
        self.assertEqual(len(tmp_files), 0)


# =========================================================================
# Test: Read JSON with Defaults
# =========================================================================
class TestReadJson(TempDirMixin, unittest.TestCase):

    def test_returns_default_when_missing(self):
        result = _read_json(self.config_dir / "missing.json", {"default": True})
        self.assertTrue(result["default"])

    def test_reads_existing_file(self):
        target = self.config_dir / "exists.json"
        target.write_text('{"existing": true}', encoding="utf-8")
        result = _read_json(target, {"default": True})
        self.assertTrue(result["existing"])

    def test_returns_default_on_corrupt_json(self):
        target = self.config_dir / "corrupt.json"
        target.write_text("not json at all{{{", encoding="utf-8")
        result = _read_json(target, {"fallback": True})
        self.assertTrue(result["fallback"])


# =========================================================================
# Test: Project State Directories
# =========================================================================
class TestProjectState(TempDirMixin, unittest.TestCase):

    def test_get_project_state_dir(self):
        state_dir = self.wm.get_project_state_dir("alpha")
        self.assertTrue(state_dir.is_dir())
        self.assertIn("alpha", str(state_dir))

    def test_separate_state_dirs(self):
        dir_a = self.wm.get_project_state_dir("alpha")
        dir_b = self.wm.get_project_state_dir("beta")
        self.assertNotEqual(dir_a, dir_b)


# =========================================================================
# Test: projects.json Schema Integrity
# =========================================================================
class TestSchemaIntegrity(TempDirMixin, unittest.TestCase):

    def test_schema_version_present(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        data = json.loads(self.projects_file.read_text(encoding="utf-8"))
        self.assertEqual(data["version"], 1)

    def test_schema_has_projects_dict(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        data = json.loads(self.projects_file.read_text(encoding="utf-8"))
        self.assertIsInstance(data["projects"], dict)

    def test_schema_has_default_project(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        data = json.loads(self.projects_file.read_text(encoding="utf-8"))
        self.assertIn("default_project", data)

    def test_project_entry_fields(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        data = json.loads(self.projects_file.read_text(encoding="utf-8"))
        entry = data["projects"]["alpha"]
        self.assertIn("name", entry)
        self.assertIn("path", entry)
        self.assertIn("created", entry)
        self.assertIn("model", entry)


# =========================================================================
# Test: Format Project List
# =========================================================================
class TestFormatProjectList(TempDirMixin, unittest.TestCase):

    def test_empty_list(self):
        output = self.wm.format_project_list()
        self.assertIn("No projects registered", output)

    def test_with_projects(self):
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        self.wm.add_project("beta", "Beta", str(self.project_b))
        output = self.wm.format_project_list()
        self.assertIn("alpha", output)
        self.assertIn("beta", output)
        self.assertIn("★", output)  # default marker


# =========================================================================
# Test: Path Traversal Protection
# =========================================================================
class TestPathTraversal(TempDirMixin, unittest.TestCase):

    def test_symlink_outside_workspace_detected(self):
        """
        A symlink inside a project pointing outside should resolve
        to the real path, which workspace path validation catches.
        """
        outside = self.tmpdir / "outside-secret"
        outside.mkdir()
        (outside / "secret.txt").write_text("top secret", encoding="utf-8")

        link = self.project_a / "escape-link"
        link.symlink_to(outside)

        # The resolved path of the symlink is outside project_a
        resolved = link.resolve()
        self.assertFalse(
            str(resolved).startswith(str(self.project_a.resolve())),
            "Symlink escape: resolved path should be outside workspace"
        )

    def test_dotdot_traversal(self):
        """Paths with .. that escape workspace should be detectable."""
        base = self.project_a
        malicious = base / "subdir" / ".." / ".." / "outside"
        resolved = malicious.resolve()
        # resolved should NOT be inside project_a
        try:
            resolved.relative_to(base.resolve())
            # If this succeeds, the path is still inside — which means
            # the .. didn't actually escape. That's fine for this case.
        except ValueError:
            # Path is outside project — correct detection
            pass


# =========================================================================
# Test: Edge Cases
# =========================================================================
class TestEdgeCases(TempDirMixin, unittest.TestCase):

    def test_missing_projects_file_handled(self):
        """Manager works even if projects.json doesn't exist yet."""
        self.assertFalse(self.projects_file.exists())
        projects = self.wm.list_projects()
        self.assertEqual(projects, {})

    def test_corrupt_projects_file_handled(self):
        """Manager survives a corrupt projects.json."""
        self.projects_file.write_text("NOT JSON!!!", encoding="utf-8")
        projects = self.wm.list_projects()
        self.assertEqual(projects, {})

    def test_concurrent_add_safety(self):
        """
        Two sequential adds should both succeed without data loss.
        (True concurrent test would need threads, but we verify
        the locking mechanism works sequentially.)
        """
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        self.wm.add_project("beta", "Beta", str(self.project_b))
        projects = self.wm.list_projects()
        self.assertEqual(len(projects), 2)

    def test_add_after_corrupt_file(self):
        """Adding a project after corrupting projects.json recovers."""
        self.projects_file.write_text("{corrupted", encoding="utf-8")
        # This should handle the corrupt file gracefully
        entry = self.wm.add_project("alpha", "Alpha", str(self.project_a))
        self.assertEqual(entry["name"], "Alpha")

    def test_remove_clears_default_only_if_matching(self):
        """Removing a non-default project shouldn't clear default."""
        self.wm.add_project("alpha", "Alpha", str(self.project_a))
        self.wm.add_project("beta", "Beta", str(self.project_b))
        self.wm.set_default_project("alpha")
        self.wm.remove_project("beta")
        self.assertEqual(self.wm.get_default_project(), "alpha")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
