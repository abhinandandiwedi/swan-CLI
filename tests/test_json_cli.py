"""Unit tests for Structured JSON CLI and idempotency controls.

Tests non-interactive JSON input/output, clear exit codes, idempotency caching,
security violation exit codes, state conflict handling, and backward compatibility.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from core.json_cli import (
    JsonCli,
    main as json_cli_main,
    EXIT_OK,
    EXIT_VALIDATION_ERROR,
    EXIT_SECURITY_VIOLATION,
    EXIT_CONFLICT,
)
from core.task_orchestrator import (
    TaskOrchestrator,
    TaskSpec,
    TaskState,
)
from core.validator import (
    ValidationReport,
    ValidationStatus,
)


class TestJsonCliExecution(unittest.TestCase):
    """Test JSON CLI command processing and exit codes."""

    def setUp(self):
        self.orchestrator = TaskOrchestrator()
        self.cli = JsonCli(orchestrator=self.orchestrator)

    def tearDown(self):
        self.orchestrator.close()

    def test_tools_command_succeeds(self):
        exit_code, resp = self.cli.run_command("tools", {})
        self.assertEqual(exit_code, EXIT_OK)
        self.assertEqual(resp["status"], "success")
        self.assertTrue(len(resp["data"]["tools"]) >= 8)

    def test_submit_task_via_cli(self):
        payload = {
            "project_id": "test-proj",
            "title": "CLI task",
            "description": "Desc",
            "role": "coder",
            "write_scope": ["src/app.py"],
        }
        exit_code, resp = self.cli.run_command("submit_task", payload)
        self.assertEqual(exit_code, EXIT_OK)
        self.assertEqual(resp["status"], "success")
        self.assertIn("task_id", resp["data"])

    def test_security_violation_returns_exit_code_2(self):
        # Setup task in validation pending with security violation
        task = TaskSpec(
            task_id="t-sec-cli",
            batch_id="b-sec-cli",
            project_id="proj",
            title="Sec",
            description="D",
            role="coder",
        )
        self.orchestrator.submit_batch("b-sec-cli", "proj", [task])
        self.orchestrator.transition_task("t-sec-cli", TaskState.RUNNING)
        self.orchestrator.transition_task("t-sec-cli", TaskState.WORKER_COMPLETED)
        self.orchestrator.transition_task("t-sec-cli", TaskState.VALIDATION_PENDING)

        rep = ValidationReport(
            batch_id="b-sec-cli",
            task_id="t-sec-cli",
            validation_status=ValidationStatus.FAILED.value,
            out_of_scope_changes=["unauthorized_file.py"],
        )
        self.orchestrator.record_validation_report(rep)

        # Attempt to ACCEPT
        payload = {
            "batch_id": "b-sec-cli",
            "task_id": "t-sec-cli",
            "verdict": "ACCEPT",
            "reason": "Override security",
        }
        exit_code, resp = self.cli.run_command("submit_verdict", payload)
        self.assertEqual(exit_code, EXIT_SECURITY_VIOLATION)
        self.assertEqual(resp["status"], "error")
        self.assertEqual(resp["error"]["code"], "SECURITY_VIOLATION")

    def test_duplicate_task_id_returns_exit_code_3(self):
        task1 = {
            "project_id": "proj",
            "task_id": "t-dup",
            "batch_id": "b-dup",
            "title": "T1",
            "description": "D",
            "role": "coder",
            "write_scope": ["src/a.py"],
        }
        exit_code1, _ = self.cli.run_command("submit_task", task1)
        self.assertEqual(exit_code1, EXIT_OK)

        # Duplicate submission with same task_id
        task2 = {
            "project_id": "proj",
            "task_id": "t-dup",
            "batch_id": "b-dup-2",
            "title": "T2",
            "description": "D",
            "role": "coder",
            "write_scope": ["src/b.py"],
        }
        exit_code2, resp2 = self.cli.run_command("submit_task", task2)
        self.assertEqual(exit_code2, EXIT_CONFLICT)
        self.assertEqual(resp2["error"]["code"], "CONFLICT")


class TestJsonCliIdempotency(unittest.TestCase):
    """Test idempotent request caching and conflict detection."""

    def setUp(self):
        self.orchestrator = TaskOrchestrator()
        self.cli = JsonCli(orchestrator=self.orchestrator)

    def tearDown(self):
        self.orchestrator.close()

    def test_idempotent_request_returns_cached_response(self):
        payload = {
            "idempotency_key": "req-uniq-12345",
            "project_id": "test-proj",
            "title": "Idempotent Task",
            "description": "Desc",
            "role": "coder",
            "write_scope": ["src/idemp.py"],
        }
        code1, resp1 = self.cli.run_command("submit_task", payload)
        self.assertEqual(code1, EXIT_OK)
        task_id = resp1["data"]["task_id"]

        # Re-submit with same key and same payload
        code2, resp2 = self.cli.run_command("submit_task", payload)
        self.assertEqual(code2, EXIT_OK)
        self.assertEqual(resp2["data"]["task_id"], task_id)

    def test_idempotent_request_conflict_detected(self):
        payload1 = {
            "idempotency_key": "req-key-conflict",
            "project_id": "test-proj",
            "title": "Original Title",
            "description": "Desc",
            "role": "coder",
            "write_scope": ["src/a.py"],
        }
        code1, _ = self.cli.run_command("submit_task", payload1)
        self.assertEqual(code1, EXIT_OK)

        # Submit same key with DIFFERENT payload
        payload2 = {
            "idempotency_key": "req-key-conflict",
            "project_id": "test-proj",
            "title": "Tampered Title",
            "description": "Desc",
            "role": "coder",
            "write_scope": ["src/a.py"],
        }
        code2, resp2 = self.cli.run_command("submit_task", payload2)
        self.assertEqual(code2, EXIT_CONFLICT)
        self.assertEqual(resp2["error"]["code"], "IDEMPOTENCY_CONFLICT")


class TestCliBackwardCompatibility(unittest.TestCase):
    """Test launcher backward compatibility and seamless JSON subcommand handling."""

    def test_existing_swan_help_preserved(self):
        swan_path = Path(__file__).parent.parent / "swan"
        proc = subprocess.run([sys.executable, str(swan_path)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("SWAN-ENGINE // Local Autonomous C2 & AI Core", proc.stdout)
        self.assertIn("swan task", proc.stdout)
        self.assertIn("swan map", proc.stdout)
        self.assertIn("swan interactive", proc.stdout)
        self.assertIn("swan json", proc.stdout)

    def test_swan_json_subcommand_outputs_pure_json(self):
        swan_path = Path(__file__).parent.parent / "swan"
        proc = subprocess.run([sys.executable, str(swan_path), "json", "tools"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        # Must be pure JSON (no ASCII banner!)
        parsed = json.loads(proc.stdout)
        self.assertEqual(parsed["status"], "success")
        self.assertEqual(parsed["command"], "tools")


if __name__ == "__main__":
    unittest.main()
