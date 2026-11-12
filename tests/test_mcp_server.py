"""Unit tests for MCP Server.

Tests JSON-RPC 2.0 protocol compliance, tool schema declarations,
tool invocations (submit, status, digest, report, verdict, cancel),
security boundaries, and malicious request containment.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from core.mcp_server import MCPServer
from core.task_orchestrator import (
    TaskOrchestrator,
    TaskSpec,
    TaskState,
)
from core.validator import (
    ValidationReport,
    ValidationStatus,
)


class TestMCPServerProtocol(unittest.TestCase):
    """Test standard MCP JSON-RPC 2.0 protocol compliance."""

    def setUp(self):
        self.orchestrator = TaskOrchestrator()
        self.server = MCPServer(orchestrator=self.orchestrator)

    def tearDown(self):
        self.orchestrator.close()

    def test_initialize_handshake(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resp = self.server.handle_json_rpc(req)
        self.assertEqual(resp["id"], 1)
        self.assertEqual(resp["result"]["serverInfo"]["name"], "swan-hive-mcp")
        self.assertTrue(resp["result"]["capabilities"]["tools"])

    def test_ping(self):
        req = {"jsonrpc": "2.0", "id": "p-1", "method": "ping"}
        resp = self.server.handle_json_rpc(req)
        self.assertEqual(resp["id"], "p-1")
        self.assertEqual(resp["result"], {})

    def test_tools_list(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        resp = self.server.handle_json_rpc(req)
        tools = resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        self.assertIn("submit_task", tool_names)
        self.assertIn("submit_batch", tool_names)
        self.assertIn("get_task_status", tool_names)
        self.assertIn("retrieve_compact_digest", tool_names)
        self.assertIn("retrieve_detailed_task_report", tool_names)
        self.assertIn("submit_verdict", tool_names)
        self.assertIn("cancel_task", tool_names)

    def test_unknown_method_error(self):
        req = {"jsonrpc": "2.0", "id": 99, "method": "unsupported/method"}
        resp = self.server.handle_json_rpc(req)
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32601)


class TestMCPToolExecution(unittest.TestCase):
    """Test MCP tool executions and security containment."""

    def setUp(self):
        self.orchestrator = TaskOrchestrator()
        self.server = MCPServer(orchestrator=self.orchestrator)

    def tearDown(self):
        self.orchestrator.close()

    def test_submit_task_tool(self):
        req = {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "submit_task",
                "arguments": {
                    "project_id": "test-proj",
                    "title": "Build feature",
                    "description": "Desc",
                    "role": "coder",
                    "read_scope": ["src/app.py"],
                    "write_scope": ["src/app.py"],
                },
            },
        }
        resp = self.server.handle_json_rpc(req)
        self.assertFalse(resp["result"]["isError"])
        content_text = resp["result"]["content"][0]["text"]
        data = json.loads(content_text)
        self.assertEqual(data["status"], "submitted")
        self.assertIsNotNone(data["task_id"])

    def test_submit_batch_tool(self):
        req = {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "submit_batch",
                "arguments": {
                    "batch_id": "b-mcp",
                    "project_id": "test-proj",
                    "tasks": [
                        {
                            "task_id": "t1",
                            "batch_id": "b-mcp",
                            "project_id": "test-proj",
                            "title": "T1",
                            "description": "D",
                            "role": "coder",
                            "write_scope": ["src/a.py"],
                        },
                        {
                            "task_id": "t2",
                            "batch_id": "b-mcp",
                            "project_id": "test-proj",
                            "title": "T2",
                            "description": "D",
                            "role": "tester",
                            "depends_on": ["t1"],
                            "write_scope": ["src/b.py"],
                        },
                    ],
                },
            },
        }
        resp = self.server.handle_json_rpc(req)
        self.assertFalse(resp["result"]["isError"])
        content_text = resp["result"]["content"][0]["text"]
        data = json.loads(content_text)
        self.assertEqual(data["task_count"], 2)

    def test_get_task_status_and_batch_status(self):
        # Setup task
        task = TaskSpec(
            task_id="t-stat",
            batch_id="b-stat",
            project_id="proj",
            title="Status test",
            description="desc",
            role="coder",
        )
        self.orchestrator.submit_batch("b-stat", "proj", [task])

        # Get task status
        res_task = self.server.execute_tool("get_task_status", {"task_id": "t-stat"})
        self.assertEqual(res_task["task_id"], "t-stat")
        self.assertEqual(res_task["state"], "READY")

        # Get batch status
        res_batch = self.server.execute_tool("get_batch_status", {"batch_id": "b-stat"})
        self.assertEqual(res_batch["total_tasks"], 1)

    def test_retrieve_compact_digest_and_detail(self):
        task = TaskSpec(
            task_id="t-rep",
            batch_id="b-rep",
            project_id="proj",
            title="Report test",
            description="desc",
            role="coder",
        )
        self.orchestrator.submit_batch("b-rep", "proj", [task])

        # Digest
        digest_res = self.server.execute_tool("retrieve_compact_digest", {"batch_id": "b-rep"})
        self.assertIn("digest", digest_res)
        self.assertIn("markdown", digest_res)

        # Detail
        detail_res = self.server.execute_tool("retrieve_detailed_task_report", {"task_id": "t-rep"})
        self.assertEqual(detail_res["task_id"], "t-rep")

    def test_submit_verdict_and_security_override_prevention(self):
        task = TaskSpec(
            task_id="t-sec",
            batch_id="b-sec",
            project_id="proj",
            title="Sec test",
            description="desc",
            role="coder",
        )
        self.orchestrator.submit_batch("b-sec", "proj", [task])
        self.orchestrator.transition_task("t-sec", TaskState.RUNNING)
        self.orchestrator.transition_task("t-sec", TaskState.WORKER_COMPLETED)
        self.orchestrator.transition_task("t-sec", TaskState.VALIDATION_PENDING)

        # Record validation report with security violation
        rep = ValidationReport(
            batch_id="b-sec",
            task_id="t-sec",
            validation_status=ValidationStatus.FAILED.value,
            out_of_scope_changes=["/etc/passwd"],
        )
        self.orchestrator.record_validation_report(rep)

        # Attempt to ACCEPT
        res = self.server.execute_tool("submit_verdict", {
            "batch_id": "b-sec",
            "task_id": "t-sec",
            "verdict": "ACCEPT",
            "reason": "Accept anyway",
        })
        self.assertIn("error", res)
        self.assertIn("SECURITY VIOLATION", res["error"])

    def test_cancel_task_and_batch_tools(self):
        t1 = TaskSpec(task_id="t1", batch_id="b-can", project_id="proj", title="T1", description="D", role="coder")
        t2 = TaskSpec(task_id="t2", batch_id="b-can", project_id="proj", title="T2", description="D", role="coder")
        self.orchestrator.submit_batch("b-can", "proj", [t1, t2])

        can_task = self.server.execute_tool("cancel_task", {"task_id": "t1"})
        self.assertEqual(can_task["status"], "cancelled")
        self.assertEqual(self.orchestrator.get_task_state("t1"), TaskState.CANCELLED)

        can_batch = self.server.execute_tool("cancel_batch", {"batch_id": "b-can"})
        self.assertEqual(can_batch["status"], "cancelled")
        self.assertEqual(self.orchestrator.get_task_state("t2"), TaskState.CANCELLED)

    def test_malicious_request_path_traversal_rejected(self):
        # External Manager attempts to submit a task with path traversal
        res = self.server.execute_tool("submit_task", {
            "project_id": "proj",
            "title": "Escape",
            "description": "Desc",
            "role": "coder",
            "write_scope": ["../../outside.txt"],
        })
        self.assertIn("error", res)
        self.assertIn("traversal", res["error"].lower())


if __name__ == "__main__":
    unittest.main()

