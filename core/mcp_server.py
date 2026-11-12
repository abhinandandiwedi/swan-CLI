"""Model Context Protocol (MCP) Server for SWAN-HIVE.

Exposes structured tools allowing External Managers (Claude, Gemini, etc.)
to safely submit batches, inspect status, retrieve compact digests,
request task revisions, and enforce security boundaries over JSON-RPC.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from core.task_orchestrator import (
    TaskOrchestrator,
    TaskSpec,
    TaskState,
    TaskValidationError,
    DuplicateTaskError,
    MissingDependencyError,
    DependencyCycleError,
    InvalidStateTransitionError,
    validate_scope_path,
)
from core.worker_runner import (
    WorkerRunner,
    WorkerResult,
)
from core.validator import (
    Validator,
    ValidationReport,
    WorktreeManager,
)
from core.digest_builder import (
    DigestBuilder,
    BatchDigest,
)
from core.manager_verdict import (
    ManagerVerdict,
    ManagerVerdictService,
    VerdictType,
    VerdictError,
    SecurityOverrideError,
)


class MCPServer:
    """Implements standard MCP Tool interface and JSON-RPC 2.0 server."""

    SERVER_NAME = "swan-hive-mcp"
    SERVER_VERSION = "0.2.0"

    def __init__(
        self,
        orchestrator: TaskOrchestrator,
        runner: Optional[WorkerRunner] = None,
        validator: Optional[Validator] = None,
        digest_builder: Optional[DigestBuilder] = None,
        workspace_path: Optional[Path] = None,
    ):
        self.orchestrator = orchestrator
        self.runner = runner or WorkerRunner()
        self.validator = validator or Validator()
        self.workspace_path = Path(workspace_path).resolve() if workspace_path else Path.cwd().resolve()
        self.digest_builder = digest_builder or DigestBuilder(
            orchestrator=self.orchestrator,
            validator=self.validator,
            workspace_path=self.workspace_path,
        )
        self.worktree_mgr = WorktreeManager(workspace=self.workspace_path)

    # -----------------------------------------------------------------------
    # Tool Schemas
    # -----------------------------------------------------------------------

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return MCP tools manifest with schemas."""
        return [
            {
                "name": "submit_task",
                "description": "Submit a single task to SWAN-HIVE task orchestrator.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string", "description": "Registered project ID"},
                        "title": {"type": "string", "description": "Short task title"},
                        "description": {"type": "string", "description": "Detailed task description"},
                        "role": {
                            "type": "string",
                            "enum": ["coder", "tester", "debugger", "documentation", "analysis"],
                            "description": "Worker role required",
                        },
                        "read_scope": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of allowed read paths (relative)",
                        },
                        "write_scope": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of allowed write paths (relative)",
                        },
                        "batch_id": {"type": "string", "description": "Batch ID (optional, auto-generated if omitted)"},
                        "task_id": {"type": "string", "description": "Task ID (optional, auto-generated if omitted)"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                        "validation_requirements": {"type": "array", "items": {"type": "string"}},
                        "timeout": {"type": "integer", "default": 60},
                        "retry_limits": {"type": "integer", "default": 2},
                        "model_preference": {"type": "string"},
                    },
                    "required": ["project_id", "title", "description", "role", "write_scope"],
                },
            },
            {
                "name": "submit_batch",
                "description": "Submit a batch of structured tasks with DAG dependencies.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "batch_id": {"type": "string", "description": "Unique batch identifier"},
                        "project_id": {"type": "string", "description": "Registered project ID"},
                        "tasks": {
                            "type": "array",
                            "description": "List of task specifications",
                            "items": {"type": "object"},
                        },
                        "metadata": {"type": "object", "description": "Optional batch metadata"},
                        "execute_immediately": {
                            "type": "boolean",
                            "description": "If true, execute batch sequentially after submission",
                            "default": False,
                        },
                    },
                    "required": ["batch_id", "project_id", "tasks"],
                },
            },
            {
                "name": "get_task_status",
                "description": "Retrieve execution and validation state of a specific task.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Unique task identifier"},
                    },
                    "required": ["task_id"],
                },
            },
            {
                "name": "get_batch_status",
                "description": "Retrieve progress and status of all tasks in a batch.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "batch_id": {"type": "string", "description": "Unique batch identifier"},
                    },
                    "required": ["batch_id"],
                },
            },
            {
                "name": "retrieve_compact_digest",
                "description": "Retrieve high-density summary digest for a batch.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "batch_id": {"type": "string", "description": "Unique batch identifier"},
                    },
                    "required": ["batch_id"],
                },
            },
            {
                "name": "retrieve_detailed_task_report",
                "description": "Fetch expanded task definition, worker output, validation checks, and diff.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Unique task identifier"},
                    },
                    "required": ["task_id"],
                },
            },
            {
                "name": "submit_verdict",
                "description": "Submit an External Manager verdict (ACCEPT, REQUEST_REVISION, RETRY, REJECT).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "batch_id": {"type": "string"},
                        "task_id": {"type": "string"},
                        "verdict": {
                            "type": "string",
                            "enum": ["ACCEPT", "REQUEST_REVISION", "RETRY", "REJECT"],
                        },
                        "reason": {"type": "string", "description": "Justification for verdict"},
                        "feedback": {"type": "string", "description": "Optional instructions for revision"},
                        "manager_id": {"type": "string", "default": "external-manager"},
                    },
                    "required": ["batch_id", "task_id", "verdict", "reason"],
                },
            },
            {
                "name": "cancel_task",
                "description": "Cancel an eligible pending or running task.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "reason": {"type": "string", "default": "Cancelled by External Manager"},
                    },
                    "required": ["task_id"],
                },
            },
            {
                "name": "cancel_batch",
                "description": "Cancel all active tasks within a batch.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "batch_id": {"type": "string"},
                        "reason": {"type": "string", "default": "Cancelled by External Manager"},
                    },
                    "required": ["batch_id"],
                },
            },
        ]

    # -----------------------------------------------------------------------
    # Tool Execution Handler
    # -----------------------------------------------------------------------

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute named tool with provided arguments and enforce security controls."""
        try:
            if name == "submit_task":
                return self._tool_submit_task(arguments)
            elif name == "submit_batch":
                return self._tool_submit_batch(arguments)
            elif name == "get_task_status":
                return self._tool_get_task_status(arguments)
            elif name == "get_batch_status":
                return self._tool_get_batch_status(arguments)
            elif name == "retrieve_compact_digest":
                return self._tool_retrieve_compact_digest(arguments)
            elif name == "retrieve_detailed_task_report":
                return self._tool_retrieve_detailed_task_report(arguments)
            elif name == "submit_verdict":
                return self._tool_submit_verdict(arguments)
            elif name == "cancel_task":
                return self._tool_cancel_task(arguments)
            elif name == "cancel_batch":
                return self._tool_cancel_batch(arguments)
            else:
                return {"error": f"Unknown tool: '{name}'"}
        except (TaskValidationError, DuplicateTaskError, MissingDependencyError,
                DependencyCycleError, InvalidStateTransitionError, VerdictError) as e:
            return {"error": f"Validation error: {e}", "error_type": type(e).__name__}
        except SecurityOverrideError as e:
            return {"error": f"Security violation: {e}", "error_type": "SecurityOverrideError"}
        except Exception as e:
            return {"error": f"Execution error: {e}", "error_type": type(e).__name__}

    # -----------------------------------------------------------------------
    # Individual Tool Implementations
    # -----------------------------------------------------------------------

    def _tool_submit_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        batch_id = args.get("batch_id") or f"b-{uuid.uuid4().hex[:8]}"
        task_id = args.get("task_id") or f"t-{uuid.uuid4().hex[:8]}"
        spec = TaskSpec(
            task_id=task_id,
            batch_id=batch_id,
            project_id=args["project_id"],
            title=args["title"],
            description=args["description"],
            role=args["role"],
            read_scope=args.get("read_scope", []),
            write_scope=args["write_scope"],
            depends_on=args.get("depends_on", []),
            acceptance_criteria=args.get("acceptance_criteria", []),
            validation_requirements=args.get("validation_requirements", []),
            timeout=int(args.get("timeout", 60)),
            retry_limits=int(args.get("retry_limits", 2)),
            model_preference=args.get("model_preference"),
        )
        self.orchestrator.submit_batch(batch_id, spec.project_id, [spec])
        return {
            "status": "submitted",
            "batch_id": batch_id,
            "task_id": task_id,
            "state": self.orchestrator.get_task_state(task_id).value,
        }

    def _tool_submit_batch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        batch_id = args["batch_id"]
        project_id = args["project_id"]
        task_specs = [TaskSpec.from_dict(t) for t in args["tasks"]]
        metadata = args.get("metadata", {})
        execute_immediately = bool(args.get("execute_immediately", False))

        self.orchestrator.submit_batch(batch_id, project_id, task_specs, metadata=metadata)

        execution_results = None
        validation_reports = {}
        if execute_immediately:
            execution_results = self.runner.execute_batch(
                batch_id=batch_id,
                orchestrator=self.orchestrator,
                workspace_path=self.workspace_path,
            )
            # Run validator on all completed tasks
            for task in task_specs:
                t_state = self.orchestrator.get_task_state(task.task_id)
                if t_state in (TaskState.VALIDATION_PENDING, TaskState.WORKER_COMPLETED):
                    report = self.validator.validate_task(task, workspace_path=self.workspace_path)
                    validation_reports[task.task_id] = report
                    self.orchestrator.record_validation_report(report)

        return {
            "status": "submitted",
            "batch_id": batch_id,
            "project_id": project_id,
            "task_count": len(task_specs),
            "executed": execute_immediately,
            "execution_summary": {k: v.status for k, v in (execution_results or {}).items()},
        }

    def _tool_get_task_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        task_id = args["task_id"]
        rec = self.orchestrator.get_task(task_id)
        if not rec:
            return {"error": f"Task '{task_id}' not found"}

        latest_report = self.orchestrator.get_latest_validation_report(task_id)
        worker_results = self.orchestrator.get_worker_results(task_id)

        return {
            "task_id": rec.task_id,
            "batch_id": rec.batch_id,
            "project_id": rec.project_id,
            "state": rec.state.value,
            "retry_count": rec.retry_count,
            "title": rec.spec.title,
            "role": rec.spec.role,
            "latest_worker_status": worker_results[-1]["result"].get("status") if worker_results else None,
            "latest_validation_status": latest_report["report"].get("validation_status") if latest_report else None,
        }

    def _tool_get_batch_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        batch_id = args["batch_id"]
        tasks = self.orchestrator.list_tasks(batch_id=batch_id)
        if not tasks:
            return {"error": f"Batch '{batch_id}' not found"}

        return {
            "batch_id": batch_id,
            "project_id": tasks[0].project_id,
            "total_tasks": len(tasks),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "title": t.spec.title,
                    "role": t.spec.role,
                    "state": t.state.value,
                    "retry_count": t.retry_count,
                }
                for t in tasks
            ],
        }

    def _tool_retrieve_compact_digest(self, args: Dict[str, Any]) -> Dict[str, Any]:
        batch_id = args["batch_id"]
        digest = self.digest_builder.build_digest(batch_id)
        return {
            "digest": digest.to_dict(),
            "markdown": digest.to_markdown(),
        }

    def _tool_retrieve_detailed_task_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
        task_id = args["task_id"]
        return self.digest_builder.get_task_detail(task_id)

    def _tool_submit_verdict(self, args: Dict[str, Any]) -> Dict[str, Any]:
        verdict = ManagerVerdict.from_dict(args)
        updated_task = ManagerVerdictService.apply_verdict(verdict, self.orchestrator)
        return {
            "status": "verdict_applied",
            "task_id": verdict.task_id,
            "verdict": verdict.verdict.value,
            "new_state": updated_task.state.value if updated_task else "UNKNOWN",
        }

    def _tool_cancel_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        task_id = args["task_id"]
        reason = args.get("reason", "Cancelled by External Manager")
        self.orchestrator.cancel_task(task_id, reason=reason)
        return {"status": "cancelled", "task_id": task_id}

    def _tool_cancel_batch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        batch_id = args["batch_id"]
        reason = args.get("reason", "Cancelled by External Manager")
        self.orchestrator.cancel_batch(batch_id, reason=reason)
        return {"status": "cancelled", "batch_id": batch_id}

    # -----------------------------------------------------------------------
    # JSON-RPC 2.0 Message Handler (Standard MCP)
    # -----------------------------------------------------------------------

    def handle_json_rpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming JSON-RPC 2.0 request and generate compliant response."""
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if not method or not isinstance(method, str):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": "Invalid Request: method is required"},
            }

        # Handle MCP Protocol Methods
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": self.SERVER_NAME,
                        "version": self.SERVER_VERSION,
                    },
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                },
            }

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": self.get_tool_definitions(),
                },
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            if not tool_name:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": "Invalid params: name is required"},
                }

            result = self.execute_tool(tool_name, arguments)
            is_error = "error" in result

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2),
                        }
                    ],
                    "isError": is_error,
                },
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: '{method}'"},
            }

    def run_stdio(self) -> None:
        """Run line-by-line JSON-RPC stdio event loop."""
        for line in sys.stdin:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                req = json.loads(line_str)
                resp = self.handle_json_rpc(req)
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()

