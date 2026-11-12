"""Structured JSON CLI for SWAN-HIVE.

Provides non-interactive, deterministic JSON input and output for external automation,
Claude/Gemini integration, and headless orchestration. Features clear exit codes,
idempotency tracking, and robust error formatting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from core.task_orchestrator import (
    TaskOrchestrator,
    TaskSpec,
    TaskState,
    TaskValidationError,
    DuplicateTaskError,
    MissingDependencyError,
    DependencyCycleError,
    InvalidStateTransitionError,
)
from core.worker_runner import (
    WorkerRunner,
    WorkerResult,
)
from core.validator import (
    Validator,
    ValidationReport,
)
from core.digest_builder import (
    DigestBuilder,
)
from core.manager_verdict import (
    ManagerVerdict,
    ManagerVerdictService,
    VerdictType,
    VerdictError,
    SecurityOverrideError,
    RetryLimitExceededError,
)
from core.mcp_server import MCPServer


# ---------------------------------------------------------------------------
# Exit Codes & Error Enums
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_VALIDATION_ERROR = 1
EXIT_SECURITY_VIOLATION = 2
EXIT_CONFLICT = 3
EXIT_INTERNAL_ERROR = 4


def format_response(
    status: str,
    command: str,
    data: Optional[Dict[str, Any]] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Format standard deterministic JSON response."""
    resp: Dict[str, Any] = {
        "status": status,
        "command": command,
        "data": data or {},
        "error": None,
        "timestamp": time.time(),
    }
    if error_code or error_message:
        resp["error"] = {
            "code": error_code or "ERROR",
            "message": error_message or "",
        }
    if idempotency_key:
        resp["idempotency_key"] = idempotency_key
    return resp


class JsonCli:
    """Non-interactive CLI processor with idempotency and structured output."""

    def __init__(
        self,
        orchestrator: Optional[TaskOrchestrator] = None,
        workspace_path: Optional[Path] = None,
        db_path: Optional[Union[str, Path]] = None,
    ):
        self.workspace_path = Path(workspace_path).resolve() if workspace_path else Path.cwd().resolve()
        resolved_db = db_path or (self.workspace_path / ".swan" / "tasks.db")
        if isinstance(resolved_db, Path):
            resolved_db.parent.mkdir(parents=True, exist_ok=True)

        self.orchestrator = orchestrator or TaskOrchestrator(db_path=resolved_db)
        self.mcp_server = MCPServer(
            orchestrator=self.orchestrator,
            workspace_path=self.workspace_path,
        )
        self._init_idempotency_table()

    def _init_idempotency_table(self) -> None:
        """Create idempotency persistence table if not exists."""
        try:
            with self.orchestrator.conn:
                self.orchestrator.conn.execute("""
                    CREATE TABLE IF NOT EXISTS idempotent_requests (
                        idempotency_key TEXT PRIMARY KEY,
                        command TEXT NOT NULL,
                        request_hash TEXT NOT NULL,
                        response_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );
                """)
        except Exception:
            pass

    def _check_idempotency(self, key: str, payload_str: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Check idempotency record.

        Returns (is_cached, cached_response, conflict_error).
        """
        payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
        row = self.orchestrator.conn.execute(
            "SELECT request_hash, response_json FROM idempotent_requests WHERE idempotency_key = ?",
            (key,),
        ).fetchone()
        if not row:
            return False, None, None

        if row["request_hash"] != payload_hash:
            return False, None, f"Idempotency key '{key}' was previously used with different payload"

        cached_dict = json.loads(row["response_json"])
        return True, cached_dict, None

    def _store_idempotency(self, key: str, command: str, payload_str: str, response: Dict[str, Any]) -> None:
        """Persist idempotency response."""
        payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
        now = time.time()
        try:
            with self.orchestrator.conn:
                self.orchestrator.conn.execute("""
                    INSERT OR REPLACE INTO idempotent_requests (idempotency_key, command, request_hash, response_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (key, command, payload_hash, json.dumps(response), now))
        except Exception:
            pass

    def run_command(self, command: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Process command with structured JSON input and output.

        Returns (exit_code, response_dict).
        """
        idempotency_key = payload.get("idempotency_key")
        payload_str = json.dumps(payload, sort_keys=True)

        if idempotency_key:
            is_cached, cached_resp, conflict_err = self._check_idempotency(idempotency_key, payload_str)
            if conflict_err:
                resp = format_response(
                    status="error",
                    command=command,
                    error_code="IDEMPOTENCY_CONFLICT",
                    error_message=conflict_err,
                    idempotency_key=idempotency_key,
                )
                return EXIT_CONFLICT, resp
            if is_cached and cached_resp:
                return EXIT_OK, cached_resp

        # Handle Commands
        exit_code = EXIT_OK
        try:
            # Map command names to MCP Server tools
            cmd_normalized = command.lower().replace("-", "_")
            if cmd_normalized in ("help", "tools", "list_tools"):
                tools = self.mcp_server.get_tool_definitions()
                resp = format_response(
                    status="success",
                    command=command,
                    data={"tools": tools},
                    idempotency_key=idempotency_key,
                )
            else:
                result = self.mcp_server.execute_tool(cmd_normalized, payload)
                if "error" in result:
                    err_type = result.get("error_type", "EXECUTION_ERROR")
                    if err_type == "SecurityOverrideError":
                        exit_code = EXIT_SECURITY_VIOLATION
                        std_code = "SECURITY_VIOLATION"
                    elif err_type in ("DuplicateTaskError", "InvalidStateTransitionError", "ConflictError", "RetryLimitExceededError"):
                        exit_code = EXIT_CONFLICT
                        std_code = "CONFLICT"
                    else:
                        exit_code = EXIT_VALIDATION_ERROR
                        std_code = "VALIDATION_ERROR"

                    resp = format_response(
                        status="error",
                        command=command,
                        error_code=std_code,
                        error_message=result.get("error"),
                        idempotency_key=idempotency_key,
                    )
                else:
                    resp = format_response(
                        status="success",
                        command=command,
                        data=result,
                        idempotency_key=idempotency_key,
                    )

        except SecurityOverrideError as e:
            exit_code = EXIT_SECURITY_VIOLATION
            resp = format_response(
                status="error",
                command=command,
                error_code="SECURITY_VIOLATION",
                error_message=str(e),
                idempotency_key=idempotency_key,
            )
        except (TaskValidationError, VerdictValidationError) as e:
            exit_code = EXIT_VALIDATION_ERROR
            resp = format_response(
                status="error",
                command=command,
                error_code="VALIDATION_ERROR",
                error_message=str(e),
                idempotency_key=idempotency_key,
            )
        except (DuplicateTaskError, InvalidStateTransitionError, RetryLimitExceededError) as e:
            exit_code = EXIT_CONFLICT
            resp = format_response(
                status="error",
                command=command,
                error_code="CONFLICT",
                error_message=str(e),
                idempotency_key=idempotency_key,
            )
        except Exception as e:
            exit_code = EXIT_INTERNAL_ERROR
            resp = format_response(
                status="error",
                command=command,
                error_code="INTERNAL_ERROR",
                error_message=str(e),
                idempotency_key=idempotency_key,
            )

        if exit_code == EXIT_OK and idempotency_key:
            self._store_idempotency(idempotency_key, command, payload_str, resp)

        return exit_code, resp


def main(args: Optional[List[str]] = None) -> int:
    """CLI entry point for JSON CLI."""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="SWAN-HIVE Structured JSON CLI")
    parser.add_argument("command", nargs="?", help="Command to execute (e.g. submit_task, get_batch_status)")
    parser.add_argument("--data", "-d", help="JSON string arguments")
    parser.add_argument("--file", "-f", help="JSON file containing arguments")
    parser.add_argument("--workspace", "-w", help="Workspace root path")
    parser.add_argument("--db", help="Tasks SQLite database path")

    parsed = parser.parse_args(args)

    # Determine command and payload
    command = parsed.command
    payload: Dict[str, Any] = {}

    # Check for stdin piped JSON
    if not sys.stdin.isatty():
        try:
            stdin_data = sys.stdin.read().strip()
            if stdin_data:
                parsed_stdin = json.loads(stdin_data)
                if isinstance(parsed_stdin, dict):
                    if not command and "command" in parsed_stdin:
                        command = parsed_stdin.pop("command")
                    payload.update(parsed_stdin)
        except Exception as e:
            resp = format_response(
                status="error",
                command=command or "unknown",
                error_code="INVALID_JSON",
                error_message=f"Failed to parse stdin JSON: {e}",
            )
            sys.stdout.write(json.dumps(resp, indent=2) + "\n")
            return EXIT_VALIDATION_ERROR

    if parsed.file:
        try:
            file_data = Path(parsed.file).read_text(encoding="utf-8")
            payload.update(json.loads(file_data))
        except Exception as e:
            resp = format_response(
                status="error",
                command=command or "unknown",
                error_code="INVALID_JSON",
                error_message=f"Failed to read JSON from file '{parsed.file}': {e}",
            )
            sys.stdout.write(json.dumps(resp, indent=2) + "\n")
            return EXIT_VALIDATION_ERROR

    if parsed.data:
        try:
            payload.update(json.loads(parsed.data))
        except Exception as e:
            resp = format_response(
                status="error",
                command=command or "unknown",
                error_code="INVALID_JSON",
                error_message=f"Failed to parse --data JSON: {e}",
            )
            sys.stdout.write(json.dumps(resp, indent=2) + "\n")
            return EXIT_VALIDATION_ERROR

    if not command:
        resp = format_response(
            status="error",
            command="none",
            error_code="MISSING_COMMAND",
            error_message="Command must be specified as argument or in JSON 'command' key. Use 'tools' to see available commands.",
        )
        sys.stdout.write(json.dumps(resp, indent=2) + "\n")
        return EXIT_VALIDATION_ERROR

    cli = JsonCli(workspace_path=parsed.workspace, db_path=parsed.db)
    exit_code, resp = cli.run_command(command, payload)
    sys.stdout.write(json.dumps(resp, indent=2) + "\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
