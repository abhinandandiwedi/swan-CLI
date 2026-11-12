"""Unit tests for Worker Runner.

Tests role-based model selection, hardware resource admission, context building,
Ollama model invocation (mocked), strict write-scope enforcement, command
permission filtering, failure handling, retryability classification, and batch execution.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from core.worker_runner import (
    WorkerRunner,
    WorkerResult,
    is_retryable_error,
    ModelUnavailableError,
    ScopeViolationError,
)
from core.task_orchestrator import (
    TaskOrchestrator,
    TaskSpec,
    TaskState,
)
from core.context_manager import ContextManager
from core.model_router import (
    ModelRouter,
    ModelInfo,
    ModelNotFoundError,
    TaskRole,
)
from core.resource_monitor import (
    ResourceMonitor,
    AdmissionDecision,
    AdmissionStatus,
    TelemetryStatus,
)
from core.execution_engine import (
    ExecutionEngine,
    PERMISSION_READ_ONLY,
    PERMISSION_TEST_BUILD,
    PERMISSION_MUTATING,
    PERMISSION_DESTRUCTIVE,
    PERMISSION_PRIVILEGED,
)


class TestWorkerResultSchema(unittest.TestCase):
    """Test WorkerResult contract validation and serialization."""

    def test_valid_worker_result(self):
        res = WorkerResult(
            task_id="t1",
            batch_id="b1",
            worker_id="w-coder",
            model="swan-coder",
            status="completed",
            summary="Refactored module",
            changed_files=["src/core.py"],
            commands_requested=[{"command": "pytest", "returncode": 0}],
            observations=["Tests passed"],
            uncertainties=["None"],
        )
        res.validate()
        d = res.to_dict()
        reconstructed = WorkerResult.from_dict(d)
        self.assertEqual(res.task_id, reconstructed.task_id)
        self.assertEqual(res.status, reconstructed.status)
        self.assertEqual(res.changed_files, reconstructed.changed_files)

    def test_invalid_worker_result_status(self):
        with self.assertRaises(Exception):
            WorkerResult(
                task_id="t1",
                batch_id="b1",
                worker_id="w1",
                status="invalid_status",
            ).validate()

    def test_missing_task_id(self):
        with self.assertRaises(Exception):
            WorkerResult(
                task_id="",
                status="completed",
            ).validate()


class TestRetryableErrorClassification(unittest.TestCase):
    """Test retry policy categorization for bounded retries."""

    def test_non_retryable_errors(self):
        self.assertFalse(is_retryable_error("Scope violation: File 'evil.py' outside write_scope"))
        self.assertFalse(is_retryable_error("Forbidden command requested: 'sudo rm -rf /' (privileged)"))
        self.assertFalse(is_retryable_error("Model unavailable: swan-coder not found"))
        self.assertFalse(is_retryable_error("Resource rejected: Exceeds safe limits"))
        self.assertFalse(is_retryable_error("Lock conflict: active lock held"))

    def test_retryable_errors(self):
        self.assertTrue(is_retryable_error("Ollama failure: Connection timed out"))
        self.assertTrue(is_retryable_error("Malformed model response JSON: Expecting value"))
        self.assertTrue(is_retryable_error("Transient socket error"))


class TestWorkerRunnerExecution(unittest.TestCase):
    """Test individual task execution using mocked Ollama and hardware admission."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        (self.workspace / "src").mkdir(parents=True, exist_ok=True)
        (self.workspace / "src" / "app.py").write_text("def hello(): pass\n", encoding="utf-8")

        self.mock_model_router = MagicMock(spec=ModelRouter)
        self.mock_model = ModelInfo(
            name="swan-coder:latest",
            base_name="swan-coder",
            tag="latest",
            family="qwen2",
            context_window=8192,
            supported_roles={"coder", "tester"},
        )
        self.mock_model_router.select_model_for_role.return_value = self.mock_model
        self.mock_model_router.select_explicit_model.return_value = self.mock_model

        self.mock_resource_monitor = MagicMock(spec=ResourceMonitor)
        self.mock_resource_monitor.evaluate_admission.return_value = AdmissionDecision(
            status=AdmissionStatus.ADMITTED,
            admitted=True,
            reason="Adequate memory",
            model_name="swan-coder:latest",
            target_device="GPU",
            estimated_weights_bytes=4000000000,
            estimated_kv_cache_bytes=200000000,
            estimated_runtime_buffer_bytes=500000000,
            total_required_bytes=4700000000,
            available_memory_bytes=8000000000,
            reserve_bytes=1000000000,
            active_inferences=0,
            max_concurrency=1,
            is_model_resident=True,
            telemetry_status=TelemetryStatus.KNOWN,
            timestamp=time.time(),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_successful_worker_execution(self):
        # Mock Ollama returning structured JSON
        model_output = {
            "summary": "Implemented greet function",
            "files_to_modify": [
                {"path": "src/app.py", "content": "def hello(): return 'hello world'\n"}
            ],
            "commands_to_run": [
                "python3 -c \"exit(0)\""
            ],
            "observations": ["Syntax verified"],
            "uncertainties": [],
        }

        def mock_ollama_client(payload, timeout):
            return {
                "model": "swan-coder:latest",
                "response": json.dumps(model_output),
                "prompt_eval_count": 150,
                "eval_count": 60,
                "total_duration": 1000000,
            }

        runner = WorkerRunner(
            model_router=self.mock_model_router,
            resource_monitor=self.mock_resource_monitor,
            ollama_client_fn=mock_ollama_client,
        )

        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="test-proj",
            title="Implement greeting",
            description="Update app.py to return hello world",
            role="coder",
            read_scope=["src/app.py"],
            write_scope=["src/app.py"],
        )

        res = runner.execute_task(task, workspace_path=self.workspace)
        self.assertEqual(res.status, "completed")
        self.assertEqual(res.summary, "Implemented greet function")
        self.assertIn("src/app.py", res.changed_files)
        self.assertEqual(len(res.commands_requested), 1)
        self.assertEqual(res.commands_requested[0]["returncode"], 0)

        # Verify file on disk was actually modified
        content = (self.workspace / "src" / "app.py").read_text(encoding="utf-8")
        self.assertEqual(content, "def hello(): return 'hello world'\n")

        # Verify inference acquire & release occurred
        self.mock_resource_monitor.acquire_inference.assert_called_once()
        self.mock_resource_monitor.release_inference.assert_called_once()

    def test_model_unavailable_fails_worker(self):
        self.mock_model_router.select_model_for_role.side_effect = ModelNotFoundError("swan-coder not found")

        runner = WorkerRunner(
            model_router=self.mock_model_router,
            resource_monitor=self.mock_resource_monitor,
        )

        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="test-proj",
            title="Test",
            description="Desc",
            role="coder",
        )

        res = runner.execute_task(task, workspace_path=self.workspace)
        self.assertEqual(res.status, "failed")
        self.assertIn("Model unavailable", res.error)

    def test_resource_admission_wait_defers_worker(self):
        self.mock_resource_monitor.evaluate_admission.return_value = AdmissionDecision(
            status=AdmissionStatus.WAIT,
            admitted=False,
            reason="Active inference slot busy",
            model_name="swan-coder:latest",
            target_device="GPU",
            estimated_weights_bytes=4000000000,
            estimated_kv_cache_bytes=200000000,
            estimated_runtime_buffer_bytes=500000000,
            total_required_bytes=4700000000,
            available_memory_bytes=8000000000,
            reserve_bytes=1000000000,
            active_inferences=1,
            max_concurrency=1,
            is_model_resident=True,
            telemetry_status=TelemetryStatus.KNOWN,
            timestamp=time.time(),
        )

        runner = WorkerRunner(
            model_router=self.mock_model_router,
            resource_monitor=self.mock_resource_monitor,
        )

        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="test-proj",
            title="Test",
            description="Desc",
            role="coder",
        )

        res = runner.execute_task(task, workspace_path=self.workspace)
        self.assertEqual(res.status, "resource_wait")
        self.assertIn("Resource wait", res.error)
        self.mock_resource_monitor.acquire_inference.assert_not_called()

    def test_resource_admission_reject_fails_worker(self):
        self.mock_resource_monitor.evaluate_admission.return_value = AdmissionDecision(
            status=AdmissionStatus.REJECT,
            admitted=False,
            reason="Model size exceeds available RAM",
            model_name="swan-coder:latest",
            target_device="GPU",
            estimated_weights_bytes=32000000000,
            estimated_kv_cache_bytes=200000000,
            estimated_runtime_buffer_bytes=500000000,
            total_required_bytes=32700000000,
            available_memory_bytes=8000000000,
            reserve_bytes=1000000000,
            active_inferences=0,
            max_concurrency=1,
            is_model_resident=False,
            telemetry_status=TelemetryStatus.KNOWN,
            timestamp=time.time(),
        )

        runner = WorkerRunner(
            model_router=self.mock_model_router,
            resource_monitor=self.mock_resource_monitor,
        )

        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="test-proj",
            title="Test",
            description="Desc",
            role="coder",
        )

        res = runner.execute_task(task, workspace_path=self.workspace)
        self.assertEqual(res.status, "failed")
        self.assertIn("Resource rejected", res.error)

    def test_ollama_timeout_handled(self):
        def mock_timeout_client(payload, timeout):
            raise TimeoutError("Model generation timed out after 60s")

        runner = WorkerRunner(
            model_router=self.mock_model_router,
            resource_monitor=self.mock_resource_monitor,
            ollama_client_fn=mock_timeout_client,
        )

        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="test-proj",
            title="Timeout Test",
            description="Desc",
            role="coder",
        )

        res = runner.execute_task(task, workspace_path=self.workspace)
        self.assertEqual(res.status, "failed")
        self.assertIn("Ollama failure", res.error)
        # Release inference must still be called
        self.mock_resource_monitor.release_inference.assert_called_once()

    def test_malformed_model_json_handled(self):
        def mock_malformed_client(payload, timeout):
            return {
                "model": "swan-coder:latest",
                "response": "Sorry, here is plain text instead of JSON!",
            }

        runner = WorkerRunner(
            model_router=self.mock_model_router,
            resource_monitor=self.mock_resource_monitor,
            ollama_client_fn=mock_malformed_client,
        )

        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="test-proj",
            title="Malformed Test",
            description="Desc",
            role="coder",
        )

        res = runner.execute_task(task, workspace_path=self.workspace)
        self.assertEqual(res.status, "failed")
        self.assertIn("Malformed model response JSON", res.error)

    def test_write_scope_violation_rejected(self):
        # Model tries to modify /etc/passwd or a file outside write_scope
        model_output = {
            "summary": "Attempting unauthorized write",
            "files_to_modify": [
                {"path": "src/unauthorized.py", "content": "# evil"}
            ],
            "commands_to_run": [],
        }

        runner = WorkerRunner(
            model_router=self.mock_model_router,
            resource_monitor=self.mock_resource_monitor,
            ollama_client_fn=lambda p, t: {"model": "swan-coder", "response": json.dumps(model_output)},
        )

        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="test-proj",
            title="Scope Violation Test",
            description="Desc",
            role="coder",
            write_scope=["src/app.py"],  # only app.py is authorized
        )

        res = runner.execute_task(task, workspace_path=self.workspace)
        self.assertEqual(res.status, "failed")
        self.assertIn("Scope violation", res.error)
        # File must NOT have been written
        self.assertFalse((self.workspace / "src" / "unauthorized.py").exists())

    def test_untrusted_worker_forbidden_command_blocked(self):
        # Model tries to run privileged command (sudo)
        model_output = {
            "summary": "Attempting privileged execution",
            "files_to_modify": [],
            "commands_to_run": [
                "sudo rm -rf /var/log"
            ],
        }

        runner = WorkerRunner(
            model_router=self.mock_model_router,
            resource_monitor=self.mock_resource_monitor,
            ollama_client_fn=lambda p, t: {"model": "swan-coder", "response": json.dumps(model_output)},
        )

        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="test-proj",
            title="Privileged Test",
            description="Desc",
            role="coder",
        )

        res = runner.execute_task(task, workspace_path=self.workspace)
        self.assertEqual(res.status, "failed")
        self.assertIn("Forbidden command", res.error)
        self.assertEqual(len(res.commands_requested), 1)
        self.assertTrue(res.commands_requested[0]["blocked"])


class TestWorkerRunnerBatchExecution(unittest.TestCase):
    """Test full sequential batch execution with Orchestrator integration."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        (self.workspace / "src").mkdir(parents=True, exist_ok=True)
        (self.workspace / "src" / "mod1.py").write_text("# mod1\n", encoding="utf-8")
        (self.workspace / "src" / "mod2.py").write_text("# mod2\n", encoding="utf-8")

        self.orchestrator = TaskOrchestrator()

        self.mock_model_router = MagicMock(spec=ModelRouter)
        self.mock_model = ModelInfo(
            name="swan-coder:latest",
            base_name="swan-coder",
            tag="latest",
            family="qwen2",
            context_window=8192,
            supported_roles={"coder", "tester"},
        )
        self.mock_model_router.select_model_for_role.return_value = self.mock_model
        self.mock_model_router.select_explicit_model.return_value = self.mock_model

        self.mock_resource_monitor = MagicMock(spec=ResourceMonitor)
        self.mock_resource_monitor.evaluate_admission.return_value = AdmissionDecision(
            status=AdmissionStatus.ADMITTED,
            admitted=True,
            reason="Adequate memory",
            model_name="swan-coder:latest",
            target_device="GPU",
            estimated_weights_bytes=4000000000,
            estimated_kv_cache_bytes=200000000,
            estimated_runtime_buffer_bytes=500000000,
            total_required_bytes=4700000000,
            available_memory_bytes=8000000000,
            reserve_bytes=1000000000,
            active_inferences=0,
            max_concurrency=1,
            is_model_resident=True,
            telemetry_status=TelemetryStatus.KNOWN,
            timestamp=time.time(),
        )

    def tearDown(self):
        self.orchestrator.close()
        self.temp_dir.cleanup()

    def test_execute_batch_success_pipeline(self):
        # Two tasks: t1 prepares mod1, t2 depends on t1 and tests mod1
        tasks = [
            TaskSpec(
                task_id="t1",
                batch_id="b1",
                project_id="test-proj",
                title="Task 1",
                description="Write mod1",
                role="coder",
                write_scope=["src/mod1.py"],
            ),
            TaskSpec(
                task_id="t2",
                batch_id="b1",
                project_id="test-proj",
                title="Task 2",
                description="Test mod1",
                role="tester",
                depends_on=["t1"],
                write_scope=["src/mod2.py"],
            ),
        ]
        self.orchestrator.submit_batch("b1", "test-proj", tasks)

        def mock_ollama(payload, timeout):
            return {
                "model": "swan-coder:latest",
                "response": json.dumps({
                    "summary": "Completed task",
                    "files_to_modify": [],
                    "commands_to_run": ["python3 -c \"exit(0)\""],
                }),
            }

        runner = WorkerRunner(
            model_router=self.mock_model_router,
            resource_monitor=self.mock_resource_monitor,
            ollama_client_fn=mock_ollama,
        )

        results = runner.execute_batch("b1", self.orchestrator, self.workspace)
        self.assertEqual(len(results), 2)
        self.assertEqual(results["t1"].status, "completed")
        self.assertEqual(results["t2"].status, "completed")

        # In Orchestrator, both should now be VALIDATION_PENDING
        self.assertEqual(self.orchestrator.get_task_state("t1"), TaskState.VALIDATION_PENDING)
        self.assertEqual(self.orchestrator.get_task_state("t2"), TaskState.VALIDATION_PENDING)

        # Worker results should be persisted
        t1_results = self.orchestrator.get_worker_results("t1")
        self.assertEqual(len(t1_results), 1)

    def test_execute_batch_stops_on_resource_wait(self):
        tasks = [
            TaskSpec(task_id="t1", batch_id="b1", project_id="test-proj", title="T1", description="D", role="coder"),
        ]
        self.orchestrator.submit_batch("b1", "test-proj", tasks)

        self.mock_resource_monitor.evaluate_admission.return_value = AdmissionDecision(
            status=AdmissionStatus.WAIT,
            admitted=False,
            reason="High memory pressure",
            model_name="swan-coder",
            target_device="GPU",
            estimated_weights_bytes=4000000000,
            estimated_kv_cache_bytes=200000000,
            estimated_runtime_buffer_bytes=500000000,
            total_required_bytes=4700000000,
            available_memory_bytes=100000000,
            reserve_bytes=1000000000,
            active_inferences=0,
            max_concurrency=1,
            is_model_resident=True,
            telemetry_status=TelemetryStatus.KNOWN,
            timestamp=time.time(),
        )

        runner = WorkerRunner(
            model_router=self.mock_model_router,
            resource_monitor=self.mock_resource_monitor,
        )

        results = runner.execute_batch("b1", self.orchestrator, self.workspace)
        self.assertEqual(results["t1"].status, "resource_wait")
        self.assertEqual(self.orchestrator.get_task_state("t1"), TaskState.WAITING_RESOURCE)


if __name__ == "__main__":
    unittest.main()
