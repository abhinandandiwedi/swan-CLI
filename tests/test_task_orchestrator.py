"""Unit tests for Task Orchestrator.

Tests task schema validation, DAG cycle detection, dependency resolution,
SQLite persistence, transactional rollback, state machine enforcement,
write-scope overlap detection, lock management, and cancellation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

from core.task_orchestrator import (
    TaskOrchestrator,
    TaskSpec,
    TaskState,
    TaskRecord,
    TaskValidationError,
    DuplicateTaskError,
    MissingDependencyError,
    DependencyCycleError,
    InvalidStateTransitionError,
    BatchLimitExceededError,
    validate_scope_path,
    is_path_in_scope,
    paths_overlap,
    detect_dependency_cycles,
)


class TestScopePathUtilities(unittest.TestCase):
    """Test path normalization, containment, and overlap detection."""

    def test_validate_scope_path_valid(self):
        self.assertEqual(validate_scope_path("src/core/app.py"), "src/core/app.py")
        self.assertEqual(validate_scope_path("config.json"), "config.json")
        self.assertEqual(validate_scope_path("tests/"), "tests")

    def test_validate_scope_path_rejects_empty(self):
        with self.assertRaises(TaskValidationError):
            validate_scope_path("")
        with self.assertRaises(TaskValidationError):
            validate_scope_path("   ")

    def test_validate_scope_path_rejects_absolute(self):
        with self.assertRaises(TaskValidationError):
            validate_scope_path("/etc/passwd")
        with self.assertRaises(TaskValidationError):
            validate_scope_path("\\Windows\\System32")

    def test_validate_scope_path_rejects_traversal(self):
        with self.assertRaises(TaskValidationError):
            validate_scope_path("../outside.py")
        with self.assertRaises(TaskValidationError):
            validate_scope_path("src/../../outside.py")

    def test_validate_scope_path_rejects_null_bytes(self):
        with self.assertRaises(TaskValidationError):
            validate_scope_path("src/core\x00/app.py")

    def test_is_path_in_scope_exact_file(self):
        scope = ["src/core/app.py", "config.json"]
        self.assertTrue(is_path_in_scope("src/core/app.py", scope))
        self.assertTrue(is_path_in_scope("config.json", scope))
        self.assertFalse(is_path_in_scope("src/core/other.py", scope))

    def test_is_path_in_scope_directory_containment(self):
        scope = ["src/core", "tests/"]
        self.assertTrue(is_path_in_scope("src/core/models/user.py", scope))
        self.assertTrue(is_path_in_scope("tests/test_app.py", scope))
        self.assertFalse(is_path_in_scope("src/app.py", scope))

    def test_paths_overlap_detection(self):
        # Exact match
        self.assertTrue(paths_overlap("src/core/app.py", "src/core/app.py"))
        # Directory contains file
        self.assertTrue(paths_overlap("src/core", "src/core/app.py"))
        self.assertTrue(paths_overlap("src/core/app.py", "src/core"))
        # Disjoint
        self.assertFalse(paths_overlap("src/core/a.py", "src/core/b.py"))
        self.assertFalse(paths_overlap("src/core", "tests"))


class TestTaskSpecValidation(unittest.TestCase):
    """Test schema validation for TaskSpec."""

    def test_valid_task_spec(self):
        spec = TaskSpec(
            task_id="task-01",
            batch_id="batch-01",
            project_id="test-project",
            title="Implement User Model",
            description="Create user model with fields",
            role="coder",
            read_scope=["src/models.py"],
            write_scope=["src/models.py"],
            timeout=120,
            retry_limits=3,
        )
        spec.validate()
        self.assertEqual(spec.task_id, "task-01")
        self.assertEqual(spec.timeout, 120)

    def test_invalid_task_id(self):
        with self.assertRaises(TaskValidationError):
            TaskSpec(
                task_id="bad id with spaces!",
                batch_id="b1",
                project_id="proj",
                title="Title",
                description="Desc",
                role="coder",
            ).validate()

    def test_invalid_project_id(self):
        with self.assertRaises(TaskValidationError):
            TaskSpec(
                task_id="t1",
                batch_id="b1",
                project_id="INVALID_PROJECT_ID_SPACES",
                title="Title",
                description="Desc",
                role="coder",
            ).validate()

    def test_unsupported_role(self):
        with self.assertRaises(TaskValidationError):
            TaskSpec(
                task_id="t1",
                batch_id="b1",
                project_id="proj",
                title="Title",
                description="Desc",
                role="superhero",
            ).validate()

    def test_invalid_retry_limits(self):
        with self.assertRaises(TaskValidationError):
            TaskSpec(
                task_id="t1",
                batch_id="b1",
                project_id="proj",
                title="Title",
                description="Desc",
                role="coder",
                retry_limits=999,  # exceeds MAX_RETRY_LIMIT
            ).validate()

    def test_self_dependency_rejected(self):
        with self.assertRaises(TaskValidationError):
            TaskSpec(
                task_id="task-01",
                batch_id="b1",
                project_id="proj",
                title="Title",
                description="Desc",
                role="coder",
                depends_on=["task-01"],
            ).validate()

    def test_dict_serialization_and_deserialization(self):
        original = TaskSpec(
            task_id="task-ser",
            batch_id="b1",
            project_id="proj-1",
            title="Serialization Test",
            description="Testing to_dict and from_dict",
            role="tester",
            read_scope=["tests/test_ser.py"],
            write_scope=["tests/test_ser.py"],
        )
        d = original.to_dict()
        reconstructed = TaskSpec.from_dict(d)
        self.assertEqual(original, reconstructed)


class TestDAGCycleDetection(unittest.TestCase):
    """Test cycle detection across dependent tasks."""

    def test_acyclic_graph(self):
        graph = {
            "t1": [],
            "t2": ["t1"],
            "t3": ["t1"],
            "t4": ["t2", "t3"],
        }
        cycle = detect_dependency_cycles(graph)
        self.assertEqual(cycle, [])

    def test_direct_cycle(self):
        graph = {
            "a": ["b"],
            "b": ["a"],
        }
        cycle = detect_dependency_cycles(graph)
        self.assertTrue(len(cycle) >= 2)
        self.assertEqual(cycle[0], cycle[-1])

    def test_indirect_cycle(self):
        graph = {
            "a": ["c"],
            "b": ["a"],
            "c": ["b"],
        }
        cycle = detect_dependency_cycles(graph)
        self.assertTrue(len(cycle) >= 3)
        self.assertEqual(cycle[0], cycle[-1])


class TestTaskOrchestratorBatchSubmission(unittest.TestCase):
    """Test submitting task batches and handling integrity checks."""

    def setUp(self):
        self.orchestrator = TaskOrchestrator()

    def tearDown(self):
        self.orchestrator.close()

    def test_submit_valid_batch(self):
        tasks = [
            TaskSpec(
                task_id="t1",
                batch_id="b1",
                project_id="proj",
                title="Task 1",
                description="First task",
                role="coder",
            ),
            TaskSpec(
                task_id="t2",
                batch_id="b1",
                project_id="proj",
                title="Task 2",
                description="Second task",
                role="tester",
                depends_on=["t1"],
            ),
        ]
        submitted = self.orchestrator.submit_batch("b1", "proj", tasks)
        self.assertEqual(len(submitted), 2)

        # t1 should be READY, t2 should be WAITING_DEPENDENCY
        self.assertEqual(self.orchestrator.get_task_state("t1"), TaskState.READY)
        self.assertEqual(self.orchestrator.get_task_state("t2"), TaskState.WAITING_DEPENDENCY)

    def test_duplicate_task_id_in_batch_rejected(self):
        tasks = [
            TaskSpec(task_id="t1", batch_id="b1", project_id="proj", title="T1", description="D", role="coder"),
            TaskSpec(task_id="t1", batch_id="b1", project_id="proj", title="T1 Duplicate", description="D", role="coder"),
        ]
        with self.assertRaises(DuplicateTaskError):
            self.orchestrator.submit_batch("b1", "proj", tasks)

    def test_duplicate_task_id_in_database_rejected(self):
        tasks1 = [
            TaskSpec(task_id="t1", batch_id="b1", project_id="proj", title="T1", description="D", role="coder"),
        ]
        self.orchestrator.submit_batch("b1", "proj", tasks1)

        tasks2 = [
            TaskSpec(task_id="t1", batch_id="b2", project_id="proj", title="T1 Again", description="D", role="coder"),
        ]
        with self.assertRaises(DuplicateTaskError):
            self.orchestrator.submit_batch("b2", "proj", tasks2)

    def test_missing_dependency_rejected(self):
        tasks = [
            TaskSpec(
                task_id="t1",
                batch_id="b1",
                project_id="proj",
                title="T1",
                description="D",
                role="coder",
                depends_on=["non_existent_task"],
            ),
        ]
        with self.assertRaises(MissingDependencyError):
            self.orchestrator.submit_batch("b1", "proj", tasks)

    def test_cyclic_dependencies_in_batch_rejected(self):
        tasks = [
            TaskSpec(task_id="t1", batch_id="b1", project_id="proj", title="T1", description="D", role="coder", depends_on=["t2"]),
            TaskSpec(task_id="t2", batch_id="b1", project_id="proj", title="T2", description="D", role="coder", depends_on=["t1"]),
        ]
        with self.assertRaises(DependencyCycleError):
            self.orchestrator.submit_batch("b1", "proj", tasks)

    def test_batch_limit_exceeded(self):
        small_orch = TaskOrchestrator(max_batch_size=3)
        tasks = [
            TaskSpec(task_id=f"t{i}", batch_id="b1", project_id="proj", title=f"T{i}", description="D", role="coder")
            for i in range(5)
        ]
        with self.assertRaises(BatchLimitExceededError):
            small_orch.submit_batch("b1", "proj", tasks)
        small_orch.close()

    def test_transaction_rollback_on_failure(self):
        # A batch where the second task has an invalid spec
        tasks = [
            TaskSpec(task_id="t1", batch_id="b1", project_id="proj", title="T1", description="D", role="coder"),
            TaskSpec(task_id="t2", batch_id="b1", project_id="proj", title="T2", description="D", role="coder", depends_on=["missing"]),
        ]
        with self.assertRaises(MissingDependencyError):
            self.orchestrator.submit_batch("b1", "proj", tasks)

        # Ensure t1 was NOT committed
        self.assertIsNone(self.orchestrator.get_task("t1"))


class TestTaskStateMachineAndTransitions(unittest.TestCase):
    """Test strict state machine enforcement and audit event logging."""

    def setUp(self):
        self.orchestrator = TaskOrchestrator()
        task = TaskSpec(task_id="t1", batch_id="b1", project_id="proj", title="T1", description="D", role="coder")
        self.orchestrator.submit_batch("b1", "proj", [task])

    def tearDown(self):
        self.orchestrator.close()

    def test_valid_lifecycle_transitions(self):
        self.assertEqual(self.orchestrator.get_task_state("t1"), TaskState.READY)

        # READY -> RUNNING
        self.orchestrator.transition_task("t1", TaskState.RUNNING, reason="Started")
        self.assertEqual(self.orchestrator.get_task_state("t1"), TaskState.RUNNING)

        # RUNNING -> WORKER_COMPLETED
        self.orchestrator.transition_task("t1", TaskState.WORKER_COMPLETED, reason="Done")
        self.assertEqual(self.orchestrator.get_task_state("t1"), TaskState.WORKER_COMPLETED)

        # WORKER_COMPLETED -> VALIDATION_PENDING
        self.orchestrator.transition_task("t1", TaskState.VALIDATION_PENDING, reason="Validating")
        self.assertEqual(self.orchestrator.get_task_state("t1"), TaskState.VALIDATION_PENDING)

    def test_invalid_transition_rejected(self):
        # Current state is READY; cannot transition directly to WORKER_COMPLETED
        with self.assertRaises(InvalidStateTransitionError):
            self.orchestrator.transition_task("t1", TaskState.WORKER_COMPLETED)

    def test_retry_pending_increments_retry_count(self):
        self.orchestrator.transition_task("t1", TaskState.RUNNING)
        self.orchestrator.transition_task("t1", TaskState.WORKER_FAILED)

        rec_before = self.orchestrator.get_task("t1")
        self.assertEqual(rec_before.retry_count, 0)

        self.orchestrator.transition_task("t1", TaskState.RETRY_PENDING)
        rec_after = self.orchestrator.get_task("t1")
        self.assertEqual(rec_after.retry_count, 1)

    def test_audit_event_logged(self):
        self.orchestrator.transition_task("t1", TaskState.RUNNING, reason="Agent started")
        events = self.orchestrator.get_events(task_id="t1")
        self.assertTrue(len(events) >= 2)  # SUBMITTED + STATE_CHANGE
        last_event = events[-1]
        self.assertEqual(last_event["new_state"], "RUNNING")
        self.assertEqual(last_event["message"], "Agent started")


class TestDependencyAdvancementAndFailurePropagation(unittest.TestCase):
    """Test advancement of ready tasks and failure propagation down the DAG."""

    def setUp(self):
        self.orchestrator = TaskOrchestrator()
        # Diamond dependency:
        #        t1
        #       /  \
        #     t2    t3
        #       \  /
        #        t4
        tasks = [
            TaskSpec(task_id="t1", batch_id="b1", project_id="proj", title="T1", description="Root", role="coder"),
            TaskSpec(task_id="t2", batch_id="b1", project_id="proj", title="T2", description="Branch 1", role="coder", depends_on=["t1"]),
            TaskSpec(task_id="t3", batch_id="b1", project_id="proj", title="T3", description="Branch 2", role="coder", depends_on=["t1"]),
            TaskSpec(task_id="t4", batch_id="b1", project_id="proj", title="T4", description="Join", role="tester", depends_on=["t2", "t3"]),
        ]
        self.orchestrator.submit_batch("b1", "proj", tasks)

    def tearDown(self):
        self.orchestrator.close()

    def test_diamond_advancement(self):
        # Initially, only t1 is READY
        ready = [t.task_id for t in self.orchestrator.get_ready_tasks("b1")]
        self.assertEqual(ready, ["t1"])

        # Complete t1
        self.orchestrator.transition_task("t1", TaskState.RUNNING)
        self.orchestrator.transition_task("t1", TaskState.WORKER_COMPLETED)

        # Advance dependencies
        advanced = self.orchestrator.evaluate_dependencies_and_advance("b1")
        self.assertCountEqual(advanced, ["t2", "t3"])
        self.assertEqual(self.orchestrator.get_task_state("t4"), TaskState.WAITING_DEPENDENCY)

        # Complete t2
        self.orchestrator.transition_task("t2", TaskState.RUNNING)
        self.orchestrator.transition_task("t2", TaskState.WORKER_COMPLETED)
        self.orchestrator.evaluate_dependencies_and_advance("b1")
        self.assertEqual(self.orchestrator.get_task_state("t4"), TaskState.WAITING_DEPENDENCY)

        # Complete t3
        self.orchestrator.transition_task("t3", TaskState.RUNNING)
        self.orchestrator.transition_task("t3", TaskState.WORKER_COMPLETED)
        advanced_final = self.orchestrator.evaluate_dependencies_and_advance("b1")
        self.assertEqual(advanced_final, ["t4"])
        self.assertEqual(self.orchestrator.get_task_state("t4"), TaskState.READY)

    def test_failure_propagation_blocks_downstream(self):
        # If t1 fails permanently, t2, t3, and t4 must be marked BLOCKED
        self.orchestrator.transition_task("t1", TaskState.RUNNING)
        self.orchestrator.transition_task("t1", TaskState.WORKER_FAILED)
        self.orchestrator.transition_task("t1", TaskState.BLOCKED, reason="Exhausted retries")

        self.orchestrator.evaluate_dependencies_and_advance("b1")
        self.assertEqual(self.orchestrator.get_task_state("t2"), TaskState.BLOCKED)
        self.assertEqual(self.orchestrator.get_task_state("t3"), TaskState.BLOCKED)

        # Now evaluate t4, which depends on t2 and t3
        self.orchestrator.evaluate_dependencies_and_advance("b1")
        self.assertEqual(self.orchestrator.get_task_state("t4"), TaskState.BLOCKED)


class TestWriteScopeLockManagement(unittest.TestCase):
    """Test mutual exclusion locks based on task write scopes."""

    def setUp(self):
        self.orchestrator = TaskOrchestrator()

    def tearDown(self):
        self.orchestrator.close()

    def test_disjoint_write_scopes_both_acquire(self):
        t1 = TaskSpec(task_id="t1", batch_id="b1", project_id="proj", title="T1", description="D", role="coder", write_scope=["src/a.py"])
        t2 = TaskSpec(task_id="t2", batch_id="b1", project_id="proj", title="T2", description="D", role="coder", write_scope=["src/b.py"])
        self.orchestrator.submit_batch("b1", "proj", [t1, t2])

        self.assertTrue(self.orchestrator.acquire_task_locks("t1"))
        self.assertTrue(self.orchestrator.acquire_task_locks("t2"))
        locks = self.orchestrator.get_held_locks()
        self.assertEqual(len(locks), 2)

    def test_overlapping_file_locks_conflict(self):
        t1 = TaskSpec(task_id="t1", batch_id="b1", project_id="proj", title="T1", description="D", role="coder", write_scope=["src/shared.py"])
        t2 = TaskSpec(task_id="t2", batch_id="b1", project_id="proj", title="T2", description="D", role="coder", write_scope=["src/shared.py"])
        self.orchestrator.submit_batch("b1", "proj", [t1, t2])

        self.assertTrue(self.orchestrator.acquire_task_locks("t1"))
        # t2 should fail lock acquisition
        self.assertFalse(self.orchestrator.acquire_task_locks("t2"))

        # Release t1
        self.orchestrator.release_task_locks("t1")
        # Now t2 can acquire
        self.assertTrue(self.orchestrator.acquire_task_locks("t2"))

    def test_directory_level_lock_conflict(self):
        # t1 locks "src/models", t2 locks "src/models/user.py"
        t1 = TaskSpec(task_id="t1", batch_id="b1", project_id="proj", title="T1", description="D", role="coder", write_scope=["src/models"])
        t2 = TaskSpec(task_id="t2", batch_id="b1", project_id="proj", title="T2", description="D", role="coder", write_scope=["src/models/user.py"])
        self.orchestrator.submit_batch("b1", "proj", [t1, t2])

        self.assertTrue(self.orchestrator.acquire_task_locks("t1"))
        self.assertFalse(self.orchestrator.acquire_task_locks("t2"))

        self.orchestrator.release_task_locks("t1")
        self.assertTrue(self.orchestrator.acquire_task_locks("t2"))


class TestCancellationAndPersistenceRecovery(unittest.TestCase):
    """Test task and batch cancellation and database restart recovery."""

    def test_cancel_task_releases_locks(self):
        orch = TaskOrchestrator()
        t1 = TaskSpec(task_id="t1", batch_id="b1", project_id="proj", title="T1", description="D", role="coder", write_scope=["src/a.py"])
        orch.submit_batch("b1", "proj", [t1])
        orch.acquire_task_locks("t1")
        self.assertEqual(len(orch.get_held_locks()), 1)

        orch.cancel_task("t1", reason="User aborted")
        self.assertEqual(orch.get_task_state("t1"), TaskState.CANCELLED)
        self.assertEqual(len(orch.get_held_locks()), 0)
        orch.close()

    def test_cancel_batch(self):
        orch = TaskOrchestrator()
        tasks = [
            TaskSpec(task_id="t1", batch_id="b1", project_id="proj", title="T1", description="D", role="coder"),
            TaskSpec(task_id="t2", batch_id="b1", project_id="proj", title="T2", description="D", role="coder"),
        ]
        orch.submit_batch("b1", "proj", tasks)
        orch.cancel_batch("b1", reason="Batch aborted")

        self.assertEqual(orch.get_task_state("t1"), TaskState.CANCELLED)
        self.assertEqual(orch.get_task_state("t2"), TaskState.CANCELLED)
        orch.close()

    def test_persistence_recovery_after_reopen(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name

        try:
            orch1 = TaskOrchestrator(db_path=db_path)
            t1 = TaskSpec(task_id="t1", batch_id="b1", project_id="proj", title="T1", description="D", role="coder")
            orch1.submit_batch("b1", "proj", [t1])
            orch1.transition_task("t1", TaskState.RUNNING)
            orch1.close()

            # Re-open database
            orch2 = TaskOrchestrator(db_path=db_path)
            t1_rec = orch2.get_task("t1")
            self.assertIsNotNone(t1_rec)
            self.assertEqual(t1_rec.state, TaskState.RUNNING)
            self.assertEqual(t1_rec.spec.title, "T1")
            orch2.close()
        finally:
            Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
