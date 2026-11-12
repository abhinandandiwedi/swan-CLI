"""Unit tests for Manager Verdict Protocol.

Tests verdict schema validation, state transitions (ACCEPT, REQUEST_REVISION,
RETRY, REJECT), security supremacy enforcement (verdict cannot override security
or scope failures), and bounded retry accounting.
"""

from __future__ import annotations

import unittest

from core.manager_verdict import (
    ManagerVerdict,
    ManagerVerdictService,
    VerdictType,
    VerdictValidationError,
    SecurityOverrideError,
    RetryLimitExceededError,
)
from core.task_orchestrator import (
    TaskOrchestrator,
    TaskSpec,
    TaskState,
)
from core.validator import (
    ValidationReport,
    ValidationStatus,
    ValidationCheckResult,
)


class TestManagerVerdictValidation(unittest.TestCase):
    """Test schema validation and serialization for ManagerVerdict."""

    def test_valid_verdict(self):
        v = ManagerVerdict(
            batch_id="b1",
            task_id="t1",
            verdict=VerdictType.ACCEPT,
            reason="All acceptance tests and validation checks passed cleanly",
        )
        v.validate()
        d = v.to_dict()
        reconstructed = ManagerVerdict.from_dict(d)
        self.assertEqual(v.task_id, reconstructed.task_id)
        self.assertEqual(v.verdict, reconstructed.verdict)

    def test_invalid_verdict_string_rejected(self):
        with self.assertRaises(VerdictValidationError):
            ManagerVerdict(
                batch_id="b1",
                task_id="t1",
                verdict="INVALID_VERDICT_XYZ",  # type: ignore
                reason="Reason",
            ).validate()

    def test_missing_reason_rejected(self):
        with self.assertRaises(VerdictValidationError):
            ManagerVerdict(
                batch_id="b1",
                task_id="t1",
                verdict=VerdictType.ACCEPT,
                reason="",  # Empty reason rejected
            ).validate()


class TestManagerVerdictService(unittest.TestCase):
    """Test verdict processing and security supremacy guardrails."""

    def setUp(self):
        self.orchestrator = TaskOrchestrator()
        self.task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="proj",
            title="Task 1",
            description="Desc",
            role="coder",
            write_scope=["src/app.py"],
            retry_limits=2,
        )
        self.orchestrator.submit_batch("b1", "proj", [self.task])

    def tearDown(self):
        self.orchestrator.close()

    def test_accept_verdict_on_passed_validation_succeeds(self):
        # Move task to VALIDATION_PENDING
        self.orchestrator.transition_task("t1", TaskState.RUNNING)
        self.orchestrator.transition_task("t1", TaskState.WORKER_COMPLETED)
        self.orchestrator.transition_task("t1", TaskState.VALIDATION_PENDING)

        # Record PASSED validation report
        rep = ValidationReport(
            batch_id="b1",
            task_id="t1",
            validation_status=ValidationStatus.PASSED.value,
        )
        self.orchestrator.record_validation_report(rep)

        verdict = ManagerVerdict(
            batch_id="b1",
            task_id="t1",
            verdict=VerdictType.ACCEPT,
            reason="Passed all tests and checks",
        )
        updated_task = ManagerVerdictService.apply_verdict(verdict, self.orchestrator)
        self.assertEqual(updated_task.state, TaskState.VALIDATION_PENDING)

    def test_accept_verdict_blocked_on_scope_violation(self):
        # Task completed but had unauthorized out-of-scope changes
        self.orchestrator.transition_task("t1", TaskState.RUNNING)
        self.orchestrator.transition_task("t1", TaskState.WORKER_COMPLETED)
        self.orchestrator.transition_task("t1", TaskState.VALIDATION_PENDING)

        rep = ValidationReport(
            batch_id="b1",
            task_id="t1",
            validation_status=ValidationStatus.FAILED.value,
            out_of_scope_changes=["/etc/shadow", "other/file.py"],
        )
        self.orchestrator.record_validation_report(rep)

        # Manager verdict ACCEPT must be REJECTED! Manager cannot override security!
        verdict = ManagerVerdict(
            batch_id="b1",
            task_id="t1",
            verdict=VerdictType.ACCEPT,
            reason="I don't care about scope, accept it anyway!",
        )
        with self.assertRaises(SecurityOverrideError):
            ManagerVerdictService.apply_verdict(verdict, self.orchestrator)

    def test_accept_verdict_blocked_on_failed_validation_status(self):
        self.orchestrator.transition_task("t1", TaskState.RUNNING)
        self.orchestrator.transition_task("t1", TaskState.WORKER_COMPLETED)
        self.orchestrator.transition_task("t1", TaskState.VALIDATION_PENDING)

        rep = ValidationReport(
            batch_id="b1",
            task_id="t1",
            validation_status=ValidationStatus.FAILED.value,
            checks=[ValidationCheckResult(name="tests", passed=False, errors=["Unit test failure"])],
        )
        self.orchestrator.record_validation_report(rep)

        verdict = ManagerVerdict(
            batch_id="b1",
            task_id="t1",
            verdict=VerdictType.ACCEPT,
            reason="Accept failing test",
        )
        with self.assertRaises(SecurityOverrideError):
            ManagerVerdictService.apply_verdict(verdict, self.orchestrator)

    def test_request_revision_transitions_to_ready(self):
        self.orchestrator.transition_task("t1", TaskState.RUNNING)
        self.orchestrator.transition_task("t1", TaskState.WORKER_COMPLETED)
        self.orchestrator.transition_task("t1", TaskState.VALIDATION_PENDING)

        self.assertEqual(self.orchestrator.get_task("t1").retry_count, 0)

        verdict = ManagerVerdict(
            batch_id="b1",
            task_id="t1",
            verdict=VerdictType.REQUEST_REVISION,
            reason="Please add docstrings and handle None inputs",
            feedback="Add input validation for None",
        )
        updated = ManagerVerdictService.apply_verdict(verdict, self.orchestrator)
        self.assertEqual(updated.state, TaskState.READY)
        self.assertEqual(updated.retry_count, 1)

    def test_retry_limit_exhausted_blocks_task(self):
        # Exhaust retries on task
        self.orchestrator.transition_task("t1", TaskState.RUNNING)
        self.orchestrator.transition_task("t1", TaskState.WORKER_FAILED)
        self.orchestrator.transition_task("t1", TaskState.RETRY_PENDING)  # retry 1
        self.orchestrator.transition_task("t1", TaskState.READY)
        self.orchestrator.transition_task("t1", TaskState.RUNNING)
        self.orchestrator.transition_task("t1", TaskState.WORKER_FAILED)
        self.orchestrator.transition_task("t1", TaskState.RETRY_PENDING)  # retry 2 (max)
        self.orchestrator.transition_task("t1", TaskState.READY)
        self.orchestrator.transition_task("t1", TaskState.RUNNING)
        self.orchestrator.transition_task("t1", TaskState.WORKER_COMPLETED)
        self.orchestrator.transition_task("t1", TaskState.VALIDATION_PENDING)

        self.assertEqual(self.orchestrator.get_task("t1").retry_count, 2)

        verdict = ManagerVerdict(
            batch_id="b1",
            task_id="t1",
            verdict=VerdictType.RETRY,
            reason="Try again one more time",
        )
        with self.assertRaises(RetryLimitExceededError):
            ManagerVerdictService.apply_verdict(verdict, self.orchestrator)

        # Task must now be in BLOCKED state
        self.assertEqual(self.orchestrator.get_task_state("t1"), TaskState.BLOCKED)

    def test_reject_verdict_blocks_task(self):
        self.orchestrator.transition_task("t1", TaskState.RUNNING)
        self.orchestrator.transition_task("t1", TaskState.WORKER_COMPLETED)
        self.orchestrator.transition_task("t1", TaskState.VALIDATION_PENDING)

        verdict = ManagerVerdict(
            batch_id="b1",
            task_id="t1",
            verdict=VerdictType.REJECT,
            reason="Unsuitable approach, cancelling task",
        )
        updated = ManagerVerdictService.apply_verdict(verdict, self.orchestrator)
        self.assertEqual(updated.state, TaskState.BLOCKED)

    def test_verdict_on_running_task_rejected(self):
        # Task is in READY/RUNNING, not finished yet
        self.orchestrator.transition_task("t1", TaskState.RUNNING)

        verdict = ManagerVerdict(
            batch_id="b1",
            task_id="t1",
            verdict=VerdictType.ACCEPT,
            reason="Premature accept",
        )
        with self.assertRaises(VerdictValidationError):
            ManagerVerdictService.apply_verdict(verdict, self.orchestrator)


if __name__ == "__main__":
    unittest.main()
