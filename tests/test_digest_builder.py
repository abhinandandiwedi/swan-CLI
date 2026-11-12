"""Unit tests for Compact Manager Digest Builder.

Tests batch-level metrics, aggregation of independent validation findings,
diff statistics summarization, actionable error extraction, Manager action triggers,
markdown rendering, and on-demand task detail expansion.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest

from core.digest_builder import (
    BatchDigest,
    DigestBuilder,
)
from core.validator import (
    ValidationReport,
    ValidationCheckResult,
    DiffStats,
)
from core.task_orchestrator import (
    TaskOrchestrator,
    TaskSpec,
    TaskState,
)
from core.worker_runner import (
    WorkerResult,
)


class TestBatchDigestSerialization(unittest.TestCase):
    """Test serialization, formatting, and compactness of BatchDigest."""

    def test_digest_to_dict_and_json(self):
        digest = BatchDigest(
            batch_id="batch-001",
            project_id="test-proj",
            base_commit="abc1234",
            total_tasks=2,
            passed_count=2,
            validation_status="passed",
            diff_statistics={"total_files_changed": 3, "total_insertions": 40, "total_deletions": 5},
            manager_actions_required=["Review and approve merge"],
        )
        d = digest.to_dict()
        self.assertEqual(d["batch_id"], "batch-001")
        self.assertEqual(d["passed_count"], 2)

        j_str = digest.to_json()
        self.assertIn("batch-001", j_str)
        self.assertIn("abc1234", j_str)

    def test_compactness_no_large_source_or_conversations(self):
        digest = BatchDigest(
            batch_id="b1",
            project_id="proj",
            total_tasks=1,
            passed_count=1,
            short_task_summaries=[{
                "task_id": "t1",
                "role": "coder",
                "title": "Short title",
                "state": "VALIDATION_PENDING",
                "validation": "passed",
                "summary": "Concise summary",
                "changed_files": ["src/app.py"],
            }],
        )
        md = digest.to_markdown()
        # Verify markdown is concise
        self.assertLess(len(md.splitlines()), 35)
        self.assertIn("SWAN Batch Execution Digest", md)
        self.assertIn("src/app.py", md)


class TestDigestBuilderAggregation(unittest.TestCase):
    """Test gathering metrics, diff statistics, errors, and actions from Orchestrator."""

    def setUp(self):
        self.orchestrator = TaskOrchestrator()
        self.builder = DigestBuilder(orchestrator=self.orchestrator)

    def tearDown(self):
        self.orchestrator.close()

    def test_all_tasks_passed_digest(self):
        t1 = TaskSpec(task_id="t1", batch_id="b1", project_id="proj", title="T1", description="D", role="coder")
        t2 = TaskSpec(task_id="t2", batch_id="b1", project_id="proj", title="T2", description="D", role="tester", depends_on=["t1"])
        self.orchestrator.submit_batch("b1", "proj", [t1, t2])

        # Simulate t1 complete and validated
        self.orchestrator.transition_task("t1", TaskState.RUNNING)
        self.orchestrator.transition_task("t1", TaskState.WORKER_COMPLETED)
        self.orchestrator.transition_task("t1", TaskState.VALIDATION_PENDING)

        rep1 = ValidationReport(
            batch_id="b1",
            task_id="t1",
            validation_status="passed",
            diff_stats=DiffStats(files_changed=1, insertions=15, deletions=2, changed_files=["src/t1.py"]),
        )
        self.orchestrator.record_validation_report(rep1)

        # Advance dependencies so t2 moves from WAITING_DEPENDENCY to READY
        self.orchestrator.evaluate_dependencies_and_advance("b1")
        self.assertEqual(self.orchestrator.get_task_state("t2"), TaskState.READY)

        # Simulate t2 complete and validated
        self.orchestrator.transition_task("t2", TaskState.RUNNING)
        self.orchestrator.transition_task("t2", TaskState.WORKER_COMPLETED)
        self.orchestrator.transition_task("t2", TaskState.VALIDATION_PENDING)

        rep2 = ValidationReport(
            batch_id="b1",
            task_id="t2",
            validation_status="passed",
            diff_stats=DiffStats(files_changed=1, insertions=30, deletions=0, changed_files=["tests/test_t1.py"]),
        )
        self.orchestrator.record_validation_report(rep2)

        digest = self.builder.build_digest("b1")
        self.assertEqual(digest.total_tasks, 2)
        self.assertEqual(digest.passed_count, 2)
        self.assertEqual(digest.failed_count, 0)
        self.assertEqual(digest.blocked_count, 0)
        self.assertEqual(digest.validation_status, "passed")
        self.assertEqual(digest.diff_statistics["total_files_changed"], 2)
        self.assertEqual(digest.diff_statistics["total_insertions"], 45)
        self.assertEqual(digest.diff_statistics["total_deletions"], 2)

        # Manager action for successful batch must require human approval before merge
        self.assertTrue(any("human approval" in a.lower() for a in digest.manager_actions_required))

    def test_blocked_and_failed_tasks_digest(self):
        t1 = TaskSpec(task_id="t1", batch_id="b2", project_id="proj", title="T1", description="D", role="coder")
        t2 = TaskSpec(task_id="t2", batch_id="b2", project_id="proj", title="T2", description="D", role="coder", depends_on=["t1"])
        self.orchestrator.submit_batch("b2", "proj", [t1, t2])

        # t1 failed permanently
        self.orchestrator.transition_task("t1", TaskState.RUNNING)
        self.orchestrator.transition_task("t1", TaskState.WORKER_FAILED)
        self.orchestrator.transition_task("t1", TaskState.BLOCKED, reason="Exhausted retries")
        self.orchestrator.evaluate_dependencies_and_advance("b2")

        # t2 is blocked by t1
        self.assertEqual(self.orchestrator.get_task_state("t2"), TaskState.BLOCKED)

        digest = self.builder.build_digest("b2")
        self.assertEqual(digest.total_tasks, 2)
        self.assertEqual(digest.blocked_count, 2)
        self.assertEqual(digest.validation_status, "blocked")
        self.assertTrue(any("blocked" in a.lower() for a in digest.manager_actions_required))

    def test_out_of_scope_security_alert_action(self):
        t1 = TaskSpec(task_id="t1", batch_id="b3", project_id="proj", title="T1", description="D", role="coder")
        self.orchestrator.submit_batch("b3", "proj", [t1])
        self.orchestrator.transition_task("t1", TaskState.RUNNING)
        self.orchestrator.transition_task("t1", TaskState.WORKER_COMPLETED)

        # Record validation report with out_of_scope_changes
        rep = ValidationReport(
            batch_id="b3",
            task_id="t1",
            validation_status="failed",
            out_of_scope_changes=["/etc/shadow", "other/secret.py"],
            checks=[
                ValidationCheckResult(name="scope", passed=False, errors=["Unauthorized file modification"]),
            ],
        )
        self.orchestrator.record_validation_report(rep)

        digest = self.builder.build_digest("b3")
        self.assertTrue(any("CRITICAL" in a or "Out-of-scope" in a for a in digest.manager_actions_required))
        self.assertTrue(len(digest.important_error_lines) > 0)


class TestTaskDetailExpansion(unittest.TestCase):
    """Test expanding on-demand details for an individual task."""

    def setUp(self):
        self.orchestrator = TaskOrchestrator()
        self.builder = DigestBuilder(orchestrator=self.orchestrator)

    def tearDown(self):
        self.orchestrator.close()

    def test_get_task_detail_complete(self):
        task = TaskSpec(
            task_id="t1",
            batch_id="b1",
            project_id="proj",
            title="Detail Test",
            description="Testing task detail expansion",
            role="coder",
            read_scope=["src/a.py"],
            write_scope=["src/a.py"],
        )
        self.orchestrator.submit_batch("b1", "proj", [task])
        self.orchestrator.transition_task("t1", TaskState.RUNNING)

        # Record worker result
        w_res = WorkerResult(
            task_id="t1",
            batch_id="b1",
            worker_id="w1",
            model="swan-coder",
            status="completed",
            summary="Refactored src/a.py",
            changed_files=["src/a.py"],
        )
        self.orchestrator.record_worker_result(w_res)

        # Record validation report
        val_rep = ValidationReport(
            batch_id="b1",
            task_id="t1",
            validation_status="passed",
            diff_stats=DiffStats(files_changed=1, insertions=10, deletions=2),
        )
        self.orchestrator.record_validation_report(val_rep)

        detail = self.builder.get_task_detail("t1")
        self.assertEqual(detail["task_id"], "t1")
        self.assertEqual(detail["state"], "RUNNING")
        self.assertEqual(len(detail["worker_results"]), 1)
        self.assertEqual(detail["worker_results"][0]["summary"], "Refactored src/a.py")
        self.assertEqual(len(detail["validation_reports"]), 1)
        self.assertEqual(detail["validation_reports"][0]["validation_status"], "passed")
        self.assertTrue(len(detail["events"]) >= 2)

    def test_get_task_detail_not_found(self):
        detail = self.builder.get_task_detail("missing_task_id")
        self.assertIn("error", detail)


if __name__ == "__main__":
    unittest.main()
