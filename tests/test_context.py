"""
SWAN-ENGINE v0.2 — Context Manager & Budgeting Tests
=====================================================

Comprehensive test suite covering:
  - Basic context construction and section rendering
  - Critical section preservation (Objective and System Instructions never dropped)
  - Strict context budget enforcement
  - Output token reservation (e.g. 2048 tokens reserved for completion)
  - Old observation eviction (FIFO within same priority)
  - Priority-based observation retention
  - Large repository AST map truncation
  - Large tool-output bounding
  - 20+ observation overflow handling
  - Empty context resilience
  - Unicode and multi-byte UTF-8 resilience
  - Multi-session and multi-project isolation
  - Worker-result integration (future HIVE compatibility)
  - Malformed input and edge case handling
  - Session clearing and resetting
"""

import sys
import unittest
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.context_manager import (
    ContextManager,
    ContextSession,
    Observation,
    ObservationType,
    SessionRole,
    WorkerResult,
    estimate_tokens,
    truncate_text_to_tokens,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_RESERVED_OUTPUT,
    DEFAULT_CHARS_PER_TOKEN,
)


class TestContextConstruction(unittest.TestCase):

    def setUp(self):
        self.session = ContextSession(session_id="test-session", project_id="test-proj")

    def test_basic_context_construction(self):
        self.session.set_system_instructions("You are SWAN-ENGINE.")
        self.session.set_project_info(name="TIDDA-Core", path="/home/user/tidda", project_id="tidda")
        self.session.set_objective("Refactor websocket handler.")
        self.session.set_task_specification("1. Fix socket leak\n2. Add timeout")

        prompt = self.session.build_prompt()

        self.assertIn("You are SWAN-ENGINE.", prompt)
        self.assertIn("Active Project Context:", prompt)
        self.assertIn("TIDDA-Core", prompt)
        self.assertIn("Refactor websocket handler.", prompt)
        self.assertIn("Task Specification:", prompt)

    def test_objective_preservation(self):
        """User objective must NEVER be dropped, even under tight budgets."""
        # Create a small budget session
        tiny_session = ContextSession(session_id="tiny", context_budget=1000, reserved_output=200)
        tiny_session.set_system_instructions("System instructions")
        tiny_session.set_objective("CRITICAL MISSION: DO NOT DROP THIS OBJECTIVE")

        # Overload with observations
        for i in range(15):
            tiny_session.add_observation(f"Long observation text that takes up space in budget {i} " * 20)

        prompt = tiny_session.build_prompt()
        self.assertIn("CRITICAL MISSION: DO NOT DROP THIS OBJECTIVE", prompt)

    def test_system_instruction_preservation(self):
        """System instructions must be preserved."""
        self.session.set_system_instructions("CRITICAL SYSTEM RULE: STAY IN WORKSPACE")
        self.session.set_objective("Run tests")

        prompt = self.session.build_prompt()
        self.assertIn("CRITICAL SYSTEM RULE: STAY IN WORKSPACE", prompt)

    def test_empty_context_resilience(self):
        """Empty context session builds safely without crashing."""
        empty_session = ContextSession(session_id="empty")
        prompt = empty_session.build_prompt()
        self.assertEqual(prompt, "")


class TestContextBudgeting(unittest.TestCase):

    def setUp(self):
        self.session = ContextSession(
            session_id="budget-session",
            context_budget=1000,
            reserved_output=200,
            chars_per_token=3.5,
        )

    def test_output_token_reservation(self):
        """Available input tokens = context_budget - reserved_output."""
        report = self.session.get_budget_report()
        self.assertEqual(report.total_budget_tokens, 1000)
        self.assertEqual(report.reserved_output_tokens, 200)
        self.assertEqual(report.available_input_tokens, 800)

    def test_context_budget_enforcement(self):
        """Total built prompt must never exceed available_input_tokens."""
        self.session.set_system_instructions("System prompt")
        self.session.set_objective("Objective")

        # Add many large observations
        for i in range(25):
            self.session.add_observation(f"Step {i} executed command with large result: " + ("x" * 200))

        prompt = self.session.build_prompt()
        est_tokens = self.session.estimate(prompt)

        self.assertLessEqual(est_tokens, 800)

    def test_old_observation_eviction(self):
        """Oldest observations are evicted first when priorities are equal."""
        self.session.set_objective("Test FIFO eviction")

        # Add 5 observations of equal priority, large enough to exceed 800 available tokens
        # Each observation ~700 chars / 3.5 = ~200 tokens. 5 * 200 = ~1000 tokens > 800 tokens.
        for i in range(1, 6):
            self.session.add_observation(f"UniqueMarkerStep{i} observation content " + ("a" * 650), step=i, priority=1)

        prompt = self.session.build_prompt()

        # Step 5 (newest) should be retained
        self.assertIn("UniqueMarkerStep5", prompt)
        # Step 1 (oldest) should have been evicted due to budget limit
        self.assertNotIn("UniqueMarkerStep1", prompt)

    def test_priority_based_retention(self):
        """High priority observations are retained even if older."""
        self.session.set_objective("Priority test")

        # Step 1 is high priority (3)
        self.session.add_observation("CRITICAL_STEP_1: Found root cause bug in auth.py", step=1, priority=3)

        # Steps 2, 3, 4 are low priority (1) but newer
        for i in range(2, 5):
            self.session.add_observation(f"LOW_PRIORITY_STEP_{i}: " + ("x" * 300), step=i, priority=1)

        prompt = self.session.build_prompt()

        # Step 1 MUST be retained because of priority 3
        self.assertIn("CRITICAL_STEP_1", prompt)

    def test_20_plus_observation_overflow(self):
        """A session with 25+ observations overflows gracefully with accurate tracking."""
        self.session.set_objective("Overflow mission")

        for i in range(1, 26):
            self.session.add_observation(f"Observation step {i}: result details " + ("z" * 150), step=i)

        report = self.session.get_budget_report()

        self.assertEqual(report.total_observations, 25)
        self.assertGreater(report.evicted_observations, 0)
        self.assertGreater(report.retained_observations, 0)
        self.assertEqual(report.total_observations, report.evicted_observations + report.retained_observations)
        self.assertTrue(report.truncation.was_truncated)
        self.assertIn("observations", report.truncation.truncated_sections)


class TestTruncationAndBounding(unittest.TestCase):

    def setUp(self):
        self.session = ContextSession(session_id="truncation-session")

    def test_large_repo_map_truncation_by_entries(self):
        """Repo map with > 200 lines is capped at max_entries."""
        lines = [f"- `src/module_{i}.py` -> Funcs: f{i}" for i in range(350)]
        big_map = "\n".join(lines)

        self.session.set_repo_map(big_map, max_entries=200)

        self.assertIn("more files in repository map", self.session._repo_map)
        # Should contain at most 201 lines (200 + truncation note)
        self.assertLessEqual(len(self.session._repo_map.splitlines()), 202)

    def test_large_tool_output_bounding(self):
        """Individual tool output exceeding max_tool_output_tokens is bounded on addition."""
        session = ContextSession(session_id="tool-bound", max_tool_output_tokens=100)
        huge_output = "stdout line\n" * 500  # ~6000 chars

        obs = session.add_observation(huge_output, tool_name="run_command")

        self.assertIn("tool output bounded at 100 estimated tokens", obs.content)
        self.assertLessEqual(session.estimate(obs.content), 120)

    def test_truncate_text_to_tokens_utility(self):
        text = "Hello world! " * 100
        truncated, was_trunc = truncate_text_to_tokens(text, max_tokens=20, chars_per_token=3.5)

        self.assertTrue(was_trunc)
        self.assertIn("[truncated]", truncated)
        self.assertLessEqual(estimate_tokens(truncated, 3.5), 25)


class TestUnicodeAndTokenEstimation(unittest.TestCase):

    def test_estimate_tokens_ascii(self):
        text = "hello world"  # 11 chars
        tokens = estimate_tokens(text, chars_per_token=3.5)
        self.assertGreater(tokens, 0)
        self.assertEqual(tokens, 4)

    def test_estimate_tokens_empty(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens(None), 0)

    def test_unicode_and_emojis(self):
        """Multi-byte UTF-8 (Devanagari, emojis) handled safely."""
        hindi_text = "नमस्ते दुनिया 🚀 SWAN-ENGINE"
        tokens = estimate_tokens(hindi_text)
        self.assertGreater(tokens, 0)

        session = ContextSession(session_id="unicode-test")
        session.set_objective(hindi_text)
        session.add_observation("अपेक्षित परिणाम प्राप्त हुआ ✅", tool_name="check")

        prompt = session.build_prompt()
        self.assertIn("नमस्ते दुनिया 🚀", prompt)
        self.assertIn("अपेक्षित परिणाम", prompt)


class TestMultiSessionAndIsolation(unittest.TestCase):

    def setUp(self):
        self.manager = ContextManager(default_budget=4000, default_reserved_output=500)

    def test_multiple_isolated_sessions(self):
        session_a = self.manager.create_session("session-a", project_id="proj-a")
        session_b = self.manager.create_session("session-b", project_id="proj-b")

        session_a.set_objective("Mission A: Fix login")
        session_a.add_observation("Obs in A")

        session_b.set_objective("Mission B: Refactor database")
        session_b.add_observation("Obs in B")

        prompt_a = session_a.build_prompt()
        prompt_b = session_b.build_prompt()

        self.assertIn("Mission A: Fix login", prompt_a)
        self.assertNotIn("Mission B", prompt_a)
        self.assertNotIn("Obs in B", prompt_a)

        self.assertIn("Mission B: Refactor database", prompt_b)
        self.assertNotIn("Mission A", prompt_b)
        self.assertNotIn("Obs in A", prompt_b)

    def test_duplicate_session_rejected(self):
        self.manager.create_session("sess-1")
        with self.assertRaises(ValueError):
            self.manager.create_session("sess-1")

    def test_get_sessions_for_project(self):
        self.manager.create_session("s1", project_id="tidda")
        self.manager.create_session("s2", project_id="tidda")
        self.manager.create_session("s3", project_id="other")

        tidda_sessions = self.manager.get_sessions_for_project("tidda")
        self.assertEqual(len(tidda_sessions), 2)
        self.assertEqual({s.session_id for s in tidda_sessions}, {"s1", "s2"})

    def test_delete_and_clear_sessions(self):
        self.manager.create_session("s1")
        self.manager.create_session("s2")

        self.assertTrue(self.manager.delete_session("s1"))
        self.assertIsNone(self.manager.get_session("s1"))
        self.assertEqual(self.manager.list_sessions(), ["s2"])

        self.manager.clear()
        self.assertEqual(self.manager.list_sessions(), [])


class TestFutureHIVECompatibility(unittest.TestCase):

    def setUp(self):
        self.session = ContextSession(session_id="hive-session", role=SessionRole.MANAGER)

    def test_worker_result_formatting_and_budgeting(self):
        self.session.set_objective("Manager coordinating workers")

        res = self.session.add_worker_result(
            worker_id="worker-coder-1",
            task_id="ast-indexing",
            status="success",
            summary="AST index completed for 45 files",
            artifacts=["docs/repo_map.md"],
            priority=2,
        )

        self.assertEqual(res.worker_id, "worker-coder-1")
        self.assertEqual(self.session.worker_result_count, 1)

        prompt = self.session.build_prompt()
        self.assertIn("SWAN-HIVE Worker Results", prompt)
        self.assertIn("[worker-coder-1] Task: ast-indexing (SUCCESS)", prompt)
        self.assertIn("Artifacts: docs/repo_map.md", prompt)

    def test_session_role_assignment(self):
        manager_sess = ContextSession("m1", role=SessionRole.MANAGER)
        worker_sess = ContextSession("w1", role=SessionRole.WORKER)

        self.assertEqual(manager_sess.role, SessionRole.MANAGER)
        self.assertEqual(worker_sess.role, SessionRole.WORKER)


class TestStateManagementAndReset(unittest.TestCase):

    def setUp(self):
        self.session = ContextSession(session_id="state-session")

    def test_clear_observations(self):
        self.session.set_objective("Keep objective")
        self.session.set_system_instructions("Keep system")
        self.session.add_observation("Obs 1")
        self.session.add_worker_result("w1", "summary")

        self.session.clear_observations()

        self.assertEqual(self.session.observation_count, 0)
        self.assertEqual(self.session.worker_result_count, 0)
        self.assertEqual(self.session.get_objective(), "Keep objective")
        self.assertEqual(self.session.get_system_instructions(), "Keep system")

    def test_full_reset(self):
        self.session.set_objective("Clear all")
        self.session.set_system_instructions("Clear system")
        self.session.add_observation("Obs 1")

        self.session.reset()

        self.assertEqual(self.session.observation_count, 0)
        self.assertEqual(self.session.get_objective(), "")
        self.assertEqual(self.session.get_system_instructions(), "")

    def test_budget_report_diagnostics(self):
        self.session.set_system_instructions("System")
        self.session.set_objective("Objective")
        self.session.add_observation("Some tool output")

        report = self.session.get_budget_report()

        self.assertTrue(report.is_estimate)
        self.assertEqual(report.estimation_method, "character_heuristic")
        self.assertGreater(report.system_tokens, 0)
        self.assertGreater(report.objective_tokens, 0)
        self.assertGreater(report.observations_tokens, 0)
        self.assertGreater(report.remaining_tokens, 0)


class TestMalformedInputHandling(unittest.TestCase):

    def test_none_and_invalid_inputs(self):
        session = ContextSession(
            session_id="robust",
            context_budget=-10,        # Should clamp to >= 512
            reserved_output=-5,        # Should clamp
            chars_per_token=-2.0,      # Should clamp to >= 1.0
        )

        self.assertGreaterEqual(session.context_budget, 512)
        self.assertGreaterEqual(session.reserved_output, 128)
        self.assertGreaterEqual(session.chars_per_token, 1.0)

        # None inputs shouldn't crash
        session.set_system_instructions(None)
        session.set_objective(None)
        session.set_repo_map(None)
        session.set_task_specification(None)

        obs = session.add_observation(None)
        self.assertEqual(obs.content, "")

        prompt = session.build_prompt()
        self.assertIsInstance(prompt, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
