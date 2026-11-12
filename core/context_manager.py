"""
SWAN-ENGINE v0.2 — Context Manager & Budgeting
===============================================

Manages bounded model context independently of the agent core.
Designed for 16K context windows (Ollama / Qwen 2.5 Coder 7B) with
architectural hooks for hierarchical multi-agent SWAN-HIVE.

Context Sections Supported:
  1. System Instructions (Critical — preserved)
  2. Active Project Information (Isolated per workspace)
  3. Repository Map (AST index — bounded and truncated)
  4. User Objective (Critical — preserved)
  5. Task Specification (Detailed task bounds)
  6. Relevant File Observations (Bounded content)
  7. Tool Observations (Bounded, prioritized, chronologically evicted)
  8. Worker Results (Future HIVE compatibility)

Budget Guarantee:
  - Context Budget is strictly enforced.
  - Output tokens are reserved upfront (default: 2048 tokens).
  - Dynamic content fits within remaining budget.
  - Oldest low-priority observations are evicted first.
  - Explicit truncation metadata is tracked and reported.
  - Conservative character-based token estimation clearly identified as an estimate.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_WINDOW: int = 16384       # 16K target for Qwen 2.5 Coder
DEFAULT_RESERVED_OUTPUT: int = 2048       # Reserved for LLM completion
DEFAULT_CHARS_PER_TOKEN: float = 3.5      # Conservative estimate for code/English
DEFAULT_MAX_REPO_MAP_TOKENS: int = 2000   # Max budget for repo map preview
DEFAULT_MAX_TOOL_OUTPUT_TOKENS: int = 1500 # Max tokens per single tool output
DEFAULT_MAX_REPO_MAP_ENTRIES: int = 200   # Hard entry limit


# ---------------------------------------------------------------------------
# Enums and Typed Data Structures
# ---------------------------------------------------------------------------

class SessionRole(str, Enum):
    """Role classification for future HIVE compatibility."""
    COORDINATOR = "coordinator"
    MANAGER = "manager"
    WORKER = "worker"
    SOLO = "solo"


class ObservationType(str, Enum):
    """Classification of context observations."""
    TOOL = "tool"
    FILE = "file"
    WORKER = "worker"
    SYSTEM = "system"
    ERROR = "error"


@dataclass
class Observation:
    """An individual observation within a context session."""
    content: str
    obs_type: ObservationType = ObservationType.TOOL
    step: int = 0
    priority: int = 1            # Higher integer = higher retention priority
    timestamp: float = field(default_factory=time.time)
    tool_name: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.obs_type, ObservationType):
            try:
                self.obs_type = ObservationType(str(self.obs_type).lower())
            except ValueError:
                self.obs_type = ObservationType.TOOL


@dataclass
class WorkerResult:
    """Worker result summary for future HIVE hierarchy."""
    worker_id: str
    task_id: str
    status: str                  # "success", "failure", "in_progress"
    summary: str
    artifacts: List[str] = field(default_factory=list)
    priority: int = 1
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TruncationMetadata:
    """Explicit metadata on what was truncated or evicted."""
    was_truncated: bool = False
    evicted_observation_count: int = 0
    retained_observation_count: int = 0
    truncated_sections: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class BudgetReport:
    """Detailed context budget accounting."""
    total_budget_tokens: int
    reserved_output_tokens: int
    available_input_tokens: int
    estimated_used_tokens: int
    remaining_tokens: int
    chars_per_token: float
    is_estimate: bool
    estimation_method: str
    system_tokens: int
    project_tokens: int
    repo_map_tokens: int
    objective_tokens: int
    task_spec_tokens: int
    observations_tokens: int
    worker_results_tokens: int
    total_observations: int
    retained_observations: int
    evicted_observations: int
    truncation: TruncationMetadata


# ---------------------------------------------------------------------------
# Token Estimator
# ---------------------------------------------------------------------------

def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """
    Conservatively estimate token count from text using character heuristic.

    Clearly identified as an estimate: does not claim exact BPE token counts.
    Accounts for multibyte UTF-8 characters conservatively.
    """
    if not text:
        return 0

    if chars_per_token <= 0:
        chars_per_token = DEFAULT_CHARS_PER_TOKEN

    char_est = math.ceil(len(text) / chars_per_token)
    # Check byte length to prevent underestimating multi-byte UTF-8 sequences
    byte_est = math.ceil(len(text.encode("utf-8", errors="replace")) / 4.0)

    return max(char_est, byte_est)


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    suffix: str = "\n... [truncated]",
) -> Tuple[str, bool]:
    """
    Truncate text so its estimated token count does not exceed max_tokens.

    Returns (truncated_text, was_truncated).
    """
    if not text or max_tokens <= 0:
        return "", bool(text)

    current_tokens = estimate_tokens(text, chars_per_token)
    if current_tokens <= max_tokens:
        return text, False

    suffix_tokens = estimate_tokens(suffix, chars_per_token)
    allowed_content_tokens = max(1, max_tokens - suffix_tokens)
    target_chars = int(allowed_content_tokens * chars_per_token)

    truncated = text[:target_chars] + suffix
    return truncated, True


# ---------------------------------------------------------------------------
# Context Session
# ---------------------------------------------------------------------------

class ContextSession:
    """
    A single bounded context session for an agent or worker.
    Maintains isolation across tasks, workspaces, and roles.
    """

    def __init__(
        self,
        session_id: str,
        project_id: Optional[str] = None,
        role: SessionRole = SessionRole.SOLO,
        context_budget: int = DEFAULT_CONTEXT_WINDOW,
        reserved_output: int = DEFAULT_RESERVED_OUTPUT,
        chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
        max_repo_map_tokens: int = DEFAULT_MAX_REPO_MAP_TOKENS,
        max_tool_output_tokens: int = DEFAULT_MAX_TOOL_OUTPUT_TOKENS,
    ):
        self.session_id = str(session_id)
        self.project_id = str(project_id) if project_id else None
        self.role = role if isinstance(role, SessionRole) else SessionRole(str(role).lower())
        self.context_budget = max(512, int(context_budget))
        self.reserved_output = max(128, min(int(reserved_output), self.context_budget // 2))
        self.chars_per_token = max(1.0, float(chars_per_token))
        self.max_repo_map_tokens = max(100, int(max_repo_map_tokens))
        self.max_tool_output_tokens = max(50, int(max_tool_output_tokens))

        # Context components
        self._system_instructions: str = ""
        self._project_name: Optional[str] = None
        self._project_path: Optional[str] = None
        self._project_metadata: Dict[str, Any] = {}
        self._repo_map: str = ""
        self._objective: str = ""
        self._task_specification: str = ""
        self._observations: List[Observation] = []
        self._worker_results: List[WorkerResult] = []

    # ------ Configuration & Setting APIs ------

    def set_system_instructions(self, instructions: str) -> None:
        """Set system instructions. Never silently truncated or dropped."""
        self._system_instructions = str(instructions).strip() if instructions else ""

    def set_project_info(
        self,
        name: str,
        path: str,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set active project information."""
        self._project_name = str(name).strip() if name else None
        self._project_path = str(path).strip() if path else None
        if project_id:
            self.project_id = str(project_id)
        self._project_metadata = dict(metadata) if metadata else {}

    def set_repo_map(self, repo_map: str, max_entries: int = DEFAULT_MAX_REPO_MAP_ENTRIES) -> None:
        """
        Set the repository AST map.
        Bounds entries to max_entries to prevent monorepo token explosion.
        """
        if not repo_map:
            self._repo_map = ""
            return

        lines = repo_map.strip().splitlines()
        if len(lines) > max_entries:
            truncated_lines = lines[:max_entries]
            truncated_lines.append(f"... [{len(lines) - max_entries} more files in repository map]")
            self._repo_map = "\n".join(truncated_lines)
        else:
            self._repo_map = repo_map.strip()

    def set_objective(self, objective: str) -> None:
        """Set user objective. Critical section — never evicted."""
        self._objective = str(objective).strip() if objective else ""

    def set_task_specification(self, spec: str) -> None:
        """Set task specification."""
        self._task_specification = str(spec).strip() if spec else ""

    def add_observation(
        self,
        content: str,
        obs_type: ObservationType = ObservationType.TOOL,
        step: int = 0,
        priority: int = 1,
        tool_name: Optional[str] = None,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Observation:
        """
        Add a bounded observation (tool output, file read, system note).
        Single tool outputs exceeding max_tool_output_tokens are truncated.
        """
        raw_text = str(content) if content is not None else ""
        bounded_text, _ = truncate_text_to_tokens(
            raw_text,
            max_tokens=self.max_tool_output_tokens,
            chars_per_token=self.chars_per_token,
            suffix=f"\n... [tool output bounded at {self.max_tool_output_tokens} estimated tokens]",
        )

        obs = Observation(
            content=bounded_text,
            obs_type=obs_type,
            step=int(step),
            priority=int(priority),
            timestamp=time.time(),
            tool_name=str(tool_name) if tool_name else None,
            file_path=str(file_path) if file_path else None,
            metadata=dict(metadata) if metadata else {},
        )
        self._observations.append(obs)
        return obs

    def add_worker_result(
        self,
        worker_id: str,
        summary: str,
        status: str = "success",
        task_id: Optional[str] = None,
        artifacts: Optional[List[str]] = None,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkerResult:
        """Add a worker summary result (future HIVE compatibility)."""
        res = WorkerResult(
            worker_id=str(worker_id),
            task_id=str(task_id or f"task-{len(self._worker_results) + 1}"),
            status=str(status),
            summary=str(summary).strip() if summary else "",
            artifacts=list(artifacts) if artifacts else [],
            priority=int(priority),
            timestamp=time.time(),
            metadata=dict(metadata) if metadata else {},
        )
        self._worker_results.append(res)
        return res

    # ------ Inspection & State Queries ------

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    @property
    def worker_result_count(self) -> int:
        return len(self._worker_results)

    def get_objective(self) -> str:
        return self._objective

    def get_system_instructions(self) -> str:
        return self._system_instructions

    def clear_observations(self) -> None:
        """Clear observations while retaining objective, system instructions, and project info."""
        self._observations.clear()
        self._worker_results.clear()

    def reset(self) -> None:
        """Full reset of the session state."""
        self._system_instructions = ""
        self._project_name = None
        self._project_path = None
        self._project_metadata.clear()
        self._repo_map = ""
        self._objective = ""
        self._task_specification = ""
        self._observations.clear()
        self._worker_results.clear()

    # ------ Context Budget & Building APIs ------

    def estimate(self, text: str) -> int:
        """Estimate tokens using session's chars_per_token."""
        return estimate_tokens(text, self.chars_per_token)

    def _build_project_header(self) -> str:
        """Build formatted active project block."""
        if not self._project_name and not self._project_path:
            return ""
        parts = ["Active Project Context:"]
        if self._project_name:
            parts.append(f"  Name: {self._project_name}")
        if self.project_id:
            parts.append(f"  ID: {self.project_id}")
        if self._project_path:
            parts.append(f"  Root: {self._project_path}")
        return "\n".join(parts)

    def _build_worker_results_section(self, max_tokens: int) -> Tuple[str, bool]:
        """Format worker results, bounded by max_tokens."""
        if not self._worker_results or max_tokens <= 0:
            return "", False

        lines = ["=== SWAN-HIVE Worker Results ==="]
        for res in self._worker_results:
            art_str = f" | Artifacts: {', '.join(res.artifacts)}" if res.artifacts else ""
            lines.append(f"[{res.worker_id}] Task: {res.task_id} ({res.status.upper()}){art_str}")
            lines.append(f"  Summary: {res.summary}")

        content = "\n".join(lines)
        return truncate_text_to_tokens(content, max_tokens, self.chars_per_token)

    def build_prompt(self) -> str:
        """
        Assemble the bounded context prompt strictly respecting the context budget.

        Preservation Order:
          1. System instructions & User objective (Never dropped)
          2. Active project context & Task specification (High priority)
          3. Repo map (Bounded by max_repo_map_tokens)
          4. Worker results (Bounded by remaining dynamic budget)
          5. Observations (Chronological with priority; lowest priority / oldest evicted first)

        Returns the assembled prompt string.
        """
        available_input_tokens = self.context_budget - self.reserved_output

        # 1. Critical Header & Objective
        sections: Dict[str, str] = {}

        if self._system_instructions:
            sections["system"] = self._system_instructions

        project_hdr = self._build_project_header()
        if project_hdr:
            sections["project"] = project_hdr

        if self._objective:
            sections["objective"] = f"User Objective:\n{self._objective}"

        if self._task_specification:
            sections["task_spec"] = f"Task Specification:\n{self._task_specification}"

        # Calculate committed tokens for critical sections
        committed_text = "\n\n".join(sections.values())
        committed_tokens = self.estimate(committed_text)

        # Dynamic budget remaining for repo map, worker results, observations
        dynamic_budget = max(0, available_input_tokens - committed_tokens)

        # 2. Repo Map Budgeting
        repo_map_budget = min(self.max_repo_map_tokens, dynamic_budget // 3)
        if self._repo_map and repo_map_budget > 50:
            bounded_map, _ = truncate_text_to_tokens(
                f"Repository AST Map:\n{self._repo_map}",
                repo_map_budget,
                self.chars_per_token,
                suffix="\n... [AST Map truncated to fit budget]"
            )
            sections["repo_map"] = bounded_map
            dynamic_budget = max(0, dynamic_budget - self.estimate(bounded_map))

        # 3. Worker Results Budgeting
        worker_budget = min(2000, dynamic_budget // 3)
        if self._worker_results and worker_budget > 50:
            bounded_workers, _ = self._build_worker_results_section(worker_budget)
            if bounded_workers:
                sections["workers"] = bounded_workers
                dynamic_budget = max(0, dynamic_budget - self.estimate(bounded_workers))

        # 4. Observations Budgeting (Drop oldest low-priority first)
        retained_obs, evicted_count = self._select_observations(dynamic_budget)
        if retained_obs:
            obs_lines = ["Observations & Actions History:"]
            if evicted_count > 0:
                obs_lines.append(f"[Note: {evicted_count} older observation(s) evicted to maintain {self.context_budget} token budget]")

            for obs in retained_obs:
                prefix = f"Step {obs.step} " if obs.step > 0 else ""
                tool_label = f"[{obs.tool_name}] " if obs.tool_name else ""
                obs_lines.append(f"--- {prefix}{tool_label}(Type: {obs.obs_type.value}) ---")
                obs_lines.append(obs.content)

            sections["observations"] = "\n".join(obs_lines)

        # Combine in logical semantic order
        ordered_keys = ["system", "project", "repo_map", "task_spec", "workers", "observations", "objective"]
        assembled_parts = [sections[k] for k in ordered_keys if k in sections]
        prompt = "\n\n".join(assembled_parts)

        # Enforce strict boundary: if inter-section formatting/newlines exceeded budget, evict further
        while retained_obs and self.estimate(prompt) > available_input_tokens:
            retained_obs.pop(0)  # Evict oldest remaining
            evicted_count += 1
            if retained_obs:
                obs_lines = ["Observations & Actions History:"]
                obs_lines.append(f"[Note: {evicted_count} older observation(s) evicted to maintain {self.context_budget} token budget]")
                for obs in retained_obs:
                    prefix = f"Step {obs.step} " if obs.step > 0 else ""
                    tool_label = f"[{obs.tool_name}] " if obs.tool_name else ""
                    obs_lines.append(f"--- {prefix}{tool_label}(Type: {obs.obs_type.value}) ---")
                    obs_lines.append(obs.content)
                sections["observations"] = "\n".join(obs_lines)
            else:
                sections.pop("observations", None)

            assembled_parts = [sections[k] for k in ordered_keys if k in sections]
            prompt = "\n\n".join(assembled_parts)

        return prompt

    def _select_observations(self, budget_tokens: int) -> Tuple[List[Observation], int]:
        """
        Select which observations to retain within budget_tokens.
        Strategy:
          - Retain highest priority observations.
          - Within the same priority, retain newest observations.
          - Return retained observations in original chronological order.
        """
        if not self._observations or budget_tokens <= 0:
            return [], len(self._observations)

        # Candidate indexing: list of (index, Observation, estimated_tokens)
        candidates = [
            (i, obs, self.estimate(f"--- Step {obs.step} --- \n{obs.content}"))
            for i, obs in enumerate(self._observations)
        ]

        # Sort candidates for eviction: lowest priority first, then oldest timestamp/index first
        # Items sorted at the end are most protected
        protected_order = sorted(
            candidates,
            key=lambda x: (x[1].priority, x[0]),  # higher priority and newer index at end
            reverse=True                          # most desired first
        )

        retained_indices = set()
        consumed_tokens = 0

        for idx, obs, tokens in protected_order:
            if consumed_tokens + tokens <= budget_tokens:
                retained_indices.add(idx)
                consumed_tokens += tokens

        evicted_count = len(self._observations) - len(retained_indices)
        # Restore chronological order
        retained_obs = [self._observations[i] for i in sorted(retained_indices)]

        return retained_obs, evicted_count

    def get_budget_report(self) -> BudgetReport:
        """
        Generate detailed budget accounting and truncation diagnostics.
        """
        available_input = self.context_budget - self.reserved_output

        sys_tokens = self.estimate(self._system_instructions)
        proj_tokens = self.estimate(self._build_project_header())
        obj_tokens = self.estimate(self._objective)
        spec_tokens = self.estimate(self._task_specification)
        repo_tokens = self.estimate(self._repo_map)

        # Build prompt to verify actual allocation and eviction
        prompt = self.build_prompt()
        used_tokens = self.estimate(prompt)
        remaining = max(0, available_input - used_tokens)

        # Count evicted observations
        dynamic_budget = max(0, available_input - (sys_tokens + proj_tokens + obj_tokens + spec_tokens))
        retained_obs, evicted_count = self._select_observations(dynamic_budget)
        obs_tokens = sum(self.estimate(o.content) for o in retained_obs)
        worker_tokens = sum(self.estimate(w.summary) for w in self._worker_results)

        truncation_sections = []
        if evicted_count > 0:
            truncation_sections.append("observations")
        if self._repo_map and repo_tokens > self.max_repo_map_tokens:
            truncation_sections.append("repo_map")

        truncation = TruncationMetadata(
            was_truncated=(len(truncation_sections) > 0),
            evicted_observation_count=evicted_count,
            retained_observation_count=len(retained_obs),
            truncated_sections=truncation_sections,
            notes=[
                f"Chars per token heuristic: {self.chars_per_token}",
                f"Reserved output: {self.reserved_output} tokens",
            ],
        )

        return BudgetReport(
            total_budget_tokens=self.context_budget,
            reserved_output_tokens=self.reserved_output,
            available_input_tokens=available_input,
            estimated_used_tokens=used_tokens,
            remaining_tokens=remaining,
            chars_per_token=self.chars_per_token,
            is_estimate=True,
            estimation_method="character_heuristic",
            system_tokens=sys_tokens,
            project_tokens=proj_tokens,
            repo_map_tokens=repo_tokens,
            objective_tokens=obj_tokens,
            task_spec_tokens=spec_tokens,
            observations_tokens=obs_tokens,
            worker_results_tokens=worker_tokens,
            total_observations=len(self._observations),
            retained_observations=len(retained_obs),
            evicted_observations=evicted_count,
            truncation=truncation,
        )


# ---------------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------------

class ContextManager:
    """
    Manages isolated ContextSessions across workspaces, tasks, and agents.
    Provides session creation, retrieval, listing, and lifecycle management.
    """

    def __init__(
        self,
        default_budget: int = DEFAULT_CONTEXT_WINDOW,
        default_reserved_output: int = DEFAULT_RESERVED_OUTPUT,
        chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    ):
        self.default_budget = default_budget
        self.default_reserved_output = default_reserved_output
        self.chars_per_token = chars_per_token
        self._sessions: Dict[str, ContextSession] = {}

    def create_session(
        self,
        session_id: str,
        project_id: Optional[str] = None,
        role: SessionRole = SessionRole.SOLO,
        context_budget: Optional[int] = None,
        reserved_output: Optional[int] = None,
    ) -> ContextSession:
        """
        Create and register a new isolated ContextSession.
        Raises ValueError if session_id is empty or duplicate.
        """
        sid = str(session_id).strip()
        if not sid:
            raise ValueError("session_id cannot be empty.")
        if sid in self._sessions:
            raise ValueError(f"Session '{sid}' already exists.")

        session = ContextSession(
            session_id=sid,
            project_id=project_id,
            role=role,
            context_budget=context_budget or self.default_budget,
            reserved_output=reserved_output or self.default_reserved_output,
            chars_per_token=self.chars_per_token,
        )
        self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Optional[ContextSession]:
        """Get an existing session by ID or return None."""
        return self._sessions.get(str(session_id).strip())

    def get_or_create_session(
        self,
        session_id: str,
        project_id: Optional[str] = None,
        role: SessionRole = SessionRole.SOLO,
    ) -> ContextSession:
        """Return existing session or create a new one."""
        sid = str(session_id).strip()
        if sid in self._sessions:
            return self._sessions[sid]
        return self.create_session(sid, project_id=project_id, role=role)

    def delete_session(self, session_id: str) -> bool:
        """Remove a session from the manager."""
        return bool(self._sessions.pop(str(session_id).strip(), None))

    def list_sessions(self) -> List[str]:
        """List all registered session IDs."""
        return sorted(self._sessions.keys())

    def get_sessions_for_project(self, project_id: str) -> List[ContextSession]:
        """Return all sessions associated with a specific project ID."""
        pid = str(project_id).strip()
        return [s for s in self._sessions.values() if s.project_id == pid]

    def clear(self) -> None:
        """Clear all registered sessions."""
        self._sessions.clear()
