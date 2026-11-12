"""
SWAN-ENGINE v0.2 — Model Router & Ollama Discovery
===================================================

Phase 3: Multi-model discovery and role-based routing for SWAN-HIVE.

Features:
- Ollama runtime discovery via /api/tags and /api/ps
- Resilient error handling (connection failures, timeouts, malformed JSON)
- Zero automatic downloads (prohibits auto-pulling)
- Model registry with family, parameter size, quantization, context window, and capabilities
- Role-based model selection (coder, tester, debugger, documentation, analysis, manager)
- Explicit model selection with strict availability validation (no silent substitutions)
- Configurable fallback chains per role
- Interfaces for future HIVE Task Orchestrator and Context Manager integration
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Roles & Constants
# ---------------------------------------------------------------------------

class TaskRole(str, Enum):
    CODER = "coder"
    TESTER = "tester"
    DEBUGGER = "debugger"
    DOCUMENTATION = "documentation"
    ANALYSIS = "analysis"
    MANAGER = "manager"


SUPPORTED_ROLES: Set[str] = {
    TaskRole.CODER.value,
    TaskRole.TESTER.value,
    TaskRole.DEBUGGER.value,
    TaskRole.DOCUMENTATION.value,
    TaskRole.ANALYSIS.value,
    TaskRole.MANAGER.value,
}

# Default fallback chains for roles
DEFAULT_ROLE_CHAINS: Dict[str, List[str]] = {
    TaskRole.CODER.value: ["swan-coder", "qwen2.5-coder:7b", "codellama", "deepseek-coder"],
    TaskRole.TESTER.value: ["swan-coder", "qwen2.5-coder:7b", "llama3.2", "codellama"],
    TaskRole.DEBUGGER.value: ["swan-coder", "qwen2.5-coder:7b", "llama3.1:8b"],
    TaskRole.DOCUMENTATION.value: ["qwen2.5-coder:7b", "swan-coder", "llama3.2:3b", "llama3.1:8b"],
    TaskRole.ANALYSIS.value: ["qwen2.5-coder:7b", "swan-coder", "llama3.1:8b"],
    TaskRole.MANAGER.value: ["swan-coder", "llama3.1:8b", "qwen2.5:14b", "qwen2.5-coder:7b"],
}

# Known context window configurations (tokens)
KNOWN_MODEL_CONTEXTS: Dict[str, int] = {
    "swan-coder": 16384,
    "qwen2.5-coder:7b": 16384,
    "qwen2.5-coder": 16384,
    "qwen2.5:14b": 32768,
    "llama3.1:8b": 8192,
    "llama3.2:3b": 8192,
    "codellama": 4096,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ModelRouterError(Exception):
    """Base exception for model router errors."""
    pass


class OllamaConnectionError(ModelRouterError):
    """Raised when Ollama is unreachable, timed out, or connection refused."""
    pass


class ModelNotFoundError(ModelRouterError):
    """Raised when an explicit model or all models in a fallback chain are unavailable."""
    pass


class ConfigurationError(ModelRouterError):
    """Raised when configuration or role mappings are invalid."""
    pass


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Rich metadata for an installed or discovered model."""
    name: str                           # Full identifier, e.g. "swan-coder:latest"
    base_name: str                      # Tag-stripped name, e.g. "swan-coder"
    tag: str                            # Tag portion, e.g. "latest" or "7b"
    family: str = "unknown"             # Model family (e.g. "qwen2", "llama")
    parameter_size: Optional[str] = None # e.g. "7.6B"
    quantization: Optional[str] = None  # e.g. "Q4_K_M"
    size_bytes: int = 0
    is_installed: bool = False
    is_loaded: bool = False             # True if model is currently active in memory (/api/ps)
    context_window: int = 16384         # Configured or known context window
    capabilities: Set[str] = field(default_factory=set) # e.g. {"code", "tools", "json"}
    supported_roles: Set[str] = field(default_factory=set) # e.g. {"coder", "tester"}
    modified_at: Optional[str] = None
    digest: Optional[str] = None


@dataclass
class DiscoveryResult:
    """Outcome of an Ollama endpoint discovery query."""
    success: bool
    ollama_reachable: bool
    installed_models: List[ModelInfo] = field(default_factory=list)
    loaded_models: List[ModelInfo] = field(default_factory=list)
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Name Normalization Helper
# ---------------------------------------------------------------------------

def split_model_tag(model_name: str) -> Tuple[str, str]:
    """
    Split a model name into (base_name, tag).
    If no tag is present, tag defaults to 'latest'.
    """
    cleaned = str(model_name).strip()
    if ":" in cleaned:
        base, tag = cleaned.split(":", 1)
        return base, tag or "latest"
    return cleaned, "latest"


def matches_model_identifier(requested: str, candidate: str) -> bool:
    """
    Determine if a candidate model satisfies a requested identifier.
    e.g.:
      requested 'swan-coder' matches candidate 'swan-coder:latest' and 'swan-coder'
      requested 'qwen2.5-coder:7b' matches candidate 'qwen2.5-coder:7b'
      requested 'qwen2.5-coder:7b' does NOT match candidate 'qwen2.5-coder:14b'
    """
    req_clean = requested.strip().lower()
    cand_clean = candidate.strip().lower()

    if req_clean == cand_clean:
        return True

    req_base, req_tag = split_model_tag(req_clean)
    cand_base, cand_tag = split_model_tag(cand_clean)

    if ":" not in req_clean:
        # User requested without explicit tag: match any tag or latest
        return req_base == cand_base
    else:
        # User requested with explicit tag: must match both
        return (req_base == cand_base) and (req_tag == cand_tag)


# ---------------------------------------------------------------------------
# Model Router
# ---------------------------------------------------------------------------

class ModelRouter:
    """
    Discovers installed Ollama models and routes tasks to appropriate models.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        timeout: float = 5.0,
        default_context_window: int = 16384,
        role_chains: Optional[Dict[str, List[str]]] = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = max(0.5, float(timeout))
        self.default_context_window = max(512, int(default_context_window))

        # Role mappings
        self._role_chains: Dict[str, List[str]] = dict(DEFAULT_ROLE_CHAINS)
        if role_chains:
            for role, chain in role_chains.items():
                self.set_role_chain(role, chain)

        # In-memory registry: key = canonical full model name
        self._registry: Dict[str, ModelInfo] = {}
        self._last_discovery: Optional[DiscoveryResult] = None

    # ------ Configuration APIs ------

    def set_role_chain(self, role: str, chain: List[str]) -> None:
        """Configure the fallback chain for a role."""
        role_clean = str(role).strip().lower()
        if not role_clean:
            raise ConfigurationError("Role name cannot be empty.")
        if not chain or not isinstance(chain, list):
            raise ConfigurationError(f"Fallback chain for role '{role}' must be a non-empty list of model names.")

        cleaned_chain = [str(m).strip() for m in chain if str(m).strip()]
        if not cleaned_chain:
            raise ConfigurationError(f"Fallback chain for role '{role}' contains no valid model names.")

        self._role_chains[role_clean] = cleaned_chain

    def get_role_chain(self, role: str) -> List[str]:
        """Get the configured fallback chain for a role."""
        role_clean = str(role).strip().lower()
        if role_clean not in self._role_chains:
            raise ConfigurationError(
                f"Role '{role}' is not configured. Supported roles: {sorted(self._role_chains.keys())}"
            )
        return list(self._role_chains[role_clean])

    def get_supported_roles(self) -> List[str]:
        """Return all currently configured task roles."""
        return sorted(self._role_chains.keys())

    # ------ Ollama Discovery API ------

    def _http_get_json(self, path: str) -> Any:
        """Execute HTTP GET to Ollama endpoint and parse JSON response."""
        url = f"{self.endpoint}{path}"
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "SWAN-ENGINE/0.2"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
                return json.loads(data.decode("utf-8"))
        except urllib.error.URLError as e:
            if isinstance(e.reason, socket.timeout) or isinstance(e, TimeoutError):
                raise OllamaConnectionError(f"Connection to Ollama timed out after {self.timeout}s: {url}")
            raise OllamaConnectionError(f"Failed to connect to Ollama at {url}: {e.reason}")
        except TimeoutError:
            raise OllamaConnectionError(f"Connection to Ollama timed out after {self.timeout}s: {url}")
        except json.JSONDecodeError as e:
            raise ModelRouterError(f"Malformed JSON response from Ollama at {url}: {e}")

    def discover_models(self) -> DiscoveryResult:
        """
        Discover installed models (/api/tags) and currently loaded models (/api/ps).
        Updates internal model registry.

        Never assumes models are installed.
        Never downloads or pulls models automatically.
        """
        # 1. Query installed models (/api/tags)
        try:
            tags_resp = self._http_get_json("/api/tags")
        except OllamaConnectionError as e:
            result = DiscoveryResult(
                success=False,
                ollama_reachable=False,
                error_message=str(e),
            )
            self._last_discovery = result
            return result
        except ModelRouterError as e:
            result = DiscoveryResult(
                success=False,
                ollama_reachable=True,
                error_message=str(e),
            )
            self._last_discovery = result
            return result

        if not isinstance(tags_resp, dict) or "models" not in tags_resp or not isinstance(tags_resp["models"], list):
            result = DiscoveryResult(
                success=False,
                ollama_reachable=True,
                error_message="Malformed response from /api/tags: missing 'models' list.",
            )
            self._last_discovery = result
            return result

        # 2. Query currently loaded models (/api/ps)
        loaded_names: Set[str] = set()
        try:
            ps_resp = self._http_get_json("/api/ps")
            if isinstance(ps_resp, dict) and isinstance(ps_resp.get("models"), list):
                for item in ps_resp["models"]:
                    if isinstance(item, dict) and "name" in item:
                        loaded_names.add(item["name"].strip().lower())
                        loaded_names.add(split_model_tag(item["name"])[0].lower())
        except Exception:
            # Failure of /api/ps is non-fatal: models are still installed, just not loaded
            pass

        # 3. Parse installed models into registry
        new_registry: Dict[str, ModelInfo] = {}
        installed_list: List[ModelInfo] = []
        loaded_list: List[ModelInfo] = []

        for item in tags_resp["models"]:
            if not isinstance(item, dict) or "name" not in item:
                continue

            full_name = item["name"].strip()
            base_name, tag = split_model_tag(full_name)
            details = item.get("details", {}) if isinstance(item.get("details"), dict) else {}

            family = details.get("family") or "unknown"
            param_size = details.get("parameter_size")
            quant = details.get("quantization_level")
            size_bytes = int(item.get("size", 0))

            # Determine loaded status
            is_loaded = (full_name.lower() in loaded_names) or (base_name.lower() in loaded_names)

            # Determine context window
            ctx = (
                KNOWN_MODEL_CONTEXTS.get(full_name)
                or KNOWN_MODEL_CONTEXTS.get(base_name)
                or self.default_context_window
            )

            # Determine capabilities
            caps: Set[str] = {"text"}
            roles: Set[str] = set()

            name_lower = full_name.lower()
            if "coder" in name_lower or "code" in name_lower:
                caps.update({"code", "tools", "json"})
                roles.update({TaskRole.CODER.value, TaskRole.TESTER.value, TaskRole.DEBUGGER.value})

            if "swan" in name_lower:
                caps.update({"code", "tools", "json", "ast"})
                roles.update(SUPPORTED_ROLES)

            if "llama" in name_lower or "qwen" in name_lower:
                caps.update({"json", "general"})
                roles.update({TaskRole.DOCUMENTATION.value, TaskRole.ANALYSIS.value})

            info = ModelInfo(
                name=full_name,
                base_name=base_name,
                tag=tag,
                family=family,
                parameter_size=param_size,
                quantization=quant,
                size_bytes=size_bytes,
                is_installed=True,
                is_loaded=is_loaded,
                context_window=ctx,
                capabilities=caps,
                supported_roles=roles,
                modified_at=item.get("modified_at"),
                digest=item.get("digest"),
            )

            new_registry[full_name] = info
            installed_list.append(info)
            if is_loaded:
                loaded_list.append(info)

        self._registry = new_registry
        result = DiscoveryResult(
            success=True,
            ollama_reachable=True,
            installed_models=installed_list,
            loaded_models=loaded_list,
        )
        self._last_discovery = result
        return result

    # ------ Manual Registry Registration (Testing & Offline Operation) ------

    def register_model(self, model: ModelInfo) -> None:
        """Register a model directly into the registry."""
        if not model or not model.name:
            raise ValueError("Invalid ModelInfo object.")
        self._registry[model.name] = model

    def clear_registry(self) -> None:
        """Clear all registered models."""
        self._registry.clear()

    # ------ Model Query & Retrieval APIs ------

    def list_models(self, installed_only: bool = True, loaded_only: bool = False) -> List[ModelInfo]:
        """List registered models according to filters."""
        models = list(self._registry.values())
        if installed_only:
            models = [m for m in models if m.is_installed]
        if loaded_only:
            models = [m for m in models if m.is_loaded]
        return sorted(models, key=lambda m: m.name)

    def is_model_available(self, model_name: str) -> bool:
        """Check whether a model matching model_name is installed and registered."""
        return self.find_model(model_name) is not None

    def find_model(self, model_name: str) -> Optional[ModelInfo]:
        """Find a model by name or base name."""
        target = str(model_name).strip()
        if not target:
            return None

        # 1. Exact match
        if target in self._registry:
            return self._registry[target]

        # 2. Case-insensitive exact match
        for name, info in self._registry.items():
            if name.lower() == target.lower():
                return info

        # 3. Matching base name / tag normalization
        for name, info in self._registry.items():
            if matches_model_identifier(target, name):
                return info

        return None

    # ------ Selection Strategy APIs ------

    def select_explicit_model(self, model_name: str) -> ModelInfo:
        """
        Explicitly select a specific model.

        Raises ModelNotFoundError if the model is not installed.
        Never silently substitutes with an alternative model.
        """
        target = str(model_name).strip()
        if not target:
            raise ModelNotFoundError("Model name cannot be empty.")

        model = self.find_model(target)
        if model is None or not model.is_installed:
            installed = [m.name for m in self.list_models(installed_only=True)]
            raise ModelNotFoundError(
                f"Model '{target}' is not installed in Ollama.\n"
                f"  Installed models: {installed or '(none)'}\n"
                f"  Resolution: Run 'ollama pull {target}' to install it."
            )
        return model

    def select_model_for_role(self, role: str, prefer_loaded: bool = True) -> ModelInfo:
        """
        Select an appropriate model for a task role using the configured fallback chain.

        Parameters
        ----------
        role : str
            Role name (e.g. 'coder', 'tester', 'debugger', 'manager')
        prefer_loaded : bool
            If True, prioritizes an installed model that is already resident in VRAM.

        Returns
        -------
        ModelInfo
            The selected model.

        Raises
        ------
        ConfigurationError
            If the role is not configured.
        ModelNotFoundError
            If none of the candidates in the role's fallback chain are installed.
        """
        role_clean = str(role).strip().lower()
        chain = self.get_role_chain(role_clean)

        # Gather installed candidate matches in chain order
        candidates: List[ModelInfo] = []
        for candidate_name in chain:
            model = self.find_model(candidate_name)
            if model is not None and model.is_installed:
                candidates.append(model)

        if not candidates:
            installed = [m.name for m in self.list_models(installed_only=True)]
            raise ModelNotFoundError(
                f"No available model found for role '{role_clean}'.\n"
                f"  Attempted fallback chain: {chain}\n"
                f"  Installed models available: {installed or '(none)'}\n"
                f"  Resolution: Run 'ollama pull {chain[0]}' to install the primary model for this role."
            )

        # If prefer_loaded is requested, pick the first candidate that is already in VRAM
        if prefer_loaded:
            for cand in candidates:
                if cand.is_loaded:
                    return cand

        # Otherwise pick the highest-priority installed candidate in the chain
        return candidates[0]

    # ------ Prohibited Actions Safeguard ------

    def pull_model(self, model_name: str) -> None:
        """
        Prohibits automatic downloading of models.
        """
        raise ModelRouterError(
            f"🚫 Automatic downloading is prohibited.\n"
            f"  Model '{model_name}' must be pulled manually by the user:\n"
            f"    ollama pull {model_name}"
        )

