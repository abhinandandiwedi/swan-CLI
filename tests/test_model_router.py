"""
SWAN-ENGINE v0.2 — Model Router & Ollama Discovery Tests
=========================================================

Comprehensive test suite for core/model_router.py:
- Ollama runtime discovery via /api/tags and /api/ps
- Empty model lists and malformed JSON responses
- Connection failures, connection refused, and timeout handling
- Loaded vs installed model status tracking
- Role-based routing across coder, tester, debugger, doc, analysis, manager
- Explicit model selection and missing model rejection
- Fallback chain traversal and exhausted chain failure
- Prefer-loaded model optimization
- Model identifier and tag normalization
- Prohibition of automatic model downloads
- Configuration validation and offline operation
"""

import json
import socket
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_router import (
    ModelRouter,
    ModelInfo,
    DiscoveryResult,
    ModelRouterError,
    OllamaConnectionError,
    ModelNotFoundError,
    ConfigurationError,
    TaskRole,
    split_model_tag,
    matches_model_identifier,
)


class TestModelIdentifierMatching(unittest.TestCase):

    def test_split_model_tag(self):
        self.assertEqual(split_model_tag("swan-coder"), ("swan-coder", "latest"))
        self.assertEqual(split_model_tag("swan-coder:latest"), ("swan-coder", "latest"))
        self.assertEqual(split_model_tag("qwen2.5-coder:7b"), ("qwen2.5-coder", "7b"))

    def test_matches_model_identifier_without_tag(self):
        # When request omits tag, match any tag with same base name
        self.assertTrue(matches_model_identifier("swan-coder", "swan-coder:latest"))
        self.assertTrue(matches_model_identifier("swan-coder", "swan-coder"))
        self.assertTrue(matches_model_identifier("swan-coder", "swan-coder:v2"))
        self.assertFalse(matches_model_identifier("swan-coder", "qwen2.5-coder:7b"))

    def test_matches_model_identifier_with_tag(self):
        # When request specifies tag, must match exact tag
        self.assertTrue(matches_model_identifier("qwen2.5-coder:7b", "qwen2.5-coder:7b"))
        self.assertFalse(matches_model_identifier("qwen2.5-coder:7b", "qwen2.5-coder:14b"))
        self.assertFalse(matches_model_identifier("swan-coder:v1", "swan-coder:v2"))


class TestOllamaDiscovery(unittest.TestCase):

    def setUp(self):
        self.router = ModelRouter(endpoint="http://localhost:11434", timeout=2.0)

    @mock.patch.object(ModelRouter, "_http_get_json")
    def test_discovery_success(self, mock_http):
        # Mock /api/tags and /api/ps
        tags_data = {
            "models": [
                {
                    "name": "qwen2.5-coder:7b",
                    "model": "qwen2.5-coder:7b",
                    "size": 4683075276,
                    "digest": "2b04965141f0",
                    "details": {
                        "family": "qwen2",
                        "parameter_size": "7.6B",
                        "quantization_level": "Q4_K_M"
                    }
                },
                {
                    "name": "swan-coder:latest",
                    "model": "swan-coder:latest",
                    "size": 4683075276,
                    "digest": "1a2b3c4d5e",
                    "details": {
                        "family": "qwen2",
                        "parameter_size": "7.6B",
                        "quantization_level": "Q4_K_M"
                    }
                }
            ]
        }
        ps_data = {
            "models": [
                {
                    "name": "qwen2.5-coder:7b",
                    "size_vram": 4683075276
                }
            ]
        }

        mock_http.side_effect = [tags_data, ps_data]

        result = self.router.discover_models()

        self.assertTrue(result.success)
        self.assertTrue(result.ollama_reachable)
        self.assertEqual(len(result.installed_models), 2)
        self.assertEqual(len(result.loaded_models), 1)

        # qwen2.5-coder:7b should be loaded
        qwen = self.router.find_model("qwen2.5-coder:7b")
        self.assertIsNotNone(qwen)
        self.assertTrue(qwen.is_installed)
        self.assertTrue(qwen.is_loaded)
        self.assertEqual(qwen.family, "qwen2")
        self.assertEqual(qwen.parameter_size, "7.6B")
        self.assertEqual(qwen.quantization, "Q4_K_M")

        # swan-coder should be installed but not loaded
        swan = self.router.find_model("swan-coder")
        self.assertIsNotNone(swan)
        self.assertTrue(swan.is_installed)
        self.assertFalse(swan.is_loaded)

    @mock.patch.object(ModelRouter, "_http_get_json")
    def test_empty_model_list(self, mock_http):
        mock_http.side_effect = [{"models": []}, {"models": []}]

        result = self.router.discover_models()

        self.assertTrue(result.success)
        self.assertTrue(result.ollama_reachable)
        self.assertEqual(len(result.installed_models), 0)
        self.assertEqual(len(self.router.list_models()), 0)

    @mock.patch.object(ModelRouter, "_http_get_json")
    def test_malformed_response_missing_models_key(self, mock_http):
        mock_http.return_value = {"error": "unexpected payload"}

        result = self.router.discover_models()

        self.assertFalse(result.success)
        self.assertTrue(result.ollama_reachable)
        self.assertIn("missing 'models' list", result.error_message)

    @mock.patch.object(ModelRouter, "_http_get_json")
    def test_malformed_response_models_not_list(self, mock_http):
        mock_http.return_value = {"models": "not-a-list"}

        result = self.router.discover_models()

        self.assertFalse(result.success)
        self.assertIn("missing 'models' list", result.error_message)

    @mock.patch.object(ModelRouter, "_http_get_json")
    def test_connection_refused(self, mock_http):
        mock_http.side_effect = OllamaConnectionError("Failed to connect to Ollama: Connection refused")

        result = self.router.discover_models()

        self.assertFalse(result.success)
        self.assertFalse(result.ollama_reachable)
        self.assertIn("Connection refused", result.error_message)

    @mock.patch.object(ModelRouter, "_http_get_json")
    def test_connection_timeout(self, mock_http):
        mock_http.side_effect = OllamaConnectionError("Connection to Ollama timed out after 2.0s")

        result = self.router.discover_models()

        self.assertFalse(result.success)
        self.assertFalse(result.ollama_reachable)
        self.assertIn("timed out", result.error_message)


class TestModelSelection(unittest.TestCase):

    def setUp(self):
        self.router = ModelRouter()
        # Pre-populate registry with mock models
        self.m_swan = ModelInfo(
            name="swan-coder:latest",
            base_name="swan-coder",
            tag="latest",
            family="qwen2",
            parameter_size="7B",
            is_installed=True,
            is_loaded=False,
            context_window=16384,
            capabilities={"code", "tools", "json"},
            supported_roles={"coder", "tester", "debugger"},
        )
        self.m_qwen = ModelInfo(
            name="qwen2.5-coder:7b",
            base_name="qwen2.5-coder",
            tag="7b",
            family="qwen2",
            parameter_size="7B",
            is_installed=True,
            is_loaded=True,
            context_window=16384,
            capabilities={"code", "tools", "json"},
            supported_roles={"coder", "documentation", "analysis"},
        )
        self.m_llama = ModelInfo(
            name="llama3.1:8b",
            base_name="llama3.1",
            tag="8b",
            family="llama",
            parameter_size="8B",
            is_installed=True,
            is_loaded=False,
            context_window=8192,
            capabilities={"general", "json"},
            supported_roles={"documentation", "analysis", "manager"},
        )
        self.router.register_model(self.m_swan)
        self.router.register_model(self.m_qwen)
        self.router.register_model(self.m_llama)

    def test_explicit_selection_success(self):
        model = self.router.select_explicit_model("swan-coder")
        self.assertEqual(model.name, "swan-coder:latest")

        model_tagged = self.router.select_explicit_model("qwen2.5-coder:7b")
        self.assertEqual(model_tagged.name, "qwen2.5-coder:7b")

    def test_explicit_selection_missing_raises_model_not_found(self):
        with self.assertRaises(ModelNotFoundError) as ctx:
            self.router.select_explicit_model("deepseek-coder:33b")
        self.assertIn("not installed in Ollama", str(ctx.exception))
        self.assertIn("ollama pull", str(ctx.exception))

    def test_explicit_selection_empty_raises(self):
        with self.assertRaises(ModelNotFoundError):
            self.router.select_explicit_model("")

    def test_role_based_routing_coder(self):
        # Default chain: ["swan-coder", "qwen2.5-coder:7b", "codellama", "deepseek-coder"]
        # With prefer_loaded=False, picks primary "swan-coder"
        model = self.router.select_model_for_role(TaskRole.CODER.value, prefer_loaded=False)
        self.assertEqual(model.base_name, "swan-coder")

    def test_role_based_routing_prefer_loaded(self):
        # "qwen2.5-coder:7b" is loaded in memory.
        # With prefer_loaded=True, picks qwen2.5-coder:7b even though swan-coder is earlier in chain
        model = self.router.select_model_for_role(TaskRole.CODER.value, prefer_loaded=True)
        self.assertEqual(model.name, "qwen2.5-coder:7b")

    def test_fallback_chain_traversal(self):
        # Remove swan-coder from registry
        self.router.clear_registry()
        # Only register second candidate
        self.router.register_model(self.m_qwen)

        model = self.router.select_model_for_role(TaskRole.CODER.value, prefer_loaded=False)
        self.assertEqual(model.name, "qwen2.5-coder:7b")

    def test_fallback_chain_exhausted_raises_model_not_found(self):
        self.router.clear_registry()
        # Only register a model NOT in coder chain
        unrelated = ModelInfo(name="phi3:mini", base_name="phi3", tag="mini", is_installed=True)
        self.router.register_model(unrelated)

        with self.assertRaises(ModelNotFoundError) as ctx:
            self.router.select_model_for_role(TaskRole.CODER.value)
        self.assertIn("No available model found for role 'coder'", str(ctx.exception))
        self.assertIn("Attempted fallback chain", str(ctx.exception))

    def test_unknown_role_raises_configuration_error(self):
        with self.assertRaises(ConfigurationError) as ctx:
            self.router.select_model_for_role("quantum_physicist")
        self.assertIn("is not configured", str(ctx.exception))

    def test_custom_role_mapping(self):
        self.router.set_role_chain("security_auditor", ["llama3.1:8b", "swan-coder"])
        self.assertIn("security_auditor", self.router.get_supported_roles())

        model = self.router.select_model_for_role("security_auditor", prefer_loaded=False)
        self.assertEqual(model.name, "llama3.1:8b")

    def test_invalid_role_configuration_raises(self):
        with self.assertRaises(ConfigurationError):
            self.router.set_role_chain("bad_role", [])
        with self.assertRaises(ConfigurationError):
            self.router.set_role_chain("", ["swan-coder"])


class TestProhibitedActionsAndSafety(unittest.TestCase):

    def setUp(self):
        self.router = ModelRouter()

    def test_automatic_downloads_prohibited(self):
        """Router must explicitly reject pulling/downloading models automatically."""
        with self.assertRaises(ModelRouterError) as ctx:
            self.router.pull_model("qwen2.5-coder:32b")
        self.assertIn("Automatic downloading is prohibited", str(ctx.exception))
        self.assertIn("ollama pull", str(ctx.exception))


class TestModelRegistryAndContext(unittest.TestCase):

    def setUp(self):
        self.router = ModelRouter(default_context_window=16384)

    def test_context_window_resolution(self):
        m1 = ModelInfo(name="qwen2.5:14b", base_name="qwen2.5", tag="14b", context_window=32768, is_installed=True)
        m2 = ModelInfo(name="unknown-model:latest", base_name="unknown-model", tag="latest", context_window=16384, is_installed=True)

        self.router.register_model(m1)
        self.router.register_model(m2)

        self.assertEqual(self.router.select_explicit_model("qwen2.5:14b").context_window, 32768)
        self.assertEqual(self.router.select_explicit_model("unknown-model").context_window, 16384)

    def test_list_models_filters(self):
        m_inst = ModelInfo(name="m1", base_name="m1", tag="latest", is_installed=True, is_loaded=False)
        m_load = ModelInfo(name="m2", base_name="m2", tag="latest", is_installed=True, is_loaded=True)
        m_uninst = ModelInfo(name="m3", base_name="m3", tag="latest", is_installed=False, is_loaded=False)

        self.router.register_model(m_inst)
        self.router.register_model(m_load)
        self.router.register_model(m_uninst)

        installed = self.router.list_models(installed_only=True)
        self.assertEqual(len(installed), 2)

        loaded = self.router.list_models(loaded_only=True)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].name, "m2")


if __name__ == "__main__":
    unittest.main(verbosity=2)

