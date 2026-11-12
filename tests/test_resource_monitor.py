"""
SWAN-ENGINE v0.2 — Hardware Resource Monitor & Admission Controller Tests
==========================================================================

Comprehensive test suite covering:
  - Linux /proc/meminfo parsing, missing files, malformed files, and MemAvailable fallback
  - AMD GPU sysfs discovery under /sys/class/drm/*/device
  - Multi-GPU scenarios (e.g. Intel integrated + AMD discrete)
  - Missing or non-numeric sysfs VRAM telemetry
  - Strict telemetry reliability classification (KNOWN, UNKNOWN, UNAVAILABLE, ERROR)
  - Integration with Ollama ModelRouter for loaded-model residency
  - Resource profiles: TINY, STANDARD, LARGE, and custom profiles
  - Admission controller decisions:
      * ADMITTED when VRAM has sufficient headroom
      * WAIT when active inferences hit concurrency limits
      * WAIT when VRAM is temporarily full
      * REJECT when context exceeds profile limit
      * REJECT when total required memory exceeds physical capacity
      * Model residency benefit: weights already loaded in VRAM
      * Verification: never admits merely because model weight fits without KV cache
      * Context size scaling effects on KV cache
      * Telemetry unavailable: conservative admission behavior
      * CPU-only fallback: safe admission using system RAM
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.resource_monitor import (
    ResourceMonitor,
    ResourceProfile,
    HardwareSnapshot,
    RAMTelemetry,
    GPUTelemetry,
    Metric,
    TelemetryStatus,
    ProfileName,
    AdmissionStatus,
    AdmissionDecision,
    read_proc_meminfo,
    discover_amd_gpus,
    estimate_kv_cache_bytes,
    estimate_model_weight_bytes,
    PRESET_PROFILES,
    AMD_VENDOR_ID,
)


class TestProcMeminfoParsing(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="swan_meminfo_test_")
        self.meminfo_file = Path(self.tmpdir) / "meminfo"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_valid_meminfo_parsing(self):
        content = """MemTotal:       16384000 kB
MemFree:         4096000 kB
MemAvailable:    8192000 kB
Buffers:          100000 kB
Cached:          1000000 kB
"""
        self.meminfo_file.write_text(content, encoding="utf-8")
        ram = read_proc_meminfo(str(self.meminfo_file))

        self.assertTrue(ram.is_reliable)
        self.assertEqual(ram.total.value, 16384000)
        self.assertEqual(ram.total.unit, "kB")
        self.assertEqual(ram.total.in_bytes(), 16384000 * 1024)
        self.assertEqual(ram.total.in_mb(), 16000.0)

        self.assertEqual(ram.available.value, 8192000)
        self.assertEqual(ram.available.in_bytes(), 8192000 * 1024)
        self.assertEqual(ram.available.in_mb(), 8000.0)

        # Used = Total - Available = 16384000 - 8192000 = 8192000 kB
        self.assertEqual(ram.used.value, 8192000)

    def test_meminfo_missing_file(self):
        ram = read_proc_meminfo(str(Path(self.tmpdir) / "nonexistent"))
        self.assertFalse(ram.is_reliable)
        self.assertEqual(ram.total.status, TelemetryStatus.UNAVAILABLE)
        self.assertIsNone(ram.total.in_bytes())

    def test_meminfo_malformed_missing_total(self):
        self.meminfo_file.write_text("Buffers: 1000 kB\nCached: 2000 kB\n", encoding="utf-8")
        ram = read_proc_meminfo(str(self.meminfo_file))
        self.assertFalse(ram.is_reliable)
        self.assertEqual(ram.total.status, TelemetryStatus.ERROR)

    def test_older_kernel_memavailable_fallback(self):
        content = """MemTotal:       10000000 kB
MemFree:         2000000 kB
Buffers:          500000 kB
Cached:          1500000 kB
"""
        self.meminfo_file.write_text(content, encoding="utf-8")
        ram = read_proc_meminfo(str(self.meminfo_file))

        self.assertTrue(ram.is_reliable)
        # Available fallback = Free (2000000) + Buffers (500000) + Cached (1500000) = 4000000 kB
        self.assertEqual(ram.available.value, 4000000)


class TestAMDGPUDiscovery(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="swan_drm_test_")
        self.drm_dir = Path(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_gpu_sysfs(self, card_name: str, vendor: str, vram_total: int, vram_used: int, is_amd: bool = True):
        dev = self.drm_dir / card_name / "device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text(vendor + "\n")
        (dev / "mem_info_vram_total").write_text(str(vram_total) + "\n")
        (dev / "mem_info_vram_used").write_text(str(vram_used) + "\n")
        (dev / "product_name").write_text("AMD Radeon RX 9060 XT\n" if is_amd else "Intel UHD\n")

    def test_amd_gpu_discovery_success(self):
        # 16 GB total, 2 GB used
        total = 16 * 1024 * 1024 * 1024
        used = 2 * 1024 * 1024 * 1024
        self._create_gpu_sysfs("card1", "0x1002", total, used)

        gpus = discover_amd_gpus(str(self.drm_dir))
        self.assertEqual(len(gpus), 1)

        gpu = gpus[0]
        self.assertEqual(gpu.card_id, "card1")
        self.assertTrue(gpu.is_amd)
        self.assertTrue(gpu.is_reliable)
        self.assertEqual(gpu.vram_total.in_bytes(), total)
        self.assertEqual(gpu.vram_used.in_bytes(), used)
        self.assertEqual(gpu.vram_free.in_bytes(), total - used)
        self.assertEqual(gpu.product_name, "AMD Radeon RX 9060 XT")

    def test_multi_gpu_discovery_filters_non_amd(self):
        # card0 = Intel (0x8086), card1 = AMD (0x1002)
        self._create_gpu_sysfs("card0", "0x8086", 0, 0, is_amd=False)
        self._create_gpu_sysfs("card1", "0x1002", 16 * 1024**3, 4 * 1024**3, is_amd=True)

        gpus = discover_amd_gpus(str(self.drm_dir))
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].card_id, "card1")

    def test_ignores_non_card_connectors(self):
        # Entries like card1-DP-1 or renderD128
        (self.drm_dir / "card1-DP-1").mkdir()
        (self.drm_dir / "renderD128").mkdir()
        (self.drm_dir / "version").write_text("drm 1.0")

        gpus = discover_amd_gpus(str(self.drm_dir))
        self.assertEqual(len(gpus), 0)

    def test_missing_vram_telemetry_file(self):
        dev = self.drm_dir / "card1" / "device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text("0x1002\n")
        # Do not create mem_info_vram_total

        gpus = discover_amd_gpus(str(self.drm_dir))
        self.assertEqual(len(gpus), 1)
        gpu = gpus[0]
        self.assertFalse(gpu.is_reliable)
        self.assertEqual(gpu.vram_total.status, TelemetryStatus.UNAVAILABLE)

    def test_corrupt_non_numeric_vram_file(self):
        dev = self.drm_dir / "card1" / "device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text("0x1002\n")
        (dev / "mem_info_vram_total").write_text("CORRUPTED_NOT_INT\n")

        gpus = discover_amd_gpus(str(self.drm_dir))
        self.assertEqual(len(gpus), 1)
        self.assertFalse(gpus[0].is_reliable)
        self.assertEqual(gpus[0].vram_total.status, TelemetryStatus.ERROR)


class TestMetricsAndConversions(unittest.TestCase):

    def test_metric_conversions(self):
        m_bytes = Metric(1073741824, "bytes", "test", TelemetryStatus.KNOWN)
        self.assertEqual(m_bytes.in_bytes(), 1073741824)
        self.assertEqual(m_bytes.in_mb(), 1024.0)
        self.assertEqual(m_bytes.in_gb(), 1.0)

        m_kb = Metric(1048576, "kB", "test", TelemetryStatus.KNOWN)
        self.assertEqual(m_kb.in_bytes(), 1073741824)
        self.assertEqual(m_kb.in_mb(), 1024.0)
        self.assertEqual(m_kb.in_gb(), 1.0)

    def test_unreliable_metric_returns_none(self):
        m_unavail = Metric(None, "bytes", "test", TelemetryStatus.UNAVAILABLE)
        self.assertIsNone(m_unavail.in_bytes())
        self.assertIsNone(m_unavail.in_mb())
        self.assertIsNone(m_unavail.in_gb())


class TestKVAndWeightEstimates(unittest.TestCase):

    def test_kv_cache_estimate_scaling(self):
        # 16k context vs 32k context: 32k should be exactly double
        kv_16k = estimate_kv_cache_bytes(16384)
        kv_32k = estimate_kv_cache_bytes(32768)

        self.assertGreater(kv_16k, 0)
        self.assertEqual(kv_32k, kv_16k * 2)

    def test_model_weight_estimate(self):
        w_7b = estimate_model_weight_bytes("qwen2.5-coder:7b")
        w_14b = estimate_model_weight_bytes("qwen2.5:14b")
        w_3b = estimate_model_weight_bytes("llama3.2:3b")

        self.assertGreater(w_14b, w_7b)
        self.assertGreater(w_7b, w_3b)


class TestResourceProfiles(unittest.TestCase):

    def test_preset_profiles_exist(self):
        self.assertIn("tiny", PRESET_PROFILES)
        self.assertIn("standard", PRESET_PROFILES)
        self.assertIn("large", PRESET_PROFILES)

        tiny = PRESET_PROFILES["tiny"]
        std = PRESET_PROFILES["standard"]
        large = PRESET_PROFILES["large"]

        self.assertEqual(tiny.max_context_window, 4096)
        self.assertEqual(std.max_context_window, 16384)
        self.assertEqual(large.max_context_window, 32768)
        self.assertFalse(std.allow_concurrency)
        self.assertTrue(large.allow_concurrency)

    def test_set_custom_profile(self):
        monitor = ResourceMonitor()
        custom = ResourceProfile(
            name="custom-low-ram",
            max_active_inferences=1,
            max_context_window=8192,
            vram_reserve_bytes=500 * 1024**2,
        )
        monitor.set_profile(custom)
        self.assertEqual(monitor.profile.name, "custom-low-ram")
        self.assertEqual(monitor.profile.max_context_window, 8192)


class TestAdmissionController(unittest.TestCase):

    def setUp(self):
        self.monitor = ResourceMonitor()

    def _make_snapshot(
        self,
        ram_total_gb: float = 16.0,
        ram_avail_gb: float = 8.0,
        vram_total_gb: float = 16.0,
        vram_used_gb: float = 2.0,
        loaded_models: list = None,
        gpu_present: bool = True,
    ) -> HardwareSnapshot:
        ram = RAMTelemetry(
            total=Metric(ram_total_gb * 1024**3, "bytes", "test", TelemetryStatus.KNOWN),
            free=Metric(ram_avail_gb * 1024**3, "bytes", "test", TelemetryStatus.KNOWN),
            available=Metric(ram_avail_gb * 1024**3, "bytes", "test", TelemetryStatus.KNOWN),
            used=Metric((ram_total_gb - ram_avail_gb) * 1024**3, "bytes", "test", TelemetryStatus.KNOWN),
        )
        gpus = []
        if gpu_present:
            gpus.append(
                GPUTelemetry(
                    card_id="card1",
                    device_path="/sys/class/drm/card1/device",
                    vendor_id="0x1002",
                    is_amd=True,
                    vram_total=Metric(vram_total_gb * 1024**3, "bytes", "test", TelemetryStatus.KNOWN),
                    vram_used=Metric(vram_used_gb * 1024**3, "bytes", "test", TelemetryStatus.KNOWN),
                    vram_free=Metric((vram_total_gb - vram_used_gb) * 1024**3, "bytes", "test", TelemetryStatus.KNOWN),
                )
            )
        return HardwareSnapshot(ram=ram, gpus=gpus, loaded_models=loaded_models or [])

    def test_admitted_when_vram_sufficient(self):
        # 16GB total, 2GB used, 14GB free. 7B model requires ~6.5GB total (weights + kv + buffer)
        snap = self._make_snapshot(vram_total_gb=16.0, vram_used_gb=2.0)
        decision = self.monitor.evaluate_admission("swan-coder", context_tokens=16384, snapshot=snap)

        self.assertTrue(decision.admitted)
        self.assertEqual(decision.status, AdmissionStatus.ADMITTED)
        self.assertEqual(decision.target_device, "gpu")
        self.assertIn("admitted to gpu vram", decision.reason.lower())

    def test_wait_when_concurrency_limit_reached(self):
        snap = self._make_snapshot()
        self.monitor.acquire_inference()  # active = 1 (standard profile limit = 1)

        decision = self.monitor.evaluate_admission("swan-coder", snapshot=snap)

        self.assertFalse(decision.admitted)
        self.assertEqual(decision.status, AdmissionStatus.WAIT)
        self.assertIn("concurrency limit", decision.reason.lower())

        self.monitor.release_inference()  # active = 0

    def test_reject_when_context_exceeds_profile(self):
        snap = self._make_snapshot()
        # Standard profile cap is 16384 tokens
        decision = self.monitor.evaluate_admission("swan-coder", context_tokens=32768, snapshot=snap)

        self.assertFalse(decision.admitted)
        self.assertEqual(decision.status, AdmissionStatus.REJECT)
        self.assertIn("exceeds active profile limit", decision.reason.lower())

    def test_wait_when_vram_temporarily_full(self):
        # 16GB VRAM, but 14.5GB already used -> only 1.5GB free (1GB reserve means 0.5GB usable)
        # 7B model physically fits in 16GB, but VRAM is temporarily full
        snap = self._make_snapshot(vram_total_gb=16.0, vram_used_gb=14.5)
        decision = self.monitor.evaluate_admission("swan-coder", snapshot=snap)

        self.assertFalse(decision.admitted)
        self.assertEqual(decision.status, AdmissionStatus.WAIT)
        self.assertIn("waiting for memory", decision.reason.lower())

    def test_reject_when_model_exceeds_physical_vram_and_ram(self):
        # Massive 70B model on 8GB VRAM and 8GB RAM
        snap = self._make_snapshot(ram_total_gb=8.0, ram_avail_gb=4.0, vram_total_gb=8.0, vram_used_gb=1.0)
        huge_bytes = 40 * 1024**3  # 40 GB

        decision = self.monitor.evaluate_admission(
            "llama3.3:70b",
            size_bytes=huge_bytes,
            snapshot=snap,
        )

        self.assertFalse(decision.admitted)
        self.assertEqual(decision.status, AdmissionStatus.REJECT)
        self.assertIn("exceeds physical vram", decision.reason.lower())

    def test_model_residency_benefit(self):
        """If model is already resident in Ollama VRAM, weight loading cost is 0."""
        # 16GB VRAM, 12GB used (which includes the resident model weights!)
        # Free VRAM is only 4GB.
        # If cold: 4.7GB weights + 1GB KV + 0.7GB buffer = 6.4GB > 3GB usable -> would WAIT
        # But if RESIDENT: weight is 0 -> required is only 1.7GB <= 3GB usable -> ADMITTED!
        snap_cold = self._make_snapshot(vram_total_gb=16.0, vram_used_gb=12.0, loaded_models=[])
        decision_cold = self.monitor.evaluate_admission("swan-coder", snapshot=snap_cold)
        self.assertFalse(decision_cold.admitted)
        self.assertEqual(decision_cold.status, AdmissionStatus.WAIT)

        # Now test resident
        snap_resident = self._make_snapshot(vram_total_gb=16.0, vram_used_gb=12.0, loaded_models=["swan-coder:latest"])
        decision_resident = self.monitor.evaluate_admission("swan-coder", snapshot=snap_resident)
        self.assertTrue(decision_resident.admitted)
        self.assertEqual(decision_resident.status, AdmissionStatus.ADMITTED)
        self.assertTrue(decision_resident.is_model_resident)

    def test_never_admits_merely_because_weights_fit(self):
        """Weights alone might fit, but weights + KV cache + buffers + reserve exceed available."""
        # VRAM free = 5.5 GB. Usable after 1.0GB reserve = 4.5 GB.
        # 7B Model weight = 4.0 GB. (Weight alone: 4.0 GB <= 4.5 GB: FITS!)
        # BUT with 16k KV cache (0.9GB) + runtime buffer (0.75GB) = 5.65 GB total required.
        # 5.65 GB > 4.5 GB usable -> MUST NOT BE ADMITTED!
        snap = self._make_snapshot(vram_total_gb=16.0, vram_used_gb=10.5)  # 5.5 GB free
        decision = self.monitor.evaluate_admission("qwen2.5-coder:7b", size_bytes=int(4.0 * 1024**3), snapshot=snap)

        self.assertFalse(decision.admitted)
        self.assertEqual(decision.status, AdmissionStatus.WAIT)

    def test_cpu_only_fallback(self):
        """When no GPU is present, successfully admits to system RAM if available."""
        snap_cpu = self._make_snapshot(ram_total_gb=32.0, ram_avail_gb=16.0, gpu_present=False)
        decision = self.monitor.evaluate_admission("swan-coder", snapshot=snap_cpu)

        self.assertTrue(decision.admitted)
        self.assertEqual(decision.target_device, "cpu")
        self.assertIn("admitted to system ram", decision.reason.lower())

    def test_telemetry_unavailable_conservative_admission(self):
        """When hardware telemetry is unavailable, admits small models conservatively."""
        err_m = Metric(None, "bytes", "unavailable", TelemetryStatus.UNAVAILABLE)
        bad_ram = RAMTelemetry(err_m, err_m, err_m, err_m)
        snap_bad = HardwareSnapshot(ram=bad_ram, gpus=[], loaded_models=[])

        # Small 3B model, 2k context: admitted conservatively
        decision_small = self.monitor.evaluate_admission("llama3.2:3b", context_tokens=2048, snapshot=snap_bad)
        self.assertTrue(decision_small.admitted)
        self.assertEqual(decision_small.status, AdmissionStatus.CONSERVATIVE)

        # Huge model: rejected under conservative policy
        decision_huge = self.monitor.evaluate_admission("llama3:70b", size_bytes=40 * 1024**3, snapshot=snap_bad)
        self.assertFalse(decision_huge.admitted)
        self.assertEqual(decision_huge.status, AdmissionStatus.CONSERVATIVE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
