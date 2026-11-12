"""
SWAN-ENGINE v0.2 — Hardware Resource Monitor & Admission Controller
=====================================================================

Phase 4: Hardware-aware Resource Monitor and Adaptive Resource Profiles for SWAN-HIVE.

Features:
- System RAM detection via /proc/meminfo with graceful error handling
- AMD GPU discovery and VRAM telemetry via /sys/class/drm/*/device sysfs (no rocm-smi required)
- Strict telemetry reliability classification: KNOWN, UNKNOWN, UNAVAILABLE, ERROR (never fabricates data)
- Clean fallback for CPU-only systems and non-AMD architectures
- Ollama resident model tracking integrated with ModelRouter
- Configurable Resource Profiles: TINY, STANDARD, LARGE, and custom profiles
- Safe Admission Controller evaluating:
    * Model weights
    * Context-scaled KV cache
    * Runtime & activation buffers
    * VRAM & RAM safety reserves
    * Current model residency (in-VRAM vs cold load)
    * Active inference concurrency limits
- Structured admission decisions: ADMITTED, WAIT, REJECT, CONSERVATIVE
- Prohibits automated model downloads and silent model unloads
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Optional integration with ModelRouter from Phase 3
try:
    from core.model_router import ModelRouter, ModelInfo
except ImportError:
    ModelRouter = None
    ModelInfo = None


# ---------------------------------------------------------------------------
# Enums & Status Constants
# ---------------------------------------------------------------------------

class TelemetryStatus(str, Enum):
    """Reliability classification for telemetry data."""
    KNOWN = "known"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class ProfileName(str, Enum):
    """Preset resource profile names."""
    TINY = "tiny"
    STANDARD = "standard"
    LARGE = "large"
    CUSTOM = "custom"


class AdmissionStatus(str, Enum):
    """Structured decision returned by the Admission Controller."""
    ADMITTED = "admitted"
    WAIT = "wait"
    REJECT = "reject"
    CONSERVATIVE = "unknown/conservative"


# AMD PCI Vendor ID in sysfs
AMD_VENDOR_ID = "0x1002"

_CARD_REGEX = re.compile(r"^card\d+$")


# ---------------------------------------------------------------------------
# Telemetry Metrics & Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Metric:
    """An individual hardware metric with explicit provenance and reliability."""
    value: Optional[float]
    unit: str                               # "bytes", "kB", "MB", "GB", "percent"
    source: str                             # e.g. "/proc/meminfo", "sysfs:mem_info_vram_total"
    status: TelemetryStatus = TelemetryStatus.KNOWN
    message: Optional[str] = None

    @property
    def is_reliable(self) -> bool:
        return self.status == TelemetryStatus.KNOWN and self.value is not None

    def in_bytes(self) -> Optional[int]:
        if self.value is None or not self.is_reliable:
            return None
        u = self.unit.lower()
        if u in ("bytes", "b"):
            return int(self.value)
        elif u == "kb":
            return int(self.value * 1024)
        elif u == "mb":
            return int(self.value * 1024 * 1024)
        elif u == "gb":
            return int(self.value * 1024 * 1024 * 1024)
        return int(self.value)

    def in_mb(self) -> Optional[float]:
        b = self.in_bytes()
        return round(b / (1024 * 1024), 2) if b is not None else None

    def in_gb(self) -> Optional[float]:
        b = self.in_bytes()
        return round(b / (1024 * 1024 * 1024), 2) if b is not None else None


@dataclass
class RAMTelemetry:
    """System memory snapshot."""
    total: Metric
    free: Metric
    available: Metric
    used: Metric

    @property
    def is_reliable(self) -> bool:
        return self.total.is_reliable and self.available.is_reliable


@dataclass
class GPUTelemetry:
    """AMD GPU device telemetry."""
    card_id: str                            # e.g. "card1"
    device_path: str                        # e.g. "/sys/class/drm/card1/device"
    vendor_id: str                          # e.g. "0x1002"
    is_amd: bool = True
    product_name: Optional[str] = None
    vram_total: Metric = field(default_factory=lambda: Metric(None, "bytes", "sysfs", TelemetryStatus.UNAVAILABLE))
    vram_used: Metric = field(default_factory=lambda: Metric(None, "bytes", "sysfs", TelemetryStatus.UNAVAILABLE))
    vram_free: Metric = field(default_factory=lambda: Metric(None, "bytes", "sysfs", TelemetryStatus.UNAVAILABLE))
    gtt_total: Optional[Metric] = None
    gtt_used: Optional[Metric] = None

    @property
    def is_reliable(self) -> bool:
        return self.is_amd and self.vram_total.is_reliable and self.vram_used.is_reliable


@dataclass
class HardwareSnapshot:
    """Full host hardware snapshot at a point in time."""
    ram: RAMTelemetry
    gpus: List[GPUTelemetry]
    loaded_models: List[str]                # Names of models currently resident in VRAM/RAM
    timestamp: float = field(default_factory=time.time)

    @property
    def primary_gpu(self) -> Optional[GPUTelemetry]:
        """Return the primary reliable AMD GPU, or None if CPU-only."""
        for gpu in self.gpus:
            if gpu.is_amd and gpu.is_reliable:
                return gpu
        return None

    @property
    def has_reliable_gpu(self) -> bool:
        return self.primary_gpu is not None


# ---------------------------------------------------------------------------
# Resource Profiles
# ---------------------------------------------------------------------------

@dataclass
class ResourceProfile:
    """Configurable profile governing concurrency, context bounds, and reserves."""
    name: str = ProfileName.STANDARD.value
    max_active_inferences: int = 1
    max_context_window: int = 16384
    vram_reserve_bytes: int = 1024 * 1024 * 1024       # 1.0 GB reserved for OS/display
    ram_reserve_bytes: int = 2048 * 1024 * 1024        # 2.0 GB reserved for system
    runtime_buffer_bytes: int = 512 * 1024 * 1024      # 512 MB activations & scratch
    max_model_weight_bytes: Optional[int] = None       # Optional hard weight limit
    allow_concurrency: bool = False
    conservative_on_unknown: bool = True


# Preset Profile Configurations
PRESET_PROFILES: Dict[str, ResourceProfile] = {
    ProfileName.TINY.value: ResourceProfile(
        name=ProfileName.TINY.value,
        max_active_inferences=1,
        max_context_window=4096,
        vram_reserve_bytes=1536 * 1024 * 1024,         # 1.5 GB
        ram_reserve_bytes=3072 * 1024 * 1024,          # 3.0 GB
        runtime_buffer_bytes=512 * 1024 * 1024,
        allow_concurrency=False,
        conservative_on_unknown=True,
    ),
    ProfileName.STANDARD.value: ResourceProfile(
        name=ProfileName.STANDARD.value,
        max_active_inferences=1,
        max_context_window=16384,
        vram_reserve_bytes=1024 * 1024 * 1024,         # 1.0 GB
        ram_reserve_bytes=2048 * 1024 * 1024,          # 2.0 GB
        runtime_buffer_bytes=768 * 1024 * 1024,        # 768 MB
        allow_concurrency=False,
        conservative_on_unknown=True,
    ),
    ProfileName.LARGE.value: ResourceProfile(
        name=ProfileName.LARGE.value,
        max_active_inferences=2,
        max_context_window=32768,
        vram_reserve_bytes=1536 * 1024 * 1024,         # 1.5 GB
        ram_reserve_bytes=2560 * 1024 * 1024,          # 2.5 GB
        runtime_buffer_bytes=1024 * 1024 * 1024,       # 1.0 GB
        allow_concurrency=True,
        conservative_on_unknown=True,
    ),
}


# ---------------------------------------------------------------------------
# Admission Decision
# ---------------------------------------------------------------------------

@dataclass
class AdmissionDecision:
    """Detailed result of an inference admission request."""
    status: AdmissionStatus
    admitted: bool
    reason: str
    model_name: str
    target_device: str                                 # "gpu", "cpu", "unknown"
    estimated_weights_bytes: int
    estimated_kv_cache_bytes: int
    estimated_runtime_buffer_bytes: int
    total_required_bytes: int
    available_memory_bytes: Optional[int]
    reserve_bytes: int
    active_inferences: int
    max_concurrency: int
    is_model_resident: bool                            # True if already resident in memory
    telemetry_status: TelemetryStatus
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Telemetry Readers (Proc & Sysfs)
# ---------------------------------------------------------------------------

def read_proc_meminfo(proc_path: str = "/proc/meminfo") -> RAMTelemetry:
    """
    Parse Linux /proc/meminfo. Never fabricates values.
    Returns RAMTelemetry with explicit status.
    """
    p = Path(proc_path)
    if not p.is_file():
        err_m = Metric(None, "bytes", proc_path, TelemetryStatus.UNAVAILABLE, f"File not found: {proc_path}")
        return RAMTelemetry(total=err_m, free=err_m, available=err_m, used=err_m)

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        err_m = Metric(None, "bytes", proc_path, TelemetryStatus.ERROR, f"Failed reading: {e}")
        return RAMTelemetry(total=err_m, free=err_m, available=err_m, used=err_m)

    data: Dict[str, float] = {}
    for line in content.splitlines():
        parts = line.split(":")
        if len(parts) == 2:
            key = parts[0].strip()
            val_parts = parts[1].strip().split()
            if val_parts:
                try:
                    data[key] = float(val_parts[0])
                except ValueError:
                    continue

    if "MemTotal" not in data:
        err_m = Metric(None, "bytes", proc_path, TelemetryStatus.ERROR, "Malformed: missing MemTotal")
        return RAMTelemetry(total=err_m, free=err_m, available=err_m, used=err_m)

    total_kb = data["MemTotal"]
    free_kb = data.get("MemFree", 0.0)

    # MemAvailable fallback for older kernels: MemFree + Buffers + Cached
    if "MemAvailable" in data:
        avail_kb = data["MemAvailable"]
    else:
        avail_kb = free_kb + data.get("Buffers", 0.0) + data.get("Cached", 0.0)

    used_kb = max(0.0, total_kb - avail_kb)

    return RAMTelemetry(
        total=Metric(total_kb, "kB", proc_path, TelemetryStatus.KNOWN),
        free=Metric(free_kb, "kB", proc_path, TelemetryStatus.KNOWN),
        available=Metric(avail_kb, "kB", proc_path, TelemetryStatus.KNOWN),
        used=Metric(used_kb, "kB", proc_path, TelemetryStatus.KNOWN),
    )


def discover_amd_gpus(drm_dir: str = "/sys/class/drm") -> List[GPUTelemetry]:
    """
    Discover AMD GPUs and VRAM telemetry under /sys/class/drm/*/device.
    Does not assume card index. Does not require rocm-smi.
    """
    p = Path(drm_dir)
    if not p.is_dir():
        return []

    gpus: List[GPUTelemetry] = []

    try:
        entries = sorted(p.iterdir())
    except OSError:
        return []

    for card in entries:
        if not _CARD_REGEX.match(card.name):
            continue

        dev = card / "device"
        if not dev.is_dir():
            continue

        # Check vendor ID
        vendor_file = dev / "vendor"
        vendor_id = ""
        if vendor_file.is_file():
            try:
                vendor_id = vendor_file.read_text(encoding="utf-8").strip().lower()
            except OSError:
                pass

        is_amd = (vendor_id == AMD_VENDOR_ID.lower())
        if not is_amd:
            continue

        # Product name (if present)
        prod_name = None
        for name_candidate in ["product_name", "device"]:
            nf = dev / name_candidate
            if nf.is_file():
                try:
                    prod_name = nf.read_text(encoding="utf-8").strip()
                    break
                except OSError:
                    pass

        # Read VRAM telemetry
        vram_total_f = dev / "mem_info_vram_total"
        vram_used_f = dev / "mem_info_vram_used"

        total_m = _read_sysfs_metric(vram_total_f)
        used_m = _read_sysfs_metric(vram_used_f)

        if total_m.is_reliable and used_m.is_reliable and total_m.value is not None and used_m.value is not None:
            free_bytes = max(0.0, total_m.value - used_m.value)
            free_m = Metric(free_bytes, "bytes", "computed:total-used", TelemetryStatus.KNOWN)
        else:
            free_m = Metric(None, "bytes", "sysfs", TelemetryStatus.UNAVAILABLE)

        # Optional GTT telemetry
        gtt_total_m = _read_sysfs_metric(dev / "mem_info_gtt_total")
        gtt_used_m = _read_sysfs_metric(dev / "mem_info_gtt_used")

        gpu = GPUTelemetry(
            card_id=card.name,
            device_path=str(dev),
            vendor_id=vendor_id,
            is_amd=True,
            product_name=prod_name,
            vram_total=total_m,
            vram_used=used_m,
            vram_free=free_m,
            gtt_total=gtt_total_m if gtt_total_m.is_reliable else None,
            gtt_used=gtt_used_m if gtt_used_m.is_reliable else None,
        )
        gpus.append(gpu)

    return gpus


def _read_sysfs_metric(path: Path) -> Metric:
    """Helper to read numeric sysfs file safely."""
    if not path.is_file():
        return Metric(None, "bytes", str(path), TelemetryStatus.UNAVAILABLE, "File not present")

    try:
        val_str = path.read_text(encoding="utf-8").strip()
        val = float(val_str)
        return Metric(val, "bytes", str(path), TelemetryStatus.KNOWN)
    except (ValueError, OSError) as e:
        return Metric(None, "bytes", str(path), TelemetryStatus.ERROR, str(e))


# ---------------------------------------------------------------------------
# KV Cache & Model Memory Estimators
# ---------------------------------------------------------------------------

def estimate_kv_cache_bytes(
    context_tokens: int,
    num_layers: int = 28,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    bytes_per_element: int = 2,
    batch_size: int = 1,
) -> int:
    """
    Estimate KV cache memory requirement in bytes.
    Default parameters approximate Qwen 2.5 7B GQA architecture (fp16).
    Formula: 2 * num_layers * num_kv_heads * head_dim * context_tokens * bytes_per_element * batch_size
    """
    ctx = max(0, int(context_tokens))
    return 2 * num_layers * num_kv_heads * head_dim * ctx * bytes_per_element * batch_size


def estimate_model_weight_bytes(
    model_name: str,
    parameter_size_str: Optional[str] = None,
    size_bytes: Optional[int] = None,
) -> int:
    """
    Estimate or resolve model weights size in bytes.
    Uses known model size, size_bytes from Ollama, or parses parameter strings (e.g. '7.6B' Q4).
    """
    if size_bytes and size_bytes > 0:
        return int(size_bytes)

    name_lower = model_name.lower()

    # Known standard defaults (Q4_K_M weights in bytes)
    if "7b" in name_lower or "swan-coder" in name_lower:
        return int(4.7 * 1024 * 1024 * 1024)   # ~4.7 GB
    elif "14b" in name_lower:
        return int(9.0 * 1024 * 1024 * 1024)   # ~9.0 GB
    elif "32b" in name_lower:
        return int(20.0 * 1024 * 1024 * 1024)  # ~20.0 GB
    elif "3b" in name_lower:
        return int(2.2 * 1024 * 1024 * 1024)   # ~2.2 GB
    elif "1.5b" in name_lower or "1b" in name_lower:
        return int(1.2 * 1024 * 1024 * 1024)   # ~1.2 GB

    # Fallback to parameter size parse if available
    if parameter_size_str:
        m = re.match(r"^([\d.]+)\s*([BMKbmk])?$", parameter_size_str.strip())
        if m:
            num = float(m.group(1))
            unit = (m.group(2) or "b").lower()
            if unit == "b":
                # Assuming 4-bit quantization (~0.6 bytes per param)
                return int(num * 1e9 * 0.6)
            elif unit == "m":
                return int(num * 1e6 * 0.6)

    # Conservative default: 5 GB
    return int(5.0 * 1024 * 1024 * 1024)


# ---------------------------------------------------------------------------
# Resource Monitor & Admission Controller
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """
    Hardware-aware resource monitor and admission controller.
    """

    def __init__(
        self,
        proc_meminfo_path: str = "/proc/meminfo",
        drm_dir: str = "/sys/class/drm",
        model_router: Optional[Any] = None,
        profile: Optional[ResourceProfile] = None,
    ):
        self.proc_meminfo_path = proc_meminfo_path
        self.drm_dir = drm_dir
        self.model_router = model_router
        self.profile = profile or PRESET_PROFILES[ProfileName.STANDARD.value]
        self._active_inferences: int = 0

    # ------ Profile Configuration ------

    def set_profile(self, profile: ResourceProfile) -> None:
        """Set active resource profile."""
        if not isinstance(profile, ResourceProfile):
            raise ValueError("Profile must be an instance of ResourceProfile.")
        self.profile = profile

    def set_preset_profile(self, preset_name: str) -> None:
        """Set profile from preset name ('tiny', 'standard', 'large')."""
        name_clean = preset_name.strip().lower()
        if name_clean not in PRESET_PROFILES:
            raise ValueError(f"Unknown preset '{preset_name}'. Supported: {list(PRESET_PROFILES.keys())}")
        self.profile = PRESET_PROFILES[name_clean]

    # ------ Concurrency Tracking ------

    @property
    def active_inferences(self) -> int:
        return self._active_inferences

    def acquire_inference(self) -> None:
        """Register the start of an active model inference."""
        self._active_inferences += 1

    def release_inference(self) -> None:
        """Register the completion of an active model inference."""
        self._active_inferences = max(0, self._active_inferences - 1)

    # ------ Hardware Telemetry Snapshot ------

    def get_hardware_snapshot(self) -> HardwareSnapshot:
        """
        Collect live hardware telemetry without fabricating readings.
        """
        ram = read_proc_meminfo(self.proc_meminfo_path)
        gpus = discover_amd_gpus(self.drm_dir)

        loaded_models: List[str] = []
        if self.model_router:
            try:
                # Query resident models in ModelRouter if available
                models = self.model_router.list_models(loaded_only=True)
                loaded_models = [m.name for m in models]
            except Exception:
                pass

        return HardwareSnapshot(
            ram=ram,
            gpus=gpus,
            loaded_models=loaded_models,
            timestamp=time.time(),
        )

    # ------ Admission Controller ------

    def evaluate_admission(
        self,
        model_name: str,
        context_tokens: Optional[int] = None,
        parameter_size_str: Optional[str] = None,
        size_bytes: Optional[int] = None,
        snapshot: Optional[HardwareSnapshot] = None,
    ) -> AdmissionDecision:
        """
        Evaluate whether an inference request can safely be admitted without OOM.

        Evaluates:
          1. Concurrency limit against active inferences.
          2. Context bounds against active profile.
          3. Model weight estimation.
          4. KV cache estimation based on context length.
          5. Runtime buffers and OS safety reserves.
          6. Model residency in Ollama (if already loaded in VRAM, weight loading overhead is 0).
          7. Live VRAM (AMD GPU) or RAM (CPU fallback) capacity.

        Returns AdmissionDecision.
        """
        snap = snapshot or self.get_hardware_snapshot()
        ctx = context_tokens or self.profile.max_context_window

        # 1. Concurrency Check
        if self._active_inferences >= self.profile.max_active_inferences:
            return AdmissionDecision(
                status=AdmissionStatus.WAIT,
                admitted=False,
                reason=(
                    f"Active inferences ({self._active_inferences}) at or exceeding "
                    f"profile concurrency limit ({self.profile.max_active_inferences})."
                ),
                model_name=model_name,
                target_device="unknown",
                estimated_weights_bytes=0,
                estimated_kv_cache_bytes=0,
                estimated_runtime_buffer_bytes=0,
                total_required_bytes=0,
                available_memory_bytes=None,
                reserve_bytes=0,
                active_inferences=self._active_inferences,
                max_concurrency=self.profile.max_active_inferences,
                is_model_resident=False,
                telemetry_status=TelemetryStatus.KNOWN,
            )

        # 2. Context Size Bound Check
        if ctx > self.profile.max_context_window:
            return AdmissionDecision(
                status=AdmissionStatus.REJECT,
                admitted=False,
                reason=(
                    f"Requested context size ({ctx} tokens) exceeds active profile limit "
                    f"({self.profile.max_context_window} tokens)."
                ),
                model_name=model_name,
                target_device="unknown",
                estimated_weights_bytes=0,
                estimated_kv_cache_bytes=0,
                estimated_runtime_buffer_bytes=0,
                total_required_bytes=0,
                available_memory_bytes=None,
                reserve_bytes=0,
                active_inferences=self._active_inferences,
                max_concurrency=self.profile.max_active_inferences,
                is_model_resident=False,
                telemetry_status=TelemetryStatus.KNOWN,
            )

        # 3. Model Weight & KV Cache Estimation
        weight_bytes = estimate_model_weight_bytes(model_name, parameter_size_str, size_bytes)
        kv_cache_bytes = estimate_kv_cache_bytes(ctx)
        runtime_buffer = self.profile.runtime_buffer_bytes

        # Check if model is already loaded in Ollama VRAM/RAM
        is_resident = any(
            (model_name.lower() in m.lower() or m.lower() in model_name.lower())
            for m in snap.loaded_models
        )

        # If model is already resident in memory, weights are already committed!
        effective_weight_bytes = 0 if is_resident else weight_bytes
        total_required_bytes = effective_weight_bytes + kv_cache_bytes + runtime_buffer

        # Hard model weight limit check if configured in profile
        if self.profile.max_model_weight_bytes and weight_bytes > self.profile.max_model_weight_bytes:
            return AdmissionDecision(
                status=AdmissionStatus.REJECT,
                admitted=False,
                reason=(
                    f"Model weight ({weight_bytes / (1024**3):.2f} GB) exceeds profile "
                    f"weight cap ({self.profile.max_model_weight_bytes / (1024**3):.2f} GB)."
                ),
                model_name=model_name,
                target_device="unknown",
                estimated_weights_bytes=weight_bytes,
                estimated_kv_cache_bytes=kv_cache_bytes,
                estimated_runtime_buffer_bytes=runtime_buffer,
                total_required_bytes=total_required_bytes,
                available_memory_bytes=None,
                reserve_bytes=0,
                active_inferences=self._active_inferences,
                max_concurrency=self.profile.max_active_inferences,
                is_model_resident=is_resident,
                telemetry_status=TelemetryStatus.KNOWN,
            )

        # 4. Device Target & Memory Admission
        gpu = snap.primary_gpu
        ram = snap.ram

        # Target: AMD GPU VRAM
        if gpu and gpu.is_reliable:
            target_device = "gpu"
            vram_free = gpu.vram_free.in_bytes() or 0
            vram_total = gpu.vram_total.in_bytes() or 0
            reserve = self.profile.vram_reserve_bytes
            usable_vram = max(0, vram_free - reserve)

            # Check if it fits in VRAM
            if total_required_bytes <= usable_vram:
                return AdmissionDecision(
                    status=AdmissionStatus.ADMITTED,
                    admitted=True,
                    reason="Request admitted to GPU VRAM.",
                    model_name=model_name,
                    target_device="gpu",
                    estimated_weights_bytes=weight_bytes,
                    estimated_kv_cache_bytes=kv_cache_bytes,
                    estimated_runtime_buffer_bytes=runtime_buffer,
                    total_required_bytes=total_required_bytes,
                    available_memory_bytes=vram_free,
                    reserve_bytes=reserve,
                    active_inferences=self._active_inferences,
                    max_concurrency=self.profile.max_active_inferences,
                    is_model_resident=is_resident,
                    telemetry_status=TelemetryStatus.KNOWN,
                )

            # VRAM insufficient: check if total model exceeds total physical VRAM
            if (weight_bytes + kv_cache_bytes + runtime_buffer) > vram_total:
                # Impossible to fit in VRAM even if empty: check RAM fallback
                ram_available = ram.available.in_bytes() if ram.is_reliable else None
                if ram_available and total_required_bytes <= max(0, ram_available - self.profile.ram_reserve_bytes):
                    return AdmissionDecision(
                        status=AdmissionStatus.ADMITTED,
                        admitted=True,
                        reason="Admitted to CPU RAM (exceeds physical VRAM capacity).",
                        model_name=model_name,
                        target_device="cpu",
                        estimated_weights_bytes=weight_bytes,
                        estimated_kv_cache_bytes=kv_cache_bytes,
                        estimated_runtime_buffer_bytes=runtime_buffer,
                        total_required_bytes=total_required_bytes,
                        available_memory_bytes=ram_available,
                        reserve_bytes=self.profile.ram_reserve_bytes,
                        active_inferences=self._active_inferences,
                        max_concurrency=self.profile.max_active_inferences,
                        is_model_resident=is_resident,
                        telemetry_status=TelemetryStatus.KNOWN,
                    )
                else:
                    return AdmissionDecision(
                        status=AdmissionStatus.REJECT,
                        admitted=False,
                        reason=(
                            f"Total required memory ({total_required_bytes / (1024**3):.2f} GB) "
                            f"exceeds physical VRAM capacity ({vram_total / (1024**3):.2f} GB) "
                            "and insufficient system RAM."
                        ),
                        model_name=model_name,
                        target_device="gpu",
                        estimated_weights_bytes=weight_bytes,
                        estimated_kv_cache_bytes=kv_cache_bytes,
                        estimated_runtime_buffer_bytes=runtime_buffer,
                        total_required_bytes=total_required_bytes,
                        available_memory_bytes=vram_free,
                        reserve_bytes=reserve,
                        active_inferences=self._active_inferences,
                        max_concurrency=self.profile.max_active_inferences,
                        is_model_resident=is_resident,
                        telemetry_status=TelemetryStatus.KNOWN,
                    )

            # Could fit physically, but VRAM is currently full -> WAIT
            return AdmissionDecision(
                status=AdmissionStatus.WAIT,
                admitted=False,
                reason=(
                    f"Insufficient free VRAM ({vram_free / (1024**3):.2f} GB free, "
                    f"{usable_vram / (1024**3):.2f} GB usable after {reserve / (1024**3):.2f} GB reserve; "
                    f"required: {total_required_bytes / (1024**3):.2f} GB). Waiting for memory to free."
                ),
                model_name=model_name,
                target_device="gpu",
                estimated_weights_bytes=weight_bytes,
                estimated_kv_cache_bytes=kv_cache_bytes,
                estimated_runtime_buffer_bytes=runtime_buffer,
                total_required_bytes=total_required_bytes,
                available_memory_bytes=vram_free,
                reserve_bytes=reserve,
                active_inferences=self._active_inferences,
                max_concurrency=self.profile.max_active_inferences,
                is_model_resident=is_resident,
                telemetry_status=TelemetryStatus.KNOWN,
            )

        # Target: CPU RAM Fallback (No GPU or unreliable GPU telemetry)
        if ram.is_reliable:
            target_device = "cpu"
            ram_avail = ram.available.in_bytes() or 0
            ram_total = ram.total.in_bytes() or 0
            reserve = self.profile.ram_reserve_bytes
            usable_ram = max(0, ram_avail - reserve)

            if total_required_bytes <= usable_ram:
                return AdmissionDecision(
                    status=AdmissionStatus.ADMITTED,
                    admitted=True,
                    reason="Request admitted to system RAM (CPU fallback).",
                    model_name=model_name,
                    target_device="cpu",
                    estimated_weights_bytes=weight_bytes,
                    estimated_kv_cache_bytes=kv_cache_bytes,
                    estimated_runtime_buffer_bytes=runtime_buffer,
                    total_required_bytes=total_required_bytes,
                    available_memory_bytes=ram_avail,
                    reserve_bytes=reserve,
                    active_inferences=self._active_inferences,
                    max_concurrency=self.profile.max_active_inferences,
                    is_model_resident=is_resident,
                    telemetry_status=TelemetryStatus.KNOWN,
                )

            # Check if total physical RAM is smaller than model
            if (weight_bytes + kv_cache_bytes + runtime_buffer) > ram_total:
                return AdmissionDecision(
                    status=AdmissionStatus.REJECT,
                    admitted=False,
                    reason=(
                        f"Total required memory ({total_required_bytes / (1024**3):.2f} GB) "
                        f"exceeds total physical RAM ({ram_total / (1024**3):.2f} GB)."
                    ),
                    model_name=model_name,
                    target_device="cpu",
                    estimated_weights_bytes=weight_bytes,
                    estimated_kv_cache_bytes=kv_cache_bytes,
                    estimated_runtime_buffer_bytes=runtime_buffer,
                    total_required_bytes=total_required_bytes,
                    available_memory_bytes=ram_avail,
                    reserve_bytes=reserve,
                    active_inferences=self._active_inferences,
                    max_concurrency=self.profile.max_active_inferences,
                    is_model_resident=is_resident,
                    telemetry_status=TelemetryStatus.KNOWN,
                )

            return AdmissionDecision(
                status=AdmissionStatus.WAIT,
                admitted=False,
                reason=(
                    f"Insufficient available RAM ({ram_avail / (1024**3):.2f} GB available, "
                    f"{usable_ram / (1024**3):.2f} GB usable after {reserve / (1024**3):.2f} GB reserve; "
                    f"required: {total_required_bytes / (1024**3):.2f} GB). Waiting for memory."
                ),
                model_name=model_name,
                target_device="cpu",
                estimated_weights_bytes=weight_bytes,
                estimated_kv_cache_bytes=kv_cache_bytes,
                estimated_runtime_buffer_bytes=runtime_buffer,
                total_required_bytes=total_required_bytes,
                available_memory_bytes=ram_avail,
                reserve_bytes=reserve,
                active_inferences=self._active_inferences,
                max_concurrency=self.profile.max_active_inferences,
                is_model_resident=is_resident,
                telemetry_status=TelemetryStatus.KNOWN,
            )

        # Telemetry is UNKNOWN, UNAVAILABLE, or in ERROR state
        if self.profile.conservative_on_unknown:
            # Conservative rule: Admit only small models (<= 5GB) under sequential inference
            conservative_cap = 5 * 1024 * 1024 * 1024
            can_admit_conservatively = (
                self._active_inferences == 0 and
                weight_bytes <= conservative_cap and
                ctx <= 4096
            )
            return AdmissionDecision(
                status=AdmissionStatus.CONSERVATIVE,
                admitted=can_admit_conservatively,
                reason=(
                    "Telemetry unavailable. Admitted under conservative sequential limit."
                    if can_admit_conservatively else
                    "Telemetry unavailable. Request rejected as it exceeds conservative safety limits (max 5GB, 4k ctx)."
                ),
                model_name=model_name,
                target_device="unknown",
                estimated_weights_bytes=weight_bytes,
                estimated_kv_cache_bytes=kv_cache_bytes,
                estimated_runtime_buffer_bytes=runtime_buffer,
                total_required_bytes=total_required_bytes,
                available_memory_bytes=None,
                reserve_bytes=0,
                active_inferences=self._active_inferences,
                max_concurrency=self.profile.max_active_inferences,
                is_model_resident=is_resident,
                telemetry_status=TelemetryStatus.UNAVAILABLE,
            )

        return AdmissionDecision(
            status=AdmissionStatus.REJECT,
            admitted=False,
            reason="Telemetry unavailable and conservative scheduling disabled.",
            model_name=model_name,
            target_device="unknown",
            estimated_weights_bytes=weight_bytes,
            estimated_kv_cache_bytes=kv_cache_bytes,
            estimated_runtime_buffer_bytes=runtime_buffer,
            total_required_bytes=total_required_bytes,
            available_memory_bytes=None,
            reserve_bytes=0,
            active_inferences=self._active_inferences,
            max_concurrency=self.profile.max_active_inferences,
            is_model_resident=is_resident,
            telemetry_status=TelemetryStatus.UNAVAILABLE,
        )
