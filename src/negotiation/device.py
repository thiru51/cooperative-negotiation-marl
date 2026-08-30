from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass
class DeviceSetup:
    """Everything the rest of the code needs to know about where it is running."""

    device: torch.device
    amp_dtype: torch.dtype | None
    name: str
    capability: tuple[int, int] | None = None
    total_vram_gb: float | None = None

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def needs_scaler(self) -> bool:
        # bfloat16 has the same exponent range as float32, so gradients cannot underflow
        # the way they do in fp16 and a loss scaler would be pure overhead. Scaling when
        # you do not need to is the classic way to get silently wrong grad norms.
        return self.amp_dtype is torch.float16

    def autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def scaler(self) -> torch.amp.GradScaler:
        return torch.amp.GradScaler(self.device.type, enabled=self.needs_scaler)

    def describe(self) -> str:
        amp = "off" if self.amp_dtype is None else str(self.amp_dtype).replace("torch.", "")
        vram = "" if self.total_vram_gb is None else f" {self.total_vram_gb:.1f}GB"
        cc = "" if self.capability is None else f" sm_{self.capability[0]}{self.capability[1]}"
        return f"{self.device.type}: {self.name}{cc}{vram}  amp={amp}"


def setup_device(requested: str = "auto", amp: bool = True) -> DeviceSetup:
    """Resolve the device once, turn on the fast paths, and report what we got.

    Call this exactly once per process, before any tensors are allocated.
    """
    if requested in ("auto", None, ""):
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "cuda was requested but torch.cuda.is_available() is False. "
            "Run `python scripts/check_gpu.py` to see why."
        )

    if device.type != "cuda":
        return DeviceSetup(device=device, amp_dtype=None, name="cpu")

    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # Every tensor shape in a run is pinned by horizon / num_envs / minibatch count, so
    # cudnn's autotuner pays its search cost once and then always picks a good kernel.
    torch.backends.cudnn.benchmark = True

    dtype = None
    if amp:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    return DeviceSetup(
        device=torch.device("cuda", index),
        amp_dtype=dtype,
        name=props.name,
        capability=(props.major, props.minor),
        total_vram_gb=props.total_memory / 1024**3,
    )


def reset_peak_memory(setup: DeviceSetup) -> None:
    if setup.is_cuda:
        torch.cuda.reset_peak_memory_stats(setup.device)


def peak_memory(setup: DeviceSetup) -> dict:
    """Peak VRAM this process actually touched, in GB.

    `allocated` is what the tensors needed; `reserved` is what the caching allocator held
    from the driver, which is the number nvidia-smi shows and the one that OOMs you.
    """
    if not setup.is_cuda:
        return {"peak_allocated_gb": None, "peak_reserved_gb": None}
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated(setup.device) / 1024**3, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved(setup.device) / 1024**3, 3),
    }
