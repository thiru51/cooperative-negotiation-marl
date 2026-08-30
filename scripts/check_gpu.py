"""Doctor command: tell me what this machine can actually do before I start a run.

Run this first on a fresh clone. If it says cuda available: False, nothing else in the
repo will use the GPU and everything will silently fall back to the CPU.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from negotiation.device import setup_device


def matmul_benchmark(device: torch.device, dtype: torch.dtype, n: int = 4096,
                     iters: int = 30) -> float:
    """Achieved TFLOP/s on an n x n square matmul. 2*n^3 flops per multiply."""
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(5):
        a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return 2.0 * n**3 * iters / seconds / 1e12


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=4096, help="matmul benchmark dimension")
    p.add_argument("--no-benchmark", dest="benchmark", action="store_false")
    args = p.parse_args(argv)

    print(f"torch                {torch.__version__}")
    print(f"torch cuda build     {torch.version.cuda or 'cpu-only wheel'}")
    print(f"cuda available       {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print()
        print("No CUDA device. Training still runs on the CPU, just far slower.")
        print("Check in this order:")
        print("  1. nvidia-smi           driver installed and the GPU is visible")
        print("  2. python -c 'import torch; print(torch.version.cuda)'")
        print("     -> None means you installed a CPU-only wheel; reinstall from the")
        print("        cu124/cu126 index (see README, install path b)")
        print("  3. CUDA_VISIBLE_DEVICES is not set to an empty string")
        return 1

    dev = setup_device("cuda")
    props = torch.cuda.get_device_properties(dev.device)
    free, total = torch.cuda.mem_get_info(dev.device)

    print(f"device count         {torch.cuda.device_count()}")
    print(f"device name          {props.name}")
    print(f"compute capability   {props.major}.{props.minor}")
    print(f"multiprocessors      {props.multi_processor_count}")
    print(f"total VRAM           {total / 1024**3:.2f} GB")
    print(f"free VRAM            {free / 1024**3:.2f} GB")
    print(f"bf16 supported       {torch.cuda.is_bf16_supported()}")
    print(f"amp dtype chosen     {dev.amp_dtype}")
    print(f"tf32 matmul          {torch.backends.cuda.matmul.allow_tf32}")
    print(f"tf32 cudnn           {torch.backends.cudnn.allow_tf32}")
    print(f"cudnn benchmark      {torch.backends.cudnn.benchmark}")
    print(f"float32 precision    {torch.get_float32_matmul_precision()}")

    if args.benchmark:
        n = args.size
        print()
        print(f"matmul benchmark, {n}x{n}:")
        print(f"  fp32 (tf32 on)     {matmul_benchmark(dev.device, torch.float32, n):8.1f} TFLOP/s")
        if torch.cuda.is_bf16_supported():
            print(f"  bf16               {matmul_benchmark(dev.device, torch.bfloat16, n):8.1f} TFLOP/s")
        else:
            print(f"  fp16               {matmul_benchmark(dev.device, torch.float16, n):8.1f} TFLOP/s")
        print(f"  peak allocated     {torch.cuda.max_memory_allocated(dev.device) / 1024**3:.2f} GB")

    print()
    print("GPU looks usable. Next: python scripts/smoke_test.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
