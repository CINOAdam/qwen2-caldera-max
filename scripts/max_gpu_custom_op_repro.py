#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch


def _resolve_dtype(value: str) -> torch.dtype:
    value = value.lower()
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="MAX custom-op GPU smoke repro.")
    parser.add_argument("--device", default="cuda", help="Device to run on.")
    parser.add_argument("--dtype", default="float32", help="Tensor dtype.")
    parser.add_argument("--size", type=int, default=2, help="Square tensor size.")
    parser.add_argument("--no-float8-alias", action="store_true", help="Disable float8 alias.")
    args = parser.parse_args()

    if not args.no_float8_alias:
        if not hasattr(torch, "float8_e8m0fnu") and hasattr(torch, "float8_e5m2"):
            torch.float8_e8m0fnu = torch.float8_e5m2  # type: ignore[attr-defined]

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)
    print(f"torch={torch.__version__} cuda={torch.version.cuda} device={device} dtype={dtype}")

    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available in this environment.", file=sys.stderr)
        return 2

    from max.torch import CustomOpLibrary  # import after torch alias

    mojo_src = """
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

@compiler.register("copy_first")
struct CopyFirst:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype = DType.float32, rank=2],
        input: InputTensor[dtype = DType.float32, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        output[0, 0] = input[0, 0]
"""

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "op_pkg"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.mojo").write_text(mojo_src.strip() + "\n", encoding="utf-8")

        lib = CustomOpLibrary(pkg)
        op = lib.copy_first
        x = torch.randn(args.size, args.size, device=device, dtype=dtype)
        y = torch.empty_like(x)
        op(y, x)
        print("ok", y[0, 0].item(), x[0, 0].item())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
