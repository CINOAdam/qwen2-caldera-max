#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_SHAPES = [
    (512, 512, 64),
    (1024, 1024, 128),
    (2048, 2048, 128),
    (4096, 4096, 128),
    (8192, 8192, 256),
]
DEFAULT_BITS = [
    (2, 4),
    (4, 4),
]


def _parse_triplet(value: str) -> tuple[int, int, int]:
    if "x" in value:
        parts = value.split("x")
    else:
        parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected format out,in,rank or outxinxrank")
    try:
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Shape values must be integers") from exc


def _parse_bits(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected format bits_q,bits_lr")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Bit values must be integers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small sweep of the Q + LR kernel benchmark."
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_triplet,
        help="Override shapes as out,in,rank (repeatable)",
    )
    parser.add_argument(
        "--bits",
        action="append",
        type=_parse_bits,
        help="Override bit pairs as bits_q,bits_lr (repeatable)",
    )
    parser.add_argument(
        "--backend",
        choices=("reference", "mojo", "mojo_cached", "both"),
        default="both",
        help="Which backend to run",
    )
    parser.add_argument("--iters", type=int, default=3, help="Iterations per run")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--group-size", type=int, default=128, help="Quant group size")
    parser.add_argument("--validate", action="store_true", help="Validate outputs")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("artifacts/kernel_bench.csv"),
        help="CSV output path",
    )
    parser.add_argument("--tag", default="sweep", help="Tag for CSV logging")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    shapes = args.shape or DEFAULT_SHAPES
    bits_list = args.bits or DEFAULT_BITS

    root = Path(__file__).resolve().parents[1]
    bench_script = root / "benchmarks" / "run_kernel_bench.py"
    python = sys.executable

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) if not existing else f"{root}{os.pathsep}{existing}"

    for (out_dim, in_dim, rank) in shapes:
        for (bits_q, bits_lr) in bits_list:
            cmd = [
                python,
                str(bench_script),
                "--backend",
                args.backend,
                "--iters",
                str(args.iters),
                "--warmup",
                str(args.warmup),
                "--out-dim",
                str(out_dim),
                "--in-dim",
                str(in_dim),
                "--rank",
                str(rank),
                "--group-size",
                str(args.group_size),
                "--bits-q",
                str(bits_q),
                "--bits-lr",
                str(bits_lr),
                "--log-csv",
                str(args.csv),
                "--tag",
                args.tag,
            ]
            if args.validate:
                cmd.append("--validate")
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True, cwd=root, env=env)


if __name__ == "__main__":
    main()
