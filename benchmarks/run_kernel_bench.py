#!/usr/bin/env python3
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from src.q_lr_reference import q_lr_forward, quantize_groupwise
from src.caldera_runtime import CalderaLinear, QuantizedTensor


DEFAULTS = {
    "out_dim": 4096,
    "in_dim": 4096,
    "rank": 128,
    "group_size": 128,
    "bits_q": 2,
    "bits_lr": 4,
}

PRESETS = {
    "large": DEFAULTS,
    "xlarge": {
        "out_dim": 8192,
        "in_dim": 8192,
        "rank": 256,
        "group_size": 128,
        "bits_q": 2,
        "bits_lr": 4,
    },
}


def _run_reference(
    q_q,
    l_q,
    r_q,
    x,
    *,
    iters: int,
    warmup: int,
) -> tuple[float, np.ndarray]:
    for _ in range(warmup):
        _ = q_lr_forward(q_q, l_q, r_q, x)

    timings = []
    out = None
    for _ in range(iters):
        start = time.perf_counter()
        out = q_lr_forward(q_q, l_q, r_q, x)
        timings.append(time.perf_counter() - start)
    avg = sum(timings) / len(timings)
    if out is None:
        out = q_lr_forward(q_q, l_q, r_q, x)
    return avg, out


def _run_mojo(
    q_q,
    l_q,
    r_q,
    x,
    *,
    iters: int,
    warmup: int,
) -> tuple[float, np.ndarray]:
    from src.q_lr_mojo import q_lr_fused

    for _ in range(warmup):
        _ = q_lr_fused(q_q, l_q, r_q, x)

    timings = []
    out = None
    for _ in range(iters):
        start = time.perf_counter()
        out = q_lr_fused(q_q, l_q, r_q, x)
        timings.append(time.perf_counter() - start)
    avg = sum(timings) / len(timings)
    if out is None:
        out = q_lr_fused(q_q, l_q, r_q, x)
    return avg, out


def _run_mojo_cached(
    q_q,
    l_q,
    r_q,
    x,
    *,
    iters: int,
    warmup: int,
) -> tuple[float, np.ndarray]:
    from src.q_lr_mojo import q_lr_fused_batched

    if warmup:
        _ = q_lr_fused_batched(q_q, l_q, r_q, x, iters=warmup)

    start = time.perf_counter()
    out = q_lr_fused_batched(q_q, l_q, r_q, x, iters=iters)
    elapsed = time.perf_counter() - start
    avg = elapsed / iters
    return avg, out


def _run_mojo_auto(
    q_q,
    l_q,
    r_q,
    x,
    *,
    iters: int,
    warmup: int,
) -> tuple[float, np.ndarray]:
    from src.q_lr_mojo import q_lr_auto

    for _ in range(warmup):
        _ = q_lr_auto(q_q, l_q, r_q, x)

    timings = []
    out = None
    for _ in range(iters):
        start = time.perf_counter()
        out = q_lr_auto(q_q, l_q, r_q, x)
        timings.append(time.perf_counter() - start)
    avg = sum(timings) / len(timings)
    if out is None:
        out = q_lr_auto(q_q, l_q, r_q, x)
    return avg, out


def _run_caldera_cached(
    q_q,
    l_q,
    r_q,
    x,
    *,
    iters: int,
    warmup: int,
    cache_dequant: bool,
    cache_mode: str | None,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for Caldera benchmark but is not available.")
    q_t = QuantizedTensor(
        values=torch.from_numpy(q_q.values),
        scales=torch.from_numpy(q_q.scales),
        group_size=q_q.group_size,
        num_bits=q_q.num_bits,
    )
    l_t = QuantizedTensor(
        values=torch.from_numpy(l_q.values),
        scales=torch.from_numpy(l_q.scales),
        group_size=l_q.group_size,
        num_bits=l_q.num_bits,
    )
    r_t = QuantizedTensor(
        values=torch.from_numpy(r_q.values),
        scales=torch.from_numpy(r_q.scales),
        group_size=r_q.group_size,
        num_bits=r_q.num_bits,
    )
    layer = CalderaLinear(
        q_t,
        l_t,
        r_t,
        cache_dequant=cache_dequant,
        cache_mode=cache_mode,
        chunk_size=None,
    ).to(device)
    x_t = torch.from_numpy(x).to(device)

    for _ in range(warmup):
        _ = layer(x_t)

    timings = []
    out = None
    for _ in range(iters):
        start = time.perf_counter()
        out = layer(x_t)
        timings.append(time.perf_counter() - start)
    avg = sum(timings) / len(timings)
    if out is None:
        out = layer(x_t)
    return avg, out.detach().cpu().numpy()


def _maybe_gflops(rows: int, cols: int, rank: int, avg_time_s: float) -> float | None:
    if avg_time_s <= 0:
        return None
    ops = 2 * rows * cols + 2 * rank * cols + 2 * rows * rank
    return ops / avg_time_s / 1.0e9


def _print_summary(
    label: str,
    *,
    avg_time_s: float,
    rows: int,
    cols: int,
    rank: int,
    group_size: int,
    bits_q: int,
    bits_lr: int,
    warmup: int,
    iters: int,
) -> None:
    gflops = _maybe_gflops(rows, cols, rank, avg_time_s)
    gflops_str = f"{gflops:.3f}" if gflops is not None else "n/a"
    print(
        f"[{label}] avg_time_s={avg_time_s:.6f} "
        f"gflops={gflops_str} rows={rows} cols={cols} rank={rank} "
        f"group={group_size} bits_q={bits_q} bits_lr={bits_lr} "
        f"warmup={warmup} iters={iters}"
    )


def _append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    fieldnames = list(rows[0].keys())
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fused vs unfused Q + LR inference."
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS.keys()),
        default=None,
        help="Use a preset shape (overrides unspecified dim/bit flags)",
    )
    parser.add_argument("--out-dim", type=int, default=None, help="Output dimension")
    parser.add_argument("--in-dim", type=int, default=None, help="Input dimension")
    parser.add_argument("--rank", type=int, default=None, help="Low-rank size")
    parser.add_argument("--group-size", type=int, default=None, help="Quant group size")
    parser.add_argument("--bits-q", type=int, default=None, help="Bits for Q")
    parser.add_argument("--bits-lr", type=int, default=None, help="Bits for L/R")
    parser.add_argument("--iters", type=int, default=5, help="Iterations")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument(
        "--backend",
        choices=("reference", "mojo", "mojo_cached", "mojo_auto", "caldera_cached", "both"),
        default="both",
        help="Which backend to run",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Compare mojo output to reference output",
    )
    parser.add_argument(
        "--caldera-cache-dequant",
        action="store_true",
        help="Cache dequantized Q/L/R for the Caldera baseline",
    )
    parser.add_argument(
        "--caldera-cache-mode",
        choices=("none", "r", "lr", "qlr"),
        default=None,
        help="Selective dequant cache for Caldera (overrides --caldera-cache-dequant)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for Caldera benchmark (e.g., cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help="Append results to a CSV file (created if missing)",
    )
    parser.add_argument("--tag", default="", help="Optional tag for CSV logging")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    if args.caldera_cache_mode is not None and args.caldera_cache_dequant:
        raise SystemExit("Use either --caldera-cache-mode or --caldera-cache-dequant, not both.")
    caldera_cache_mode = args.caldera_cache_mode
    if caldera_cache_mode is None and args.caldera_cache_dequant:
        caldera_cache_mode = "qlr"

    config = DEFAULTS.copy()
    if args.preset is not None:
        config.update(PRESETS[args.preset])
    for key in ("out_dim", "in_dim", "rank", "group_size", "bits_q", "bits_lr"):
        val = getattr(args, key)
        if val is not None:
            config[key] = val

    out_dim = config["out_dim"]
    in_dim = config["in_dim"]
    rank = config["rank"]
    group_size = config["group_size"]
    bits_q = config["bits_q"]
    bits_lr = config["bits_lr"]

    rng = np.random.default_rng(args.seed)
    q = rng.standard_normal((out_dim, in_dim), dtype=np.float32)
    l = rng.standard_normal((out_dim, rank), dtype=np.float32)
    r = rng.standard_normal((rank, in_dim), dtype=np.float32)
    x = rng.standard_normal((in_dim,), dtype=np.float32)

    q_q = quantize_groupwise(q, group_size, bits_q)
    l_q = quantize_groupwise(l, group_size, bits_lr)
    r_q = quantize_groupwise(r, group_size, bits_lr)

    ref_time = None
    ref_out = None
    mojo_time = None
    mojo_out = None
    mojo_cached_time = None
    mojo_cached_out = None
    mojo_auto_time = None
    mojo_auto_out = None
    caldera_time = None
    caldera_out = None
    val_max = None
    val_mean = None

    if args.backend in ("reference", "both") or args.validate:
        ref_time, ref_out = _run_reference(
            q_q,
            l_q,
            r_q,
            x,
            iters=args.iters,
            warmup=args.warmup,
        )
        _print_summary(
            "reference",
            avg_time_s=ref_time,
            rows=out_dim,
            cols=in_dim,
            rank=rank,
            group_size=group_size,
            bits_q=bits_q,
            bits_lr=bits_lr,
            warmup=args.warmup,
            iters=args.iters,
        )

    if args.backend in ("mojo", "both"):
        mojo_time, mojo_out = _run_mojo(
            q_q,
            l_q,
            r_q,
            x,
            iters=args.iters,
            warmup=args.warmup,
        )
        _print_summary(
            "mojo",
            avg_time_s=mojo_time,
            rows=out_dim,
            cols=in_dim,
            rank=rank,
            group_size=group_size,
            bits_q=bits_q,
            bits_lr=bits_lr,
            warmup=args.warmup,
            iters=args.iters,
        )

    if args.backend == "mojo_auto":
        mojo_auto_time, mojo_auto_out = _run_mojo_auto(
            q_q,
            l_q,
            r_q,
            x,
            iters=args.iters,
            warmup=args.warmup,
        )
        _print_summary(
            "mojo_auto",
            avg_time_s=mojo_auto_time,
            rows=out_dim,
            cols=in_dim,
            rank=rank,
            group_size=group_size,
            bits_q=bits_q,
            bits_lr=bits_lr,
            warmup=args.warmup,
            iters=args.iters,
        )

    if args.backend == "mojo_cached":
        mojo_cached_time, mojo_cached_out = _run_mojo_cached(
            q_q,
            l_q,
            r_q,
            x,
            iters=args.iters,
            warmup=args.warmup,
        )
        _print_summary(
            "mojo_cached",
            avg_time_s=mojo_cached_time,
            rows=out_dim,
            cols=in_dim,
            rank=rank,
            group_size=group_size,
            bits_q=bits_q,
            bits_lr=bits_lr,
            warmup=args.warmup,
            iters=args.iters,
        )

    if args.backend == "caldera_cached":
        caldera_time, caldera_out = _run_caldera_cached(
            q_q,
            l_q,
            r_q,
            x,
            iters=args.iters,
            warmup=args.warmup,
            cache_dequant=args.caldera_cache_dequant,
            cache_mode=caldera_cache_mode,
            device=torch.device(args.device),
        )
        _print_summary(
            "caldera_cached",
            avg_time_s=caldera_time,
            rows=out_dim,
            cols=in_dim,
            rank=rank,
            group_size=group_size,
            bits_q=bits_q,
            bits_lr=bits_lr,
            warmup=args.warmup,
            iters=args.iters,
        )

    if args.validate:
        if ref_out is None:
            _, ref_out = _run_reference(
                q_q,
                l_q,
                r_q,
                x,
                iters=1,
                warmup=0,
            )
        if (
            mojo_out is None
            and mojo_cached_out is None
            and mojo_auto_out is None
            and caldera_out is None
        ):
            _, mojo_out = _run_mojo(
                q_q,
                l_q,
                r_q,
                x,
                iters=1,
                warmup=0,
            )
        if caldera_out is not None:
            cmp_out = caldera_out
        else:
            if mojo_out is not None:
                cmp_out = mojo_out
            elif mojo_cached_out is not None:
                cmp_out = mojo_cached_out
            else:
                cmp_out = mojo_auto_out
        diff = np.abs(cmp_out - ref_out)
        val_max = float(diff.max())
        val_mean = float(diff.mean())
        print(f"[validate] max_abs={val_max:.6e} mean_abs={val_mean:.6e}")

    if args.log_csv is not None:
        timestamp = time.time()
        rows = []
        if ref_time is not None:
            rows.append(
                {
                    "timestamp": timestamp,
                    "backend": "reference",
                    "avg_time_s": ref_time,
                    "gflops": _maybe_gflops(out_dim, in_dim, rank, ref_time),
                    "rows": out_dim,
                    "cols": in_dim,
                    "rank": rank,
                    "group_size": group_size,
                    "bits_q": bits_q,
                    "bits_lr": bits_lr,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "seed": args.seed,
                    "preset": args.preset or "",
                    "tag": args.tag,
                    "val_max_abs": val_max,
                    "val_mean_abs": val_mean,
                }
            )
        if mojo_time is not None:
            rows.append(
                {
                    "timestamp": timestamp,
                    "backend": "mojo",
                    "avg_time_s": mojo_time,
                    "gflops": _maybe_gflops(out_dim, in_dim, rank, mojo_time),
                    "rows": out_dim,
                    "cols": in_dim,
                    "rank": rank,
                    "group_size": group_size,
                    "bits_q": bits_q,
                    "bits_lr": bits_lr,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "seed": args.seed,
                    "preset": args.preset or "",
                    "tag": args.tag,
                    "val_max_abs": val_max,
                    "val_mean_abs": val_mean,
                }
            )
        if mojo_cached_time is not None:
            rows.append(
                {
                    "timestamp": timestamp,
                    "backend": "mojo_cached",
                    "avg_time_s": mojo_cached_time,
                    "gflops": _maybe_gflops(out_dim, in_dim, rank, mojo_cached_time),
                    "rows": out_dim,
                    "cols": in_dim,
                    "rank": rank,
                    "group_size": group_size,
                    "bits_q": bits_q,
                    "bits_lr": bits_lr,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "seed": args.seed,
                    "preset": args.preset or "",
                    "tag": args.tag,
                    "val_max_abs": val_max,
                    "val_mean_abs": val_mean,
                }
            )
        if mojo_auto_time is not None:
            rows.append(
                {
                    "timestamp": timestamp,
                    "backend": "mojo_auto",
                    "avg_time_s": mojo_auto_time,
                    "gflops": _maybe_gflops(out_dim, in_dim, rank, mojo_auto_time),
                    "rows": out_dim,
                    "cols": in_dim,
                    "rank": rank,
                    "group_size": group_size,
                    "bits_q": bits_q,
                    "bits_lr": bits_lr,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "seed": args.seed,
                    "preset": args.preset or "",
                    "tag": args.tag,
                    "val_max_abs": val_max,
                    "val_mean_abs": val_mean,
                }
            )
        if caldera_time is not None:
            rows.append(
                {
                    "timestamp": timestamp,
                    "backend": "caldera_cached",
                    "avg_time_s": caldera_time,
                    "gflops": _maybe_gflops(out_dim, in_dim, rank, caldera_time),
                    "rows": out_dim,
                    "cols": in_dim,
                    "rank": rank,
                    "group_size": group_size,
                    "bits_q": bits_q,
                    "bits_lr": bits_lr,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "seed": args.seed,
                    "preset": args.preset or "",
                    "tag": args.tag,
                    "val_max_abs": val_max,
                    "val_mean_abs": val_mean,
                }
            )
        _append_csv(args.log_csv, rows)


if __name__ == "__main__":
    main()
