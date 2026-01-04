from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

from src.q_lr_reference import QuantizedGroupwise, q_lr_forward, quantize_groupwise


_LIB_NAME = "libq_lr_fused_gpu.so"


def _lib_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "artifacts" / "mojo" / _LIB_NAME


def _build_lib() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_q_lr_kernels.sh"
    subprocess.run([str(script)], check=True)


def _load_lib() -> ctypes.CDLL:
    lib_path = _lib_path()
    if not lib_path.exists():
        _build_lib()
    lib = ctypes.CDLL(str(lib_path))
    func = lib.q_lr_fused_host
    func.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    ]
    func.restype = ctypes.c_int64
    func_tiled = getattr(lib, "q_lr_fused_host_tiled", None)
    if func_tiled is not None:
        func_tiled.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        func_tiled.restype = ctypes.c_int64
    func_batched = getattr(lib, "q_lr_fused_host_batched", None)
    if func_batched is not None:
        func_batched.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        func_batched.restype = ctypes.c_int64
    return lib


def q_lr_fused(
    q_weight: QuantizedGroupwise,
    l_weight: QuantizedGroupwise,
    r_weight: QuantizedGroupwise,
    x: np.ndarray,
    *,
    use_tiled: bool = True,
) -> np.ndarray:
    if q_weight.group_size != l_weight.group_size or q_weight.group_size != r_weight.group_size:
        raise ValueError("Q/L/R group_size must match for fused kernel.")

    q_vals = np.ascontiguousarray(q_weight.values.astype(np.int32))
    q_scales = np.ascontiguousarray(q_weight.scales.astype(np.float32))
    l_vals = np.ascontiguousarray(l_weight.values.astype(np.int32))
    l_scales = np.ascontiguousarray(l_weight.scales.astype(np.float32))
    r_vals = np.ascontiguousarray(r_weight.values.astype(np.int32))
    r_scales = np.ascontiguousarray(r_weight.scales.astype(np.float32))
    x = np.ascontiguousarray(x.astype(np.float32))

    if q_vals.ndim != 2 or l_vals.ndim != 2 or r_vals.ndim != 2:
        raise ValueError("Q/L/R values must be 2D arrays.")

    rows, cols = q_vals.shape
    rank, r_cols = r_vals.shape
    if l_vals.shape != (rows, rank):
        raise ValueError("L shape must be (rows, rank).")
    if r_cols != cols:
        raise ValueError("R shape must be (rank, cols).")
    if x.shape != (cols,):
        raise ValueError("x must be a 1D vector of shape (cols,).")

    group_size = int(q_weight.group_size)
    q_groups = int(q_scales.shape[1])
    l_groups = int(l_scales.shape[1])
    r_groups = int(r_scales.shape[1])

    out = np.empty((rows,), dtype=np.float32)
    lib = _load_lib()
    func = getattr(lib, "q_lr_fused_host_tiled", None) if use_tiled else None
    if func is None:
        func = lib.q_lr_fused_host
    status = func(
        q_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        q_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        l_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        l_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        r_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        r_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(rows),
        ctypes.c_int64(cols),
        ctypes.c_int64(rank),
        ctypes.c_int64(group_size),
        ctypes.c_int64(q_groups),
        ctypes.c_int64(l_groups),
        ctypes.c_int64(r_groups),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if status != 0:
        raise RuntimeError(f"q_lr_fused_host failed with status {status}")
    return out


def q_lr_fused_batched(
    q_weight: QuantizedGroupwise,
    l_weight: QuantizedGroupwise,
    r_weight: QuantizedGroupwise,
    x: np.ndarray,
    *,
    iters: int,
) -> np.ndarray:
    if iters < 1:
        raise ValueError("iters must be >= 1")
    q_vals = np.ascontiguousarray(q_weight.values.astype(np.int32))
    q_scales = np.ascontiguousarray(q_weight.scales.astype(np.float32))
    l_vals = np.ascontiguousarray(l_weight.values.astype(np.int32))
    l_scales = np.ascontiguousarray(l_weight.scales.astype(np.float32))
    r_vals = np.ascontiguousarray(r_weight.values.astype(np.int32))
    r_scales = np.ascontiguousarray(r_weight.scales.astype(np.float32))
    x = np.ascontiguousarray(x.astype(np.float32))

    rows, cols = q_vals.shape
    rank, r_cols = r_vals.shape
    if l_vals.shape != (rows, rank):
        raise ValueError("L shape must be (rows, rank).")
    if r_cols != cols:
        raise ValueError("R shape must be (rank, cols).")
    if x.shape != (cols,):
        raise ValueError("x must be a 1D vector of shape (cols,).")

    group_size = int(q_weight.group_size)
    q_groups = int(q_scales.shape[1])
    l_groups = int(l_scales.shape[1])
    r_groups = int(r_scales.shape[1])

    out = np.empty((rows,), dtype=np.float32)
    lib = _load_lib()
    func = getattr(lib, "q_lr_fused_host_batched", None)
    if func is None:
        for _ in range(iters):
            out = q_lr_fused(q_weight, l_weight, r_weight, x, use_tiled=True)
        return out

    status = func(
        q_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        q_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        l_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        l_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        r_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        r_scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(rows),
        ctypes.c_int64(cols),
        ctypes.c_int64(rank),
        ctypes.c_int64(group_size),
        ctypes.c_int64(q_groups),
        ctypes.c_int64(l_groups),
        ctypes.c_int64(r_groups),
        ctypes.c_int64(iters),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if status != 0:
        raise RuntimeError(f"q_lr_fused_host_batched failed with status {status}")
    return out


def _should_use_fused(
    rows: int,
    cols: int,
    rank: int,
    *,
    min_dim: int = 2048,
    fuse_rank_threshold: int = 128,
    batch: int = 1,
    seq: int = 1,
    max_tokens: int = 128,
) -> bool:
    if rank > fuse_rank_threshold:
        return False
    if rows * cols < min_dim * min_dim:
        return False
    return batch * seq <= max_tokens


def q_lr_auto(
    q_weight: QuantizedGroupwise,
    l_weight: QuantizedGroupwise,
    r_weight: QuantizedGroupwise,
    x: np.ndarray,
    *,
    batch: int = 1,
    seq: int = 1,
    use_tiled: bool = True,
    allow_fused: bool = True,
) -> np.ndarray:
    rows, cols = q_weight.values.shape
    rank, r_cols = r_weight.values.shape
    if r_cols != cols:
        raise ValueError("R shape must be (rank, cols).")
    if allow_fused and _should_use_fused(
        rows,
        cols,
        rank,
        batch=batch,
        seq=seq,
    ):
        try:
            return q_lr_fused(q_weight, l_weight, r_weight, x, use_tiled=use_tiled)
        except Exception:
            pass
    return q_lr_forward(q_weight, l_weight, r_weight, x)


def q_lr_fused_from_float(
    q: np.ndarray,
    l: np.ndarray,
    r: np.ndarray,
    x: np.ndarray,
    *,
    group_size: int,
    bits_q: int,
    bits_lr: int,
    use_tiled: bool = True,
) -> np.ndarray:
    q_weight = quantize_groupwise(q, group_size, bits_q)
    l_weight = quantize_groupwise(l, group_size, bits_lr)
    r_weight = quantize_groupwise(r, group_size, bits_lr)
    return q_lr_fused(q_weight, l_weight, r_weight, x, use_tiled=use_tiled)
