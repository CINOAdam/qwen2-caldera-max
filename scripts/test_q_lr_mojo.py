from __future__ import annotations

import numpy as np

from src.q_lr_mojo import q_lr_fused
from src.q_lr_reference import q_lr_forward, quantize_groupwise


def main() -> None:
    rng = np.random.default_rng(0)
    rows = 128
    cols = 256
    rank = 32
    group_size = 64
    bits_q = 2
    bits_lr = 4

    q = rng.normal(scale=0.02, size=(rows, cols)).astype(np.float32)
    l = rng.normal(scale=0.02, size=(rows, rank)).astype(np.float32)
    r = rng.normal(scale=0.02, size=(rank, cols)).astype(np.float32)
    x = rng.normal(scale=1.0, size=(cols,)).astype(np.float32)

    q_weight = quantize_groupwise(q, group_size, bits_q)
    l_weight = quantize_groupwise(l, group_size, bits_lr)
    r_weight = quantize_groupwise(r, group_size, bits_lr)

    ref = q_lr_forward(q_weight, l_weight, r_weight, x)
    out = q_lr_fused(q_weight, l_weight, r_weight, x)

    max_err = float(np.max(np.abs(out - ref)))
    mean_err = float(np.mean(np.abs(out - ref)))
    print(f"max_abs={max_err:.6e} mean_abs={mean_err:.6e}")
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    main()
