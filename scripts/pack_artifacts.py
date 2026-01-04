#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.packing import pack_packed


def pack_tensor(values: torch.Tensor, bits: int) -> tuple[torch.Tensor, int, int]:
    packed, pad = pack_packed(values, bits)
    return packed, values.shape[1], pad


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack CALDERA artifacts to 2/4-bit.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Artifacts directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--pack-bits-q", type=int, default=2, help="Pack bits for Q weights.")
    parser.add_argument("--pack-bits-lr", type=int, default=4, help="Pack bits for L/R weights.")
    args = parser.parse_args()

    input_layers = args.input_dir / "layers"
    output_layers = args.output_dir / "layers"
    output_layers.mkdir(parents=True, exist_ok=True)

    for path in input_layers.glob("*.pt"):
        payload = torch.load(path, map_location="cpu")
        meta = payload.get("meta", {})

        q_packed, q_cols, q_pad = pack_tensor(payload["q_values"], args.pack_bits_q)
        l_packed, l_cols, l_pad = pack_tensor(payload["l_values"], args.pack_bits_lr)
        r_packed, r_cols, r_pad = pack_tensor(payload["r_values"], args.pack_bits_lr)

        packed_payload = {
            "q_packed": q_packed,
            "q_packed_bits": args.pack_bits_q,
            "q_packed_cols": q_cols,
            "q_packed_pad": q_pad,
            "l_packed": l_packed,
            "l_packed_bits": args.pack_bits_lr,
            "l_packed_cols": l_cols,
            "l_packed_pad": l_pad,
            "r_packed": r_packed,
            "r_packed_bits": args.pack_bits_lr,
            "r_packed_cols": r_cols,
            "r_packed_pad": r_pad,
            "q_scales": payload["q_scales"],
            "l_scales": payload["l_scales"],
            "r_scales": payload["r_scales"],
            "meta": meta,
        }

        torch.save(packed_payload, output_layers / path.name)


if __name__ == "__main__":
    main()
