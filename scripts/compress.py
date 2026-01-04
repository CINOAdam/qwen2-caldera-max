#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from src.caldera_pipeline import compress_model
from src.calibration import build_calibration
from src.env import load_env_file


def load_config(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Missing dependency: pyyaml. Install with `pip install pyyaml`."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Placeholder entrypoint for CALDERA-style compression."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--skip-compress",
        action="store_true",
        help="Only build calibration set and skip compression.",
    )
    args = parser.parse_args()

    load_env_file(Path(".env"))

    config = load_config(Path(args.config))
    print("Loaded config:")
    print(json.dumps(config, indent=2, sort_keys=True))

    output_dir = Path(config["output_dir"])
    calibration_cfg = config["calibration"]
    stats = build_calibration(
        model_id=config["model_id"],
        dataset_items=calibration_cfg["datasets"],
        output_dir=output_dir,
        samples=int(calibration_cfg["samples"]),
        sequence_length=int(calibration_cfg["sequence_length"]),
        progress_every=int(calibration_cfg.get("progress_every", 200)),
    )

    with (output_dir / "calibration_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)

    print("\nCalibration dataset built. Stats:")
    print(json.dumps(stats, indent=2, sort_keys=True))

    if args.skip_compress:
        print("\nSkipping compression step.")
        return

    compress_model(config)


if __name__ == "__main__":
    main()
