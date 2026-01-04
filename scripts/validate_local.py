#!/usr/bin/env python3
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Placeholder local validation entrypoint."
    )
    parser.add_argument("--model", required=True, help="Path to compressed artifacts")
    args = parser.parse_args()

    print(f"TODO: Load model from {args.model} and run a small eval.")


if __name__ == "__main__":
    main()
