#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.runpod_client import (
    build_update_payload,
    create_pod,
    delete_pod,
    get_pod,
    list_pods,
    load_runpod_config,
    reset_pod,
    restart_pod,
    start_pod,
    stop_pod,
    update_pod,
)
from src.env import load_env_file


def _redact_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        redacted = {}
        for key, value in payload.items():
            if key == "env" and isinstance(value, dict):
                redacted[key] = {k: "***" for k in value}
            else:
                redacted[key] = _redact_payload(value)
        return redacted
    if isinstance(payload, list):
        return [_redact_payload(item) for item in payload]
    return payload


def _write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_redact_payload(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _maybe_state_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage RunPod pods via REST API.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_list = subparsers.add_parser("list", help="List pods")
    parser_list.add_argument("--output", help="Optional path to write JSON output")

    parser_create = subparsers.add_parser("create", help="Create a pod")
    parser_create.add_argument("--config", required=True, help="Path to runpod YAML config")
    parser_create.add_argument("--state", help="Optional path to write pod state JSON")

    parser_get = subparsers.add_parser("get", help="Get pod by id")
    parser_get.add_argument("pod_id", help="Pod ID")
    parser_get.add_argument("--output", help="Optional path to write JSON output")

    parser_update = subparsers.add_parser("update", help="Update pod image/env")
    parser_update.add_argument("pod_id", help="Pod ID")
    parser_update.add_argument("--config", required=True, help="Path to runpod YAML config")
    parser_update.add_argument("--output", help="Optional path to write JSON output")

    for name in ("start", "stop", "restart", "reset", "delete"):
        sub = subparsers.add_parser(name, help=f"{name.capitalize()} a pod")
        sub.add_argument("pod_id", help="Pod ID")
        sub.add_argument("--output", help="Optional path to write JSON output")

    args = parser.parse_args()
    load_env_file(Path(".env"))

    if args.command == "list":
        pods = list_pods()
        if args.output:
            _write_state(Path(args.output), pods)
        else:
            print(json.dumps(_redact_payload(pods), indent=2, sort_keys=True))
        return

    if args.command == "create":
        config = load_runpod_config(Path(args.config))
        payload = create_pod(config)
        state_path = _maybe_state_path(args.state)
        if state_path:
            _write_state(state_path, payload)
        print(json.dumps(_redact_payload(payload), indent=2, sort_keys=True))
        return

    if args.command == "get":
        payload = get_pod(args.pod_id)
        if args.output:
            _write_state(Path(args.output), payload)
        else:
            print(json.dumps(_redact_payload(payload), indent=2, sort_keys=True))
        return

    if args.command == "update":
        config = load_runpod_config(Path(args.config))
        payload = update_pod(args.pod_id, build_update_payload(config))
        if args.output:
            _write_state(Path(args.output), payload)
        else:
            print(json.dumps(_redact_payload(payload), indent=2, sort_keys=True))
        return

    action_map = {
        "start": start_pod,
        "stop": stop_pod,
        "restart": restart_pod,
        "reset": reset_pod,
        "delete": delete_pod,
    }

    payload = action_map[args.command](args.pod_id)
    if args.output:
        _write_state(Path(args.output), payload)
    else:
        print(json.dumps(_redact_payload(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
