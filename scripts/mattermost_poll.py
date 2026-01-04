#!/usr/bin/env python3
"""Poll Mattermost channel for new posts and append to a local log."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


class ApiError(RuntimeError):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _api_request(base: str, token: str, method: str, path: str, payload: dict | None = None):
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            if not body:
                return None
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        message = body
        if body:
            try:
                message = json.loads(body).get("message", body)
            except json.JSONDecodeError:
                message = body
        raise ApiError(exc.code, message) from exc
    except urllib.error.URLError as exc:
        raise ApiError(None, str(exc)) from exc


def _get_channel_by_name(base: str, token: str, team_id: str, name: str):
    try:
        return _api_request(base, token, "GET", f"/api/v4/teams/{team_id}/channels/name/{name}")
    except ApiError as exc:
        if exc.status == 404:
            return None
        raise


def _get_posts_since(base: str, token: str, channel_id: str, since_ms: int):
    return _api_request(
        base,
        token,
        "GET",
        f"/api/v4/channels/{channel_id}/posts?since={since_ms}",
    )


def _create_post(base: str, token: str, channel_id: str, message: str):
    payload = {"channel_id": channel_id, "message": message}
    return _api_request(base, token, "POST", "/api/v4/posts", payload)


def _load_state(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return int(data.get("last_seen_ms", 0))


def _save_state(path: Path, last_seen_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"last_seen_ms": int(last_seen_ms)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _format_post(post: dict) -> str:
    created = datetime.fromtimestamp(post["create_at"] / 1000, tz=timezone.utc)
    stamp = created.strftime("%Y-%m-%d %H:%M:%S UTC")
    user_id = post.get("user_id", "unknown")
    message = post.get("message", "").replace("\n", " ")
    return f"[{stamp}] user={user_id} {message}"


def _emit(line: str, log_path: Path | None) -> None:
    print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll Mattermost for new posts.")
    parser.add_argument("--channel-id", help="Channel ID to poll.")
    parser.add_argument("--channel-name", help="Channel name to look up.")
    parser.add_argument("--interval", type=float, default=30.0, help="Polling interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Run a single poll and exit.")
    parser.add_argument(
        "--state",
        default="artifacts/mattermost_poll_state.json",
        help="Path to persist last seen timestamp.",
    )
    parser.add_argument(
        "--log",
        default="artifacts/mattermost_poll.log",
        help="Path to append new messages (empty to disable).",
    )
    parser.add_argument(
        "--since-ms",
        type=int,
        default=None,
        help="Override initial last-seen timestamp in ms.",
    )
    parser.add_argument(
        "--auto-ack",
        action="store_true",
        help="Post a short acknowledgement for new messages.",
    )
    parser.add_argument(
        "--ack-message",
        default="Message received. I will respond as soon as I can.",
        help="Acknowledgement message text.",
    )
    args = parser.parse_args()

    base = _get_env("MATTERMOST_URL")
    token = _get_env("MATTERMOST_CATEGORY_TOKEN")
    team_id = _get_env("MATTERMOST_TEAM_ID")
    self_user_id = os.getenv("MATTERMOST_CATEGORY_USER_ID")

    channel_id = args.channel_id
    if channel_id is None:
        if not args.channel_name:
            raise RuntimeError("Provide --channel-id or --channel-name.")
        channel = _get_channel_by_name(base, token, team_id, args.channel_name)
        if channel is None:
            raise RuntimeError(f"Channel '{args.channel_name}' not found.")
        channel_id = channel["id"]

    state_path = Path(args.state)
    log_path = Path(args.log) if args.log else None
    last_seen = _load_state(state_path)
    if last_seen is None:
        if args.since_ms is not None:
            last_seen = args.since_ms
        else:
            last_seen = int(time.time() * 1000)
        _save_state(state_path, last_seen)

    while True:
        response = _get_posts_since(base, token, channel_id, last_seen)
        posts = response.get("posts", {}) if response else {}
        order = response.get("order", []) if response else []
        new_max = last_seen
        ack_prefix = "[codex-auto-ack]"
        ignore_prefixes = (ack_prefix, "[codex]")
        for post_id in reversed(order):
            post = posts.get(post_id)
            if not post:
                continue
            created_at = int(post.get("create_at", 0))
            if created_at > new_max:
                new_max = created_at
            _emit(_format_post(post), log_path)
            if args.auto_ack:
                message = post.get("message", "")
                if any(message.startswith(prefix) for prefix in ignore_prefixes):
                    continue
                if self_user_id and post.get("user_id") == self_user_id:
                    continue
                ack = f"{ack_prefix} {args.ack_message}"
                _create_post(base, token, channel_id, ack)
        if new_max > last_seen:
            last_seen = new_max
            _save_state(state_path, last_seen)

        if args.once:
            break
        time.sleep(max(args.interval, 1.0))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ApiError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        raise SystemExit(1)
