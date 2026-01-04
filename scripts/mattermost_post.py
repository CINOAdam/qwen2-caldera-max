#!/usr/bin/env python3
"""Create channels and post updates to Mattermost using env credentials."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone


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


def _create_channel(base: str, token: str, team_id: str, name: str, display_name: str, channel_type: str):
    payload = {
        "team_id": team_id,
        "name": name,
        "display_name": display_name,
        "type": channel_type,
    }
    return _api_request(base, token, "POST", "/api/v4/channels", payload)


def _add_member(base: str, token: str, channel_id: str, user_id: str):
    payload = {"user_id": user_id}
    return _api_request(base, token, "POST", f"/api/v4/channels/{channel_id}/members", payload)


def _create_post(base: str, token: str, channel_id: str, message: str):
    payload = {"channel_id": channel_id, "message": message}
    return _api_request(base, token, "POST", "/api/v4/posts", payload)


def _get_posts(base: str, token: str, channel_id: str, limit: int):
    return _api_request(
        base,
        token,
        "GET",
        f"/api/v4/channels/{channel_id}/posts?per_page={limit}",
    )


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def cmd_create_channel(args: argparse.Namespace) -> int:
    base = _get_env("MATTERMOST_URL")
    token = _get_env("MATTERMOST_CATEGORY_TOKEN")
    team_id = _get_env("MATTERMOST_TEAM_ID")

    channel = _get_channel_by_name(base, token, team_id, args.name)
    if channel is None:
        channel = _create_channel(base, token, team_id, args.name, args.display_name, args.channel_type)
    channel_id = channel["id"]

    if args.auto_add:
        for user_id in _split_csv(os.getenv("MATTERMOST_AUTO_ADD_USERS")):
            try:
                _add_member(base, token, channel_id, user_id)
            except ApiError:
                pass

    print(channel_id)
    return 0


def cmd_post(args: argparse.Namespace) -> int:
    base = _get_env("MATTERMOST_URL")
    token = _get_env("MATTERMOST_CATEGORY_TOKEN")
    team_id = _get_env("MATTERMOST_TEAM_ID")

    channel_id = args.channel_id
    if channel_id is None:
        channel = _get_channel_by_name(base, token, team_id, args.channel_name)
        if channel is None:
            raise RuntimeError(f"Channel '{args.channel_name}' not found.")
        channel_id = channel["id"]

    _create_post(base, token, channel_id, args.message)
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    base = _get_env("MATTERMOST_URL")
    token = _get_env("MATTERMOST_CATEGORY_TOKEN")
    team_id = _get_env("MATTERMOST_TEAM_ID")

    channel_id = args.channel_id
    if channel_id is None:
        channel = _get_channel_by_name(base, token, team_id, args.channel_name)
        if channel is None:
            raise RuntimeError(f"Channel '{args.channel_name}' not found.")
        channel_id = channel["id"]

    response = _get_posts(base, token, channel_id, args.limit)
    posts = response.get("posts", {})
    order = response.get("order", [])

    for post_id in reversed(order):
        post = posts.get(post_id)
        if not post:
            continue
        created = datetime.fromtimestamp(post["create_at"] / 1000, tz=timezone.utc)
        stamp = created.strftime("%Y-%m-%d %H:%M:%S UTC")
        message = post.get("message", "").replace("\n", " ")
        print(f"[{stamp}] {message}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Mattermost helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create-channel", help="Create a channel if needed.")
    create_parser.add_argument("--name", required=True, help="Channel name (lowercase, no spaces).")
    create_parser.add_argument("--display-name", required=True, help="Channel display name.")
    create_parser.add_argument("--channel-type", default="O", choices=("O", "P"), help="O=public, P=private.")
    create_parser.add_argument("--auto-add", action="store_true", help="Auto-add users from env.")
    create_parser.set_defaults(func=cmd_create_channel)

    post_parser = subparsers.add_parser("post", help="Post a message to a channel.")
    post_parser.add_argument("--channel-id", help="Channel ID to post to.")
    post_parser.add_argument("--channel-name", help="Channel name to look up.")
    post_parser.add_argument("--message", required=True, help="Message text.")
    post_parser.set_defaults(func=cmd_post)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch recent posts from a channel.")
    fetch_parser.add_argument("--channel-id", help="Channel ID to fetch from.")
    fetch_parser.add_argument("--channel-name", help="Channel name to look up.")
    fetch_parser.add_argument("--limit", type=int, default=20, help="Number of posts to fetch.")
    fetch_parser.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    try:
        return args.func(args)
    except (RuntimeError, ApiError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
