from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .env import load_env_file

API_BASE = "https://rest.runpod.io/v1"


@dataclass(frozen=True)
class RunpodConfig:
    name: str
    cloud_type: str
    compute_type: str
    gpu_type_ids: list[str]
    gpu_type_priority: str
    gpu_count: int
    interruptible: bool
    image_name: str
    volume_in_gb: int | None
    container_disk_in_gb: int | None
    volume_mount_path: str
    ports: list[str]
    min_vcpu_per_gpu: int | None
    min_ram_per_gpu: int | None
    support_public_ip: bool | None
    docker_start_cmd: list[str]
    env: dict[str, str]
    data_center_ids: list[str] | None
    country_codes: list[str] | None


def _require_token() -> str:
    token = os.getenv("RUNPOD_API_KEY")
    if not token:
        raise SystemExit("RUNPOD_API_KEY is missing from the environment.")
    return token


def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    token = _require_token()
    url = f"{API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Runpod API error {exc.code}: {body}") from exc
    if not raw:
        return None
    return json.loads(raw)


def list_pods() -> list[dict[str, Any]]:
    return _request("GET", "/pods")


def get_pod(pod_id: str) -> dict[str, Any]:
    return _request("GET", f"/pods/{pod_id}")


def create_pod(config: RunpodConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": config.name,
        "cloudType": config.cloud_type,
        "computeType": config.compute_type,
        "gpuTypeIds": config.gpu_type_ids,
        "gpuTypePriority": config.gpu_type_priority,
        "gpuCount": config.gpu_count,
        "interruptible": config.interruptible,
        "imageName": config.image_name,
        "ports": config.ports,
        "volumeInGb": config.volume_in_gb,
        "containerDiskInGb": config.container_disk_in_gb,
        "volumeMountPath": config.volume_mount_path,
        "minVCPUPerGPU": config.min_vcpu_per_gpu,
        "minRAMPerGPU": config.min_ram_per_gpu,
        "supportPublicIp": config.support_public_ip,
        "dockerStartCmd": config.docker_start_cmd,
        "env": config.env,
    }
    if config.data_center_ids:
        payload["dataCenterIds"] = config.data_center_ids
    if config.country_codes:
        payload["countryCodes"] = config.country_codes

    return _request("POST", "/pods", payload)


def build_update_payload(config: RunpodConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "imageName": config.image_name,
        "ports": config.ports,
        "volumeInGb": config.volume_in_gb,
        "containerDiskInGb": config.container_disk_in_gb,
        "volumeMountPath": config.volume_mount_path,
        "dockerStartCmd": config.docker_start_cmd,
        "env": config.env,
        "locked": False,
    }


def update_pod(pod_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _request("POST", f"/pods/{pod_id}/update", payload)


def start_pod(pod_id: str) -> dict[str, Any]:
    return _request("POST", f"/pods/{pod_id}/start")


def stop_pod(pod_id: str) -> dict[str, Any]:
    return _request("POST", f"/pods/{pod_id}/stop")


def restart_pod(pod_id: str) -> dict[str, Any]:
    return _request("POST", f"/pods/{pod_id}/restart")


def reset_pod(pod_id: str) -> dict[str, Any]:
    return _request("POST", f"/pods/{pod_id}/reset")


def delete_pod(pod_id: str) -> dict[str, Any]:
    return _request("DELETE", f"/pods/{pod_id}")


def _interpolate_env(values: dict[str, Any]) -> dict[str, str]:
    interpolated: dict[str, str] = {}
    pattern = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
    for key, value in values.items():
        raw = str(value)
        match = pattern.match(raw)
        if match:
            env_key = match.group(1)
            env_value = os.getenv(env_key)
            if env_value is None:
                raise SystemExit(f"Missing env var for {key}: {env_key}")
            interpolated[key] = env_value
        else:
            interpolated[key] = raw
    return interpolated


def load_runpod_config(path: Path) -> RunpodConfig:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: pyyaml. Install with `pip install pyyaml`."
        ) from exc

    load_env_file(Path(".env"))

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    env_values = data.get("env", {}) or {}

    return RunpodConfig(
        name=data["name"],
        cloud_type=data.get("cloud_type", "SECURE"),
        compute_type=data.get("compute_type", "GPU"),
        gpu_type_ids=list(data.get("gpu_type_ids") or []),
        gpu_type_priority=data.get("gpu_type_priority", "availability"),
        gpu_count=int(data.get("gpu_count", 1)),
        interruptible=bool(data.get("interruptible", False)),
        image_name=data["image_name"],
        volume_in_gb=data.get("volume_in_gb"),
        container_disk_in_gb=data.get("container_disk_in_gb"),
        volume_mount_path=data.get("volume_mount_path", "/workspace"),
        ports=list(data.get("ports") or []),
        min_vcpu_per_gpu=data.get("min_vcpu_per_gpu"),
        min_ram_per_gpu=data.get("min_ram_per_gpu"),
        support_public_ip=data.get("support_public_ip"),
        docker_start_cmd=list(data.get("docker_start_cmd") or []),
        env=_interpolate_env(env_values),
        data_center_ids=data.get("data_center_ids"),
        country_codes=data.get("country_codes"),
    )
