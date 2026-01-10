#!/usr/bin/env python3
"""Launch RunPod for N=100 Goodhart Gap experiment.

Usage:
    python scripts/launch_n100_pod.py --action create
    python scripts/launch_n100_pod.py --action status
    python scripts/launch_n100_pod.py --action stop
    python scripts/launch_n100_pod.py --action delete
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.runpod_client import (
    create_pod,
    delete_pod,
    get_pod,
    list_pods,
    load_runpod_config,
    start_pod,
    stop_pod,
)


def wait_for_ready(pod_id: str, timeout: int = 600) -> dict:
    """Wait for pod to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        pod = get_pod(pod_id)
        status = pod.get("desiredStatus", "")
        runtime = pod.get("runtime", {}) or {}

        print(f"  Status: {status}, Runtime: {bool(runtime)}")

        if status == "RUNNING" and runtime:
            ports = runtime.get("ports", [])
            ssh_port = None
            for p in ports:
                if p.get("privatePort") == 22:
                    ssh_port = p.get("publicPort")
                    break

            if ssh_port:
                ip = runtime.get("gpus", [{}])[0].get("podIp") or pod.get("machine", {}).get("ip", "")
                return {
                    "id": pod_id,
                    "ip": ip,
                    "ssh_port": ssh_port,
                    "ssh_cmd": f"ssh -p {ssh_port} root@{ip}",
                }

        time.sleep(10)

    raise TimeoutError(f"Pod {pod_id} did not become ready in {timeout}s")


def main():
    parser = argparse.ArgumentParser(description="Manage RunPod for N=100 experiment")
    parser.add_argument("--action", required=True, choices=["create", "status", "stop", "delete", "list"])
    parser.add_argument("--pod-id", default=None, help="Pod ID (for status/stop/delete)")
    parser.add_argument("--config", default="configs/runpod_n100_experiment.yaml")
    args = parser.parse_args()

    if args.action == "list":
        pods = list_pods()
        if not pods:
            print("No pods found")
        for pod in pods:
            status = pod.get("desiredStatus", "UNKNOWN")
            name = pod.get("name", "unnamed")
            print(f"  {pod['id']}: {name} [{status}]")
        return

    if args.action == "create":
        config = load_runpod_config(Path(args.config))
        print(f"Creating pod: {config.name}")
        print(f"  GPUs: {config.gpu_count}x {config.gpu_type_ids}")
        print(f"  Volume: {config.volume_in_gb}GB")

        result = create_pod(config)
        pod_id = result.get("id")
        print(f"  Pod ID: {pod_id}")

        print("Waiting for pod to be ready...")
        info = wait_for_ready(pod_id)

        print(f"\n{'='*50}")
        print("POD READY")
        print(f"{'='*50}")
        print(f"Pod ID: {info['id']}")
        print(f"SSH: {info['ssh_cmd']}")
        print(f"\nNext steps:")
        print(f"  1. SSH into pod: {info['ssh_cmd']}")
        print(f"  2. Clone repo: git clone https://github.com/your-repo/qwen2-caldera-max.git")
        print(f"  3. Setup: cd qwen2-caldera-max && pip install -e .")
        print(f"  4. Run: bash scripts/run_n100_experiment.sh")
        return

    pod_id = args.pod_id
    if not pod_id:
        pods = list_pods()
        if len(pods) == 1:
            pod_id = pods[0]["id"]
            print(f"Using pod: {pod_id}")
        else:
            print("Multiple pods found. Specify --pod-id")
            for pod in pods:
                print(f"  {pod['id']}: {pod.get('name', 'unnamed')}")
            return

    if args.action == "status":
        pod = get_pod(pod_id)
        print(f"Pod: {pod_id}")
        print(f"  Status: {pod.get('desiredStatus')}")
        runtime = pod.get("runtime", {}) or {}
        if runtime:
            gpus = runtime.get("gpus", [])
            print(f"  GPUs: {len(gpus)}")
            for gpu in gpus:
                print(f"    - {gpu.get('gpuTypeId', 'unknown')}")

    elif args.action == "stop":
        print(f"Stopping pod: {pod_id}")
        stop_pod(pod_id)
        print("Stopped")

    elif args.action == "delete":
        print(f"Deleting pod: {pod_id}")
        delete_pod(pod_id)
        print("Deleted")


if __name__ == "__main__":
    main()
