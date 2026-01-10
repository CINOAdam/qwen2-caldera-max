#!/bin/bash
# Enhanced health monitor for long-running jobs on RunPod
# Logs disk/memory/GPU every 60s with detailed breakdown
# Run in background: ./scripts/runpod_health_monitor.sh &

LOGFILE=${1:-/workspace/out/health.log}
INTERVAL=${2:-60}

# Ensure output directory exists
mkdir -p "$(dirname "$LOGFILE")"

echo "Health monitor started. Logging to $LOGFILE every ${INTERVAL}s"
echo "Kill with: pkill -f runpod_health_monitor"

while true; do
    {
        echo "=== $(date -Iseconds) ==="

        echo "-- Disk --"
        df -h / /workspace 2>/dev/null || df -h

        echo "-- Inodes --"
        df -i / /workspace 2>/dev/null || df -i

        echo "-- Workspace breakdown (top 20) --"
        du -sh /workspace/* 2>/dev/null | sort -h | tail -n 20

        echo "-- Artifacts count --"
        find /workspace -name "*.pt" 2>/dev/null | wc -l

        echo "-- Memory --"
        free -h

        echo "-- GPU --"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null || echo "N/A"

        echo "-- Running compression processes --"
        ps aux | grep -E "compress|python" | grep -v grep | head -5

        echo ""
    } >> "$LOGFILE"
    sleep "$INTERVAL"
done
