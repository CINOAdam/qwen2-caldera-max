#!/bin/bash
# RunPod environment setup - run before any compression jobs
# Belt + suspenders: forces ALL caches and temp writes onto /workspace

set -e

# Detect workspace (network volume mounts at /workspace or /runpod-volume)
if [ -d "/workspace" ] && [ "$(df /workspace | tail -1 | awk '{print $1}')" != "overlay" ]; then
    WORK=/workspace
elif [ -d "/runpod-volume" ]; then
    WORK=/runpod-volume
else
    echo "ERROR: No writable volume found at /workspace or /runpod-volume"
    exit 1
fi

echo "=== RunPod Environment Setup ==="
echo "Using workspace: $WORK"

# Create cache directories on volume
mkdir -p $WORK/.cache/huggingface
mkdir -p $WORK/.cache/huggingface/transformers
mkdir -p $WORK/.cache/huggingface/datasets
mkdir -p $WORK/.cache/torch
mkdir -p $WORK/.cache/triton
mkdir -p $WORK/.cache/pip
mkdir -p $WORK/tmp
mkdir -p $WORK/out

# Redirect all caches to workspace (belt + suspenders)
export TMPDIR=$WORK/tmp
export XDG_CACHE_HOME=$WORK/.cache
export HF_HOME=$WORK/.cache/huggingface
export TRANSFORMERS_CACHE=$WORK/.cache/huggingface/transformers
export HF_DATASETS_CACHE=$WORK/.cache/huggingface/datasets
export TORCH_HOME=$WORK/.cache/torch
export TRITON_CACHE_DIR=$WORK/.cache/triton
export PIP_CACHE_DIR=$WORK/.cache/pip

# Ensure Python output is unbuffered for better logging
export PYTHONUNBUFFERED=1

# Verify setup
echo ""
echo "Cache locations:"
echo "  HF_HOME=$HF_HOME"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  TORCH_HOME=$TORCH_HOME"
echo "  TMPDIR=$TMPDIR"
echo ""
echo "Disk usage:"
df -h / $WORK 2>/dev/null || df -h
echo ""
echo "Inode usage:"
df -i / $WORK 2>/dev/null || df -i
echo ""

# Check Python temp directory
python3 -c "import tempfile; print(f'Python tempdir: {tempfile.gettempdir()}')" 2>/dev/null || true

echo "=== Setup complete. Source this file: source scripts/runpod_setup.sh ==="
