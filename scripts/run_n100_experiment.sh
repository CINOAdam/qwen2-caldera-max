#!/bin/bash
# N=100 Goodhart Gap Experiment on RunPod
# Run this script on the pod after setup
set -e

# Configuration
MODEL_ID="Qwen/Qwen2-72B-Instruct"
NUM_RUNS=5  # 20 tests × 5 runs = 100 evaluations
TEMPERATURE=0.3
SEED=42
RESULTS_DIR="/workspace/results"
REPO_DIR="/workspace/qwen2-caldera-max"

# CALDERA artifact paths (on RunPod volume)
CALDERA_UNIFORM="/workspace/artifacts/qwen2-72b/caldera-4bit-uniform"
CALDERA_SELECTIVE="/workspace/artifacts/qwen2-72b/caldera-4bit-selective"
S3_BUCKET="s3://caldera-artifacts-20260107"

# Setup environment
source /workspace/qwen2-caldera-max/scripts/runpod_setup.sh

# Sync CALDERA artifacts from S3 if not present
sync_artifacts() {
    local name=$1
    local s3_path=$2
    local local_path=$3

    if [ ! -d "$local_path/layers" ]; then
        echo "Syncing $name artifacts from S3..."
        mkdir -p "$local_path"
        aws s3 sync "$s3_path" "$local_path" --quiet
        echo "  Done: $(ls $local_path/layers/*.pt 2>/dev/null | wc -l) layer files"
    else
        echo "$name artifacts already present: $(ls $local_path/layers/*.pt 2>/dev/null | wc -l) layers"
    fi
}

echo "=== Syncing CALDERA artifacts from S3 ==="
sync_artifacts "4-bit Uniform" "${S3_BUCKET}/qwen2-72b-4bit-uniform" "$CALDERA_UNIFORM"
sync_artifacts "4-bit Selective" "${S3_BUCKET}/qwen2-72b-4bit-selective/qwen2-72b/caldera-4bit-selective" "$CALDERA_SELECTIVE"

cd $REPO_DIR

# Create results directory
mkdir -p $RESULTS_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "==========================================="
echo "N=100 Goodhart Gap Experiment"
echo "==========================================="
echo "Model: $MODEL_ID"
echo "Runs per test: $NUM_RUNS"
echo "Temperature: $TEMPERATURE"
echo "Timestamp: $TIMESTAMP"
echo "==========================================="

# Function to run evaluation
run_eval() {
    local name=$1
    local caldera_dir=$2
    local output_file="${RESULTS_DIR}/goodhart_n100_${name}_${TIMESTAMP}.json"

    echo ""
    echo ">>> Running: $name"
    echo ">>> Output: $output_file"

    if [ -z "$caldera_dir" ]; then
        python scripts/eval_goodhart_gap.py \
            --model-id "$MODEL_ID" \
            --device-map auto \
            --num-runs $NUM_RUNS \
            --temperature $TEMPERATURE \
            --seed $SEED \
            --output "$output_file"
    else
        python scripts/eval_goodhart_gap.py \
            --model-id "$MODEL_ID" \
            --caldera-dir "$caldera_dir" \
            --device-map auto \
            --num-runs $NUM_RUNS \
            --temperature $TEMPERATURE \
            --seed $SEED \
            --output "$output_file"
    fi

    echo ">>> Completed: $name"
    echo ""
}

# Run all 3 conditions
echo ""
echo "=== PHASE 1: Baseline (no compression) ==="
run_eval "baseline" ""

echo ""
echo "=== PHASE 2: 4-bit Uniform Compression ==="
run_eval "4bit_uniform" "$CALDERA_UNIFORM"

echo ""
echo "=== PHASE 3: 4-bit Selective Compression ==="
run_eval "4bit_selective" "$CALDERA_SELECTIVE"

# Generate summary
echo ""
echo "==========================================="
echo "EXPERIMENT COMPLETE"
echo "==========================================="
echo "Results saved to: $RESULTS_DIR"
ls -la $RESULTS_DIR/goodhart_n100_*_${TIMESTAMP}.json

# Create quick summary
python3 << EOF
import json
import glob

files = sorted(glob.glob("${RESULTS_DIR}/goodhart_n100_*_${TIMESTAMP}.json"))
print("\n=== QUICK SUMMARY ===")
print(f"{'Condition':<20} {'Understands':<12} {'Executes':<10} {'Gaps':<8} {'DBDK Rate':<10}")
print("-" * 60)

for f in files:
    with open(f) as fp:
        data = json.load(fp)
    s = data['summary']
    name = f.split('/')[-1].replace('goodhart_n100_', '').replace('_${TIMESTAMP}.json', '')
    print(f"{name:<20} {s['understands']}/{s['total']} ({100*s['understands']/s['total']:.0f}%)   "
          f"{s['executes']}/{s['total']} ({100*s['executes']/s['total']:.0f}%)   "
          f"{s['gaps']:<8} {100*s['dbdk_rate']:.0f}%")
EOF

echo ""
echo "Don't forget to copy results back before terminating the pod!"
echo "  scp -r root@<pod-ip>:${RESULTS_DIR}/goodhart_n100_*_${TIMESTAMP}.json ./results/"
