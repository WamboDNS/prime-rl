#!/usr/bin/env bash
# End-to-end SDPO experiment pipeline.
#
# Usage:
#   ./scripts/sdft/run_sdpo.sh <config> <dataset_dir> [extra_args...]
#
# Example:
#   ./scripts/sdft/run_sdpo.sh configs/sdft/generalization.toml ../SDPO/datasets/sciknoweval/biology
#   ./scripts/sdft/run_sdpo.sh configs/sdft/generalization.toml ../SDPO/datasets/sciknoweval/biology --trainer.model.name=allenai/Olmo-3-7B-Instruct
#
# Steps:
#   1. Start inference server → baseline eval on test.json
#   2. Stop inference server
#   3. Run SDFT training
#   4. Start inference with trained checkpoint → eval on test.json
#   5. Print comparison table

set -euo pipefail

CONFIG="${1:?Usage: $0 <config> <dataset_dir> [extra_args...]}"
DATASET_DIR="${2:?Usage: $0 <config> <dataset_dir> [extra_args...]}"
shift 2
EXTRA_ARGS=("$@")

# Derive experiment name from dataset path (e.g. "biology", "tooluse")
NAME=$(basename "$DATASET_DIR")
TEST_DATA="$DATASET_DIR/test.json"
TRAIN_DATA="$DATASET_DIR/train.json"
RESULTS_DIR="results/${NAME}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"

# Extract config values using Python (handles TOML correctly)
read -r MODEL INFERENCE_GPUS OUTPUT_DIR MAX_MODEL_LEN PORT < <(python3 -c "
import tomllib
with open('$CONFIG', 'rb') as f:
    cfg = tomllib.load(f)
model = cfg.get('trainer', {}).get('model', {}).get('name', 'Qwen/Qwen3-8B')
gpus = ','.join(str(g) for g in cfg.get('inference_gpu_ids', [0]))
output_dir = cfg.get('trainer', {}).get('output_dir', 'outputs')
max_model_len = cfg.get('inference', {}).get('model', {}).get('max_model_len', '')
port = cfg.get('inference', {}).get('server', {}).get('port', 8000)
print(model, gpus, output_dir, max_model_len, port)
")

mkdir -p "$RESULTS_DIR"

echo "=================================================="
echo "  SDPO Experiment: $NAME"
echo "  Config: $CONFIG"
echo "  Model: $MODEL"
echo "  Train: $TRAIN_DATA"
echo "  Test:  $TEST_DATA"
echo "=================================================="

if [ ! -f "$TEST_DATA" ]; then
    echo "ERROR: Test data not found: $TEST_DATA"
    exit 1
fi
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Train data not found: $TRAIN_DATA"
    exit 1
fi

wait_for_server() {
    echo "Waiting for inference server..."
    for i in $(seq 1 120); do
        if curl -s "$BASE_URL/models" > /dev/null 2>&1; then
            echo "Server ready."
            return 0
        fi
        sleep 5
    done
    echo "ERROR: Server did not start within 10 minutes"
    exit 1
}

start_inference() {
    local model_name="$1"
    local inference_args=(
        --model.name "$model_name"
        --server.port "$PORT"
    )
    if [ -n "$MAX_MODEL_LEN" ]; then
        inference_args+=(--model.max-model-len "$MAX_MODEL_LEN")
    fi
    CUDA_VISIBLE_DEVICES="$INFERENCE_GPUS" uv run inference "${inference_args[@]}" &
    INFERENCE_PID=$!
    wait_for_server
}

stop_inference() {
    kill $INFERENCE_PID
    wait $INFERENCE_PID 2>/dev/null || true
}

# === 1. Baseline evaluation ===
echo ""
echo "=== 1. Starting inference server (baseline) ==="
start_inference "$MODEL"

echo ""
echo "=== 2. Baseline evaluation ==="
uv run python scripts/sdft/evaluate.py \
    --test-data "$TEST_DATA" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --num-completions 1 \
    --temperature 0.0 \
    --label baseline \
    --output "$RESULTS_DIR/baseline.json"

echo ""
echo "=== 3. Stopping inference server ==="
stop_inference

# === 2. Training ===
echo ""
echo "=== 4. Starting SDFT training ==="
uv run sdft @ "$CONFIG" \
    --trainer.data.dataset_name="$TRAIN_DATA" \
    "${EXTRA_ARGS[@]}"

# Find latest checkpoint
LATEST_STEP=$(cat "$OUTPUT_DIR/weights/latest_step.txt" 2>/dev/null || ls -1 "$OUTPUT_DIR/weights/" | sort -n | tail -1)
CHECKPOINT="$OUTPUT_DIR/weights/$LATEST_STEP"

if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found at $CHECKPOINT"
    exit 1
fi
echo "Using checkpoint: $CHECKPOINT"

# === 3. Trained model evaluation ===
echo ""
echo "=== 5. Starting inference server (trained) ==="
start_inference "$CHECKPOINT"

echo ""
echo "=== 6. Trained model evaluation ==="
uv run python scripts/sdft/evaluate.py \
    --test-data "$TEST_DATA" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --num-completions 1 \
    --temperature 0.0 \
    --label trained \
    --output "$RESULTS_DIR/trained.json"

echo ""
echo "=== 7. Stopping inference server ==="
stop_inference

# === 4. Print comparison ===
echo ""
echo "=================================================="
echo "  Results: $NAME"
echo "=================================================="
python3 -c "
import json
baseline = json.load(open('$RESULTS_DIR/baseline.json'))
trained = json.load(open('$RESULTS_DIR/trained.json'))
bm = baseline['metrics']
tm = trained['metrics']
print(f\"{'Metric':<20} {'Baseline':>10} {'Trained':>10} {'Delta':>10}\")
print('-' * 52)
for key in bm:
    if key == 'total':
        continue
    bv = bm[key]
    tv = tm.get(key, 'N/A')
    if isinstance(bv, (int, float)) and isinstance(tv, (int, float)):
        delta = tv - bv
        sign = '+' if delta >= 0 else ''
        print(f'{key:<20} {bv:>10.4f} {tv:>10.4f} {sign}{delta:>9.4f}')
    else:
        print(f'{key:<20} {bv!s:>10} {tv!s:>10}')
print(f\"{'total':<20} {bm['total']:>10}\")
"
echo ""
echo "Results saved to $RESULTS_DIR/"
