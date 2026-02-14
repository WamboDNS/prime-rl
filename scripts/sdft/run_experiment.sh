#!/usr/bin/env bash
set -euo pipefail

EVAL_DATA="${EVAL_DATA:-../Self-Distillation/data/tooluse_data/eval_data.json}"
TRAIN_DATA="${TRAIN_DATA:-../Self-Distillation/data/tooluse_data/train_data.json}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"

echo "=== 1. Install dependencies ==="
uv sync --all-extras

echo "=== 2. Prepare dataset ==="
uv run python scripts/sdft/prepare_tooluse.py \
    --input "$TRAIN_DATA" \
    --output data/tooluse_sdft

echo "=== 3. Start inference server for baseline eval ==="
uv run inference @ configs/sdft/inference.toml &
INFERENCE_PID=$!

echo "Waiting for inference server to be ready..."
until curl -s "$BASE_URL/models" > /dev/null 2>&1; do
    sleep 5
done
echo "Inference server ready."

echo "=== 4. Baseline eval ==="
uv run python scripts/sdft/eval_tooluse.py \
    --eval-data "$EVAL_DATA" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --label baseline --output results/baseline.json

echo "=== 5. Kill inference server ==="
kill $INFERENCE_PID
wait $INFERENCE_PID 2>/dev/null || true

echo "=== 6. Start SDFT training ==="
uv run sdft @ configs/sdft/tooluse.toml

echo "=== 7. Start inference server for SDFT eval ==="
uv run inference @ configs/sdft/inference.toml \
    --model.name outputs/sdft-tooluse/weights/step_126 &
INFERENCE_PID=$!

echo "Waiting for inference server to be ready..."
until curl -s "$BASE_URL/models" > /dev/null 2>&1; do
    sleep 5
done
echo "Inference server ready."

echo "=== 8. SDFT eval ==="
uv run python scripts/sdft/eval_tooluse.py \
    --eval-data "$EVAL_DATA" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --label sdft --output results/sdft.json

echo "=== 9. Kill inference server ==="
kill $INFERENCE_PID
wait $INFERENCE_PID 2>/dev/null || true

echo "=== Done ==="
echo "Baseline: $(python -c "import json; d=json.load(open('results/baseline.json')); print(f\"{d['correct']}/{d['total']} = {d['accuracy']}%\")")"
echo "SDFT:     $(python -c "import json; d=json.load(open('results/sdft.json')); print(f\"{d['correct']}/{d['total']} = {d['accuracy']}%\")")"
