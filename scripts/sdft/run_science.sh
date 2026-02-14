#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
PRIME_API_KEY="${PRIME_API_KEY:?Set PRIME_API_KEY for GPT-4o demo generation}"

echo "=== 1. Install dependencies ==="
uv sync --all-extras

echo "=== 2. Prepare dataset ==="
uv run python scripts/sdft/prepare_science.py \
    --output data/science_sdft \
    --test-output data/science_test.json \
    --val-output data/science_val.json \
    --api-key "$PRIME_API_KEY"

echo "=== 3. Start inference server for baseline eval ==="
uv run inference @ configs/sdft/inference.toml &
INFERENCE_PID=$!

echo "Waiting for inference server to be ready..."
until curl -s "$BASE_URL/models" > /dev/null 2>&1; do
    sleep 5
done
echo "Inference server ready."

echo "=== 4. Baseline eval ==="
uv run python scripts/sdft/eval_science.py \
    --eval-data data/science_test.json \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --label baseline --output results/science_baseline.json

echo "=== 5. Kill inference server ==="
kill $INFERENCE_PID
wait $INFERENCE_PID 2>/dev/null || true

echo "=== 6. Start SDFT training ==="
# NOTE: update --trainer.max_steps based on actual dataset size
uv run sdft @ configs/sdft/science.toml

echo "=== 7. Find latest checkpoint ==="
LAST_CKPT=$(ls -d outputs/sdft-science/weights/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
if [ -z "$LAST_CKPT" ]; then
    echo "ERROR: No checkpoint found in outputs/sdft-science/weights/"
    exit 1
fi
echo "Using checkpoint: $LAST_CKPT"

echo "=== 8. Start inference server for SDFT eval ==="
uv run inference @ configs/sdft/inference.toml \
    --model.name "$LAST_CKPT" &
INFERENCE_PID=$!

echo "Waiting for inference server to be ready..."
until curl -s "$BASE_URL/models" > /dev/null 2>&1; do
    sleep 5
done
echo "Inference server ready."

echo "=== 9. SDFT eval ==="
uv run python scripts/sdft/eval_science.py \
    --eval-data data/science_test.json \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --label sdft --output results/science_sdft.json

echo "=== 10. Kill inference server ==="
kill $INFERENCE_PID
wait $INFERENCE_PID 2>/dev/null || true

echo "=== Done ==="
echo "Baseline: $(python -c "import json; d=json.load(open('results/science_baseline.json')); print(f\"{d['correct']}/{d['total']} = {d['accuracy']}%\")")"
echo "SDFT:     $(python -c "import json; d=json.load(open('results/science_sdft.json')); print(f\"{d['correct']}/{d['total']} = {d['accuracy']}%\")")"
