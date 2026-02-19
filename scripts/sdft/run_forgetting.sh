#!/usr/bin/env bash
set -euo pipefail

# Run forgetting eval against base model + SDFT checkpoints.
#
# Usage:
#   BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
#   CHECKPOINTS=outputs/sdft-tooluse/weights/step_50,outputs/sdft-tooluse/weights/step_126 \
#   ./scripts/sdft/run_forgetting.sh

BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL}"
CHECKPOINTS="${CHECKPOINTS:-}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
OUTPUT="${OUTPUT:-results/forgetting}"

echo "=== Forgetting eval ==="
echo "Base model: $BASE_MODEL"
echo "Checkpoints: $CHECKPOINTS"

# Evaluate base model
echo "=== Evaluating base model ==="
uv run inference @ configs/sdft/inference.toml \
    --model.name "$BASE_MODEL" &
INFERENCE_PID=$!

echo "Waiting for inference server to be ready..."
until curl -s "$BASE_URL/models" > /dev/null 2>&1; do
    sleep 5
done

./scripts/sdft/eval_forgetting.sh \
    --base-url "$BASE_URL" \
    --model "$BASE_MODEL" \
    --label base \
    --output "$OUTPUT"

kill $INFERENCE_PID
wait $INFERENCE_PID 2>/dev/null || true

# Evaluate each checkpoint
if [ -n "$CHECKPOINTS" ]; then
    IFS=',' read -ra CKPT_LIST <<< "$CHECKPOINTS"
    for ckpt in "${CKPT_LIST[@]}"; do
        ckpt_name=$(basename "$ckpt")
        echo "=== Evaluating checkpoint: $ckpt_name ==="

        uv run inference @ configs/sdft/inference.toml \
            --model.name "$ckpt" &
        INFERENCE_PID=$!

        echo "Waiting for inference server to be ready..."
        until curl -s "$BASE_URL/models" > /dev/null 2>&1; do
            sleep 5
        done

        ./scripts/sdft/eval_forgetting.sh \
            --base-url "$BASE_URL" \
            --model "$BASE_MODEL" \
            --label "$ckpt_name" \
            --output "$OUTPUT"

        kill $INFERENCE_PID
        wait $INFERENCE_PID 2>/dev/null || true
    done
fi

echo "=== Done ==="
echo "Results in: $OUTPUT"
