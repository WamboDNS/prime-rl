#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-allenai/OLMo-3-7B-Think}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-$PRIME_API_KEY}"

echo "=== 1. Install dependencies ==="
uv sync --all-extras

echo "=== 2. Prepare dataset (skip if exists) ==="
if [ ! -d "data/medical_sdft" ]; then
    uv run python scripts/sdft/prepare_medical.py \
        --output data/medical_sdft \
        --eval-output data/medical_eval.json
else
    echo "Dataset already exists, skipping."
fi

echo "=== 3. Start inference server for baseline eval ==="
uv run inference @ configs/sdft/inference.toml \
    --model.name "$MODEL" --model.max_model_len 8192 &
INFERENCE_PID=$!

echo "Waiting for inference server to be ready..."
until curl -s "$BASE_URL/models" > /dev/null 2>&1; do
    sleep 5
done
echo "Inference server ready."

echo "=== 4. Baseline eval ==="
uv run python scripts/sdft/eval_medical.py \
    --eval-data data/medical_eval.json \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --max-tokens 4096 \
    --judge-api-key "$JUDGE_API_KEY" \
    --report-token-stats \
    --label baseline --output results/medical_olmo_baseline.json

echo "=== 5. Kill inference server ==="
kill $INFERENCE_PID
wait $INFERENCE_PID 2>/dev/null || true

echo "=== 6. Start SDFT training ==="
uv run sdft @ configs/sdft/medical_olmo.toml

echo "=== 7. Start inference server for SDFT eval ==="
uv run inference @ configs/sdft/inference.toml \
    --model.name outputs/sdft-medical-olmo/weights/step_585 \
    --model.max_model_len 8192 &
INFERENCE_PID=$!

echo "Waiting for inference server to be ready..."
until curl -s "$BASE_URL/models" > /dev/null 2>&1; do
    sleep 5
done
echo "Inference server ready."

echo "=== 8. SDFT eval ==="
uv run python scripts/sdft/eval_medical.py \
    --eval-data data/medical_eval.json \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --max-tokens 4096 \
    --judge-api-key "$JUDGE_API_KEY" \
    --report-token-stats \
    --label sdft --output results/medical_olmo_sdft.json

echo "=== 9. Kill inference server ==="
kill $INFERENCE_PID
wait $INFERENCE_PID 2>/dev/null || true

echo "=== Done ==="
echo "Baseline: $(python -c "import json; d=json.load(open('results/medical_olmo_baseline.json')); print(f\"{d['correct']}/{d['total']} = {d['accuracy']}%\")")"
echo "SDFT:     $(python -c "import json; d=json.load(open('results/medical_olmo_sdft.json')); print(f\"{d['correct']}/{d['total']} = {d['accuracy']}%\")")"
