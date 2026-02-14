#!/usr/bin/env bash
set -euo pipefail

# Evaluate forgetting via lm-evaluation-harness benchmarks.
# Requires lm-evaluation-harness to be installed separately.
#
# Usage:
#   ./scripts/sdft/eval_forgetting.sh \
#       --base-url http://localhost:8000/v1 \
#       --model Qwen/Qwen2.5-7B-Instruct \
#       --label baseline \
#       --output results/forgetting

BASE_URL=""
MODEL=""
LABEL=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-url) BASE_URL="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$BASE_URL" ] || [ -z "$MODEL" ] || [ -z "$LABEL" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 --base-url URL --model MODEL --label LABEL --output DIR"
    exit 1
fi

OUTPUT_DIR="${OUTPUT}/${LABEL}"
mkdir -p "$OUTPUT_DIR"

echo "Running lm-evaluation-harness for $LABEL ($MODEL)..."

lm_eval --model local-chat-completions \
    --tasks hellaswag,truthfulqa_mc2,mmlu,ifeval,winogrande,humaneval \
    --model_args "model=$MODEL,base_url=${BASE_URL}/chat/completions,num_concurrent=32" \
    --batch_size auto \
    --output_path "$OUTPUT_DIR" \
    --log_samples

echo "Results saved to $OUTPUT_DIR"
