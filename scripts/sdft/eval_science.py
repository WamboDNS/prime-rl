"""Evaluate Science Q&A with MCQ exact-match accuracy.

Generates completions via an OpenAI-compatible API (vLLM), extracts the
answer letter from model output, and compares to the golden answerKey.

Usage:
    uv run python scripts/sdft/eval_science.py \
        --eval-data data/science_test.json \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-7B-Instruct \
        --label baseline --output results/science_baseline.json
"""

import argparse
import json
import re
from pathlib import Path

from openai import OpenAI


def extract_answer_letter(text: str) -> str | None:
    """Extract a single answer letter (A-D) from model output."""
    patterns = [
        r"(?:the\s+)?answer\s+is\s*[:\s]*([A-D])",
        r"Answer\s*:\s*([A-D])",
        r"\b([A-D])\s*$",
        r"^\s*([A-D])\s*[.\)]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Science Q&A accuracy")
    parser.add_argument("--eval-data", type=str, required=True, help="Path to eval JSON")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="vLLM API base URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--label", type=str, default=None, help="Label for this run")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    args = parser.parse_args()

    with open(args.eval_data) as f:
        eval_data = json.load(f)

    client = OpenAI(base_url=args.base_url, api_key="unused")

    correct = 0
    total = len(eval_data)
    samples = []

    for i, example in enumerate(eval_data):
        prompt = example["prompt"]
        answer_key = example["answerKey"]

        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            temperature=0,
        )
        completion = response.choices[0].message.content or ""

        predicted = extract_answer_letter(completion)
        match = predicted == answer_key

        if match:
            correct += 1
        else:
            print(f"[{i}] WRONG | predicted={predicted} expected={answer_key}")

        samples.append({
            "index": i,
            "correct": match,
            "predicted": predicted,
            "expected": answer_key,
            "completion": completion,
        })

    accuracy = correct / total * 100
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "label": args.label,
            "model": args.model,
            "accuracy": round(accuracy, 1),
            "correct": correct,
            "total": total,
            "samples": samples,
        }
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
