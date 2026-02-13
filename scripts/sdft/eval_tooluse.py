"""Evaluate tool-use accuracy for the Self-Distillation paper reproduction.

Generates completions via an OpenAI-compatible API, parses Action/Action Input
from model output, and compares against golden answers with JSON normalization.

Usage:
    uv run python scripts/sdft/eval_tooluse.py \
        --eval-data /path/to/eval_data.json \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import re

from openai import OpenAI


def parse_actions(text: str) -> list[dict]:
    """Parse Action and Action Input pairs from model output."""
    actions = []
    # Match lines like "Action: toolName" followed by "Action Input: {...}"
    pattern = re.compile(
        r"Action:\s*(.+?)\s*\n\s*Action Input:\s*(.+?)(?:\n|$)",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        action_name = match.group(1).strip()
        action_input_raw = match.group(2).strip()
        actions.append({"Action": action_name, "Action_Input": action_input_raw})
    return actions


def normalize_json(s: str) -> dict | list | str:
    """Parse JSON string, returning the parsed object or the original string on failure."""
    s = s.strip()
    if not s:
        return {}
    for candidate in [s, s.rstrip(",")]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return s


def actions_match(predicted: list[dict], golden: list[dict]) -> bool:
    """Check if predicted actions match golden actions (order-sensitive)."""
    if len(predicted) != len(golden):
        return False
    for pred, gold in zip(predicted, golden):
        if pred["Action"] != gold["Action"]:
            return False
        pred_input = normalize_json(pred["Action_Input"])
        gold_input = normalize_json(gold["Action_Input"])
        if pred_input != gold_input:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Evaluate tool-use accuracy")
    parser.add_argument("--eval-data", type=str, required=True, help="Path to eval_data.json")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate")
    args = parser.parse_args()

    with open(args.eval_data) as f:
        eval_data = json.load(f)

    client = OpenAI(base_url=args.base_url, api_key="unused")

    correct = 0
    total = len(eval_data)

    for i, example in enumerate(eval_data):
        prompt = example["prompt"]
        golden = example["golden_answer"]

        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            temperature=0,
        )
        completion = response.choices[0].message.content or ""

        predicted = parse_actions(completion)
        match = actions_match(predicted, golden)

        if match:
            correct += 1
        else:
            print(f"[{i}] WRONG | predicted {len(predicted)} actions, expected {len(golden)}")
            if predicted:
                print(f"       pred: {predicted[0]['Action']}({predicted[0]['Action_Input']})")
            print(f"       gold: {golden[0]['Action']}({golden[0]['Action_Input']})")

    accuracy = correct / total * 100
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")


if __name__ == "__main__":
    main()
