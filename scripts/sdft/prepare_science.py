"""Prepare the Science Q&A dataset for SDFT training.

Downloads SciKnowEval Chemistry L3 questions, generates GPT-4o demonstrations
for each, and saves a dual-prompt dataset for self-distillation.

Usage:
    uv run python scripts/sdft/prepare_science.py \
        --output data/science_sdft \
        --test-output data/science_test.json \
        --val-output data/science_val.json \
        --api-key $PRIME_API_KEY
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

from datasets import Dataset, load_dataset
from openai import OpenAI


def format_choices(choices: dict) -> str:
    """Format answer choices as 'A. text\\nB. text\\n...'."""
    labels = sorted(choices["label"])
    texts = choices["text"]
    label_to_text = dict(zip(choices["label"], texts))
    return "\n".join(f"{label}. {label_to_text[label]}" for label in labels)


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


def generate_demo(client: OpenAI, model: str, question: str, choices_str: str, answer_key: str, num_samples: int) -> str | None:
    """Generate a demonstration response that gets the correct answer."""
    demo_prompt = (
        f"{question}\n\n{choices_str}\n\n"
        "Think through this step by step, then state your final answer as a single letter (A/B/C/D)."
    )
    for _ in range(num_samples):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": demo_prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        text = response.choices[0].message.content or ""
        extracted = extract_answer_letter(text)
        if extracted == answer_key:
            return text
    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare Science Q&A dataset for SDFT")
    parser.add_argument("--output", type=str, required=True, help="Output path for HuggingFace dataset (train)")
    parser.add_argument("--test-output", type=str, required=True, help="Output path for test JSON")
    parser.add_argument("--val-output", type=str, required=True, help="Output path for val JSON")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--api-key", type=str, default=None, help="API key (defaults to PRIME_API_KEY env var)")
    parser.add_argument("--api-base-url", type=str, default="https://api.primeintellect.ai/v1", help="API base URL")
    parser.add_argument("--demo-model", type=str, default="gpt-4o", help="Model for generating demos")
    parser.add_argument("--num-samples", type=int, default=8, help="Max attempts per question for demo generation")
    args = parser.parse_args()

    print("Loading SciKnowEval (v2, test split)...")
    raw = load_dataset("hicai-zju/SciKnowEval", "v2", split="test")
    print(f"Loaded {len(raw)} examples")

    filtered = [ex for ex in raw if ex["domain"] == "Chemistry" and ex["details"]["level"] == "L3"]
    print(f"Filtered to {len(filtered)} Chemistry L3 examples")

    api_key = args.api_key or os.environ.get("PRIME_API_KEY")
    client = OpenAI(base_url=args.api_base_url, api_key=api_key)

    records = []
    skipped = 0
    for i, ex in enumerate(filtered):
        choices_str = format_choices(ex["choices"])
        answer_key = ex["answerKey"]

        demo = generate_demo(client, args.demo_model, ex["question"], choices_str, answer_key, args.num_samples)
        if demo is None:
            skipped += 1
            print(f"[{i}] Skipped (no correct demo after {args.num_samples} attempts)")
            continue

        student_prompt = f"{ex['question']}\n\n{choices_str}\n\nState your final answer as a single letter (A/B/C/D)."
        teacher_prompt = (
            f"{ex['question']}\n\n{choices_str}\n\n"
            f"This is an example of a correct response:\n{demo}\n\n"
            "Now answer with a response of your own. State your final answer as a single letter (A/B/C/D)."
        )

        records.append({
            "prompt": student_prompt,
            "teacher_prompt": teacher_prompt,
            "answerKey": answer_key,
        })

        if (i + 1) % 20 == 0:
            print(f"Progress: {i+1}/{len(filtered)} | kept {len(records)} | skipped {skipped}")

    print(f"\nTotal: {len(records)} examples (skipped {skipped})")

    random.seed(args.seed)
    random.shuffle(records)

    n = len(records)
    n_train = int(n * 0.75)
    n_val = int(n * 0.05)

    train_records = records[:n_train]
    val_records = records[n_train:n_train + n_val]
    test_records = records[n_train + n_val:]

    train_dataset = Dataset.from_list([{"prompt": r["prompt"], "teacher_prompt": r["teacher_prompt"]} for r in train_records])
    train_dataset.save_to_disk(args.output)
    print(f"Saved {len(train_dataset)} train examples to {args.output}")

    for path, data, label in [
        (args.val_output, val_records, "val"),
        (args.test_output, test_records, "test"),
    ]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        eval_data = [{"prompt": r["prompt"], "answerKey": r["answerKey"]} for r in data]
        p.write_text(json.dumps(eval_data, indent=2))
        print(f"Saved {len(eval_data)} {label} examples to {p}")

    print(f"\nColumns: {train_dataset.column_names}")
    print(f"Sample prompt[:200]: {train_dataset[0]['prompt'][:200]}")
    print(f"Sample teacher_prompt[:200]: {train_dataset[0]['teacher_prompt'][:200]}")


if __name__ == "__main__":
    main()
