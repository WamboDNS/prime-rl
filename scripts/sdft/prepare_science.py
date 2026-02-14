"""Prepare the Science Q&A dataset for SDFT training.

Downloads SciKnowEval Chemistry L3 questions, generates demonstrations
for each via an LLM API, and saves a dual-prompt dataset for self-distillation.

Supports resuming from a checkpoint file to avoid re-doing API calls.

Usage:
    uv run python scripts/sdft/prepare_science.py \
        --output data/science_sdft \
        --test-output data/science_test.json \
        --val-output data/science_val.json
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
    label_to_text = dict(zip(choices["label"], choices["text"]))
    return "\n".join(f"{label}. {label_to_text[label]}" for label in labels)


def extract_answer_letter(text: str) -> str | None:
    """Extract a single answer letter (A-D) from model output."""
    patterns = [
        r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:\s]*\(?([A-D])\)?",
        r"(?:Answer|ANSWER)\s*:\s*\(?([A-D])\)?",
        r"\*\*([A-D])\*\*\s*$",
        r"\b([A-D])\s*$",
        r"^\s*\(?([A-D])\)?\s*[.\)]",
        r"\\boxed\{([A-D])\}",
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
    parser.add_argument("--api-base-url", type=str, default="https://api.pinference.ai/api/v1", help="API base URL")
    parser.add_argument("--demo-model", type=str, default="openai/gpt-4.1-mini", help="Model for generating demos")
    parser.add_argument("--num-samples", type=int, default=8, help="Max attempts per question for demo generation")
    parser.add_argument("--checkpoint", type=str, default="data/science_checkpoint.jsonl", help="Checkpoint file for resuming")
    args = parser.parse_args()

    print("Loading SciKnowEval (v2, test split)...")
    raw = load_dataset("hicai-zju/SciKnowEval", "v2", split="test")
    print(f"Loaded {len(raw)} examples")

    filtered = [ex for ex in raw if ex["domain"] == "Chemistry" and ex["details"]["level"] == "L3"]
    print(f"Filtered to {len(filtered)} Chemistry L3 examples")

    # Load checkpoint if it exists
    checkpoint_path = Path(args.checkpoint)
    completed = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                entry = json.loads(line)
                completed[entry["index"]] = entry
        print(f"Resumed from checkpoint: {len(completed)} examples already processed")

    api_key = args.api_key or os.environ.get("PRIME_API_KEY")
    client = OpenAI(base_url=args.api_base_url, api_key=api_key)

    records = []
    skipped = 0

    # Collect already-completed successes
    for entry in completed.values():
        if entry.get("demo"):
            records.append(entry["record"])

    checkpoint_file = open(checkpoint_path, "a")

    for i, ex in enumerate(filtered):
        if i in completed:
            continue

        choices_str = format_choices(ex["choices"])
        answer_key = ex["answerKey"]

        demo = generate_demo(client, args.demo_model, ex["question"], choices_str, answer_key, args.num_samples)

        entry = {"index": i, "demo": demo is not None}
        if demo is None:
            skipped += 1
            print(f"[{i}] Skipped (no correct demo after {args.num_samples} attempts)")
        else:
            student_prompt = f"{ex['question']}\n\n{choices_str}\n\nState your final answer as a single letter (A/B/C/D)."
            teacher_prompt = (
                f"{ex['question']}\n\n{choices_str}\n\n"
                f"This is an example of a correct response:\n{demo}\n\n"
                "Now answer with a response of your own. State your final answer as a single letter (A/B/C/D)."
            )
            record = {"prompt": student_prompt, "teacher_prompt": teacher_prompt, "answerKey": answer_key}
            entry["record"] = record
            records.append(record)

        checkpoint_file.write(json.dumps(entry) + "\n")
        checkpoint_file.flush()

        if (i + 1) % 20 == 0:
            total_done = len(completed) + (i - len(completed) + 1)
            print(f"Progress: {total_done}/{len(filtered)} | kept {len(records)} | skipped {skipped}")

    checkpoint_file.close()
    skipped += sum(1 for e in completed.values() if not e.get("demo"))
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
