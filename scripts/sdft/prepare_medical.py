"""Prepare the Medical dataset for SDFT training.

Downloads HuatuoGPT-o1 English medical reasoning data and converts it into
a HuggingFace dataset with `prompt` and `teacher_prompt` columns.

Usage:
    uv run python scripts/sdft/prepare_medical.py \
        --output data/medical_sdft \
        --eval-output data/medical_eval.json
"""

import argparse
import json
from pathlib import Path
from string import Template

from datasets import Dataset, load_dataset


TEACHER_TEMPLATE = Template(
    "$question\n\n"
    "This is an example of a correct response:\n"
    "$response\n\n"
    "Now answer with a response of your own, including your reasoning."
)


def format_example(example: dict) -> dict:
    teacher_prompt = TEACHER_TEMPLATE.substitute(
        question=example["Question"],
        response=example["Response"],
    )
    return {
        "prompt": example["Question"],
        "teacher_prompt": teacher_prompt,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Medical dataset for SDFT")
    parser.add_argument("--output", type=str, required=True, help="Output path for HuggingFace dataset (train)")
    parser.add_argument("--eval-output", type=str, required=True, help="Output path for eval JSON")
    parser.add_argument("--eval-size", type=int, default=1000, help="Number of examples to hold out for eval")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    print("Loading FreedomIntelligence/medical-o1-reasoning-SFT (en)...")
    raw = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    print(f"Loaded {len(raw)} examples")

    dataset = raw.map(format_example, remove_columns=raw.column_names)
    dataset = dataset.shuffle(seed=args.seed)

    train_dataset = dataset.select(range(len(dataset) - args.eval_size))
    eval_dataset = dataset.select(range(len(dataset) - args.eval_size, len(dataset)))

    train_dataset.save_to_disk(args.output)
    print(f"Saved {len(train_dataset)} train examples to {args.output}")

    eval_path = Path(args.eval_output)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    eval_records = []
    for i in range(len(eval_dataset)):
        row = eval_dataset[i]
        raw_row = raw[len(train_dataset) + i]
        eval_records.append({
            "prompt": row["prompt"],
            "reference_answer": raw_row["Response"],
        })

    eval_path.write_text(json.dumps(eval_records, indent=2))
    print(f"Saved {len(eval_records)} eval examples to {eval_path}")

    print(f"\nColumns: {train_dataset.column_names}")
    print(f"Sample prompt[:200]: {train_dataset[0]['prompt'][:200]}")
    print(f"Sample teacher_prompt[:200]: {train_dataset[0]['teacher_prompt'][:200]}")


if __name__ == "__main__":
    main()
