"""Prepare the Tool Use dataset for SDFT training.

Converts raw JSON from the Self-Distillation paper into a HuggingFace dataset
with `prompt` and `teacher_prompt` columns.

Usage:
    uv run python scripts/sdft/prepare_tooluse.py \
        --input /path/to/train_data.json \
        --output /path/to/output_dataset
"""

import argparse
from string import Template

from datasets import Dataset


TEACHER_TEMPLATE = Template(
    "$orig_content\n\n"
    "This is an example for a response to the question:\n"
    "$output_text\n\n"
    "Now answer with a response of your own, including the thinking process."
)


def format_example(example: dict) -> dict:
    teacher_prompt = TEACHER_TEMPLATE.substitute(
        orig_content=example["prompt"],
        output_text="\n".join(example["golden_response"]),
    )
    return {
        "prompt": example["prompt"],
        "teacher_prompt": teacher_prompt,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Tool Use dataset for SDFT")
    parser.add_argument("--input", type=str, required=True, help="Path to train_data.json")
    parser.add_argument("--output", type=str, required=True, help="Output path for HuggingFace dataset")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    dataset = Dataset.from_json(args.input)
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=args.seed)

    dataset.save_to_disk(args.output)
    print(f"Saved {len(dataset)} examples to {args.output}")
    print(f"Columns: {dataset.column_names}")
    print(f"Sample prompt[:100]: {dataset[0]['prompt'][:100]}")
    print(f"Sample teacher_prompt[:100]: {dataset[0]['teacher_prompt'][:100]}")


if __name__ == "__main__":
    main()
