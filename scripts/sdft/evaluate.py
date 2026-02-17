"""Generic evaluation script for SDFT-trained models.

Reuses scoring.py to evaluate any task type (mcq, tooluse, exact match).
Generates completions via an OpenAI-compatible API, scores each using
score_completion(), and computes accuracy, pass@k, and average score.

Usage:
    uv run python scripts/sdft/evaluate.py \
        --test-data ../SDPO/datasets/sciknoweval/biology/test.json \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-8B \
        --num-completions 16 \
        --temperature 0.0 \
        --output results/biology_baseline.json
"""

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from prime_rl.trainer.sdft.scoring import score_completion


async def generate_completions(
    client: AsyncOpenAI,
    prompt: str,
    system: str | None,
    model: str,
    max_tokens: int,
    temperature: float,
    num_completions: int,
) -> list[str]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    tasks = [
        client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for _ in range(num_completions)
    ]
    responses = await asyncio.gather(*tasks)
    return [r.choices[0].message.content or "" for r in responses]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al. 2021).

    n: total completions, c: number correct, k: desired pass@k.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model using scoring.py")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test JSONL/JSON file")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--num-completions", type=int, default=1, help="Completions per prompt")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=16, help="Max concurrent API calls")
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="HTTP timeout per request")
    args = parser.parse_args()

    test_path = Path(args.test_data)
    with open(test_path) as f:
        text = f.read()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = [data]
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]

    client = AsyncOpenAI(base_url=args.base_url, api_key="unused", timeout=args.timeout_seconds)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def eval_one(idx: int, example: dict) -> dict:
        prompt = example["prompt"]
        answer = example["answer"]
        kind = example.get("kind")
        system = example.get("system")

        async with semaphore:
            completions = await generate_completions(
                client, prompt, system, args.model,
                args.max_tokens, args.temperature, args.num_completions,
            )

        scores = [score_completion(c, answer, kind) for c in completions]
        score_vals = [s["score"] for s in scores]
        best_score = max(score_vals)

        return {
            "idx": idx,
            "prompt": prompt[:200],
            "answer": answer,
            "kind": kind,
            "scores": score_vals,
            "best_score": best_score,
        }

    async def run_all():
        tasks = [eval_one(i, ex) for i, ex in enumerate(data)]
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            done = len(results)
            if done % 50 == 0 or done == len(data):
                acc = sum(1 for r in results if r["best_score"] >= 1.0) / done
                print(f"Progress: {done}/{len(data)} | Running accuracy: {acc:.1%}")
        return sorted(results, key=lambda r: r["idx"])

    samples = asyncio.run(run_all())

    # Compute metrics
    n = args.num_completions
    correct_counts = [sum(1 for s in sample["scores"] if s >= 1.0) for sample in samples]
    accuracy = sum(1 for c in correct_counts if c > 0) / len(samples)
    avg_score = sum(s for sample in samples for s in sample["scores"]) / (len(samples) * n)

    metrics = {
        "accuracy": round(accuracy, 4),
        "avg_score": round(avg_score, 4),
        "total": len(samples),
    }

    if n > 1:
        for k in [1, 4, 8, 16]:
            if k <= n:
                pk = sum(pass_at_k(n, c, k) for c in correct_counts) / len(samples)
                metrics[f"pass_at_{k}"] = round(pk, 4)

    # Derive dataset name from path
    parts = test_path.parts
    dataset_name = "/".join(parts[-3:-1]) if len(parts) >= 3 else test_path.stem

    output_data = {
        "label": args.label,
        "model": args.model,
        "dataset": dataset_name,
        "num_completions": n,
        "temperature": args.temperature,
        "metrics": metrics,
        "samples": samples,
    }

    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {args.model}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
