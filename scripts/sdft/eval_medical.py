"""Evaluate medical QA using an LLM judge.

Generates completions via an OpenAI-compatible API (vLLM), then uses GPT-5-mini
as a judge to assess correctness against reference answers.

Usage:
    uv run python scripts/sdft/eval_medical.py \
        --eval-data data/medical_eval.json \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-7B-Instruct \
        --judge-api-key $PRIME_API_KEY \
        --label baseline --output results/medical_baseline.json
"""

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI


JUDGE_PROMPT = """You are evaluating the correctness of a medical answer.

Question:
{question}

Reference answer:
{reference}

Model answer:
{answer}

Is the model's answer medically correct and consistent with the reference answer? \
Consider the key medical facts, diagnosis, reasoning, and recommendations.

Reply with exactly one word: CORRECT or INCORRECT."""


def judge_answer(client: OpenAI, model: str, question: str, reference: str, answer: str) -> bool:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, reference=reference, answer=answer,
        )}],
        max_tokens=8,
        temperature=0,
    )
    verdict = response.choices[0].message.content or ""
    return "CORRECT" in verdict.upper()


def main():
    parser = argparse.ArgumentParser(description="Evaluate medical QA with LLM judge")
    parser.add_argument("--eval-data", type=str, required=True, help="Path to eval JSON")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="vLLM API base URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name for generation")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--judge-model", type=str, default="gpt-5-mini", help="Model for judging")
    parser.add_argument("--judge-api-key", type=str, default=None, help="API key for judge (defaults to PRIME_API_KEY env var)")
    parser.add_argument("--judge-base-url", type=str, default="https://api.primeintellect.ai/v1", help="API base URL for judge")
    parser.add_argument("--label", type=str, default=None, help="Label for this run")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    parser.add_argument("--report-token-stats", action="store_true", help="Report average completion token count")
    args = parser.parse_args()

    with open(args.eval_data) as f:
        eval_data = json.load(f)

    gen_client = OpenAI(base_url=args.base_url, api_key="unused")
    judge_api_key = args.judge_api_key or os.environ.get("PRIME_API_KEY")
    judge_client = OpenAI(base_url=args.judge_base_url, api_key=judge_api_key)

    correct = 0
    total = len(eval_data)
    total_tokens = 0
    samples = []

    for i, example in enumerate(eval_data):
        prompt = example["prompt"]
        reference = example["reference_answer"]

        response = gen_client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            temperature=0,
        )
        completion = response.choices[0].message.content or ""
        completion_tokens = response.usage.completion_tokens if response.usage else len(completion.split())

        is_correct = judge_answer(judge_client, args.judge_model, prompt, reference, completion)

        if is_correct:
            correct += 1
        else:
            print(f"[{i}] INCORRECT")

        total_tokens += completion_tokens
        samples.append({
            "index": i,
            "correct": is_correct,
            "completion": completion,
            "completion_tokens": completion_tokens,
        })

        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{total} | Running accuracy: {correct/(i+1)*100:.1f}%")

    accuracy = correct / total * 100
    avg_tokens = total_tokens / total
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")
    if args.report_token_stats:
        print(f"Avg completion tokens: {avg_tokens:.1f}")

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
        if args.report_token_stats:
            results["avg_tokens"] = round(avg_tokens, 1)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
