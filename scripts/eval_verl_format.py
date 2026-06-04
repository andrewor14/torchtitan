"""Evaluate RL checkpoints using the same prompt format as verl training.

Uses vLLM for fast generation with the chat template that triggers thinking mode.
Scores answers using the same regex extraction as verl's reward function.
"""
import argparse
import json
import os
import re

import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def extract_answer_gsm8k(text: str) -> str | None:
    """Extract answer after #### from GSM8K-style responses."""
    match = re.search(r"####\s*(.+?)(?:\s*$|\n)", text)
    if match:
        return match.group(1).strip().replace(",", "")
    return None


def extract_answer_math(text: str) -> str | None:
    """Extract answer from \\boxed{} in MATH-style responses."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip().lower()
    answer = answer.replace("$", "").replace(",", "").replace(" ", "")
    return answer


def check_answer(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False
    return normalize_answer(predicted) == normalize_answer(ground_truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to HF checkpoint")
    parser.add_argument("--data", default="/home/andrewor/data/gsm8k_math/test.parquet")
    parser.add_argument("--dataset", choices=["gsm8k", "math", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Max examples to eval")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    if args.dataset == "gsm8k":
        df = df[df["data_source"] == "openai/gsm8k"]
    elif args.dataset == "math":
        df = df[df["data_source"] != "openai/gsm8k"]

    if args.limit:
        df = df.head(args.limit)

    print(f"Evaluating {len(df)} examples from {args.checkpoint}")
    print(f"Dataset: {args.dataset}, GSM8K: {(df['data_source'] == 'openai/gsm8k').sum()}, MATH: {(df['data_source'] != 'openai/gsm8k').sum()}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    llm = LLM(
        model=args.checkpoint,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        enforce_eager=True,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    # Build prompts using the chat template (triggers thinking mode)
    prompts = []
    for _, row in df.iterrows():
        messages = json.loads(row["prompt"]) if isinstance(row["prompt"], str) else row["prompt"]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    outputs = llm.generate(prompts, sampling_params)

    # Score
    gsm8k_correct = 0
    gsm8k_total = 0
    math_correct = 0
    math_total = 0

    for i, (output, (_, row)) in enumerate(zip(outputs, df.iterrows())):
        response = output.outputs[0].text
        gt = json.loads(row["reward_model"]) if isinstance(row["reward_model"], str) else row["reward_model"]
        ground_truth = gt["ground_truth"]

        is_gsm8k = row["data_source"] == "openai/gsm8k"
        if is_gsm8k:
            predicted = extract_answer_gsm8k(response)
            gsm8k_total += 1
            if check_answer(predicted, ground_truth):
                gsm8k_correct += 1
        else:
            predicted = extract_answer_math(response)
            math_total += 1
            if check_answer(predicted, ground_truth):
                math_correct += 1

    total_correct = gsm8k_correct + math_correct
    total = gsm8k_total + math_total

    print(f"\n=== Results for {args.checkpoint} ===")
    if gsm8k_total > 0:
        print(f"GSM8K: {gsm8k_correct}/{gsm8k_total} = {gsm8k_correct/gsm8k_total*100:.1f}%")
    if math_total > 0:
        print(f"MATH:  {math_correct}/{math_total} = {math_correct/math_total*100:.1f}%")
    print(f"Total: {total_correct}/{total} = {total_correct/total*100:.1f}%")


if __name__ == "__main__":
    main()
