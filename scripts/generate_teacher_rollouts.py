"""Generate teacher rollouts for QAD (paper's 'generated from RL prompts' recipe).

For each GSM8K+MATH train prompt (rendered in the exact RL format), sample N
completions from the bf16 teacher (step100_bf16). Each (prompt, sampled_answer)
becomes one {"text": <full harmony sequence>} training example — far more unique,
on-teacher-distribution data than looping the 15k reference set.

Data-parallel: run one process per GPU with --shard i --num-shards K (set
CUDA_VISIBLE_DEVICES externally). Each writes shard_<i>.json; merge afterwards.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams

TEACHER = "/data/users/andrewor/logs/gpt_oss_20b_rl_6-25_eval/step100_bf16"
OUT = Path("/home/andrewor/data/math_teachergen")
GSM8K_SUFFIX = ' Let\'s think step by step and output the final answer after "####".'
MATH_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."


def gen_prompt(question: str) -> str:
    # Simplified harmony, generation prompt (assistant turn left open).
    return f"<|start|>user<|message|>{question}<|end|>\n<|start|>assistant<|message|>"


def build_prompts():
    prompts = []
    df = pd.read_parquet("/home/andrewor/data/gsm8k/train.parquet")
    for _, row in df.iterrows():
        prompts.append(gen_prompt(row["extra_info"]["question"] + GSM8K_SUFFIX))
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    for row in ds:
        prompts.append(gen_prompt(row["problem"] + MATH_SUFFIX))
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    prompts = build_prompts()
    shard = prompts[args.shard :: args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(shard)} prompts x n={args.n}")

    llm = LLM(
        model=TEACHER,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        enforce_eager=True,
    )
    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["<|end|>"],
    )
    outputs = llm.generate(shard, sp)

    samples = []
    for prompt, out in zip(shard, outputs):
        for comp in out.outputs:
            text = prompt + comp.text + "<|end|>\n"
            samples.append({"text": text})

    path = OUT / f"shard_{args.shard}.json"
    with open(path, "w") as f:
        json.dump(samples, f)
    print(f"shard {args.shard}: wrote {len(samples)} sequences -> {path}")


if __name__ == "__main__":
    main()
