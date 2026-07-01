"""Prepare GSM8K+MATH QAD data rendered in the EXACT RL prompt format.

Same problems/reference-solutions as math_sft, but the user turn carries the
RL/eval instruction suffix (so QAD distills on the same prompt format the model
was RL'd on and is evaluated with), wrapped in the simplified gpt-oss harmony
chat template. Output is a list of {"text": <full rendered sequence>} — the same
format math_sft_chat uses.

RL prompt suffixes (verified against /home/andrewor/data/{gsm8k,math}/train.parquet):
- GSM8K: ... Let's think step by step and output the final answer after "####".
- MATH : ... Let's think step by step and output the final answer within \\boxed{}.
"""

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT = Path("/home/andrewor/data/math_rlfmt")
OUT.mkdir(parents=True, exist_ok=True)

GSM8K_SUFFIX = ' Let\'s think step by step and output the final answer after "####".'
MATH_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."


def render(question: str, answer: str) -> str:
    """Simplified gpt-oss harmony template (matches the model's chat_template)."""
    return (
        f"<|start|>user<|message|>{question}<|end|>\n"
        f"<|start|>assistant<|message|>{answer}<|end|>\n"
    )


def main():
    samples = []

    df = pd.read_parquet("/home/andrewor/data/gsm8k/train.parquet")
    for _, row in df.iterrows():
        q = row["extra_info"]["question"] + GSM8K_SUFFIX
        samples.append({"text": render(q, row["extra_info"]["answer"])})
    print(f"gsm8k: {len(samples)}")

    n0 = len(samples)
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    for row in ds:
        q = row["problem"] + MATH_SUFFIX
        samples.append({"text": render(q, row["solution"])})
    print(f"math: {len(samples) - n0}")

    path = OUT / "train.json"
    with open(path, "w") as f:
        json.dump(samples, f)
    print(f"Wrote {path} ({len(samples)} examples)")
    print("--- first gsm8k sample ---")
    print(samples[0]["text"][:400])
    print("--- first math sample ---")
    print(samples[n0]["text"][:400])


if __name__ == "__main__":
    main()
