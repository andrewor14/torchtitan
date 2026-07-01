"""Prepare combined GSM8K + MATH SFT dataset for GPT-OSS.

Produces a JSON file with [user, assistant] message pairs where:
- User message: the math problem
- Assistant message: step-by-step solution ending with the answer

GSM8K solutions use "#### <number>" format.
MATH solutions use "\\boxed{<answer>}" format.
"""

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = Path("/home/andrewor/data/math_sft")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_gsm8k(parquet_path: str) -> list[dict]:
    df = pd.read_parquet(parquet_path)
    samples = []
    for _, row in df.iterrows():
        question = row["extra_info"]["question"]
        answer = row["extra_info"]["answer"]
        samples.append({
            "question": question,
            "answer": answer,
        })
    return samples


def process_math(split: str) -> list[dict]:
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split=split)
    samples = []
    for row in ds:
        samples.append({
            "question": row["problem"],
            "answer": row["solution"],
        })
    return samples


def main():
    print("Loading GSM8K train...")
    gsm_train = process_gsm8k("/home/andrewor/data/gsm8k/train.parquet")
    print(f"  {len(gsm_train)} examples")

    print("Loading GSM8K test...")
    gsm_test = process_gsm8k("/home/andrewor/data/gsm8k/test.parquet")
    print(f"  {len(gsm_test)} examples")

    print("Loading MATH train...")
    math_train = process_math("train")
    print(f"  {len(math_train)} examples")

    print("Loading MATH test...")
    math_test = process_math("test")
    print(f"  {len(math_test)} examples")

    train_data = gsm_train + math_train
    test_data = gsm_test + math_test

    print(f"\nTotal train: {len(train_data)}, test: {len(test_data)}")

    for name, data in [("train", train_data), ("test", test_data)]:
        path = OUTPUT_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Wrote {path} ({len(data)} examples)")


if __name__ == "__main__":
    main()
