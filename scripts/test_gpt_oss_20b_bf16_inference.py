"""
GPT-OSS-20B bf16 inference test with vLLM.

Tests whether the dequantized bf16 checkpoint produces correct output
using the TRITON backend (avoid FLASHINFER_CUTLASS swap issue).
"""

import os
import time

os.environ["VLLM_USE_FLASHINFER_MOE_FP16"] = "0"

from vllm import LLM, SamplingParams

MODEL = "/data/users/andrewor/checkpoints/gpt-oss-20b-bf16/"

PROMPTS = [
    "What is 25 * 17?",
    "Solve for x: 3x + 7 = 22",
    "What is the square root of 144?",
]


def main():
    print(f"Loading model: {MODEL}")

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
    )

    print(f"\nRunning {len(PROMPTS)} inference steps...\n")

    start = time.time()
    outputs = llm.generate(PROMPTS, sampling_params)
    elapsed = time.time() - start

    for i, output in enumerate(outputs):
        prompt = output.prompt
        response = output.outputs[0].text
        print(f"\n--- Step {i+1} ---")
        print(f"Prompt: {prompt}")
        print(f"Response: {response[:300]}")
        print(f"Tokens generated: {len(output.outputs[0].token_ids)}")

    print(f"\nTotal time: {elapsed:.1f}s for {len(PROMPTS)} prompts")


if __name__ == "__main__":
    main()
