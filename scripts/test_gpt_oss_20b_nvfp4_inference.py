"""
GPT-OSS-20B NVFP4 inference test with vLLM.

Uses the converted NVFP4 checkpoint (from MXFP4→bf16→ModelOpt NVFP4).
With the gpt-oss-20b NVFP4 fixes applied, the oracle auto-selects the
FLASHINFER_CUTLASS MoE backend, which correctly handles the clamped
swigluoai activation and per-expert biases. Leave the MoE backend on
auto; the EMULATION backend also works if you need a kernel-free path.
"""

import os
import time

from vllm import LLM, SamplingParams

MODEL = "/data/users/andrewor/checkpoints/gpt-oss-20b-nvfp4-modelopt/"

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
