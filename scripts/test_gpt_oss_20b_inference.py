"""
GPT-OSS-20B MXFP4 inference test with vLLM.

Uses the flashinfer TRTLLM MXFP4 kernel on B200 (SM100+).
Runs 10 inference steps with math-style prompts.
"""

import os
import time

# Force flashinfer TRTLLM MXFP4 BF16 kernel
os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_BF16"] = "1"

from vllm import LLM, SamplingParams

MODEL = "openai/gpt-oss-20b"

PROMPTS = [
    "What is 25 * 17?",
    "Solve for x: 3x + 7 = 22",
    "What is the square root of 144?",
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "What is the derivative of x^3 + 2x^2 - 5x + 1?",
    "Simplify: (2/3) + (3/4)",
    "What is the area of a circle with radius 5?",
    "How many ways can you arrange 4 books on a shelf?",
    "What is 2^10?",
    "If f(x) = x^2 + 3x + 2, what is f(5)?",
]

def main():
    print(f"Loading model: {MODEL}")
    print(f"VLLM_USE_FLASHINFER_MOE_MXFP4_BF16={os.environ.get('VLLM_USE_FLASHINFER_MOE_MXFP4_BF16')}")

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
    )

    print(f"\nRunning {len(PROMPTS)} inference steps...\n")
    print("=" * 60)

    start = time.time()
    outputs = llm.generate(PROMPTS, sampling_params)
    elapsed = time.time() - start

    for i, output in enumerate(outputs):
        prompt = output.prompt
        response = output.outputs[0].text
        print(f"\n--- Step {i+1} ---")
        print(f"Prompt: {prompt}")
        print(f"Response: {response[:200]}...")
        print(f"Tokens generated: {len(output.outputs[0].token_ids)}")

    print("\n" + "=" * 60)
    print(f"Total time: {elapsed:.1f}s for {len(PROMPTS)} prompts")
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"Total tokens: {total_tokens}, throughput: {total_tokens/elapsed:.1f} tok/s")


if __name__ == "__main__":
    main()
