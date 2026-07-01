"""
Quantize GPT-OSS-20B from bf16 to NVFP4 using NVIDIA ModelOpt.

Loads the dequantized bf16 checkpoint, runs NVFP4 quantization with
calibration, and exports a vLLM-compatible NVFP4 checkpoint.

Usage:
    python scripts/quantize_gptoss_nvfp4_modelopt.py \
        --input-dir /data/users/andrewor/checkpoints/gpt-oss-20b-bf16/ \
        --output-dir /data/users/andrewor/checkpoints/gpt-oss-20b-nvfp4-modelopt/ \
        --num-calib-samples 128
"""

import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-calib-samples", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading bf16 model from {args.input_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.input_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.input_dir)
    model.eval()
    device_map = getattr(model, "hf_device_map", "single-device")
    print(f"Model loaded. Device map: {device_map}")

    # Create calibration forward loop with simple math prompts
    print(f"Creating calibration data ({args.num_calib_samples} samples, "
          f"max_len={args.max_seq_len})...")

    calib_prompts = [
        "What is 25 * 17?",
        "Solve for x: 3x + 7 = 22",
        "What is the square root of 144?",
        "If a train travels 60 mph for 2.5 hours, how far does it go?",
        "What is the derivative of x^3 + 2x^2 - 5x + 1?",
        "Simplify: (2/3) + (3/4)",
        "What is the area of a circle with radius 5?",
        "How many ways can you arrange 4 books on a shelf?",
        "Explain the concept of integration in calculus.",
        "What is the probability of rolling a sum of 7 with two dice?",
        "Calculate the compound interest on $1000 at 5% for 3 years.",
        "What is the Fibonacci sequence?",
        "Prove that the square root of 2 is irrational.",
        "What is the binomial theorem?",
        "Explain the chain rule in differentiation.",
        "What is a prime number? List the first 20 primes.",
    ]

    # Tokenize calibration data
    calib_tokens = []
    for i in range(args.num_calib_samples):
        prompt = calib_prompts[i % len(calib_prompts)]
        tokens = tokenizer(prompt, return_tensors="pt", padding=False,
                           truncation=True, max_length=args.max_seq_len)
        calib_tokens.append(tokens)

    def forward_loop(model):
        for i, tokens in enumerate(calib_tokens):
            input_ids = tokens["input_ids"].to(model.device if hasattr(model, 'device') else "cuda:0")
            with torch.no_grad():
                model(input_ids)
            if (i + 1) % 32 == 0:
                print(f"  Calibration: {i+1}/{len(calib_tokens)}")
        print(f"  Calibration complete: {len(calib_tokens)} samples")

    # Quantize with NVFP4
    print("Running NVFP4 quantization with calibration...")
    model = mtq.quantize(
        model,
        config=mtq.NVFP4_DEFAULT_CFG,
        forward_loop=forward_loop,
    )
    print("Quantization complete.")

    # Export vLLM-compatible checkpoint
    print(f"Exporting to {args.output_dir}...")
    export_hf_checkpoint(
        model,
        dtype=torch.bfloat16,
        export_dir=args.output_dir,
    )

    # export_hf_checkpoint writes only weights/configs, not the tokenizer, so
    # downstream loaders (vLLM, lm_eval) can't instantiate it. Persist it here.
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done! NVFP4 checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
