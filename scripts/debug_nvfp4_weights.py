"""Debug NVFP4 weight loading in vLLM."""

import os
os.environ['VLLM_USE_FLASHINFER_MOE_FP16'] = '0'

from vllm import LLM, SamplingParams
import torch


MODEL = '/data/users/andrewor/checkpoints/gpt-oss-20b-nvfp4-modelopt/'


def main():
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        enforce_eager=True,
        kernel_config={'moe_backend': 'emulation'},
    )

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    named_params = dict(model.named_parameters())

    print("=== Layer 0 MLP ===")
    for key in sorted(named_params.keys()):
        if 'layers.0.mlp' in key:
            p = named_params[key]
            is_zero = (p.data == 0).all().item()
            print(f'{key}: shape={list(p.shape)} dtype={p.dtype} all_zero={is_zero}')

    print("\n=== Layer 0 Attn ===")
    for key in sorted(named_params.keys()):
        if 'layers.0.attn' in key:
            p = named_params[key]
            is_zero = (p.data == 0).all().item()
            print(f'{key}: shape={list(p.shape)} dtype={p.dtype} all_zero={is_zero}')


if __name__ == '__main__':
    main()
