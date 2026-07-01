"""Timed GPT-OSS model construction to identify bottlenecks.
Run with: torchrun --nproc_per_node=8 scripts/test_gptoss_torchtitan_build_timed.py
"""
import faulthandler
import time

faulthandler.enable()

import torch
import torch.distributed as dist

MODEL_PATH = "/data/users/andrewor/checkpoints/gpt-oss-20b-bf16"


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    t0 = time.time()

    from torchtitan.models.gpt_oss import model_registry
    from torchtitan.protocols.model import BaseModel

    spec = model_registry("20b", attn_backend="flex")
    if rank == 0:
        print(f"[{time.time()-t0:.1f}s] ModelSpec created", flush=True)

    # Step 1: Build model on meta device
    t1 = time.time()
    with torch.device("meta"), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model = spec.model.build()
    if rank == 0:
        print(f"[{time.time()-t0:.1f}s] Model built on meta ({time.time()-t1:.1f}s)", flush=True)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total params: {total_params:,}", flush=True)

    # Step 2: Init distributed
    from torchtitan.distributed import ParallelDims
    from torchtitan.config import ParallelismConfig

    parallel_config = ParallelismConfig(
        data_parallel_shard_degree=8,
        tensor_parallel_degree=1,
        expert_parallel_degree=1,
    )
    parallel_dims = ParallelDims.from_config(parallel_config, dist.get_world_size())
    if rank == 0:
        print(f"[{time.time()-t0:.1f}s] ParallelDims created", flush=True)

    # Step 3: Parallelize (FSDP)
    t3 = time.time()
    from torchtitan.config import TrainingConfig, CompileConfig, ActivationCheckpointConfig
    model = spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=TrainingConfig(enable_cpu_offload=True),
        parallelism=parallel_config,
        compile_config=CompileConfig(enable=False),
        ac_config=ActivationCheckpointConfig(),
        dump_folder="./outputs",
    )
    if rank == 0:
        print(f"[{time.time()-t0:.1f}s] Parallelized ({time.time()-t3:.1f}s)", flush=True)

    # Step 4: to_empty on CPU
    t4 = time.time()
    model.to_empty(device="cpu")
    if rank == 0:
        import os
        pid = os.getpid()
        rss_mb = int(open(f"/proc/{pid}/status").read().split("VmRSS:")[1].split()[0]) // 1024
        print(f"[{time.time()-t0:.1f}s] to_empty on CPU ({time.time()-t4:.1f}s), RSS={rss_mb}MB", flush=True)

    # Step 5: init_weights with init_params=False (skip expensive param init)
    t5 = time.time()
    with torch.no_grad():
        from typing import cast
        cast(BaseModel, model).init_weights(
            buffer_device=torch.device("cuda"),
            init_params=False,
        )
    if rank == 0:
        rss_mb = int(open(f"/proc/{pid}/status").read().split("VmRSS:")[1].split()[0]) // 1024
        print(f"[{time.time()-t0:.1f}s] init_weights(init_params=False) ({time.time()-t5:.1f}s), RSS={rss_mb}MB", flush=True)

    dist.barrier()
    if rank == 0:
        print(f"[{time.time()-t0:.1f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
