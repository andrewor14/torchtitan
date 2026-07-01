"""Test full Trainer construction for GPT-OSS-20B with checkpoint loading.
Run with: torchrun --nproc_per_node=8 scripts/test_gptoss_trainer_build.py
"""
import faulthandler
import time

faulthandler.enable()

import torch
import torch.distributed as dist

from torchtitan.train import Trainer
from torchtitan.models.gpt_oss import model_registry
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.config import TrainingConfig, ParallelismConfig, CompileConfig

MODEL_PATH = "/data/users/andrewor/checkpoints/gpt-oss-20b-bf16"


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    t0 = time.time()
    if rank == 0:
        print(f"[{time.time()-t0:.1f}s] Starting Trainer construction", flush=True)

    spec = model_registry("20b", attn_backend="flex")

    config = Trainer.Config(
        model_spec=spec,
        hf_assets_path=MODEL_PATH,
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=1,
            initial_load_in_hf=True,
            initial_load_model_only=True,
            initial_load_path=MODEL_PATH,
        ),
        training=TrainingConfig(enable_cpu_offload=True),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=8,
            tensor_parallel_degree=1,
            expert_parallel_degree=1,
        ),
        compile=CompileConfig(enable=False),
        loss=CrossEntropyLoss.Config(),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
    )

    if rank == 0:
        print(f"[{time.time()-t0:.1f}s] Config created, building Trainer...", flush=True)

    trainer = Trainer(config)

    if rank == 0:
        print(f"[{time.time()-t0:.1f}s] Trainer built!", flush=True)

    for i, part in enumerate(trainer.model_parts):
        num_params = sum(p.numel() for p in part.parameters())
        if rank == 0:
            print(f"  model_part[{i}]: {num_params:,} params", flush=True)

    # Now test checkpoint loading
    t_load = time.time()
    if rank == 0:
        print(f"[{time.time()-t0:.1f}s] Loading checkpoint...", flush=True)
    trainer.checkpointer.load()
    if rank == 0:
        print(f"[{time.time()-t0:.1f}s] Checkpoint loaded ({time.time()-t_load:.1f}s)", flush=True)

    dist.barrier()
    if rank == 0:
        print(f"[{time.time()-t0:.1f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
