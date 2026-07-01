"""Test GPT-OSS model construction with torchtitan Trainer.
Run with: torchrun --nproc_per_node=8 scripts/test_gptoss_torchtitan_build.py
"""
import faulthandler
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
    print(f"[rank {rank}] Starting model construction test")

    spec = model_registry("20b", attn_backend="flex")
    print(f"[rank {rank}] ModelSpec created: {spec.name}/{spec.flavor}")

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
    print(f"[rank {rank}] Config created, building Trainer...")

    trainer = Trainer(config)
    print(f"[rank {rank}] Trainer built successfully!")

    for i, part in enumerate(trainer.model_parts):
        num_params = sum(p.numel() for p in part.parameters())
        print(f"[rank {rank}] model_part[{i}]: {num_params:,} params")

    dist.barrier()
    print(f"[rank {rank}] DONE - model loaded and distributed successfully")


if __name__ == "__main__":
    main()
