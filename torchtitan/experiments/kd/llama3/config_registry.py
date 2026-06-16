# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config registry for knowledge distillation experiments."""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.kd.trainer import KDTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.llama3 import model_registry


def llama3_debugmodel() -> KDTrainer.Config:
    """Debug config for testing KD with a tiny Llama3 model."""
    return KDTrainer.Config(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        # KD-specific settings
        temperature=2.0,
        alpha=0.5,
    )


def llama3_8b() -> KDTrainer.Config:
    """KD config for Llama3-8B."""
    return KDTrainer.Config(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="meta-llama/Llama-3.1-8B",
        model_spec=model_registry("8B"),
        optimizer=OptimizersContainer.Config(lr=3e-5),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=100,
            decay_ratio=0.1,
            decay_type="cosine",
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=2048,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(
            interval=100,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
        ),
        # KD-specific settings
        temperature=2.0,
        alpha=0.5,
        teacher_param_offload=True,
    )
