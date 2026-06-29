# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config registry for knowledge distillation experiments with Qwen3 models."""

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
from torchtitan.models.qwen3 import model_registry


def qwen3_moe_debug() -> KDTrainer.Config:
    """Debug KD config with a tiny Qwen3 MoE model."""
    return KDTrainer.Config(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel_moe"),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
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
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        # KD-specific settings
        temperature=2.0,
        alpha=0.5,
    )


def qwen3_moe_debug_qad() -> KDTrainer.Config:
    """Debug QAD (quantization-aware distillation) config with a tiny Qwen3 MoE model.

    Same as qwen3_moe_debug but with NvFP4 fake quantization applied to
    the student model's MoE expert weights and activations. The teacher
    stays bf16.
    """
    from torchao.prototype.qat.nvfp4_moe_simple import (
        apply_simple_fp4_full_qat_torchtitan,
    )

    config = qwen3_moe_debug()
    config.post_model_init_fn = apply_simple_fp4_full_qat_torchtitan
    return config


def qwen3_30b_a3b_qad() -> KDTrainer.Config:
    """QAD config for Qwen3-30B-A3B.

    Distills from a bf16 teacher to an NvFP4 fake-quantized student.
    The teacher is initialized from the same HF checkpoint and kept frozen.
    """
    from torchao.prototype.qat.nvfp4_moe_simple import (
        apply_simple_fp4_full_qat_torchtitan,
    )

    return KDTrainer.Config(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./assets/hf/Qwen3-30B-A3B",
        model_spec=model_registry("30B-A3B", attn_backend="sdpa"),
        optimizer=OptimizersContainer.Config(
            lr=5e-6,
            beta2=0.999,
            weight_decay=0.0,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=100,
            decay_ratio=0.1,
            decay_type="cosine",
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=2048,
            steps=100,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_path="/data/users/andrewor/torchtitan/outputs/rl_continued_v3_06-09/step200_hf",
            last_save_in_hf=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            expert_parallel_degree=1,
        ),
        # KD-specific settings
        temperature=2.0,
        alpha=0.5,
        # QAD: fake quantize student MoE experts, teacher stays bf16
        post_model_init_fn=apply_simple_fp4_full_qat_torchtitan,
    )


def qwen3_30b_a3b_qad_c4_a1t1_2000_lr5e6() -> KDTrainer.Config:
    """Long pure-KL QAD for Qwen3-30B-A3B on C4, 2000 steps, lr 5e-6.

    Mirrors the gpt-oss QAD-2000 winner recipe on qwen3: alpha=1.0 (pure KL, no
    hard-label CE) and temperature=1.0 (match teacher distribution exactly), C4,
    lr 5e-6, warmup 50 + cosine to 10% floor over the full 2000 steps. Student +
    bf16 teacher both init from the original 200-step RL checkpoint; the student's
    MoE experts AND dense linears are fp4 fake-quantized (apply_full, gated by
    QAT_FP4_FAKE_QUANT_LINEARS=1). Checkpoint every 100 steps and keep the last 30
    (= all 20) so none are purged (the gpt-oss QAD-2000 run lost steps 100-900 to
    the default keep_latest_k=10).
    """
    config = qwen3_30b_a3b_qad()  # C4, init=step200_hf, fp4 fake-quant on student
    config.alpha = 1.0  # pure KL(teacher || student), drop CE term
    config.temperature = 1.0  # match teacher distribution exactly
    config.training.local_batch_size = 4  # try 4 first; orchestrator retries lbs=2 on OOM
    config.training.steps = 2000
    config.checkpoint.interval = 100
    config.checkpoint.keep_latest_k = 30  # keep all 20 ckpts (no purging)
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=50,
        decay_ratio=None,     # cosine-decay immediately after warmup
        decay_type="cosine",
        min_lr_factor=0.1,    # 10% floor over the full 2000 steps
    )
    return config
