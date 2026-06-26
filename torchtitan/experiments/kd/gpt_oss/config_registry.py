# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config registry for knowledge distillation experiments with GPT-OSS models."""

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
from torchtitan.models.gpt_oss import model_registry

# bf16 RL checkpoint (6-25 step 100) used as both the frozen teacher and the
# student's initial weights. It is a full HF checkpoint (config + tokenizer).
_GPTOSS_INIT = "/data/users/andrewor/logs/gpt_oss_20b_rl_6-25_eval/step100_bf16"


def gpt_oss_20b_qad() -> KDTrainer.Config:
    """QAD config for GPT-OSS-20B.

    Distills from a bf16 teacher to an NvFP4 fake-quantized student. Teacher and
    student are both initialized from the same bf16 HF checkpoint; the teacher
    stays frozen/bf16 while the student's MoE experts are fake-quantized via
    ``post_model_init_fn``. 100 steps, checkpoint at step 50 and 100.

    GPT-OSS only supports FlexAttention (sink attention needs the attention op's
    log-sum-exp + a BlockMask), so ``attn_backend='flex'`` (not sdpa like Qwen).
    """
    from torchao.prototype.qat.nvfp4_moe_simple import (
        apply_simple_fp4_moe_qat_torchtitan,
    )

    return KDTrainer.Config(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path=_GPTOSS_INIT,
        model_spec=model_registry("20b", attn_backend="flex"),
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
            interval=50,
            initial_load_in_hf=True,
            initial_load_path=_GPTOSS_INIT,
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
        post_model_init_fn=apply_simple_fp4_moe_qat_torchtitan,
    )


def gpt_oss_20b_qad_math() -> KDTrainer.Config:
    """QAD for GPT-OSS-20B distilled on the in-domain GSM8K+MATH SFT data
    (the same domain as QAT), instead of generic C4. Everything else matches
    gpt_oss_20b_qad. Tests whether in-domain distillation closes the nvfp4 MATH
    gap better than c4-QAD (which did not).
    """
    config = gpt_oss_20b_qad()
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset="math_sft")
    return config
