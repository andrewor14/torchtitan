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
        apply_simple_fp4_full_qat_torchtitan,
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
        # QAD: fake quantize student MoE experts AND dense linears (attention
        # q/k/v/o; excludes lm_head/router/gate) to match the nvfp4 eval scope;
        # teacher stays bf16.
        post_model_init_fn=apply_simple_fp4_full_qat_torchtitan,
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


def gpt_oss_20b_qad_math_chat() -> KDTrainer.Config:
    """QAD distilled on the GSM8K+MATH SFT data pre-rendered through the gpt-oss
    chat (harmony) template (dataset math_sft_chat), fixing the raw-text format
    mismatch in gpt_oss_20b_qad_math. Everything else matches gpt_oss_20b_qad.
    """
    config = gpt_oss_20b_qad()
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset="math_sft_chat")
    return config


def gpt_oss_20b_qad_math_chat_500() -> KDTrainer.Config:
    """Longer chat-formatted QAD: 500 steps with a real warmup + cosine-decay LR
    schedule (the 100-step runs were misconfigured: warmup_steps==steps so LR
    never decayed). Checkpoints every 50 steps. Tests whether more steps + proper
    LR annealing lets QAD close the nvfp4 MATH gap.
    """
    config = gpt_oss_20b_qad_math_chat()
    config.training.steps = 500
    config.checkpoint.interval = 50
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=50,        # ~10% warmup
        decay_ratio=None,       # cosine-decay immediately after warmup
        decay_type="cosine",
        min_lr_factor=0.0,      # anneal to ~0 by step 500
    )
    return config


def gpt_oss_20b_qad_c4_purekl_2000() -> KDTrainer.Config:
    """Long pure-KL QAD on generic C4 (not in-domain math), 2000 steps.

    alpha=1.0 (KL only, no hard-label CE) and temperature=1.0 (match the teacher
    distribution exactly) — pure distillation. Teacher and student both start from
    the bf16 RL step-100 checkpoint; the student's MoE experts are fake-quantized
    to NvFP4 while the teacher stays bf16. C4 gives a large, diverse distillation
    corpus so the student sees far more tokens than the in-domain math runs. LR
    1e-5 (paper's recommended LR for RL-trained models) with a real warmup + cosine
    decay over the full 2000 steps; checkpoint every 100 steps.
    """
    config = gpt_oss_20b_qad()  # C4, init=step100_bf16, fp4 fake-quant on student
    config.alpha = 1.0  # pure KL(teacher || student), drop CE term
    config.temperature = 1.0  # match teacher distribution exactly (no softening)
    config.optimizer.lr = 1e-5
    config.training.local_batch_size = 4  # global batch 4*8=32 seqs -> ~0.13B tokens over 2000 steps
    config.training.steps = 2000
    config.checkpoint.interval = 100
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=100,  # 5% warmup
        decay_ratio=None,  # cosine-decay immediately after warmup
        decay_type="cosine",
        min_lr_factor=0.0,  # anneal to ~0 by step 2000
    )
    return config


def gpt_oss_20b_qad_chat_purekl_500() -> KDTrainer.Config:
    """Pure-KL QAD matching the NVIDIA QAD report's recipe: alpha=1.0 (KL only,
    no hard-label cross-entropy) and temperature=1.0 (exactly match the teacher's
    output distribution). Everything else identical to gpt_oss_20b_qad_math_chat_500
    (chat-formatted GSM8K+MATH, 500 steps, cosine-decay LR, ckpt every 50), so this
    isolates the loss-formulation change vs the alpha=0.5/T=2 run.
    """
    config = gpt_oss_20b_qad_math_chat_500()
    config.alpha = 1.0        # pure KL(teacher || student), drop CE term
    config.temperature = 1.0  # match teacher distribution exactly (no softening)
    config.optimizer.lr = 1e-5  # paper's recommended LR for RL-trained models (vs 5e-6/1e-6 for SFT)
    return config


# ---------------------------------------------------------------------------
# 6-27 experiment: 3 QAD runs, all 1000 steps / lbs=4 / lr 1e-5 / ckpt every 100,
# warmup 50 + cosine to 10% floor. All use the EXACT RL prompt format (instruction
# suffix). Compares loss formulation (alpha=1/T=1 vs alpha=0.5/T=2) and data source
# (original 15k reference vs teacher-generated rollouts).
# ---------------------------------------------------------------------------
def _qad_1000_base() -> KDTrainer.Config:
    """Shared base for the 6-27 runs: alpha=1/T=1, lr 1e-5, 1000 steps, lbs=4,
    ckpt every 100, warmup 50 + cosine to 10% floor. Dataset set by callers."""
    config = gpt_oss_20b_qad()  # C4, init=step100_bf16, fp4 fake-quant on student
    config.alpha = 1.0
    config.temperature = 1.0
    config.optimizer.lr = 1e-5
    config.training.local_batch_size = 4
    config.training.steps = 1000
    config.checkpoint.interval = 100
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=50,
        decay_ratio=None,     # cosine-decay immediately after warmup
        decay_type="cosine",
        min_lr_factor=0.1,    # 10% floor (1e-6) to avoid late-LR starvation
    )
    return config


def gpt_oss_20b_qad_rlfmt_a1t1_1000() -> KDTrainer.Config:
    """Run 1: alpha=1/T=1 pure-KL QAD on the original 15k GSM8K+MATH reference
    data, rendered in the exact RL prompt format (instruction suffix)."""
    config = _qad_1000_base()
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset="math_rlfmt")
    return config


def gpt_oss_20b_qad_teachergen_a1t1_1000() -> KDTrainer.Config:
    """Run 2: alpha=1/T=1 pure-KL QAD on teacher-generated rollouts (N=8/prompt
    from the bf16 teacher), same RL prompt format. More unique, on-teacher data."""
    config = _qad_1000_base()
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset="math_teachergen")
    return config


def gpt_oss_20b_qad_rlfmt_a05t2_1000() -> KDTrainer.Config:
    """Run 3: alpha=0.5/T=2 (KL + hard-label CE) QAD on the original 15k reference
    data in RL prompt format. Tests whether the CE term still helps when format is
    aligned. Same lr 1e-5 as the alpha=1 runs (unified for comparison)."""
    config = _qad_1000_base()
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset="math_rlfmt")
    config.alpha = 0.5
    config.temperature = 2.0
    return config


# ---------------------------------------------------------------------------
# 6-28 retry: the EXACT 3 runs above but at lr 5e-6 (instead of 1e-5). The
# best prior QAD result (nvfp4 MATH 0.534) used lr 5e-6; the 6-27 runs used
# 1e-5, so LR is a confound. These isolate it — everything else (data, alpha,
# T, steps, lbs, schedule shape, 10% floor) is identical to the 6-27 configs.
# Mechanistic motivation: a gentler LR should degrade the bf16 master less
# (the 6-27 gap closed largely by bf16 sinking), so the bf16<->nvfp4
# convergence point may sit higher -> higher nvfp4.
# ---------------------------------------------------------------------------
def gpt_oss_20b_qad_rlfmt_a1t1_1000_lr5e6() -> KDTrainer.Config:
    """Run 1 @ lr 5e-6: alpha=1/T=1 pure-KL QAD on the original 15k reference data."""
    config = gpt_oss_20b_qad_rlfmt_a1t1_1000()
    config.optimizer.lr = 5e-6
    return config


def gpt_oss_20b_qad_teachergen_a1t1_1000_lr5e6() -> KDTrainer.Config:
    """Run 2 @ lr 5e-6: alpha=1/T=1 pure-KL QAD on teacher-generated rollouts."""
    config = gpt_oss_20b_qad_teachergen_a1t1_1000()
    config.optimizer.lr = 5e-6
    return config


def gpt_oss_20b_qad_rlfmt_a05t2_1000_lr5e6() -> KDTrainer.Config:
    """Run 3 @ lr 5e-6: alpha=0.5/T=2 QAD on the original 15k reference data."""
    config = gpt_oss_20b_qad_rlfmt_a05t2_1000()
    config.optimizer.lr = 5e-6
    return config


# ---------------------------------------------------------------------------
# 6-29: pure-KL QAD on generic C4 (not in-domain math), 500 steps, lr 5e-6.
# Same init/loss/schedule shape as the 6-28 lr=5e-6 runs but with the large,
# diverse C4 corpus instead of the tiny looped math set — tests whether broad
# data (no looping) at the gentle LR recovers nvfp4 MATH without the in-domain
# data the other runs used.
# ---------------------------------------------------------------------------
def gpt_oss_20b_qad_c4_a1t1_500_lr5e6() -> KDTrainer.Config:
    """alpha=1/T=1 pure-KL QAD on C4, lr 5e-6, 500 steps, lbs=4, ckpt every 100,
    warmup 50 + cosine to 10% floor. Init from the RL step-100 bf16 checkpoint."""
    config = gpt_oss_20b_qad()  # C4, init=step100_bf16, fp4 fake-quant on student
    config.alpha = 1.0
    config.temperature = 1.0
    config.optimizer.lr = 5e-6
    config.training.local_batch_size = 4
    config.training.steps = 500
    config.checkpoint.interval = 100
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=50,
        decay_ratio=None,     # cosine-decay immediately after warmup
        decay_type="cosine",
        min_lr_factor=0.1,    # 10% floor
    )
    return config


# ---------------------------------------------------------------------------
# 6-30: extend the 6-29 C4 pure-KL QAD (the best result so far, nvfp4 MATH 0.542
# @ step500 with bf16 staying healthy) to 2000 steps. Identical recipe — alpha=1/
# T=1, lr 5e-6, lbs=4, C4, fp4 fake-quant on experts+linears, warmup 50 + cosine
# to 10% floor, ckpt every 100 — only the step count (and thus the cosine decay
# horizon) changes. Tests whether more distillation tokens push nvfp4 MATH past
# the ~0.54 plateau without collapsing the bf16 master.
# ---------------------------------------------------------------------------
def gpt_oss_20b_qad_c4_a1t1_2000_lr5e6() -> KDTrainer.Config:
    """alpha=1/T=1 pure-KL QAD on C4, lr 5e-6, 2000 steps, lbs=4, ckpt every 100,
    warmup 50 + cosine to 10% floor over the full 2000 steps. Init from the RL
    step-100 bf16 checkpoint. 4x-longer extension of gpt_oss_20b_qad_c4_a1t1_500_lr5e6."""
    config = gpt_oss_20b_qad_c4_a1t1_500_lr5e6()
    config.training.steps = 2000
    return config
