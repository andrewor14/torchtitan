#!/usr/bin/env bash
# GRPO | GPT-OSS-20B MoE | vLLM rollout | TorchTitan training
# Math tasks (GSM8K + MATH combined)
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-/data/users/andrewor/checkpoints/gpt-oss-20b-sft-math}
NGPUS=${NGPUS:-8}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
rollout_n=${ROLLOUT_N:-8}

total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-10}
test_freq=${TEST_FREQ:-10}

project_name=${PROJECT_NAME:-verl_grpo_gsm8k_math}
experiment_name=${EXPERIMENT_NAME:-gpt_oss_20b_sft_rl_6-22}

gsm8k_train=$HOME/data/gsm8k/train.parquet
gsm8k_test=$HOME/data/gsm8k/test.parquet
math_train=$HOME/data/math/train.parquet
math_test=$HOME/data/math/test.parquet

train_files="['$gsm8k_train', '$math_train']"
val_files="['$gsm8k_test', '$math_test']"

# GPT-OSS uses swigluoai activation which FLASHINFER_CUTLASS maps to
# standard swiglu (wrong math). Force TRITON backend for correct output.
export VLLM_USE_FLASHINFER_MOE_FP16=0
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

python3 -m verl.trainer.main_ppo_sync \
    model_engine=torchtitan \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=torchtitan \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size=${NGPUS} \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.expert_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.use_torch_compile=False \
    actor_rollout_ref.actor.torchtitan.attn_type=flex \
    actor_rollout_ref.actor.torchtitan.param_offload=True \
    actor_rollout_ref.actor.torchtitan.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.strategy=torchtitan \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.torchtitan.data_parallel_shard_size=${NGPUS} \
    actor_rollout_ref.ref.torchtitan.tensor_parallel_size=1 \
    actor_rollout_ref.ref.torchtitan.expert_parallel_size=1 \
    actor_rollout_ref.ref.torchtitan.use_torch_compile=False \
    actor_rollout_ref.ref.torchtitan.attn_type=flex \
    actor_rollout_ref.ref.torchtitan.param_offload=False \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    trainer.balance_batch=True \
    'trainer.logger=["console"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.val_before_train=False \
    "$@"
