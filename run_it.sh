#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: [CONFIG=<config>] [RUN_TAG=<name>] $0"
  exit 2
fi

# Set workload vars (configurable)
export CONFIG="${CONFIG:-rl_grpo_qwen3_0_6b_flex}"
export RUN_TAG="${RUN_TAG:-$CONFIG}"
export NUM_TRAINING_STEPS="${NUM_TRAINING_STEPS:-10}"
MODEL_ROOT="/checkpoint/pytorch/andrewor/models"
RUN_DATE="$(date +%-m-%-d)"

# Set model specific vars
if [[ "$CONFIG" == *"qwen3_0_6b"* ]]; then
  MODEL_DIR="$MODEL_ROOT/Qwen3-0.6B"
  export RL_SLURM_GPUS_PER_NODE=2
  export RL_SLURM_TIME="00:30:00"
elif [[ "$CONFIG" == *"qwen3_14b"* ]]; then
  MODEL_DIR="$MODEL_ROOT/Qwen3-14B"
  export RL_SLURM_GPUS_PER_NODE=4
  export RL_SLURM_TIME="02:00:00"
elif [[ "$CONFIG" == *"qwen3_6_27b"* ]]; then
  MODEL_DIR="$MODEL_ROOT/Qwen3.6-27B"
  export RL_SLURM_GPUS_PER_NODE=4
  export RL_SLURM_TIME="04:00:00"
elif [[ "$CONFIG" == *"qwen3_30b_a3b"* ]]; then
  MODEL_DIR="$MODEL_ROOT/Qwen3-30B-A3B"
  export RL_SLURM_GPUS_PER_NODE=4
  export RL_SLURM_TIME="04:00:00"
else
  echo "Could not find model directory for config: $CONFIG"
  exit 2
fi

# Set up run dir and cache dirs
export RUN_DIR="/checkpoint/pytorch/andrewor/output/$RUN_DATE/$RUN_TAG"
export TRITON_CACHE_DIR="/tmp/scratch/andrewor/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/scratch/andrewor/inductor"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
if [[ ! -d "$RUN_DIR" ]]; then
  echo "RUN_DIR is not a directory: ${RUN_DIR}"
  exit 2
fi

# Unset the batch host's inherited name before Slurm launches workers
# Otherwise, TorchStore will think the trainer and the generator are
# on the same host and wrongly chooses SharedMemory for transfers
SLURM_BASH_ENV="$RUN_DIR/slurm_env.sh"
printf 'unset HOSTNAME\n' > "$SLURM_BASH_ENV"
export BASH_ENV="$SLURM_BASH_ENV"

# Other SLURM configs
export RL_SLURM_BATCH=1
export RL_SLURM_PARTITION="g3"
export RL_SLURM_ACCOUNT="faircw-pytorch-access"
export RL_SLURM_QOS="g3_lowest"
export RL_SLURM_CPUS_PER_TASK=128
export RL_SLURM_MEM=0

# Monarch and TorchStore configs
export MONARCH_RDMA_IBVERBS_TARGET="nic:ibp0p0"
export TORCHSTORE_RDMA_ENABLED=1
export USE_TORCHCOMMS=0
export USE_TORCHCOMMS_RDMA=0
export TORCHSTORE_LOG_LEVEL="INFO"

# Other configs
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export WANDB_MODE="disabled"

# Submit from RUN_DIR so SLURM writes stdout and stderr there
original_dir=$PWD
cd "$RUN_DIR"
launcher_status=0
"/home/andrewor/titan-rl/bin/python" -m torchtitan.experiments.rl.slurm_launcher \
  --module alphabet_sort \
  --config "$CONFIG" \
  --async-loop.num-training-steps "$NUM_TRAINING_STEPS" \
  --weight-sync-transport auto \
  --generator.manual-cpu-stage-weight-sync \
  --dump-folder="$RUN_DIR/output" \
  --hf_assets_path="$MODEL_DIR" || launcher_status=$?
cd "$original_dir"
echo "Job submitted, logging to ${RUN_DIR}..."
exit "$launcher_status"
