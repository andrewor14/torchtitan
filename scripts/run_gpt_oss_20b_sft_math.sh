#!/usr/bin/env bash
# SFT GPT-OSS-20B on GSM8K + MATH (step-by-step solutions)
# Teaches the model the answer format before RL training.
#
# Usage: bash scripts/run_gpt_oss_20b_sft_math.sh
set -xeuo pipefail

NGPUS=${NGPUS:-8}

torchrun --nproc_per_node=${NGPUS} \
    --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m torchtitan.train \
    --module gpt_oss --config gpt_oss_20b_sft_math \
    "$@"
