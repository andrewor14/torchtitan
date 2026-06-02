#!/bin/bash
# Run a single QAT experiment on top of an SFT checkpoint.
# Usage: ./run_qat_sweep.sh <gpus> <sft_hf_path> <lr> <steps> <output_dir>
set -e

GPUS=$1
SFT_HF_PATH=$2
LR=$3
STEPS=$4
OUTPUT_DIR=$5

echo "=== QAT: lr=${LR}, steps=${STEPS}, gpus=${GPUS} ==="
echo "Loading SFT checkpoint from: ${SFT_HF_PATH}"
echo "Output: ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${GPUS} PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 3 \
  -m torchtitan.train --module qwen3 --config sft_qwen3_30b_a3b_arc_qat \
  --training.steps ${STEPS} \
  --optimizer.lr ${LR} \
  --checkpoint.initial_load_path ${SFT_HF_PATH} \
  --dump_folder ${OUTPUT_DIR} \
  --checkpoint.no-last-save-in-hf

echo "=== QAT training done: ${OUTPUT_DIR} ==="
