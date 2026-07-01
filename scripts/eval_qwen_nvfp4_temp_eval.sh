#!/bin/bash
# Qwen3-30B-A3B QAT nvfp4 evals via torchao temp_eval.py, using the fixed
# custom tasks (gsm8k_verl, math500_verify). Reuses existing per-task-calibrated
# nvfp4 checkpoints (symlinked to the new task names; quantize step is skipped).
set -uo pipefail

AO=/home/andrewor/local/ao
CKPT=/data/users/andrewor/torchtitan/outputs/qat_rollout_fq_is_06-18
LOGDIR=/data/users/andrewor/logs/gptoss_rl_eval
EVAL=$AO/torchao/prototype/qat/temp_eval.py
mkdir -p "$LOGDIR"; cd "$AO"

GPU=0; PIDS=()
for step in 5 10 15; do
  for task in gsm8k_verl math500_verify; do
    echo "launch qwen step${step} ${task} (nvfp4) on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_FLASHINFER_MOE_FP16=0 \
      python "$EVAL" --task "$task" --checkpoint "$CKPT/step${step}_hf" \
      > "$LOGDIR/eval_qwen_step${step}_${task}_nvfp4.log" 2>&1 &
    PIDS+=($!); GPU=$((GPU+1))
  done
done

echo "waiting for ${#PIDS[@]} jobs ..."
wait
echo "=== nvfp4 RESULTS ==="
for step in 5 10 15; do
  g=$(grep -iE "exact_match" "$LOGDIR/eval_qwen_step${step}_gsm8k_verl_nvfp4.log" | grep -oP "[0-9.]+%" | head -1)
  m=$(grep -iE "math_verify" "$LOGDIR/eval_qwen_step${step}_math500_verify_nvfp4.log" | grep -oP "[0-9.]+%" | head -1)
  echo "qwen step${step} nvfp4: GSM8K=$g MATH500=$m"
done
echo "=== ALL DONE ==="
