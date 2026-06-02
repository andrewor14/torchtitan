#!/bin/bash
# Eval math tasks for pretrained and RL step-50 checkpoints
# bf16 via lm_eval HF backend, nvfp4 via temp_eval.py
set -e

PRETRAINED="/home/andrewor/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
RL_STEP50="./outputs/rl_grpo_step50_hf"
EVAL_SCRIPT="/home/andrewor/local/ao/torchao/prototype/qat/temp_eval.py"
MATH_TASKS="gsm8k,minerva_math,hendrycks_math"

echo "=== BF16 Math Evals (lm_eval, 10% limit) ==="

# Pretrained bf16
echo "--- Pretrained bf16 ---"
CUDA_VISIBLE_DEVICES=0 python -m lm_eval \
    --model hf \
    --model_args "pretrained=${PRETRAINED},dtype=bfloat16" \
    --tasks ${MATH_TASKS} \
    --limit 0.1 \
    --batch_size 4 \
    --output_path ./outputs/lm_eval_pretrained_math &
PID1=$!

# RL step50 bf16
echo "--- RL step50 bf16 ---"
CUDA_VISIBLE_DEVICES=1 python -m lm_eval \
    --model hf \
    --model_args "pretrained=${RL_STEP50},dtype=bfloat16" \
    --tasks ${MATH_TASKS} \
    --limit 0.1 \
    --batch_size 4 \
    --output_path ./outputs/lm_eval_rl_step50_math &
PID2=$!

wait $PID1 $PID2
echo "=== BF16 Math Evals Done ==="

echo ""
echo "=== NVFP4 Math Evals (temp_eval.py for gsm8k) ==="

# Pretrained nvfp4 gsm8k
echo "--- Pretrained nvfp4 gsm8k ---"
CUDA_VISIBLE_DEVICES=2 python ${EVAL_SCRIPT} --checkpoint "${PRETRAINED}" --task gsm8k &
PID3=$!

# RL step50 nvfp4 gsm8k
echo "--- RL step50 nvfp4 gsm8k ---"
CUDA_VISIBLE_DEVICES=3 python ${EVAL_SCRIPT} --checkpoint "${RL_STEP50}" --task gsm8k &
PID4=$!

wait $PID3 $PID4
echo "=== NVFP4 GSM8K Evals Done ==="
echo "=== ALL MATH EVALS COMPLETE ==="
