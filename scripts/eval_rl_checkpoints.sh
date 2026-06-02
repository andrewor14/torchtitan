#!/bin/bash
# Evaluate all RL GRPO checkpoints: convert DCP→HF, then run bf16+nvfp4 evals
set -e

EVAL_SCRIPT="/home/andrewor/local/ao/torchao/prototype/qat/temp_eval.py"
CONVERT_SCRIPT="/data/users/andrewor/torchtitan/scripts/convert_dcp_to_hf.py"
CKPT_BASE="checkpoints/verl_grpo_qwen3_30b_a3b/torchtitan_moe"
OUTPUT_BASE="./outputs/rl_grpo"
PRETRAINED="/home/andrewor/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"

# Tasks: full eval for small ones, 10% for large math ones
BF16_TASKS="arc_challenge arc_easy winogrande boolq"
MATH_TASKS_LIMITED="gsm8k minerva_math hendrycks_math"

STEPS=(10 20 30 40 50)

echo "=== Step 1: Convert DCP checkpoints to HF ==="
for step in "${STEPS[@]}"; do
    DCP_DIR="${CKPT_BASE}/global_step_${step}/step-${step}"
    HF_DIR="${OUTPUT_BASE}_step${step}_hf"
    if [ -f "${HF_DIR}/model.safetensors" ]; then
        echo "SKIP: ${HF_DIR} already exists"
        continue
    fi
    if [ ! -d "${DCP_DIR}" ]; then
        echo "SKIP: ${DCP_DIR} does not exist"
        continue
    fi
    echo "Converting step ${step}..."
    python ${CONVERT_SCRIPT} "${DCP_DIR}" "${HF_DIR}"
    echo "Done: ${HF_DIR}"
done

echo ""
echo "=== Step 2: Run bf16 evals ==="
# Eval pretrained + all RL checkpoints on 4 general tasks (bf16)
# Run up to 8 evals in parallel across GPUs
GPU=0
PIDS=()

eval_bf16() {
    local ckpt=$1
    local name=$2
    local gpu=$3
    echo "BF16 eval: ${name} on GPU ${gpu}"
    for task in ${BF16_TASKS}; do
        CUDA_VISIBLE_DEVICES=${gpu} python ${EVAL_SCRIPT} --checkpoint "${ckpt}" --bf16 --task ${task} 2>&1 | \
            grep "acc" | grep -v "stderr" | while read line; do echo "${name} bf16 ${task}: ${line}"; done
    done
    # Math tasks with --limit 0.1
    for task in ${MATH_TASKS_LIMITED}; do
        CUDA_VISIBLE_DEVICES=${gpu} python ${EVAL_SCRIPT} --checkpoint "${ckpt}" --bf16 --task ${task} --limit 0.1 2>&1 | \
            grep "acc" | grep -v "stderr" | while read line; do echo "${name} bf16 ${task}(10%): ${line}"; done
    done
}

# Pretrained baseline
eval_bf16 "${PRETRAINED}" "pretrained" 0 &
PIDS+=($!)

# RL checkpoints
gpu=1
for step in "${STEPS[@]}"; do
    HF_DIR="${OUTPUT_BASE}_step${step}_hf"
    if [ ! -f "${HF_DIR}/model.safetensors" ]; then
        echo "SKIP eval: ${HF_DIR} not found"
        continue
    fi
    eval_bf16 "${HF_DIR}" "rl_step${step}" ${gpu} &
    PIDS+=($!)
    gpu=$(( (gpu + 1) % 8 ))
    # Don't exceed 8 parallel jobs
    if [ ${#PIDS[@]} -ge 8 ]; then
        wait "${PIDS[0]}"
        PIDS=("${PIDS[@]:1}")
    fi
done

echo "Waiting for bf16 evals..."
for pid in "${PIDS[@]}"; do
    wait $pid
done
echo "=== bf16 evals complete ==="

echo ""
echo "=== Step 3: Run nvfp4 evals (step 50 and pretrained only) ==="
PIDS=()

eval_nvfp4() {
    local ckpt=$1
    local name=$2
    local gpu=$3
    echo "NVFP4 eval: ${name} on GPU ${gpu}"
    for task in ${BF16_TASKS}; do
        CUDA_VISIBLE_DEVICES=${gpu} python ${EVAL_SCRIPT} --checkpoint "${ckpt}" --task ${task} 2>&1 | \
            grep "acc" | grep -v "stderr" | while read line; do echo "${name} nvfp4 ${task}: ${line}"; done
    done
}

# nvfp4 on pretrained and step 50
eval_nvfp4 "${PRETRAINED}" "pretrained" 0 &
PIDS+=($!)

HF_STEP50="${OUTPUT_BASE}_step50_hf"
if [ -f "${HF_STEP50}/model.safetensors" ]; then
    eval_nvfp4 "${HF_STEP50}" "rl_step50" 1 &
    PIDS+=($!)
fi

echo "Waiting for nvfp4 evals..."
for pid in "${PIDS[@]}"; do
    wait $pid
done
echo "=== nvfp4 evals complete ==="
echo "=== ALL DONE ==="
