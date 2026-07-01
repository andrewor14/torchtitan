#!/bin/bash
# Eval GPT-OSS-20B RL checkpoints (steps 50/75/100), bf16 + nvfp4.
#
# bf16 HF checkpoints are pre-converted by convert_dcp_to_hf.py into
#   $OUT/step{N}_bf16. NVFP4 produced by ModelOpt. MATH-500 uses the custom
#   math500_verify task (math_verify scorer; antlr-independent) because stock
#   minerva/hendrycks need antlr 4.11 which conflicts with verl's hydra 4.9 pin.
#
# NOTE: step 100 is degenerate (grad_norm spiked to 31.98 at the final step;
#   greedy decoding collapses to a reserved token). We bf16-eval it to quantify
#   the collapse but skip NVFP4 quantization for it.
set -uo pipefail

TT=/data/users/andrewor/torchtitan
OUT=/data/users/andrewor/checkpoints/gptoss_rl_eval
LOGDIR=/data/users/andrewor/logs/gptoss_rl_eval
TASKS=gsm8k_verl,math500_verify
INCLUDE=$TT/eval_tasks
HEALTHY="50 75"          # checkpoints fit for nvfp4
ALL="50 75 100"          # bf16-eval all (100 to quantify degeneration)

mkdir -p "$LOGDIR"
cd "$TT"

echo "=== Phase 1: NVFP4 quantization (healthy steps) ===" | tee -a "$LOGDIR/driver.log"
for s in $HEALTHY; do
    NVFP4=$OUT/step${s}_nvfp4
    [ -f "$NVFP4/config.json" ] && { echo "step$s nvfp4 exists, skip" | tee -a "$LOGDIR/driver.log"; continue; }
    echo "Quantizing step$s -> nvfp4 ..." | tee -a "$LOGDIR/driver.log"
    python scripts/quantize_gptoss_nvfp4_modelopt.py \
        --input-dir "$OUT/step${s}_bf16" --output-dir "$NVFP4" --num-calib-samples 128 \
        > "$LOGDIR/quant_step${s}.log" 2>&1 || { echo "QUANT FAILED step$s"; tail -5 "$LOGDIR/quant_step${s}.log"; }
done

echo "=== Phase 2: lm_eval (parallel across GPUs) ===" | tee -a "$LOGDIR/driver.log"
GPU=0; PIDS=()
run_eval () {
    local model_path=$1 label=$2 extra=$3 gpu=$4
    [ -f "$model_path/config.json" ] || { echo "skip $label (missing)"; return; }
    echo "eval $label on GPU $gpu" | tee -a "$LOGDIR/driver.log"
    CUDA_VISIBLE_DEVICES=$gpu VLLM_USE_FLASHINFER_MOE_FP16=0 python -m lm_eval \
        --model vllm \
        --model_args "pretrained=${model_path},dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192,enforce_eager=True${extra}" \
        --tasks "$TASKS" --apply_chat_template --batch_size auto \
        --include_path "$INCLUDE" --output_path "$LOGDIR/results_${label}" \
        > "$LOGDIR/eval_${label}.log" 2>&1 &
    PIDS+=($!)
}

for s in $ALL;     do run_eval "$OUT/step${s}_bf16"  "step${s}_bf16"  "" "$GPU"; GPU=$((GPU+1)); done
for s in $HEALTHY; do run_eval "$OUT/step${s}_nvfp4" "step${s}_nvfp4" ",quantization=modelopt_fp4" "$GPU"; GPU=$((GPU+1)); done

echo "Waiting for ${#PIDS[@]} eval jobs ..." | tee -a "$LOGDIR/driver.log"
wait

echo "=== RESULTS ===" | tee -a "$LOGDIR/driver.log"
for f in "$LOGDIR"/eval_*.log; do
    lbl=$(basename "$f" .log | sed 's/^eval_//')
    g=$(grep "gsm8k_verl" "$f" | grep -oP "exact_match.*?\|\s*\K[0-9.]+" | head -1)
    m=$(grep "math500_verify" "$f" | grep -oP "math_verify.*?\|\s*\K[0-9.]+" | head -1)
    echo "$lbl  GSM8K=$g  MATH500=$m" | tee -a "$LOGDIR/driver.log"
done
echo "=== ALL DONE ===" | tee -a "$LOGDIR/driver.log"
