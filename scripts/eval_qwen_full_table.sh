#!/bin/bash
# Full Qwen3-30B-A3B comparison table: no-QAT RL (150-200), QAT+IS (5/10/15).
# (QAT+IS+FQ already evaluated separately.) bf16 + nvfp4, gsm8k_verl + math500_verify.
# nvfp4 uses per-task-calibrated checkpoints (gsm8k -> -nvfp4-gsm8k,
# math -> -nvfp4-minerva_math), matching the prior QAT eval methodology.
set -uo pipefail

OUT=/data/users/andrewor/torchtitan/outputs
LOG=/data/users/andrewor/logs/qwen_full_table
INCLUDE=/data/users/andrewor/torchtitan/eval_tasks
mkdir -p "$LOG"; cd /data/users/andrewor/torchtitan

# Build job list: "label|model_path|tasks"
JOBS=()
add() { JOBS+=("$1|$2|$3"); }

for s in 150 160 170 180 190 200; do
  d=$OUT/rl_continued_v3_06-09
  add "noqat_step${s}_bf16"        "$d/step${s}_hf"                  "gsm8k_verl,math500_verify"
  add "noqat_step${s}_gsm8k_nvfp4" "$d/step${s}_hf-nvfp4-gsm8k"      "gsm8k_verl"
  add "noqat_step${s}_math_nvfp4"  "$d/step${s}_hf-nvfp4-minerva_math" "math500_verify"
done
for s in 5 10 15; do
  d=$OUT/qat_is_v3_06-18
  add "qatis_step${s}_bf16"        "$d/step${s}_hf"                  "gsm8k_verl,math500_verify"
  add "qatis_step${s}_gsm8k_nvfp4" "$d/step${s}_hf-nvfp4-gsm8k"      "gsm8k_verl"
  add "qatis_step${s}_math_nvfp4"  "$d/step${s}_hf-nvfp4-minerva_math" "math500_verify"
done

echo "Total jobs: ${#JOBS[@]}" | tee "$LOG/driver.log"

run_one() {
  local label=$1 model=$2 tasks=$3 gpu=$4
  local extra=""
  [[ "$model" == *nvfp4* ]] && extra=",quantization=modelopt_fp4"
  CUDA_VISIBLE_DEVICES=$gpu VLLM_USE_FLASHINFER_MOE_FP16=0 python -m lm_eval \
    --model vllm \
    --model_args "pretrained=${model},dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192,enforce_eager=True${extra}" \
    --tasks "$tasks" --apply_chat_template --batch_size auto \
    --include_path "$INCLUDE" --output_path "$LOG/results_${label}" \
    > "$LOG/eval_${label}.log" 2>&1
}

# Launch in batches of 8 (one per GPU).
NG=8; i=0
while [ $i -lt ${#JOBS[@]} ]; do
  pids=()
  for g in $(seq 0 $((NG-1))); do
    [ $i -lt ${#JOBS[@]} ] || break
    IFS='|' read -r label model tasks <<< "${JOBS[$i]}"
    echo "[$i] $label -> GPU $g" | tee -a "$LOG/driver.log"
    run_one "$label" "$model" "$tasks" "$g" &
    pids+=($!); i=$((i+1))
  done
  wait "${pids[@]}"
  echo "batch done ($i/${#JOBS[@]})" | tee -a "$LOG/driver.log"
done

echo "=== ALL DONE ===" | tee -a "$LOG/driver.log"
