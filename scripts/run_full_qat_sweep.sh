#!/bin/bash
# Run all QAT experiments sequentially, converting each to HF after training.
# Usage: GPUS=2,3,4,5 ./scripts/run_full_qat_sweep.sh <sft_hf_path>
set -e

GPUS=${GPUS:-"2,3,4,5"}
SFT_HF_PATH=${1:-"./outputs/sft_s100_lr1e-5_bs2_hf"}
NPROC=4

# Define all experiments: "lr steps"
EXPERIMENTS=(
    # Wave 1: lr=1e-6
    "1e-6 5" "1e-6 10" "1e-6 15" "1e-6 20"
    # Wave 2: lr=2e-6
    "2e-6 5" "2e-6 10" "2e-6 15" "2e-6 20"
    # Wave 3: lr=3e-6
    "3e-6 5" "3e-6 10" "3e-6 15" "3e-6 20"
    # Wave 4: lr=5e-6
    "5e-6 3" "5e-6 5" "5e-6 7" "5e-6 10" "5e-6 15" "5e-6 20"
    # Wave 5: lr=1e-5 (skip 5 and 10 if already done)
    "1e-5 2" "1e-5 3" "1e-5 7" "1e-5 15"
    # Wave 6: lr=2e-5
    "2e-5 1" "2e-5 2" "2e-5 3" "2e-5 5"
)

for exp in "${EXPERIMENTS[@]}"; do
    LR=$(echo $exp | awk '{print $1}')
    STEPS=$(echo $exp | awk '{print $2}')
    OUTPUT_DIR="./outputs/qat_lr${LR}_s${STEPS}"
    HF_DIR="${OUTPUT_DIR}_hf"

    # Skip if HF checkpoint already exists
    if [ -f "${HF_DIR}/model.safetensors" ]; then
        echo "=== SKIP: ${HF_DIR} already exists ==="
        continue
    fi

    echo ""
    echo "========================================"
    echo "QAT: lr=${LR}, steps=${STEPS}"
    echo "========================================"

    # Train
    CUDA_VISIBLE_DEVICES=${GPUS} PYTORCH_ALLOC_CONF="expandable_segments:True" \
    torchrun --nproc_per_node=${NPROC} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
        --local-ranks-filter 0 --role rank --tee 3 \
        -m torchtitan.train --module qwen3 --config sft_qwen3_30b_a3b_arc_qat \
        --training.steps ${STEPS} \
        --optimizer.lr ${LR} \
        --checkpoint.initial_load_path ${SFT_HF_PATH} \
        --dump_folder ${OUTPUT_DIR} \
        --checkpoint.no-last-save-in-hf

    # Convert DCP to HF
    echo "Converting ${OUTPUT_DIR}/checkpoint/step-${STEPS} to HF..."
    python scripts/convert_dcp_to_hf.py \
        "${OUTPUT_DIR}/checkpoint/step-${STEPS}" \
        "${HF_DIR}"

    echo "=== DONE: ${HF_DIR} ==="
done

echo ""
echo "All QAT experiments complete!"
