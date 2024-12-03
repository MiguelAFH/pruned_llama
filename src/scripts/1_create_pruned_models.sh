#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd $SCRIPT_DIR

# This script will generate the three models with different pruning percentages
# The best model from these ones will determine the pruning ratio for later experiments


# Define the model name, base output directory, and pruning ratios
MODEL="meta-llama/Llama-3.2-1B-Instruct"
BASE_OUTPUT_DIR="../../models"
# PRUNING_RATIOS=(0.1 0.2 0.3 0.5 0.7)
PRUNING_RATIOS=(0.5 0.7)

# Loop over the pruning ratios
for RATIO in "${PRUNING_RATIOS[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/Llama-3.2-1B-Instruct_pruned_${RATIO}"
    echo "Pruning model: $MODEL. Output dir: $OUTPUT_DIR"
    python3 ../1_create_pruned_models.py --model "$MODEL" \
                                         --pruning_ratio "$RATIO" \
                                         --output_dir "$OUTPUT_DIR"
done