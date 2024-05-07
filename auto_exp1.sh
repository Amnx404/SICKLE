#!/bin/bash

# Define the base command
BASE_CMD="python train.py --data_dir ./dataset --satellite [S1,S2,L8] --task crop_yield --model utae"

# Array of fusion models and their corresponding names
declare -A fusion_models
fusion_models[1]="Fusion_model_PXL_ATTN"
fusion_models[5]="CrossFusionModel"
fusion_models[6]="MultiheadFusionModel"

# Loop over each fusion model
for fusion in "${!fusion_models[@]}"; do
    # Loop over each seed value
    for seed in {0..3}; do
        # Construct the run name based on the fusion model and seed
        run_name="${fusion_models[$fusion]}_seed${seed}"

        # Build the full command
        CMD="$BASE_CMD --fusion $fusion --run_name \"$run_name\" --seed $seed"

        # Echo and run the command
        echo "Running: $CMD"
        eval $CMD
    done
done
