#!/bin/bash

# Define the base command
BASE_CMD="python train.py --satellite [S1,S2,L8] --task crop_yield --model utae"

# Array of fusion models and their corresponding names
declare -A fusion_models
fusion_models[12]="CROSS_ALL_NORM_DROP"

# Array of datasets
datasets=("dataset" "dataset_20_per" "dataset_50_per")


for seed in {0..1}; do

    for data_dir in "${datasets[@]}"; do
        # Loop over each fusion model
        for fusion in "${!fusion_models[@]}"; do
        
            # Construct the run name based on the dataset, fusion model, and seed
            run_name="${fusion_models[$fusion]}_${data_dir}_seed${seed}"

            # Build the full command
            CMD="$BASE_CMD --data_dir ./$data_dir --fusion $fusion --run_name \"$run_name\" --seed $seed"

            # Echo and run the command
            echo "Running: $CMD"
            eval $CMD
        done
    done
done
