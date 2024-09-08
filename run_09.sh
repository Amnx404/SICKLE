#!/bin/bash

# Define the base command
BASE_CMD="python train.py --satellite [S1,S2,L8] --model unet3d --wandb"

# Array of fusion models and their corresponding names
declare -A fusion_models
# fusion_models[0]="CONCAT"
fusion_models[14]="CROSS_ALL_NORM_BATCH_DROP"
# fusion_models[12]="CROSS_ALL_NORM_DROP"

# Array of datasets
datasets=( "dataset")

# Array of tasks
tasks=( "crop_type" "harvesting_date" "sowing_date")
# "crop_type" "harvesting_date"
for seed in {3..4}; do

    for task in "${tasks[@]}"; do

        for data_dir in "${datasets[@]}"; do
            # Loop over each fusion model

            for fusion in "${!fusion_models[@]}"; do
            
                # Construct the run name based on the dataset, fusion model, and seed
                run_name="${fusion_models[$fusion]}_${data_dir}_seed${seed}_1_unet3d"

                # Build the full command
                CMD="$BASE_CMD --data_dir ./$data_dir --fusion $fusion --run_name \"$run_name\" --seed $seed --task $task"

                # Echo and run the command
                echo "Running: $CMD"
                eval $CMD
            done
        done
    done
done
