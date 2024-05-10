#!/bin/bash

# Define the base command
BASE_CMD="python evaluate.py --satellite [S1,S2,L8] --task crop_yield --model utae"

# Array of fusion models and their corresponding names
declare -A fusion_models
# fusion_models[0]="SICKLE"
# fusion_models[11]="GATED"
# # fusion_models[9]="UTAE_multihead_cross_attn"
# fusion_models[12]="CROSS_ALL_NORM"
fusion_models[13]="CROSS_ALL_NORM_DROP"

# fusion_models[8]="UTAE_attn_with_cross"
# fusion_models[10]="SEB_dialated_pxlwise"
# fusion_models[1]="Fusion_model_PXL_ATTN"
# fusion_models[5]="CrossFusionModel"
# fusion_models[6]="MultiheadFusionModel"

# Array of datasets
datasets=("dataset_20_per" "dataset_50_per" "dataset")

for seed in 1; do

    for data_dir_to_eval in "${datasets[@]}"; do
        # Loop over each fusion model
        # for data_dir_model in "${datasets[@]}"; do
        for data_dir_model in "dataset"; do

            for fusion in "${!fusion_models[@]}"; do
                
                # Construct the run name based on the dataset, fusion model, and seed
                run_name="val_${fusion_models[$fusion]}_direval_${data_dir_to_eval}_dirmodel_${data_dir_model}_seed${seed}"

                # Build the full command
                CMD="$BASE_CMD --data_dir ./$data_dir_to_eval --fusion $fusion --run_name \"$run_name\" --best_path \"runs/wacv_2024_seed${seed}/crop_yield/${fusion_models[$fusion]}_${data_dir_model}_seed${seed}_debug\""
                # CMD="$BASE_CMD --data_dir ./$data_dir_to_eval --fusion $fusion --run_name \"$run_name\" --best_path \"runs/wacv_2024_seed${seed}/crop_yield/${fusion_models[$fusion]}_debug\""

                # Echo and run the command
                echo "Running: $CMD"
                eval $CMD
            done
        done
    done
done
