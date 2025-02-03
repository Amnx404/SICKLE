#!/bin/bash

# python evaluate.py --data_dir ./dataset --satellite [S1,S2,L8] --task harvesting_date --run_name "evaluate_val_crop_type_CONCAT_dataset_seed0_1_dataset_unet3d" --model unet3d --fusion 0 --best_path "wacv_2024_seed3/crop_type/CONCAT_dataset_seed0_1"
python evaluate.py --data_dir ./dataset --satellite [S1,S2,L8] --task harvesting_date --run_name "evaluate_val_crop_type_CROSS_ALL_NORM_BATCH_DROP_dataset_seed0_1_dataset_unet3d" --model unet3d --fusion 14 --best_path "wacv_2024_seed3/crop_type/CROSS_ALL_NORM_BATCH_DROP_dataset_seed3_1_unet3d"