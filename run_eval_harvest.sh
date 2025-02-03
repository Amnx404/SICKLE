#!/bin/bash

python evaluate.py --data_dir ./dataset --satellite [S1,S2,L8] --task harvesting_date --run_name "evaluate_test_seed0_batch_harvesting_date_CROSS_ALL_NORM_BATCH_DROP_dataset_50_per_seed0_1_dataset_unet3d" --model unet3d --fusion 14 --best_path "latest_models/to_send5/wacv_2024_seed0/harvesting_date/CROSS_ALL_NORM_BATCH_DROP_dataset_50_per_seed0_1_unet3d"