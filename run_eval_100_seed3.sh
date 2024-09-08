#!/bin/bash

python evaluate.py --data_dir ./dataset --satellite [S1,S2,L8] --task crop_yield --run_name "evaluate_test_seed0_batch_crop_yield_CONCAT_dataset_seed3_1_dataset_utae" --model utae --fusion 0 --best_path "runs/wacv_2024_seed3/crop_yield/CONCAT_dataset_seed3_1" > output_results_verbose/evaluate_test_seed0_batch_crop_yield_CONCAT_dataset_seed3_1_dataset_utae_output.txt 2>&1
python evaluate.py --data_dir ./dataset --satellite [S1,S2,L8] --task crop_yield --run_name "evaluate_test_seed0_batch_crop_yield_CROSS_ALL_NORM_DROP_dataset_seed3_1_dataset_utae" --model utae --fusion 12 --best_path "runs/wacv_2024_seed3/crop_yield/CROSS_ALL_NORM_DROP_dataset_seed3_1" > output_results_verbose/evaluate_test_seed0_batch_crop_yield_CROSS_ALL_NORM_DROP_dataset_seed3_1_dataset_utae_output.txt 2>&1
python evaluate.py --data_dir ./dataset --satellite [S1,S2,L8] --task crop_yield --run_name "evaluate_test_seed0_batch_crop_yield_CROSS_ALL_NORM_BATCH_DROP_dataset_seed3_1_dataset_utae" --model utae --fusion 14 --best_path "runs/wacv_2024_seed3/crop_yield/CROSS_ALL_NORM_BATCH_DROP_dataset_seed3_1" > output_results_verbose/evaluate_test_seed0_batch_crop_yield_CROSS_ALL_NORM_BATCH_DROP_dataset_seed3_1_dataset_utae_output.txt 2>&1