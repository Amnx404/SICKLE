#!/bin/bash
python evaluate.py --data_dir ./dataset --satellite [S1,S2,L8] --task crop_type --run_name "evaluate_test_seed0_batch_crop_type" --model utae --fusion 0 --best_path "runs/wacv_2024_seed0/crop_type/CONCAT_dataset_seed0_1"
python evaluate.py --data_dir ./dataset --satellite [S1,S2,L8] --task crop_type --run_name "evaluate_test_seed0_batch_crop_type_CROSS_ALL_NORM_DROP_dataset_seed0_1" --model utae --fusion 12 --best_path "runs/wacv_2024_seed0/crop_type/CROSS_ALL_NORM_DROP_dataset_seed0_1"
# python original_evaluate.py --data_dir ./dataset --satellite [S1,S2,L8] --task crop_yield --run_name "evaluate_test_seed0_batch_crop_yield" --model utae --fusion 12 --best_path "runs/wacv_2024_seed1/crop_yield/CROSS_ALL_NORM_DROP_dataset_seed1_debug"
# python evaluate.py --data_dir $1 --satellite $2 --task sowing_date --run_name "${2}_${3}" --model $3 --best_path "runs/wacv_2024/sowing_date/${2}_${3}" &&
# python evaluate.py --data_dir $1 --satellite $2 --task transplanting_date --run_name "${2}_${3}" --model $3 --best_path "runs/wacv_2024/transplanting_date/${2}_${3}" &&
# python evaluate.py --data_dir $1 --satellite $2 --task harvesting_date --run_name "${2}_${3}" --model $3 --best_path "runs/wacv_2024/harvesting_date/${2}_${3}" &&
# python evaluate.py --data_dir $1 --satellite $2 --task crop_yield --run_name "${2}_${3}" --model $3 --best_path "runs/wacv_2024/crop_yield/${2}_${3}" &&
# python evaluate.py --data_dir $1 --satellite $2 --task crop_yield --run_name "${2}_${3}_season" --model $3 --best_path "runs/wacv_2024/crop_yield/${2}_${3}_season" --actual_season