python train.py --data_dir ./dataset_20_per/ --satellite "L8" --task crop_yield --run_name "utae_L8_dataset_20_per_seed_0" --model "utae" --seed 0 --wandb
python train.py --data_dir ./dataset_20_per/ --satellite "L8" --task crop_type --run_name "unet3d_L8_dataset_20_per_seed_0" --model "unet3d" --seed 0 --wandb
# python train.py --data_dir ./dataset_20_per/ --satellite "S2" --task crop_yield --run_name "utae_S2_dataset_20_per_seed_0" --model "utae" --seed 0 --wandb
# python train.py --data_dir ./dataset_20_per/ --satellite "S2" --task crop_type --run_name "unet3d_S2_data_set_20_per_seed_0" --model "unet3d" --seed 0 --wandb

# do for dataset 50 as well

python train.py --data_dir ./dataset_50_per/ --satellite "L8" --task crop_yield --run_name "utae_L8_dataset_50_per_seed_0" --model "utae" --seed 0 --wandb
python train.py --data_dir ./dataset_50_per/ --satellite "L8" --task crop_type --run_name "unet3d_L8_dataset_50_per_seed_0" --model "unet3d" --seed 0 --wandb
python train.py --data_dir ./dataset_50_per/ --satellite "S2" --task crop_yield --run_name "utae_S2_dataset_50_per_seed_0" --model "utae" --seed 0 --wandb
python train.py --data_dir ./dataset_50_per/ --satellite "S2" --task crop_type --run_name "unet3d_S2_data_set_50_per_seed_0" --model "unet3d" --seed 0 --wandb
