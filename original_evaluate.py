# 
"""
Main script for semantic experiments
Built upon Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""

import argparse
import json
import os
import copy

import matplotlib.pyplot as plt

import wandb
import pprint

import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

# Custom import
from utils.dataset import SICKLE_Dataset
from utils import utae_utils, model_utils
from utils.weight_init import weight_init
from utils.metric import get_metrics, RMSELoss
# torch
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchnet as tnt

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--model",
    default="utae",
    type=str,
    help="Type of architecture to use. Can be one of: (utae/unet3d/fpn/convlstm/convgru/uconvlstm/buconvlstm)",
)
## U-TAE Hyperparameters
parser.add_argument("--encoder_widths", default="[64,128]", type=str)
parser.add_argument("--decoder_widths", default="[32,128]", type=str)
parser.add_argument("--out_conv", default="[32, 16]")
parser.add_argument("--str_conv_k", default=4, type=int)
parser.add_argument("--str_conv_s", default=2, type=int)
parser.add_argument("--str_conv_p", default=1, type=int)
parser.add_argument("--agg_mode", default="att_group", type=str)
parser.add_argument("--encoder_norm", default="group", type=str)
parser.add_argument("--n_head", default=16, type=int)
parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--d_k", default=4, type=int)
parser.add_argument("--fusion",default=0, type=int)
parser.add_argument("--best_path", default=None, type=str)

# Set-up parameters
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of data loading workers"
)
parser.add_argument("--seed", default=0, type=int, help="Random seed")
# Training parameters
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs per fold")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--lr", default=1e-1, type=float, help="Learning rate")
# parser.add_argument("--wd", default=1e-2, type=float, help="weight decay")
parser.add_argument("--num_classes", default=2, type=int)
parser.add_argument("--ignore_index", default=-999, type=int)
parser.add_argument("--pad_value", default=0, type=float)
parser.add_argument("--padding_mode", default="reflect", type=str)
parser.add_argument("--resume", default="", type=str, help="enter run path to resume")
parser.add_argument("--run_id", default="", type=str, help="enter run id to resume")
parser.add_argument("--wandb", action='store_true', help="debug?")
parser.add_argument('--satellites', type=str, default="[S2]")
parser.add_argument('--run_name', type=str, default="trial")
parser.add_argument('--exp_name', type=str, default="utae")
parser.add_argument('--task', type=str, default="crop_type",
                    help="Available Tasks are crop_type, sowing_date, transplanting_date, harvesting_date, crop_yield")
parser.add_argument('--actual_season', action='store_true', help="whether to consider actual season or not.")
parser.add_argument('--data_dir', type=str, default="../sickle/data")
parser.add_argument('--use_augmentation', type=bool, default=False)

list_args = ["encoder_widths", "decoder_widths", "out_conv", "satellites"]
parser.set_defaults(cache=False)


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(CFG):
    if CFG.wandb:
        if not os.path.exists(CFG.run_path):
            os.makedirs(CFG.run_path)
        elif CFG.resume:
            pass
        else:
            CFG.run_path = CFG.run_path + f"_{time.time()}"
            print("Run path already exist changed run path to ", CFG.run_path)
            os.makedirs(CFG.run_path)
    else:
        CFG.run_path += "_debug"
        os.makedirs(CFG.run_path, exist_ok=True)


def checkpoint(log, config):
    with open(
            os.path.join(config.run_path, "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


def set_seed(seed=42):
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # For reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(f"> SEEDING DONE {seed}")


def log_wandb(loss, metrics, phase="train"):
    f1_macro, acc, iou, f1_paddy, f1_non_paddy, \
    acc_paddy, acc_non_paddy, iou_paddy, iou_non_paddy, (y_pred, y_true) = metrics
    y_pred, y_true = y_pred.tolist(), y_true.tolist()
    if CFG.wandb:
        wandb.log(
            {
                f"{phase}_loss": loss,
                f"{phase}_f1_macro": f1_macro,
                f"{phase}_acc": acc,
                f"{phase}_iou": iou,
                f"{phase}_f1_paddy": f1_paddy,
                f"{phase}_f1_non_paddy": f1_non_paddy,
                f"{phase}_acc": acc,
                f"{phase}_acc_paddy": acc_paddy,
                f"{phase}_acc_non_paddy": acc_non_paddy,
                f"{phase}_iou_paddy": iou_paddy,
                f"{phase}_iou_non_paddy": iou_non_paddy,
            })
        if phase == "val":
            wandb.log({f"{phase}_conf_mat": wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred, probs=None,
                                                                        class_names=["Paddy", "Non Paddy"])})


def iterate_test(
        model, data_loader, device=None, CFG=None,
):
    predictions = None
    t_start = time.time()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Test Inference")
    for i, batch in pbar:
        if device is not None:
            batch = recursive_todevice(batch, device)
        data, masks = batch
        with torch.no_grad():
            y_pred = model(data)
        
        if predictions is None:
            predictions = y_pred
        else:
            predictions = torch.cat([predictions, y_pred], dim=0)

        # Just for Monitoring
        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix(
            gpu_mem=f"{mem:0.2f} GB",
        )
        
    t_end = time.time()
    total_time = t_end - t_start
    print("Test inference time : {:.1f}s".format(total_time))
    
    # Save predictions as a pickle file
    import pickle
    with open(os.path.join(CFG.run_path, "predictions_test.pkl"), "wb") as f:
        pickle.dump(predictions, f)

    return predictions

def main(CFG):
    prepare_output(CFG)
    device = torch.device(CFG.device)

    # Dataset definition
    data_dir = CFG.data_dir
    df = pd.read_csv(os.path.join(data_dir, "sickle_dataset_tabular.csv"))
    
    # Only use the test split for inference
    test_df = df[df.SPLIT == "test"].reset_index(drop=True)

    dt_args = dict(
        data_dir=data_dir,
        satellites=CFG.satellites,
        ignore_index=CFG.ignore_index,
        actual_season=CFG.actual_season
    )
    dt_test = SICKLE_Dataset(df=test_df, **dt_args)

    collate_fn = lambda x: utae_utils.pad_collate(x, pad_value=CFG.pad_value)
    test_loader = data.DataLoader(
        dt_test,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CFG.num_workers,
    )

    if CFG.fusion == 0:
        model = model_utils.Fusion_model(CFG)
    elif CFG.fusion == 1:
        model = model_utils.Fusion_model_PXL_ATTN(CFG)
    elif CFG.fusion == 2:
        model = model_utils.Fusion_model_CONCAT_ATTN(CFG)
    elif CFG.fusion == 3:
        model = model_utils.Fusion_model_CONCAT_ATTN_PIXELWISE(CFG)
    elif CFG.fusion == 4:
        model = model_utils.fusion_model_pxl_extralayers(CFG)
    elif CFG.fusion == 5:
        model = model_utils.CrossFusionModel(CFG)
    elif CFG.fusion == 6:
        model = model_utils.MultiheadFusionModel(CFG)
    elif CFG.fusion == 7:
        model = model_utils.CHN_ATTN(CFG)
    elif CFG.fusion == 8:
        model = model_utils.CombinedFusionModel(CFG)
    elif CFG.fusion == 9:
        model = model_utils.CombinedFusionModel2(CFG)
    elif CFG.fusion == 10:
        model = model_utils.EnhancedFusionModel(CFG)
    elif CFG.fusion == 11:
        model = model_utils.EnhancedFusionModel1(CFG)
    elif CFG.fusion == 12:
        model = model_utils.CrossAttentionFusion(CFG)
    elif CFG.fusion == 13:
        model = model_utils.CrossAttentionFusionBasic(CFG)
    elif CFG.fusion == 14:
        model = model_utils.CrossAttentionFusion3(CFG)

    model = model.to(device)
    CFG.N_params = utae_utils.get_ntrainparams(model)

    # Load the best checkpoint
    best_checkpoint = torch.load(
        os.path.join(CFG.best_path, "checkpoint_41.pth.tar")
    )
    model.load_state_dict(best_checkpoint["model"])
    model.eval()

    # Run inference on the test set
    predictions = iterate_test(
        model,
        data_loader=test_loader,
        device=device,
        CFG=CFG,
    )
    
    print(f"Test inference complete. Predictions saved to {CFG.run_path}/predictions_test.pkl")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    CFG = parser.parse_args()
    set_seed(CFG.seed)
    for k, v in vars(CFG).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            try:
                CFG.__setattr__(k, list(map(int, v.split(","))))
            except:
                CFG.__setattr__(k, list(map(str, v.split(","))))

    CFG.exp_name = CFG.task

    # Adjust parameters for regression task
    if CFG.task != "crop_type":
        CFG.num_classes = 1

    CFG.run_path = f"runs/wacv_sickle_test/{CFG.exp_name}/{CFG.run_name}"
    satellite_metadata = {
        "S2": {
            "bands": ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'],
            "rgb_bands": [3, 2, 1],
            "mask_res": 10,
            "img_size": (32, 32),
        },
        "S1": {
            "bands": ['VV', 'VH'],
            "rgb_bands": [0, 1, 0],
            "mask_res": 10,
            "img_size": (32, 32),
        },
        "L8": {
            "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"],
            "rgb_bands": [3, 2, 1],
            "mask_res": 30,
            "img_size": (32, 32),
        },
    }
    required_sat_data = {}
    for satellite in CFG.satellites:
        required_sat_data[satellite] = satellite_metadata[satellite]
    CFG.satellites = required_sat_data
    CFG.primary_sat = list(required_sat_data.keys())[0]
    CFG.img_size = required_sat_data[CFG.primary_sat]["img_size"]

    main(CFG)
