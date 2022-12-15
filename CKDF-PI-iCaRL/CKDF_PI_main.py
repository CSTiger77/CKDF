import sys

sys.path.append("../CKDF-PI-iCaRL")
from lib.approach.CKDF import CKDF_handler
from lib.ExemplarManager import ExemplarManager
from lib.dataset import *
from lib.config import CKDF_cfg, update_config
from lib.utils.utils import (
    create_logger,
)
import torch
import os
import argparse
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description="codes for CKDF")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="./configs/CKDF_cifar100.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    update_config(CKDF_cfg, args)
    logger, log_file = create_logger(CKDF_cfg, "log")
    warnings.filterwarnings("ignore")
    # split_seleted_data = {0: [52, 1, 30, 96], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11], 3: [12, 13, 14, 15], 4: [16, 17, 18, 19], 5: [20, 21, 22, 23], 6: [24, 25, 26, 27], 7: [28, 29, 2, 31], 8: [32, 33, 34, 35], 9: [36, 37, 38, 39], 10: [40, 41, 42, 43], 11: [44, 45, 46, 47], 12: [48, 49, 50, 51], 13: [0, 53, 54, 55], 14: [56, 57, 58, 59], 15: [60, 61, 62, 63], 16: [64, 65, 66, 67], 17: [68, 69, 70, 71], 18: [72, 73, 74, 75], 19: [76, 77, 78, 79], 20: [80, 81, 82, 83], 21: [84, 85, 86, 87], 22: [88, 89, 90, 91], 23: [92, 93, 94, 95], 24: [3, 97, 98, 99]}
    split_seleted_data = None
    dataset_split_handler = eval(CKDF_cfg.DATASET.dataset)(CKDF_cfg, split_selected_data=split_seleted_data)
    if CKDF_cfg.availabel_cudas != "":
        os.environ['CUDA_VISIBLE_DEVICES'] = CKDF_cfg.availabel_cudas
        device_ids = [i for i in range(len(CKDF_cfg.availabel_cudas.strip().split(',')))]
        print(device_ids)
    device = torch.device("cpu" if CKDF_cfg.CPU_MODE else "cuda")
    exemplar_img_transform_for_val = dataset_split_handler.val_test_dataset_transform if \
        CKDF_cfg.exemplar_manager.store_original_imgs else None
    exemplar_img_transform_for_train = transforms.Compose([*AVAILABLE_TRANSFORMS[dataset_split_handler.dataset_name]
    ['train_transform']]) if CKDF_cfg.exemplar_manager.store_original_imgs else None
    exemplar_manager = ExemplarManager(CKDF_cfg.exemplar_manager.memory_budget,
                                       CKDF_cfg.exemplar_manager.mng_approach,
                                       CKDF_cfg.exemplar_manager.store_original_imgs,
                                       CKDF_cfg.exemplar_manager.norm_exemplars,
                                       CKDF_cfg.exemplar_manager.centroid_order,
                                       img_transform_for_val=exemplar_img_transform_for_val,
                                       img_transform_for_train=exemplar_img_transform_for_train,
                                       device=device)

    CKDF_handler = CKDF_handler(dataset_split_handler, exemplar_manager, CKDF_cfg, logger, device)
    if "imagenet" in CKDF_cfg.DATASET.dataset_name:
        CKDF_handler.CKDF_train_main_for_local_dataset()
    else:
        CKDF_handler.CKDF_train_main()
