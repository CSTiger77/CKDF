import os
import sys


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import numpy as np
from torchvision import transforms

from CIFAR.alg_model.config.config import Config
from public.path import CIFAR100_path


globle_rate = 1 / 4
global log_type
if globle_rate == 1:
    log_type = ""
elif globle_rate == 1 / 4:
    log_type = "1_4"
elif globle_rate == 1 / 8:
    log_type = "1_8"

dir = "./log_3/EFAfIL_improve_cifar10pretrain/imgs_clsdistillsim_allsimCLS_II_prefeaturedistill_sim_cls_log/rate_{}".format(log_type)


class EFAfIL_improve_config:
    model_name = "resnet34"
    MLP_name = "MLP_3_layers"
    log_post = log_type
    dataset_name = "CIFAR100"
    resume = dir + "/latest.pth"
    evaluate = None  # 测试模型，evaluate为模型地址
    dataset_path = CIFAR100_path
    train_dataset_path = CIFAR100_path
    val_dataset_path = CIFAR100_path
    num_classes = 100
    tasks = 10
    rate = globle_rate
    use_exemplars = True
    hidden_size = int(512 * rate)
    Exemple_memory_budget = 500
    Feature_memory_budget = 35000
    epochs = 120
    norm_exemplars = True
    herding = True
    batch_size = 128
    CNN_lr = 0.1
    CNN_momentum = 0.9
    CNN_weight_decay = 5e-4
    CNN_milestones = [30, 60, 90]

    MLP_epochs = 60
    MLP_lr = 0.01
    MLP_momentum = 0.9
    MLP_weight_decay = 5e-4
    MLP_milestones = [20, 50, 80, 110]

    svm_sample_type = "undersample"
    svm_max_iter = 500000

    optim_type = "sgd"
    MLP_optim_type = "adam"

    KD_temp = 1

    kd_lamb = 1
    fd_gamma = 1
    lrgamma = 0.2

    MLP_lrgamma = 0.2

    sim_alpha = 1
    availabel_cudas = "1"
    num_workers = 4
    checkpoint_path = dir + "/EFAfIL_checkpoints"  # Path to store model
    seed = 0
    print_interval = 30
    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    train_dataset_init = {
        "root": train_dataset_path,
        "train": True,
        "download": True,
        "transform": train_transform
    }
    val_dataset_init = {
        "root": val_dataset_path,
        "train": False,
        "download": True,
        "transform": val_transform
    }
    log = dir + "/{}_{}_{}_{}.log".format(dataset_name, log_post, Exemple_memory_budget, Feature_memory_budget)
    batch_train_log = dir + "/{}_batchtrain_{}_{}_{}.log".format(dataset_name, log_post, Exemple_memory_budget,
                                                                 Feature_memory_budget)  # Path to save log
    result_file = dir + "{}_{}_{}_{}.result".format(dataset_name, log_post, Exemple_memory_budget,
                                                                 Feature_memory_budget)
