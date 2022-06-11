import os
import sys
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from public.path import CIFAR100_path
import numpy as np
import torchvision.transforms as transforms

globle_rate = 1 / 4
global log_type
if globle_rate == 1:
    log_type = ""
elif globle_rate == 1 / 4:
    log_type = "1_4"
elif globle_rate == 1 / 8:
    log_type = "1_8"

dir = "./log/featureExtractor_cifar10_preTrain_log/rate_{}".format(log_type)

class Config:
    model_name = "resnet34"
    log_post = log_type
    checkpoint_path = "./checkpoints"  # Path to store model
    resume = "./checkpoints/latest.pth"
    evaluate = None  # 测试模型，evaluate为模型地址
    dataset_root = CIFAR100_path
    train_dataset_path = CIFAR100_path
    val_dataset_path = CIFAR100_path
    availabel_cudas = '0'
    rate = 1 / 4
    # download CIFAR100 from here:https://www.cs.toronto.edu/~kriz/cifar.html

    pretrained = False
    seed = 0
    num_classes = 100

    milestones = [30, 60, 90]
    epochs = 120
    batch_size = 128
    lr = 0.1
    gamma = 0.2
    momentum = 0.9
    weight_decay = 5e-4
    num_workers = 4
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
    log = dir + "/{}_{}.log".format("cifar10", log_post)
    batch_train_log = dir + "/{}_batchtrain_{}.log".format("cifar10", log_post)  # Path to save log
    result_file = dir + "/{}_{}".format("cifar10", log_post)
