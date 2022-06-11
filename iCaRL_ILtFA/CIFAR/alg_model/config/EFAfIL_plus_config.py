import os
import sys

from public.data import DATASET_CONFIGS

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import numpy as np
from torchvision import transforms

from public.path import CIFAR100_path

globle_rate = 1 / 4
global log_type
if globle_rate == 1:
    log_type = ""
elif globle_rate == 1 / 4:
    log_type = "1_4"
elif globle_rate == 1 / 8:
    log_type = "1_8"
elif globle_rate == 1 / 2:
    log_type = "1_2"

dir = "./cifar100EXP_rateCMP/cifar10pretrain/EFAfIL_plus/MLP3_Nobalance/task2/NoFMRetrain/test_4/rate_{}".format(log_type)
# dir = "./cifar100EXP_rateCMP/cifar10pretrain_EFAfIL_plus_MLP2/discriminate/task10/test_4_imgsallcls_2/rate_{}".format(log_type)


class EFAfIL_plus_config:
    note = "current model usr Not loss sim, FA Not Add loss sim"
    use_FM = False
    model_name = "resnet34"
    MLP_name = "MLP_3_layers"
    discriminate_name = "discriminate_3_layers"
    log_post = log_type
    dataset_name = "CIFAR100"
    resume = dir + "/latest.pth"
    evaluate = None  # 测试模型，evaluate为模型地址
    dataset_path = CIFAR100_path
    train_dataset_path = CIFAR100_path
    val_dataset_path = CIFAR100_path
    num_classes = 100

    tasks = 2
    rate = globle_rate
    discriminate_note_rate = 16
    use_exemplars = True

    img_size = int((DATASET_CONFIGS[dataset_name]["size"]**2) * DATASET_CONFIGS[dataset_name]["channels"])
    Exemple_memory_budget = 1800
    hidden_size = int(512 * rate)
    Feature_memory_budget = int((2000 - Exemple_memory_budget) * int(img_size / hidden_size))

    epochs = 120
    norm_exemplars = True
    herding = True
    # FM_reTrain = True
    # use_NewfeatureSpace = True
    FM_reTrain = False
    use_NewfeatureSpace = False
    use_discriminate = False
    balance = False
    batch_size = 512
    CNN_lr = 0.1
    CNN_momentum = 0.9
    CNN_weight_decay = 5e-4
    CNN_milestones = [30, 60, 90, 100, 110]

    MLP_epochs = 60
    MLP_lr = 0.001
    MLP_momentum = 0.9
    MLP_weight_decay = 5e-4
    MLP_milestones = [20, 40, 55]

    svm_sample_type = "oversample"
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
    print_interval = 10
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
