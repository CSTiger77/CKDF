import os
import sys

from CIFAR.alg_model.config.config import Config
from public.path import CIFAR100_path, ILSVRC2012_path, imagenet100_json_path

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

globle_rate = 1
global log_type
if globle_rate == 1:
    log_type = "1"
elif globle_rate == 1 / 4:
    log_type = "1_4"
elif globle_rate == 1 / 8:
    log_type = "1_8"

if "forCL_1" in imagenet100_json_path:
    dir = "./Imagenet100EXP/Imagenet100pretrain/LwF_MC_EXP/LwF_MC_DATAforCL_1/task10/rate_{}".format(log_type)
else:
    dir = "./Imagenet100EXP/Imagenet100pretrain/LwF_MC_EXP/LwF_MC_DATAforCL_2/task10/rate_{}".format(log_type)


class iCaRL_config:
    is_iCaRL = 1
    model_name = "resnet18"
    dataset_root = ILSVRC2012_path
    dataset_json_path = imagenet100_json_path
    dataset_name = "ImageNet100"
    lr = 2
    gamma = 0.2
    momentum = 0.9
    weight_decay = 1e-5
    milestones = [20, 30, 40, 50]
    epochs = 60
    batch_size = 128
    rate = globle_rate
    num_classes = 100
    hidden_size = int(512 * rate)

    num_workers = 4
    print_interval = 30
    checkpoint_path = dir + "/imagenet100_checkpoints"  # Path to store model
    seed = 0
    optim_type = "sgd"
    KD_temp = 1.
    availabel_cudas = "1,2,3,4"
    memory_budget = 2000
    norm_exemplars = True
    herding = True
    tasks = 10

    MLP_name = "MLP_2_layers"
    MLP_epochs = 60
    MLP_lr = 0.001
    MLP_momentum = 0.9
    MLP_weight_decay = 5e-4
    MLP_milestones = [20, 40, 55]
    MLP_optim_type = "adam"
    MLP_lrgamma = 0.2

    log_post = log_type
    log = dir + "/{}_{}_{}.log".format(dataset_name, log_post, memory_budget)
    batch_train_log = dir + "/{}_batchtrain_{}_{}.log".format(dataset_name, log_post, memory_budget)  # Path to save log
    result_file = dir + "/{}_{}_{}.result".format(dataset_name, log_post, memory_budget)
