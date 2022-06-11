import os
import sys

from CIFAR.alg_model.config.config import Config
from public.path import CIFAR100_path, ILSVRC2012_path, imagenet100_json_path

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

globle_rate = 1
globle_is_LwF_MC = 2  # 0 :"LwF", 2: "FC_LwF"
globle_task = 10
MLP_layers = 3
test_num = 1
globle_MLP_KD_temp = 1
globle_MLP_KD_temp_2 = 1
globle_MLP_distill_rate = 1
globle_MLP_rate = 16
global log_type
if globle_rate == 1:
    log_type = "1"
elif globle_rate == 1 / 4:
    log_type = "1_4"
elif globle_rate == 1 / 8:
    log_type = "1_8"
assert globle_is_LwF_MC == 0 or globle_is_LwF_MC == 2

if globle_is_LwF_MC == 0:
    if "forCL_1" in imagenet100_json_path:
        dir = "./Imagenet100EXP_resnet18_Aug/Imagenet100pretrain/LwF_MC_EXP/LwF_MC/DATAforCL_1/task{}/test{}/rate_{}".format(
            globle_task,
            test_num,
            log_type)
    else:
        dir = "./Imagenet100EXP_resnet18_Aug/Imagenet100pretrain/LwF_MC_EXP/LwF_MC/DATAforCL_2/task{}/test{}/rate_{}".format(
            globle_task,
            test_num,
            log_type)

elif globle_is_LwF_MC == 2:
    if "forCL_1" in imagenet100_json_path:
        dir = "./Imagenet100EXP_resnet18_Aug/Imagenet100pretrain/LwF_MC_EXP/FC_LwF_MC/DATAforCL_1/task{}/test{}/rate_{}".format(
            globle_task,
            test_num,
            log_type)
    else:
        dir = "./Imagenet100EXP_resnet18_Aug/Imagenet100pretrain/LwF_MC_EXP/FC_LwF_MC/DATAforCL_2/task{}/test{}/rate_{}".format(
            globle_task,
            test_num,
            log_type)


class LwF_config:
    MLP_distill_rate = globle_MLP_distill_rate
    MLP_KD_temp = globle_MLP_KD_temp
    MLP_KD_temp_2 = globle_MLP_KD_temp_2
    is_LwF_MC = globle_is_LwF_MC
    MLP_rate = globle_MLP_rate
    tasks = globle_task
    rate = globle_rate
    note = "examplar set not random crop but flip to train FE_cls; NCM examplar."
    feature_oversample = 0
    model_name = "resnet18"
    dataset_root = ILSVRC2012_path
    dataset_json_path = imagenet100_json_path
    dataset_name = "ImageNet100"
    lr = 0.1
    gamma = 0.2
    momentum = 0.9
    weight_decay = 5e-4
    milestones = [20, 40, 60, 70, 80]
    epochs = 90
    batch_size = 128
    num_classes = 100
    hidden_size = int(512 * rate)

    num_workers = 4
    print_interval = 30
    checkpoint_path = dir + "/imagenet100_checkpoints"  # Path to store model
    seed = 0
    optim_type = "sgd"
    KD_temp = 2.
    availabel_cudas = "1,2,3,4"
    memory_budget = 2000
    norm_exemplars = True
    herding = True

    MLP_name = "MLP_{}_layers".format(MLP_layers)
    MLP_epochs = 60
    MLP_lr = 0.001
    MLP_momentum = 0.9
    MLP_weight_decay = 1e-4
    MLP_milestones = [20, 40, 55]
    MLP_optim_type = "adam"
    MLP_lrgamma = 0.2

    log_post = log_type
    log = dir + "/{}_{}_{}.log".format(dataset_name, log_post, memory_budget)
    batch_train_log = dir + "/{}_batchtrain_{}_{}.log".format(dataset_name, log_post, memory_budget)  # Path to save log
    result_file = dir + "/{}_{}_{}.result".format(dataset_name, log_post, memory_budget)
