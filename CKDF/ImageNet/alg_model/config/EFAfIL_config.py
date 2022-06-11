import os
import sys

from public.data import DATASET_CONFIGS

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import numpy as np
from torchvision import transforms

from CIFAR.alg_model.config.config import Config
from public.path import CIFAR100_path, ILSVRC2012_path, imagenet100_json_path, imagenet100_json_path_forBiC

globle_dataset_json_path = imagenet100_json_path_forBiC
data_num = 1 if "1" in globle_dataset_json_path else 2
globle_rate = 1
globle_is_iCaRL_LwF_BiC = 0  # 0 denotes EFAfIL_iCaRL; 1 denotes EFAfIL_LwF; 2 denotes EFAfIL_BiC
globle_task = 20
MLP_layers = 3
test_num = 4
globle_train_method = 1
globle_oversample = False
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
elif globle_rate == 1 / 2:
    log_type = "1_2"
assert globle_is_iCaRL_LwF_BiC == 0 or globle_is_iCaRL_LwF_BiC == 1 or globle_is_iCaRL_LwF_BiC == 2
if globle_is_iCaRL_LwF_BiC == 0:
    if data_num == 1:
        dir = "./Imagenet100EXP_resnet18_Sep/Imagenet100pretrain/DATAforCL_1/EFAfIL_iCaRL/task{}/MLP{}/test{}/" \
              "rate_{}".format(globle_task,
                               MLP_layers,
                               test_num,
                               log_type)
    else:
        dir = "./Imagenet100EXP_resnet18_Sep/Imagenet100pretrain/DATAforCL_2/EFAfIL_iCaRL/task{}/MLP{}/test{}/" \
              "rate_{}".format(globle_task,
                               MLP_layers,
                               test_num,
                               log_type)
elif globle_is_iCaRL_LwF_BiC == 1:
    if data_num == 1:
        dir = "./Imagenet100EXP_resnet18_FR/Imagenet100pretrain/DATAforCL_1/EFAfIL_LwF/task{}/MLP{}/test{}/" \
              "rate_{}".format(globle_task,
                               MLP_layers,
                               test_num,
                               log_type)
    else:
        dir = "./Imagenet100EXP_resnet18_FR/Imagenet100pretrain/DATAforCL_2/EFAfIL_LwF/task{}/MLP{}/test{}/" \
              "rate_{}".format(globle_task,
                               MLP_layers,
                               test_num,
                               log_type)
elif globle_is_iCaRL_LwF_BiC == 2:
    if data_num == 1:
        dir = "./Imagenet100EXP_resnet18_FR/Imagenet100pretrain/DATAforCL_1/EFAfIL_BiC/task{}/MLP{}/test{}/" \
              "rate_{}".format(globle_task,
                               MLP_layers,
                               test_num,
                               log_type)
    else:
        dir = "./Imagenet100EXP_resnet18_FR/Imagenet100pretrain/DATAforCL_2/EFAfIL_BiC/task{}/MLP{}/test{}/" \
              "rate_{}".format(globle_task,
                               MLP_layers,
                               test_num,
                               log_type)


class EFAfIL_config:
    FM_oversample = False
    use_feature_replay_in_new_model = False
    dataset_json_path = globle_dataset_json_path
    MLP_distill_rate = globle_MLP_distill_rate
    oversample = globle_oversample
    train_method = globle_train_method
    is_iCaRL_LwF_BiC = globle_is_iCaRL_LwF_BiC
    tasks = globle_task
    MLP_KD_temp = globle_MLP_KD_temp
    MLP_KD_temp_2 = globle_MLP_KD_temp_2
    sample_seed = 0
    note = "EFAfIL_BiC: use flip to train FE_cls; Not oversample examplar set; soft + cls;" \
           "MLP_KD_temp={}".format(MLP_KD_temp)
    use_FM = False
    model_name = "resnet18"
    MLP_name = "MLP_{}_layers".format(MLP_layers)
    discriminate_name = "discriminate_3_layers"
    log_post = log_type
    dataset_name = "ImageNet100"
    resume = dir + "/latest.pth"
    evaluate = None  # 测试模型，evaluate为模型地址
    dataset_path = ILSVRC2012_path
    num_classes = 100
    rate = globle_rate
    discriminate_note_rate = 1
    use_exemplars = True

    Exemple_memory_budget = 1900
    img_size = int((DATASET_CONFIGS["imagenet100"]["size"] ** 2) * DATASET_CONFIGS["imagenet100"]["channels"])
    hidden_size = int(512 * rate)
    Feature_memory_budget = int((2000 - Exemple_memory_budget) * int(img_size / hidden_size))

    epochs = 90
    norm_exemplars = True
    herding = True
    FM_reTrain = True
    use_NewfeatureSpace = False
    use_discriminate = False
    batch_size = 128
    CNN_lr = 0.1
    CNN_momentum = 0.9
    CNN_weight_decay = 5e-4
    CNN_milestones = [20, 40, 60, 70, 80]

    MLP_epochs = 60
    MLP_momentum = 0.9
    if MLP_layers == 2:
        MLP_rate = globle_MLP_rate
        MLP_lr = 0.01
        MLP_weight_decay = 1e-5
        MLP_milestones = [20, 40, 50, 55]
    elif MLP_layers == 3:
        MLP_rate = globle_MLP_rate
        MLP_lr = 0.001
        MLP_weight_decay = 1e-4
        MLP_milestones = [20, 40, 55]

    svm_sample_type = "oversample"
    svm_max_iter = 500000

    optim_type = "sgd"
    MLP_optim_type = "adam"

    KD_temp = 1 if is_iCaRL_LwF_BiC == 0 else 2

    kd_lamb = 1
    fd_gamma = 1
    lrgamma = 0.2

    MLP_lrgamma = 0.2

    sim_alpha = 1
    availabel_cudas = "1"
    num_workers = 4
    checkpoint_path = dir + "/EFAfIL_checkpoints"  # Path to store model
    seed = 0
    print_interval = 20
    log = dir + "/{}_{}_{}_{}.log".format(dataset_name, log_post, Exemple_memory_budget, Feature_memory_budget)
    batch_train_log = dir + "/{}_batchtrain_{}_{}_{}.log".format(dataset_name, log_post, Exemple_memory_budget,
                                                                 Feature_memory_budget)  # Path to save log
    result_file = dir + "{}_{}_{}_{}.result".format(dataset_name, log_post, Exemple_memory_budget,
                                                    Feature_memory_budget)
