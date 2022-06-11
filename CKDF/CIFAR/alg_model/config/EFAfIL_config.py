import os
import sys

from public.data import DATASET_CONFIGS

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import numpy as np
from torchvision import transforms

from CIFAR.alg_model.config.config import Config
from public.path import CIFAR100_path

globle_rate = 1
globle_is_iCaRL_LwF_BiC = 2 # 0 denotes EFAfIL_iCaRL; 1 denotes EFAfIL_LwF; 2 denotes EFAfIL_BiC
globle_task = 5
MLP_layers = 3
test_num = 4
globle_train_method = 3
globle_oversample = True
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
    dir = "./cifar100EXP_resnet34_Orc/cifar10pretrain/EFAfIL_iCaRL_bias_cRT/task{}/MLP{}/rate_{}/test{}".format(
        globle_task,
        MLP_layers,
        log_type,
        test_num)
    # dir = "./cifar100EXP_resnet34_Orc/cifar10pretrain/EFAfIL_iCaRL/Test_split_memory/task{}/MLP{}/rate_{}/test{}".format(
    #         globle_task,
    #         MLP_layers,
    #         log_type,
    #         test_num)
elif globle_is_iCaRL_LwF_BiC == 1:
    dir = "./cifar100EXP_resnet34_Orc/cifar10pretrain/EFAfIL_LwF/Test_split_memory/task{}/MLP{}/rate_{}/test{}".format(globle_task,
                                                                                                          MLP_layers,
                                                                                                          log_type,
                                                                                                          test_num)
elif globle_is_iCaRL_LwF_BiC == 2:
    dir = "./cifar100EXP_resnet34_Orc/cifar10pretrain/EFAfIL_BiC/task{}/MLP{}/rate_{}/test{}".format(globle_task,
                                                                                                          MLP_layers,
                                                                                                          log_type,
                                                                                                          test_num)
    # dir = "./cifar100EXP_resnet34_Orc/cifar10_Test/EFAfIL_BiC/task{}/MLP{}/rate_{}/test{}".format(globle_task,
    #                                                                                                  MLP_layers,
    #                                                                                                  log_type,
    #                                                                                                  test_num)


class EFAfIL_config:
    bias_or_cRT = 1 # 0 use bias_layer methods to retrain cls of FM_cls; 1 use cRT methods to retrain cls of FM_cls
    train_bias_cls_of_FMcls = False
    feature_BatchSize_rate = 1/4
    is_iCaRL_LwF_BiC = globle_is_iCaRL_LwF_BiC
    KD_temp = 1 if is_iCaRL_LwF_BiC == 0 else 2
    test_feature_cls_sim = False
    use_feature_replay_in_new_model = False
    FM_oversample = True
    MLP_distill_rate = globle_MLP_distill_rate
    oversample = globle_oversample
    train_method = globle_train_method
    tasks = globle_task
    MLP_KD_temp = globle_MLP_KD_temp
    MLP_KD_temp_2 = globle_MLP_KD_temp_2
    sample_seed = 0
    note = "feature_batch_size=1"
    use_FM = False
    model_name = "resnet34"
    MLP_name = "MLP_{}_layers".format(MLP_layers)
    discriminate_name = "discriminate_{}_layers".format(MLP_layers)
    log_post = log_type
    dataset_name = "CIFAR100"
    resume = dir + "/latest.pth"
    evaluate = None  # 测试模型，evaluate为模型地址
    dataset_path = CIFAR100_path
    train_dataset_path = CIFAR100_path
    val_dataset_path = CIFAR100_path
    rate = globle_rate
    discriminate_note_rate = 1
    use_exemplars = True

    img_size = int((DATASET_CONFIGS[dataset_name]["size"] ** 2) * DATASET_CONFIGS[dataset_name]["channels"])
    Exemple_memory_budget = 1900
    hidden_size = int(512 * rate)
    Feature_memory_budget = int((2000 - Exemple_memory_budget) * int(img_size / hidden_size))

    norm_exemplars = True
    herding = True
    FM_reTrain = True
    use_NewfeatureSpace = False
    use_discriminate = False
    CNN_weight_decay = 1e-5 if globle_is_iCaRL_LwF_BiC == 0 else 5e-4
    # epochs = 120
    # batch_size = 128
    # CNN_lr = 0.1
    # lrgamma = 0.2
    # CNN_momentum = 0.9
    # CNN_milestones = [30, 60, 90, 100, 110]
    # num_classes = 100
    if dataset_name == "CIFAR10" or is_iCaRL_LwF_BiC != 0:
    # if True:
        epochs = 120
        batch_size = 128
        CNN_lr = 0.1
        lrgamma = 0.2
        CNN_momentum = 0.9
        CNN_milestones = [30, 60, 90, 100, 110]
        num_classes = 100 if dataset_name == "CIFAR100" else 10
    else:
        epochs = 90
        batch_size = 128
        CNN_lr = 1
        lrgamma = 0.2
        CNN_momentum = 0.9
        CNN_milestones = [49, 63, 81]
        num_classes = 100

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


    kd_lamb = 1
    fd_gamma = 1

    MLP_lrgamma = 0.2

    sim_alpha = 1
    availabel_cudas = "0"
    num_workers = 4
    checkpoint_path = dir + "/EFAfIL_checkpoints"  # Path to store model
    seed = 0
    print_interval = 20
    log = dir + "/{}_{}_{}_{}.log".format(dataset_name, log_post, Exemple_memory_budget, Feature_memory_budget)
    batch_train_log = dir + "/{}_batchtrain_{}_{}_{}.log".format(dataset_name, log_post, Exemple_memory_budget,
                                                                 Feature_memory_budget)  # Path to save log
    result_file = dir + "{}_{}_{}_{}.result".format(dataset_name, log_post, Exemple_memory_budget,
                                                    Feature_memory_budget)
