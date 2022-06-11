import os
import sys

from CIFAR.alg_model.config.config import Config
from public.path import CIFAR100_path

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

globle_rate = 1
globle_is_LwF_MC = 2
globle_task = 5
MLP_layers = 3
test_num = 2
globle_MLP_KD_temp = 2
globle_MLP_KD_temp_2 = 2
globle_KD_temp = 2
globle_MLP_distill_rate = 1
# globle_MLP_distill_rate = 1 / (globle_MLP_KD_temp ** 2)
globle_MLP_rate = 16

assert globle_is_LwF_MC == 0 or globle_is_LwF_MC == 2
global log_type
if globle_rate == 1:
    log_type = ""
elif globle_rate == 1 / 4:
    log_type = "1_4"
elif globle_rate == 1 / 8:
    log_type = "1_8"
elif globle_rate == 1 / 2:
    log_type = "1_2"

if globle_is_LwF_MC == 0:
    dir = "./cifar100EXP_resnet34_Sep/cifar10pretrain/LwF_EXP/Test_unbiasModel/task{}/rate_{}/test_{}".format(globle_task,
                                                                                                         log_type,
                                                                                                         test_num)
    # dir = "./cifar100EXP_resnet34_Sep/cifar10_Test/LwF_EXP/LwF_OD/task{}/rate_{}/test_{}".format(globle_task,
    #                                                                                                   log_type,
    #                                                                                                   test_num)
elif globle_is_LwF_MC == 2:
    # dir = "./cifar100EXP_resnet34_Sep/cifar10pretrain/LwF_EXP/Test_Ablation_NoAnchor/task{}/rate_{}/test_{}".format(
    #     globle_task,
    #     log_type,
    #     test_num)
    dir = "./cifar100EXP_resnet34_Sep/cifar10_Test/LwF_EXP/Test_Ablation_NoOversample/task{}/rate_{}/test_{}".format(
        globle_task,
        log_type,
        test_num)
    # dir = "./cifar100EXP_resnet34_Sep/cifar10_Test/LwF_EXP/FC_LwF_OD/task{}/rate_{}/test_{}".format(
    #     globle_task,
    #     log_type,
    #     test_num)
    # dir = "./cifar100EXP_resnet34_Sep/cifar10_Test/LwF_EXP/FC_LwF/task{}/rate_{}/test_{}".format(
    #     globle_task,
    #     log_type,
    #     test_num)
    # dir = "./cifar100EXP_resnet34_Aug/cifar10pretrain/LwF_EXP/Test_Ablation_Notoversample/task{}/rate_{}/test_{}".format(
    #     globle_task,
    #     log_type,
    #     test_num)

elif globle_is_LwF_MC == 3:
    dir = "./cifar100EXP_resnet34_Sep/cifar10pretrain/LwF_EXP/FC_LwF_MC_balance/B128/task{}/MLP{}/test{}/rate_{}".format(
        globle_task,
        MLP_layers,
        test_num,
        log_type)


class LwF_config:
    NewData_to_oneclass = False
    val_current_task = False
    KD_temp = globle_KD_temp
    tasks = globle_task
    MLP_rate = globle_MLP_rate
    is_LwF_MC = globle_is_LwF_MC
    MLP_KD_temp = globle_MLP_KD_temp
    MLP_KD_temp_2 = globle_MLP_KD_temp_2
    note = " Test Ablation No Anchor"
    model_name = "resnet34"
    dataset_root = CIFAR100_path
    dataset_name = "CIFAR10"
    lr = 0.1
    gamma = 0.2
    momentum = 0.9
    weight_decay = 5e-4
    milestones = [30, 60, 90, 100, 110]
    epochs = 120
    batch_size = 128
    num_classes = 10

    MLP_name = "MLP_{}_layers".format(MLP_layers)
    MLP_epochs = 60
    MLP_momentum = 0.9
    if MLP_layers == 2:
        MLP_lr = 0.01
        MLP_weight_decay = 1e-5
        MLP_milestones = [20, 40, 50, 55]
    elif MLP_layers == 3:
        MLP_lr = 0.001
        MLP_weight_decay = 1e-4
        MLP_milestones = [20, 40, 55]
    MLP_optim_type = "adam"
    MLP_lrgamma = 0.2
    MLP_distill_rate = globle_MLP_distill_rate

    num_workers = 4
    print_interval = 30
    checkpoint_path = dir + "/cifar100_checkpoints"  # Path to store model
    rate = globle_rate
    hidden_size = int(512 * rate)
    seed = 0
    optim_type = "sgd"
    availabel_cudas = "4"
    memory_budget = 2000
    norm_exemplars = True
    herding = True
    extracted_layers = ["avg_pool", "fc"]

    log_post = log_type
    log = dir + "/{}_{}_{}.log".format(dataset_name, log_post, memory_budget)
    batch_train_log = dir + "/{}_batchtrain_{}_{}.log".format(dataset_name, log_post, memory_budget)  # Path to save log
    result_file = dir + "/{}_{}_{}.result".format(dataset_name, log_post, memory_budget)
