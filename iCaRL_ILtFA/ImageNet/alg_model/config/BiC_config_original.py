import os
import sys

from public.path import ILSVRC2012_path, imagenet100_json_path, imagenet100_json_path_forBiC

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

globle_dataset_json_path = imagenet100_json_path_forBiC
data_num = 1 if "1" in globle_dataset_json_path else 2
globle_rate = 1
globle_is_BiC = 1
globle_MLP_KD_temp = 1
MLP_layers = 3
globle_task = 20
globle_oversample = False
test_num = 2
globle_train_method = 0
globle_MLP_distill_rate = 1
globle_MLP_rate = 16
global log_type
if globle_rate == 1:
    log_type = "1"
elif globle_rate == 1 / 4:
    log_type = "1_4"
elif globle_rate == 1 / 8:
    log_type = "1_8"
assert globle_is_BiC == 1 or globle_is_BiC == 3

if globle_is_BiC == 0:
    dir = "./Imagenet100EXP_July/Imagenet100pretrain/BiC_EXP/BiC/DATAforCL_{}/task{}/test{}/rate_{}".format(
        data_num, globle_task, test_num, log_type)
elif globle_is_BiC == 1:
    dir = "./Imagenet100EXP_July/Imagenet100pretrain/BiC_EXP/BiC_bias/DATAforCL_{}/task{}/test{}/rate_{}".format(
        data_num, globle_task, test_num, log_type)
elif globle_is_BiC == 2:
    dir = "./Imagenet100EXP_July/Imagenet100pretrain/BiC_EXP/FC_BiC/DATAforCL_{}/task{}/MLP{}/test{}/rate_{}".format(
        data_num, globle_task, MLP_layers, test_num, log_type)
elif globle_is_BiC == 3:
    dir = "./Imagenet100EXP_July/Imagenet100pretrain/BiC_EXP/FC_BiC_bias/DATAforCL_{}/task{}/MLP{}/test{}/rate_{}".format(
        data_num, globle_task, MLP_layers, test_num, log_type)


class BiC_config:
    tasks = globle_task
    oversample = globle_oversample
    MLP_KD_temp = globle_MLP_KD_temp
    train_method = globle_train_method
    MLP_distill_rate = globle_MLP_distill_rate
    sample_seed = 0
    note = "bias layer train_method: {}, MLP_KD_temp={}".format(train_method, MLP_KD_temp)
    is_BiC = globle_is_BiC
    model_name = "resnet18"
    dataset_root = ILSVRC2012_path
    dataset_json_path = globle_dataset_json_path
    dataset_name = "ImageNet100"
    lr = 0.1
    gamma = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    milestones = [30, 60, 80, 90]
    epochs = 100
    batch_size = 128
    num_classes = 100

    MLP_name = "MLP_{}_layers".format(MLP_layers)
    MLP_epochs = 60
    MLP_momentum = 0.9
    MLP_rate = globle_MLP_rate
    if MLP_layers == 2:
        MLP_lr = 0.01
        MLP_weight_decay = 1e-5
        MLP_milestones = [15, 25, 40, 50, 55]
    elif MLP_layers == 3:
        MLP_lr = 0.001
        MLP_weight_decay = 1e-4
        MLP_milestones = [20, 40, 55]
    MLP_optim_type = "adam"
    MLP_lrgamma = 0.2

    num_workers = 4
    print_interval = 30
    checkpoint_path = dir + "/imagenet100_checkpoints"  # Path to store model
    rate = globle_rate
    hidden_size = int(512 * rate)
    seed = 0
    optim_type = "sgd"
    KD_temp = 2.
    availabel_cudas = "4"
    memory_budget = 2000
    norm_exemplars = True
    herding = True
    extracted_layers = ["avg_pool", "fc"]

    log_post = log_type
    log = dir + "/{}_{}_{}.log".format(dataset_name, log_post, memory_budget)
    batch_train_log = dir + "/{}_batchtrain_{}_{}.log".format(dataset_name, log_post, memory_budget)  # Path to save log
    result_file = dir + "/{}_{}_{}.result".format(dataset_name, log_post, memory_budget)
