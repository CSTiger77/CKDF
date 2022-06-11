import os
import sys

from CIFAR.alg_model.config.config import Config
from public.path import CIFAR100_path

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

globle_rate = 1
globle_is_iCaRL = 1
globle_task = 20
MLP_layers = 3
test_num = 4
globle_MLP_KD_temp = 1
globle_MLP_distill_rate = 1
# globle_MLP_distill_rate = 1 / (globle_MLP_KD_temp ** 2)

assert globle_is_iCaRL == 0 or globle_is_iCaRL == 1
global log_type
if globle_rate == 1:
    log_type = ""
elif globle_rate == 1 / 4:
    log_type = "1_4"
elif globle_rate == 1 / 8:
    log_type = "1_8"
elif globle_rate == 1 / 2:
    log_type = "1_2"

# dir = "./cifar100EXP_resnet32/cifar10pretrain/visualization/task20/rate_{}".format(log_type)
if globle_is_iCaRL == 0:
    dir = "./cifar100EXP_resnet34_July/cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task{}/test{}/rate_{}".format(
        globle_task,
        test_num,
        log_type)
elif globle_is_iCaRL == 1:
    dir = "./cifar100EXP_resnet34_July/cifar10pretrain/_Test_iCaRL_original/FC_iCaRL_cmp/task{}/MLP{}/test{}/rate_{}".format(
        globle_task,
        MLP_layers,
        test_num,
        log_type)


class iCaRL_config:
    note = "loss distill * MLP_KD_temp ** 2"
    is_iCaRL = globle_is_iCaRL
    MLP_KD_temp = globle_MLP_KD_temp
    model_name = "resnet34"
    dataset_root = CIFAR100_path
    dataset_name = "CIFAR100"
    lr = 1
    gamma = 0.2
    momentum = 0.9
    weight_decay = 1e-5
    milestones = [49, 63, 81]
    epochs = 90
    batch_size = 128
    num_classes = 100

    MLP_name = "MLP_{}_layers".format(MLP_layers)
    MLP_epochs = 60
    MLP_lr = 0.001
    MLP_momentum = 0.9
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
    KD_temp = 1.
    availabel_cudas = "4"
    memory_budget = 2000
    norm_exemplars = True
    herding = True
    extracted_layers = ["avg_pool", "fc"]
    tasks = globle_task

    log_post = log_type
    log = dir + "/{}_{}_{}.log".format(dataset_name, log_post, memory_budget)
    batch_train_log = dir + "/{}_batchtrain_{}_{}.log".format(dataset_name, log_post, memory_budget)  # Path to save log
    result_file = dir + "/{}_{}_{}.result".format(dataset_name, log_post, memory_budget)
