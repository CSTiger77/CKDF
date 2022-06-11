import argparse
import sys
import os

sys.path.append("/share/home/kcli/CL_research/iCaRL_ILtFA")
# os.chdir("/share/home/kcli/CL_research/iCaRL_ILtFA")
# print(os.getcwd() )
# from imagenet.alg_model.config.LwF_config import *
from ImageNet.alg_model.config.config import Config
from ImageNet.alg_model.config.LwF_config import LwF_config
from ImageNet.alg_model.LwF import LwF
from public.data import get_multitask_experiment
from public.utils import get_logger


def parse_args():
    # model_name, dataset_name, dataset_path, num_classes, epochs, extracted_layers, rate, tasks, logger,
    # batch_train_logger, batch_size, result_file, memory_budget, norm_exemplars, herding, lr, momentum,
    # weight_decay, optim_type, milestones, KD_temp, gamma, availabel_cudas, seed
    parser = argparse.ArgumentParser(description='PyTorch imagenet100_10_tasks class incremental learning Training')
    parser.add_argument('--model_name', type=str, default=LwF_config.model_name, help='imagenet dataset path')
    parser.add_argument('--dataset_name', type=str, default=LwF_config.dataset_name, help='imagenet dataset name')
    parser.add_argument('--dataset_path', type=str, default=LwF_config.dataset_root, help='imagenet dataset path')
    parser.add_argument('--dataset_json_path', type=str, default=LwF_config.dataset_json_path,
                        help='imagenet dataset json path')
    parser.add_argument('--num_classes', type=int, default=LwF_config.num_classes, help='learning rate')
    parser.add_argument('--hidden_size', type=int, default=LwF_config.hidden_size, help='learning rate')
    parser.add_argument('--rate', type=float, default=LwF_config.rate, help='model size')
    parser.add_argument('--tasks', type=int, default=LwF_config.tasks, help='tasks num')
    parser.add_argument('--log', type=str, default=LwF_config.log, help='path to save log')
    parser.add_argument('--batch_train_log', type=str, default=LwF_config.batch_train_log,
                        help='path to save batch training log')
    parser.add_argument('--batch_size', type=int, default=LwF_config.batch_size, help='batch size')
    parser.add_argument('--result_file', type=str, default=LwF_config.result_file, help='file to save results')
    parser.add_argument('--memory_budget', type=int, default=LwF_config.memory_budget,
                        help='memory_budget to store samples')
    parser.add_argument('--norm_exemplars', type=bool, default=LwF_config.norm_exemplars,
                        help='Do normalize features')
    parser.add_argument('--herding', type=bool, default=LwF_config.herding, help='sampling strategy')
    parser.add_argument('--lr', type=float, default=LwF_config.lr, help='learning rate')
    parser.add_argument('--momentum', type=float, default=LwF_config.momentum, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=LwF_config.weight_decay, help='weight decay')
    parser.add_argument('--optim_type', type=str, default=LwF_config.optim_type, help='optimize type, sgd/adam')
    parser.add_argument('--milestones', type=list, default=LwF_config.milestones, help='optimizer milestones')
    parser.add_argument('--KD_temp', type=float, default=LwF_config.KD_temp, help='distill temp')
    parser.add_argument('--gamma', type=float, default=LwF_config.gamma, help='gamma')
    parser.add_argument('--availabel_cudas', type=str, default=LwF_config.availabel_cudas,
                        help='availabel_cudas to use')
    parser.add_argument('--seed', type=int, default=LwF_config.seed, help='seed')

    parser.add_argument('--epochs', type=int, default=LwF_config.epochs, help='num of training epochs')
    parser.add_argument('--num_workers', type=int, default=LwF_config.num_workers,
                        help='number of worker to load data')
    parser.add_argument('--checkpoint_path', type=str, default=LwF_config.checkpoint_path,
                        help='path for saving trained models')
    parser.add_argument('--print_interval', type=int, default=LwF_config.print_interval, help='print interval')

    parser.add_argument('--is_LwF_MC', type=int, default=LwF_config.is_LwF_MC, help='is_LwF_MC')

    parser.add_argument('--MLP_name', type=str, default=LwF_config.MLP_name, help='cifar dataset path')
    parser.add_argument('--MLP_KD_temp', type=float, default=LwF_config.MLP_KD_temp, help='MLP_KD_temp')
    parser.add_argument('--MLP_KD_temp_2', type=float, default=LwF_config.MLP_KD_temp_2,
                        help='MLP distill temp for input')

    parser.add_argument('--MLP_lr', type=float, default=LwF_config.MLP_lr, help='learning rate')
    parser.add_argument('--MLP_momentum', type=float, default=LwF_config.MLP_momentum, help='momentum')
    parser.add_argument('--MLP_weight_decay', type=float, default=LwF_config.MLP_weight_decay, help='weight decay')
    parser.add_argument('--MLP_epochs', type=int, default=LwF_config.MLP_epochs, help='num of training epochs')
    parser.add_argument('--MLP_milestones', type=list, default=LwF_config.MLP_milestones,
                        help='optimizer milestones')
    parser.add_argument('--MLP_lrgamma', type=float, default=LwF_config.MLP_lrgamma, help='MLP_lrgamma')
    parser.add_argument('--MLP_optim_type', type=str, default=LwF_config.MLP_optim_type,
                        help='optimize type, sgd/adam')
    parser.add_argument('--feature_oversample', type=int, default=LwF_config.feature_oversample,
                        help='feature_oversample')

    parser.add_argument('--MLP_distill_rate', type=float, default=LwF_config.MLP_distill_rate,
                        help='MLP_distill_rate')
    parser.add_argument('--note', type=str, default=LwF_config.note,
                        help='note')

    parser.add_argument('--MLP_rate', type=float, default=LwF_config.MLP_rate, help='MLP_rate')

    # parser.add_argument('--scenario', type=str, default=LwF_config.scenario, help='class or domain')

    return parser.parse_args()


def main():
    # todo
    print(Config.log)
    args = parse_args()
    print(args.log)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    logger = get_logger(args.log)
    batch_train_logger = get_logger(args.batch_train_log)
    print(args.dataset_json_path)
    LwF_model = LwF(model_name=args.model_name, dataset_name=args.dataset_name, dataset_path=args.dataset_path,
                    dataset_json_path=args.dataset_json_path, num_classes=args.num_classes,
                    hidden_size=args.hidden_size, epochs=args.epochs,
                    num_workers=args.num_workers,
                    rate=args.rate, tasks=args.tasks, logger=logger, batch_train_logger=batch_train_logger,
                    batch_size=args.batch_size, result_file=args.result_file, memory_budget=args.memory_budget,
                    norm_exemplars=args.norm_exemplars, herding=args.herding, lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay, optim_type=args.optim_type, milestones=args.milestones,
                    KD_temp=args.KD_temp, gamma=args.gamma, availabel_cudas=args.availabel_cudas,
                    MLP_name=args.MLP_name, MLP_KD_temp=args.MLP_KD_temp, MLP_KD_temp_2=args.MLP_KD_temp_2,
                    MLP_lr=args.MLP_lr, MLP_rate=args.MLP_rate, MLP_momentum=args.MLP_momentum,
                    MLP_epochs=args.MLP_epochs, MLP_milestones=args.MLP_milestones,
                    MLP_weight_decay=args.MLP_weight_decay,
                    MLP_lrgamma=args.MLP_lrgamma, MLP_optim_type=args.MLP_optim_type,
                    MLP_distill_rate=args.MLP_distill_rate,
                    seed=args.seed)
    if args.is_LwF_MC == 0:
        LwF_model.LwF_MC_train_main(args)
    elif args.is_LwF_MC == 1:
        LwF_model.LwF_MC_train_main_balance(args)
    elif args.is_LwF_MC == 2:
        LwF_model.LwF_MC_feature_relearning_train_main(args)
    elif args.is_LwF_MC == 3:
        LwF_model.LwF_MC_feature_relearning_train_main_balance(args)


if __name__ == "__main__":
    main()
