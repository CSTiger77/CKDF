import argparse
import sys

sys.path.append("/share/home/kcli/CL_research/iCaRL_ILtFA")
from CIFAR.alg_model.EFAfILBiC import EFAfILBiC
from CIFAR.alg_model.config.EFAfILBiC_config import EFAfILBiC_config
from CIFAR.alg_model.ILtFA import ILtFA

from public.utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100_10_tasks class incremental learning Training')
    parser.add_argument('--model_name', type=str, default=EFAfILBiC_config.model_name, help=' CNN model_name')
    parser.add_argument('--MLP_name', type=str, default=EFAfILBiC_config.MLP_name, help='cifar dataset path')
    parser.add_argument('--dataset_name', type=str, default=EFAfILBiC_config.dataset_name, help='cifar dataset path')
    parser.add_argument('--dataset_path', type=str, default=EFAfILBiC_config.dataset_path, help='cifar dataset path')
    parser.add_argument('--num_classes', type=int, default=EFAfILBiC_config.num_classes, help='cifar dataset path')
    parser.add_argument('--tasks', type=int, default=EFAfILBiC_config.tasks, help='learning rate')
    parser.add_argument('--rate', type=float, default=EFAfILBiC_config.rate, help='model size')
    parser.add_argument('--use_exemplars', type=bool, default=EFAfILBiC_config.use_exemplars,
                        help='Do normalize features')
    parser.add_argument('--hidden_size', type=int, default=EFAfILBiC_config.hidden_size, help='learning rate')
    parser.add_argument('--Exemple_memory_budget', type=int, default=EFAfILBiC_config.Exemple_memory_budget,
                        help='learning rate')
    parser.add_argument('--Feature_memory_budget', type=int, default=EFAfILBiC_config.Feature_memory_budget,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=EFAfILBiC_config.epochs, help='learning rate')
    parser.add_argument('--norm_exemplars', type=bool, default=EFAfILBiC_config.norm_exemplars,
                        help='Do normalize features')
    parser.add_argument('--herding', type=bool, default=EFAfILBiC_config.herding, help='sampling strategy')
    parser.add_argument('--FM_reTrain', type=bool, default=EFAfILBiC_config.FM_reTrain,
                        help='FM_reTrain or not')
    parser.add_argument('--use_NewfeatureSpace', type=bool, default=EFAfILBiC_config.use_NewfeatureSpace,
                        help='use_NewfeatureSpace or not')
    parser.add_argument('--oversample', type=bool, default=EFAfILBiC_config.oversample, help='sampling strategy')
    parser.add_argument('--batch_size', type=int, default=EFAfILBiC_config.batch_size, help='batch size')

    parser.add_argument('--CNN_lr', type=float, default=EFAfILBiC_config.CNN_lr, help='learning rate')
    parser.add_argument('--CNN_momentum', type=float, default=EFAfILBiC_config.CNN_momentum, help='momentum')
    parser.add_argument('--CNN_weight_decay', type=float, default=EFAfILBiC_config.CNN_weight_decay,
                        help='weight decay')
    parser.add_argument('--CNN_milestones', type=list, default=EFAfILBiC_config.CNN_milestones,
                        help='optimizer milestones')
    parser.add_argument('--optim_type', type=str, default=EFAfILBiC_config.optim_type, help='optimize type, sgd/adam')
    parser.add_argument('--MLP_optim_type', type=str, default=EFAfILBiC_config.MLP_optim_type,
                        help='optimize type, sgd/adam')
    parser.add_argument('--KD_temp', type=float, default=EFAfILBiC_config.KD_temp, help='distill temp')
    parser.add_argument('--kd_lamb', type=float, default=EFAfILBiC_config.kd_lamb, help='kd_lamb')
    parser.add_argument('--fd_gamma', type=float, default=EFAfILBiC_config.fd_gamma, help='fd_gamma')
    parser.add_argument('--lrgamma', type=float, default=EFAfILBiC_config.lrgamma, help='lrgamma')

    parser.add_argument('--MLP_lr', type=float, default=EFAfILBiC_config.MLP_lr, help='learning rate')
    parser.add_argument('--MLP_momentum', type=float, default=EFAfILBiC_config.MLP_momentum, help='momentum')
    parser.add_argument('--MLP_weight_decay', type=float, default=EFAfILBiC_config.MLP_weight_decay,
                        help='weight decay')
    parser.add_argument('--MLP_epochs', type=int, default=EFAfILBiC_config.MLP_epochs, help='num of training epochs')
    parser.add_argument('--MLP_milestones', type=list, default=EFAfILBiC_config.MLP_milestones,
                        help='optimizer milestones')
    parser.add_argument('--svm_sample_type', type=str, default=EFAfILBiC_config.svm_sample_type, help='svm_sample_type')
    parser.add_argument('--svm_max_iter', type=int, default=EFAfILBiC_config.svm_max_iter, help='svm_sample_type')
    parser.add_argument('--MLP_lrgamma', type=float, default=EFAfILBiC_config.MLP_lrgamma, help='MLP_lrgamma')
    parser.add_argument('--sim_alpha', type=float, default=EFAfILBiC_config.sim_alpha, help='sim_alpha')

    parser.add_argument('--availabel_cudas', type=str, default=EFAfILBiC_config.availabel_cudas, help='availabel_cudas')
    parser.add_argument('--num_workers', type=int, default=EFAfILBiC_config.num_workers,
                        help='number of worker to load data')
    parser.add_argument('--checkpoints', type=str, default=EFAfILBiC_config.checkpoint_path,
                        help='path for saving trained models')
    parser.add_argument('--log', type=str, default=EFAfILBiC_config.log, help='path to save log')
    parser.add_argument('--batch_train_log', type=str, default=EFAfILBiC_config.batch_train_log,
                        help='path to save batch training log')
    parser.add_argument('--result_file', type=str, default=EFAfILBiC_config.result_file,
                        help='file to save results')
    parser.add_argument('--seed', type=int, default=EFAfILBiC_config.seed, help='seed')
    parser.add_argument('--print_interval', type=int, default=EFAfILBiC_config.print_interval, help='print interval')
    parser.add_argument('--use_FM', type=bool, default=EFAfILBiC_config.use_FM,
                        help='use FM to get feature or use FE_cls to get feature')

    return parser.parse_args()


def main():
    # todo
    args = parse_args()
    logger = get_logger(args.log)
    batch_train_logger = get_logger(args.batch_train_log)
    EFAfILBiC_model = EFAfILBiC(model_name=args.model_name, MLP_name=args.MLP_name, dataset_name=args.dataset_name,
                                dataset_path=args.dataset_path, num_classes=args.num_classes, rate=args.rate,
                                tasks=args.tasks,
                                logger=logger, batch_train_logger=batch_train_logger, result_file=args.result_file,
                                use_exemplars=args.use_exemplars,
                                hidden_size=args.hidden_size, Exemple_memory_budget=args.Exemple_memory_budget,
                                Feature_memory_budget=args.Feature_memory_budget, optim_type=args.optim_type,
                                MLP_optim_type=args.MLP_optim_type,
                                norm_exemplars=args.norm_exemplars, herding=args.herding, FM_reTrain=args.FM_reTrain,
                                use_NewfeatureSpace=args.use_NewfeatureSpace, batch_size=args.batch_size,
                                num_workers=args.num_workers, seed=args.seed, availabel_cudas=args.availabel_cudas,

                                epochs=args.epochs, CNN_lr=args.CNN_lr, CNN_momentum=args.CNN_momentum,
                                CNN_weight_decay=args.CNN_weight_decay, CNN_milestones=args.CNN_milestones,
                                kd_lamb=args.kd_lamb, fd_gamma=args.fd_gamma, lrgamma=args.lrgamma,
                                KD_temp=args.KD_temp,

                                MLP_lr=args.MLP_lr, MLP_momentum=args.MLP_momentum,
                                MLP_epochs=args.MLP_epochs, MLP_milestones=args.MLP_milestones,
                                svm_sample_type=args.svm_sample_type, svm_max_iter=args.svm_max_iter,
                                MLP_weight_decay=args.MLP_weight_decay,
                                MLP_lrgamma=args.MLP_lrgamma, sim_alpha=args.sim_alpha, use_FM=args.use_FM,
                                oversample=args.oversample)
    EFAfILBiC_model.feature_replay_train_main(args)
    pass


if __name__ == "__main__":
    main()
