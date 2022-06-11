import copy
import datetime
import json
import os
import random
import time
import warnings

import numpy as np
from torch.nn import functional as Func
import torch
from torch.backends import cudnn
from torch.utils.data import ConcatDataset
from torchvision import transforms

from CIFAR.alg_model import resnetforcifar
from CIFAR.alg_model.resnetforcifar import resnet34, resnet18
from public import utils
from public.data import ExemplarDataset, SubDataset, get_multitask_experiment, get_dataset, AVAILABLE_TRANSFORMS, \
    inv_CIFAR_100_normalize
from exemplars import ExemplarHandler
from public.util_models import FeatureExtractor, FE_cls, SoftTarget_CrossEntropy, BiasLayer

# -------------------------------------------------------------------------------------------------#

# --------------------#
# ----- EEIL -----#
# --------------------#
from public.utils import AverageMeter, accuracy, split_array


class BiC(ExemplarHandler):
    def __init__(self, model_name, dataset_name, dataset_path, num_classes, hidden_size, epochs, num_workers,
                 extracted_layers, rate, tasks,
                 logger,
                 batch_train_logger, batch_size, result_file, memory_budget, norm_exemplars, herding, lr, momentum,
                 weight_decay, optim_type, milestones, KD_temp, gamma, availabel_cudas,
                 MLP_name=None, MLP_KD_temp=None, MLP_KD_temp_2=None, MLP_lr=None, MLP_rate=None, MLP_momentum=None,
                 MLP_epochs=None, MLP_milestones=None,
                 MLP_weight_decay=None,
                 MLP_lrgamma=None, MLP_optim_type=None, MLP_distill_rate=None,
                 seed=0, oversample=None):
        ExemplarHandler.__init__(self, memory_budget, batch_size, num_workers, norm_exemplars, herding,
                                 feature_dim=hidden_size, num_classes=num_classes,
                                 MLP_name=MLP_name, MLP_KD_temp=MLP_KD_temp, MLP_KD_temp_2=MLP_KD_temp_2,
                                 MLP_lr=MLP_lr, MLP_rate=MLP_rate, MLP_momentum=MLP_momentum,
                                 MLP_milestones=MLP_milestones,
                                 MLP_lrgamma=MLP_lrgamma, MLP_weight_decay=MLP_weight_decay,
                                 MLP_epochs=MLP_epochs, MLP_optim_type=MLP_optim_type,
                                 MLP_distill_rate=MLP_distill_rate,
                                 availabel_cudas=availabel_cudas
                                 )
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.epochs = epochs
        self.num_workers = num_workers
        self.dataset_path = dataset_path
        self.extracted_layers = extracted_layers
        self.rate = rate
        self.tasks = tasks
        self.logger = logger
        self.batch_train_logger = batch_train_logger
        self.batch_size = batch_size
        self.result_file = result_file
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optim_type = optim_type
        self.milestones = milestones
        self.KD_temp = KD_temp
        self.gamma = gamma
        self.availabel_cudas = availabel_cudas
        self.seed = seed
        self.oversample = oversample
        self.device = "cuda" if self.availabel_cudas else "cpu"

        self.pre_FE_cls = None
        self.pre_bias_layer = None
        self.original_train_datasets, self.train_datasets, self.val_datasets, self.data_config, \
        self.classes_per_task = self.get_dataset(dataset_name)
        self.category_dataset = self.get_dataset_category(dataset_name)
        self.FE_cls = self.construct_model(rate)
        # self.FE_cls = self.load_FE_cls_model("/share/home/kcli/CL_research/iCaRL_ILtFA/log/FE_cls_cifar10_preTrain_log/"
        #                                      "resnet34/rate_/"
        #                                      "cifar10_resnet34_rate_1_FE_cls.pth")
        self.bias_layer = None

    def forward(self, x):
        final_features, target = self.FE_cls(x)
        return target

    def get_dataset(self, dataset_name):
        (original_train_datasets, train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
            name=dataset_name, tasks=self.tasks, data_dir=self.dataset_path,
            exception=True if self.seed == 0 else False)
        return original_train_datasets, train_datasets, test_datasets, config, classes_per_task

    def get_dataset_category(self, dataset_name):
        if dataset_name == 'CIFAR10':
            (original_train_datasets, train_datasets,
             test_datasets), config, classes_per_task = get_multitask_experiment(
                name=dataset_name, tasks=10, data_dir=self.dataset_path,
                exception=True if self.seed == 0 else False)
            return train_datasets
        elif dataset_name == 'CIFAR100':
            (original_train_datasets, train_datasets,
             test_datasets), config, classes_per_task = get_multitask_experiment(
                name=dataset_name, tasks=100, data_dir=self.dataset_path,
                exception=True if self.seed == 0 else False)
            return train_datasets
        else:
            raise RuntimeError('dataset must be cifar10 or cifar100')

    def feature_extractor(self, images):
        mode = self.FE_cls.training
        self.FE_cls.eval()
        with torch.no_grad():
            features = self.FE_cls(images)[-2]
        self.FE_cls.train(mode=mode)
        return features

    def get_FE_cls_output(self, images):
        mode = self.training
        self.eval()
        with torch.no_grad():
            features, targets = self.FE_cls(images)
        self.train(mode=mode)
        return features, targets

    def build_optimize(self):
        # Define optimizer (only include parameters that "requires_grad")
        optim_list = [{'params': filter(lambda p: p.requires_grad, self.FE_cls.parameters()), 'lr': self.lr}]
        optimizer = None
        if self.optim_type in ("adam", "adam_reset"):
            if self.weight_decay:
                optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999), weight_decay=self.weight_decay)
            else:
                optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999))
        elif self.optim_type == "sgd":
            if self.momentum:
                optimizer = torch.optim.SGD(optim_list, momentum=self.momentum, weight_decay=self.weight_decay)
            else:
                optimizer = torch.optim.SGD(optim_list)
        else:
            raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(self.optim_type))

        return optimizer

    def build_optimize_biaslayer(self, lr):
        # Define optimizer (only include parameters that "requires_grad")
        assert self.bias_layer is not None
        optim_list = [{'params': filter(lambda p: p.requires_grad, self.bias_layer.parameters()), 'lr': lr}]
        optimizer = None
        if self.optim_type in ("adam", "adam_reset"):
            if self.weight_decay:
                optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999), weight_decay=self.weight_decay)
            else:
                optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999))
        elif self.optim_type == "sgd":
            if self.momentum:
                optimizer = torch.optim.SGD(optim_list, momentum=self.momentum, weight_decay=2e-04)
            else:
                optimizer = torch.optim.SGD(optim_list)
        else:
            raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(self.optim_type))

        return optimizer

    def construct_model(self, rate):
        if self.availabel_cudas:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
            device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
            # model = torch.nn.DataParallel(FE_cls(resnetforcifar.__dict__[self.model_name](rate=rate, get_feature=True),
            #                                      512 * self.rate, self.num_classes)).cuda()
            model = torch.nn.DataParallel(FE_cls(resnetforcifar.__dict__[self.model_name](rate=rate, get_feature=True),
                                                 512 * self.rate, self.num_classes), device_ids=device_ids).cuda()
        else:
            model = FE_cls(resnetforcifar.__dict__[self.model_name](rate=rate, get_feature=True), 512 * self.rate,
                           self.num_classes)
        # model = torch.load("/share/home/kcli/CL_research/iCaRL_ILtFA/pretrain_models/cifar10_pretrain_1_4.pth")
        print(model)
        return model

    def load_FE_cls_model(self, model_path):
        if self.availabel_cudas:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
            device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
            model = FE_cls(resnetforcifar.__dict__[self.model_name](rate=self.rate, get_feature=True),
                           int(512 * self.rate),
                           self.num_classes)
            model.load_state_dict(
                torch.load(model_path))
            # model = torch.nn.DataParallel(model).cuda()
            model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            model = FE_cls(resnetforcifar.__dict__[self.model_name](rate=self.rate, get_feature=True),
                           int(512 * self.rate),
                           self.num_classes)
            model.load_state_dict(
                torch.load(model_path))
        print(type(model))
        print(model)
        return model
        pass

    def FE_cls_forward(self, images):
        mode = self.FE_cls.training
        self.FE_cls.eval()
        with torch.no_grad():
            targets = self.FE_cls(images)[-1]
        self.FE_cls.train(mode=mode)
        return targets

    def train_main(self, args):
        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''
        assert args.is_BiC == 0 or args.is_BiC == 1
        print("seed:", self.seed)
        print("model:", self.FE_cls)
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            cudnn.deterministic = True

        gpus = torch.cuda.device_count()
        self.logger.info(f"use {gpus} gpus")
        self.logger.info(f"args: {args}")
        cudnn.benchmark = True
        cudnn.enabled = True
        # Set model in training-mode
        # Loop over all tasks.
        sample_seed = args.sample_seed
        BiC_result = {"timestamp": str(datetime.datetime.now()), "model_rate": self.rate, 'note': args.note,
                      "sample_seed": sample_seed}
        BiC_result.update(self.data_config)
        img_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS["CIFAR100_examplar"]["BiC_train_transform"],
        ])
        for task, train_dataset in enumerate(self.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')
            self.batch_train_logger.info(f'New task {task} begin.')
            print("New task %d begin." % task)
            # Add exemplars (if available) to current dataset (if requested)
            # train_size = int(0.9 * len(train_dataset))
            # val_size = len(train_dataset) - train_size
            # per_task_train_dataset, per_task_val_dataset = torch.utils.data.random_split(train_dataset,
            #                                                                              [train_size,
            #                                                                               val_size],
            #                                                                              generator=torch.Generator().manual_seed(
            #                                                                                  sample_seed))
            # self.logger.info(f'per_task_train_dataset len: {len(per_task_train_dataset)}||'
            #                  f'per_task_val_dataset len:{len(per_task_val_dataset)}')
            if task > 1 and self.memory_budget > 0:
                examplar_train_data_array, examplar_val_data_array = split_array(self.exemplar_sets,
                                                                                 sample_seed=sample_seed)
                examplar_train_dataset = ExemplarDataset(examplar_train_data_array, img_transform=img_transform,
                                                         inv_transform=inv_CIFAR_100_normalize)
                examplar_val_dataset = ExemplarDataset(examplar_val_data_array, img_transform=img_transform,
                                                       inv_transform=inv_CIFAR_100_normalize)
                examplar_val_sample_num_per_class = len(examplar_val_data_array[0])
                per_task_training_dataset = examplar_train_dataset
                per_task_valing_dataset = examplar_val_dataset

                begin_class = int((task - 1) * self.classes_per_task)
                end_class_plus_1 = int(begin_class + self.classes_per_task)
                for class_index in range(begin_class, end_class_plus_1):
                    print("category_dataset len:", len(self.category_dataset), "||", "class_index:", class_index)
                    self.logger.info(f"category_dataset len:{len(self.category_dataset)}||class_index:{class_index}")
                    train_dataset_temp, val_dataset_temp = torch.utils.data.random_split(
                        self.category_dataset[class_index], [len(self.category_dataset[class_index]) -
                                                             examplar_val_sample_num_per_class,
                                                             examplar_val_sample_num_per_class],
                        generator=torch.Generator().manual_seed(
                            0))

                    per_task_training_dataset = ConcatDataset([per_task_training_dataset, train_dataset_temp])
                    per_task_valing_dataset = ConcatDataset([per_task_valing_dataset, val_dataset_temp])

                self.logger.info(f'#############examplar_train_data_array len:{len(examplar_train_data_array)}|| '
                                 f'#############examplar_train_data_array len:{len(examplar_train_data_array[0])}|| '
                                 f'#############examplar_train_dataset len:{len(examplar_train_dataset)}|| '
                                 f'examplar_val_data_array len: {len(examplar_val_data_array)}||'
                                 f'examplar_val_data_array len: {len(examplar_val_data_array[0])}||'
                                 f'examplar_val_dataset len: {len(examplar_val_dataset)}||'
                                 f'examplar_val_sample_num_per_class: {examplar_val_sample_num_per_class}.##############')

                self.logger.info(f'#############val dataset for bias layer train task {task} length:'
                                 f'{len(per_task_valing_dataset)}.##############')
            else:
                per_task_training_dataset = train_dataset
                per_task_valing_dataset = None

                # Find [active_classes]
            active_classes = list(range(self.classes_per_task * task))
            if task > 1:
                optimizer = self.build_optimize()
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=self.milestones, gamma=self.gamma)
                for epoch in range(1, self.epochs + 1):
                    is_first_ite = True
                    iters_left = 1
                    iter_index = 0
                    iter_num = 0
                    while iters_left > 0:
                        # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                        iters_left -= 1
                        if is_first_ite:
                            is_first_ite = False
                            data_loader = iter(
                                utils.get_data_loader(per_task_training_dataset, self.batch_size, self.num_workers,
                                                      cuda=True if self.availabel_cudas else False))
                            # NOTE:  [train_dataset]  is training-set of current task
                            #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                            iter_num = iters_left = len(data_loader)
                            continue

                        #####-----CURRENT BATCH-----#####
                        try:
                            x, y = next(data_loader)  # --> sample training data of current task
                        except StopIteration:
                            raise ValueError("next(data_loader) error while read data. ")
                        x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                        if self.pre_FE_cls is not None:
                            with torch.no_grad():
                                scores = self.pre_FE_cls(x)[-1][:, :(self.classes_per_task * (task - 1))]
                                if args.is_BiC == 1 and task > 2:
                                    self.bias_layer.eval()
                                    scores = self.bias_layer(scores, self.classes_per_task)
                        else:
                            scores = None

                        # ---> Train MAIN MODEL
                        # Train the main model with this batch
                        loss_dict = self.train_a_batch(x, y, scores, active_classes, optimizer, task)
                        iter_index += 1
                        if iter_index % args.print_interval == 0:
                            results = f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                      f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                      f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}" \
                                      f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
                            self.batch_train_logger.info(
                                results
                            )
                            print(results)
                    scheduler.step()
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes)
                    self.batch_train_logger.info(
                        f"batch train current task validate || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
                        f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    )
                    self.batch_train_logger.info(f"------------------------------------------------------------------")
                    print(f'batch train task : {task:0>3d}, epoch: {epoch}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (
                        acc1, acc5))
                if task > 1:
                    self.train_BiasLayer(per_task_valing_dataset, task, active_classes, args.print_interval,
                                         train_method=args.train_method)
            else:
                # model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                #                        "cifar10pretrain/LwF_EXP/LwF_MC/task2/test1/rate_/cifar100_checkpoints/" \
                #                        "50_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                # model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                #                        "cifar10pretrain/_Test_iCaRL_original/Test_unbias/task1/rate_/test2/" \
                #                        "cifar100_checkpoints/100_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                                       "cifar10pretrain/LwF_EXP/Test_unbias/task1/rate_/test_1/cifar100_checkpoints/" \
                                       "100_cifar100_resnet34_LwF_rate_1_FE_cls_task1.pth"

                model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                       "cifar10pretrain/LwF_EXP/LwF_MC/task5/test1/rate_/cifar100_checkpoints/" \
                                       "20_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                model_path_for_task10 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                        "cifar10pretrain/LwF_EXP/LwF_MC/task10/test1/rate_/cifar100_checkpoints/" \
                                        "10_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                if self.dataset_name == "CIFAR10":
                    # model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                    #                        "cifar10_Test/LwF_EXP/Test_unbias/biasModel/task2/rate_/test_1/" \
                    #                        "cifar100_checkpoints/5_cifar100_resnet34_LwF_rate_1_FE_cls_task1.pth"
                    model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                                           "cifar10_Test/LwF_EXP/Test_unbias/task1/rate_/test_1/cifar100_checkpoints/" \
                                           "10_cifar100_resnet34_LwF_rate_1_FE_cls_task1.pth"

                    model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                                           "cifar10_Test/LwF_EXP/LwF_OD/task5/rate_/test_1/cifar100_checkpoints/" \
                                           "2_cifar100_resnet34_LwF_rate_1_FE_cls_task1.pth"

                if self.tasks == 2:
                    model_path = model_path_for_task2
                elif self.tasks == 5:
                    model_path = model_path_for_task5
                elif self.tasks == 10:
                    model_path = model_path_for_task10

                self.FE_cls = self.load_FE_cls_model(model_path)
                print(self.FE_cls)
                # self.FE_cls = torch.load("")
                # self.FE_cls = torch.load("cifar10_preTrain_task_1_FE_cls.pth")
                # continue
                print("model for tasks{}:".format(self.tasks), self.FE_cls)

            self.batch_train_logger.info(f'##########feature extractor train task {task} End.#########')
            self.logger.info(f'#############feature extractor train task {task} End.##############')
            self.logger.info(f'#############Example handler task {task} start.##############')
            print("feature extractor train task-%d End" % (task))
            print("Example handler task-%d start." % (task))
            # EXEMPLARS: update exemplar sets
            exemplars_per_class = int(np.floor(self.memory_budget / (self.classes_per_task * task)))
            # for each new class trained on, assist_current_exemplar_set
            new_classes = list(range(self.classes_per_task * (task - 1), self.classes_per_task * task))
            if task > 1:
                self.reduce_exemplar_sets(exemplars_per_class)

            '''for each new class trained on, constructexemplar_set'''
            for class_id in new_classes:
                # create new dataset containing only all examples of this class
                class_dataset = SubDataset(original_dataset=self.original_train_datasets[task - 1],
                                           sub_labels=[class_id])
                # based on this dataset, construct new exemplar-set for this class
                print("construct_exemplar_set class_id:", class_id)
                self.construct_exemplar_set(dataset=class_dataset, n=exemplars_per_class)
            '''task validate'''
            self.compute_means = True
            self.logger.info(f'#############task: {task:0>3d} is finished Test begin. ##############')
            acc_past_tasks, acc_list = self.tasks_validate(task)
            BiC_result["task_{}_results".format(task)] = acc_past_tasks
            print("task: %d ncm acc:" % task, acc_past_tasks)
            self.logger.info(f'"task: {task} classififer:{"ncm"} fc acc: {acc_past_tasks}"')
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="fc")
            BiC_result["task_{}_fc_results".format(task)] = acc_past_tasks
            print("task: %d fc acc:" % task, acc_past_tasks)
            self.logger.info(f'"task: {task} classififer:{"fc"} fc acc: {acc_past_tasks}"')
            self.pre_FE_cls = copy.deepcopy(self.FE_cls).eval()
            self.compute_means = True
            with open(self.result_file, 'w') as fw:
                json.dump(BiC_result, fw, indent=4)
        with open(self.result_file, 'w') as fw:
            json.dump(BiC_result, fw, indent=4)

    def feature_relearning_train_main(self, args):
        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''
        assert args.is_BiC == 2 or args.is_BiC == 3
        print("seed:", self.seed)
        print("model:", self.FE_cls)
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            cudnn.deterministic = True

        gpus = torch.cuda.device_count()
        self.logger.info(f"use {gpus} gpus")
        self.logger.info(f"args: {args}")
        cudnn.benchmark = True
        cudnn.enabled = True
        # Set model in training-mode
        # Loop over all tasks.
        sample_seed = args.sample_seed
        BiC_result = {"timestamp": str(datetime.datetime.now()), "model_rate": self.rate, 'note': args.note,
                      "sample_seed": sample_seed}
        BiC_result.update(self.data_config)
        img_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS["CIFAR100_examplar"]["train_transform"],
        ])
        for task, train_dataset in enumerate(self.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')
            self.batch_train_logger.info(f'New task {task} begin.')
            print("New task %d begin." % task)
            # Add exemplars (if available) to current dataset (if requested)

            if task > 1 and self.memory_budget > 0:
                examplar_train_data_array, examplar_val_data_array = split_array(self.exemplar_sets,
                                                                                 sample_seed=sample_seed)
                examplar_train_dataset = ExemplarDataset(examplar_train_data_array, img_transform=img_transform,
                                                         inv_transform=inv_CIFAR_100_normalize)
                examplar_val_dataset = ExemplarDataset(examplar_val_data_array, img_transform=img_transform,
                                                       inv_transform=inv_CIFAR_100_normalize)
                per_task_training_dataset = examplar_train_dataset
                per_task_valing_dataset = examplar_val_dataset
                examplar_val_sample_num_per_class = len(examplar_val_data_array[0])

                begin_class = int((task - 1) * self.classes_per_task)
                end_class_plus_1 = int(begin_class + self.classes_per_task)
                for class_index in range(begin_class, end_class_plus_1):
                    print("category_dataset len:", len(self.category_dataset), "||", "class_index:", class_index)
                    self.logger.info(f"category_dataset len:{len(self.category_dataset)}||class_index:{class_index}")
                    train_dataset_temp, val_dataset_temp = torch.utils.data.random_split(
                        self.category_dataset[class_index], [len(self.category_dataset[class_index]) -
                                                             examplar_val_sample_num_per_class,
                                                             examplar_val_sample_num_per_class],
                        generator=torch.Generator().manual_seed(
                            0))

                    per_task_training_dataset = ConcatDataset([per_task_training_dataset, train_dataset_temp])
                    per_task_valing_dataset = ConcatDataset([per_task_valing_dataset, val_dataset_temp])

                self.logger.info(f'#############examplar_train_data_array len:{len(examplar_train_data_array)}|| '
                                 f'#############examplar_train_data_array len:{len(examplar_train_data_array[0])}|| '
                                 f'#############examplar_train_dataset len:{len(examplar_train_dataset)}|| '
                                 f'examplar_val_data_array len: {len(examplar_val_data_array)}||'
                                 f'examplar_val_data_array len: {len(examplar_val_data_array[0])}||'
                                 f'examplar_val_dataset len: {len(examplar_val_dataset)}||'
                                 f'examplar_val_sample_num_per_class: {examplar_val_sample_num_per_class}.##############')

                self.logger.info(f'#############val dataset for bias layer train task {task} length:'
                                 f'{len(per_task_valing_dataset)}.##############')

                exemplar_sets_for_feature_relearn = copy.deepcopy(self.exemplar_sets)
                feature_relearn_exemplar_dataset = ExemplarDataset(exemplar_sets_for_feature_relearn,
                                                                   img_transform=img_transform,
                                                                   inv_transform=inv_CIFAR_100_normalize)

            else:
                per_task_training_dataset = train_dataset
                per_task_valing_dataset = None

                # Find [active_classes]
            active_classes = list(range(self.classes_per_task * task))
            if task > 1:
                if task > 1:
                    self.logger.info(f'New task {task} begin: use feature_relearn_exemplar_dataset '
                                     f'to train FA'
                                     f'use feature_relearn_exemplar_dataset to train FE_cls.')
                    assert feature_relearn_exemplar_dataset is not None
                    if args.is_BiC == 2:
                        self.EFAfIL_split_feature_mapper_cls_domain_train(train_dataset,
                                                                          feature_relearn_exemplar_dataset,
                                                                          self.val_datasets,
                                                                          self.classes_per_task,
                                                                          active_classes, task)
                    elif args.is_BiC == 3:
                        self.EFAfIL_split_feature_mapper_cls_domain_train_bias(self.bias_layer, train_dataset,
                                                                               feature_relearn_exemplar_dataset,
                                                                               self.val_datasets,
                                                                               self.classes_per_task,
                                                                               active_classes, task)
                    self.pre_FE_cls = copy.deepcopy(self.FE_cls).eval()
                optimizer = self.build_optimize()
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones,
                                                                 gamma=self.gamma)
                for epoch in range(1, self.epochs + 1):
                    is_first_ite = True
                    iters_left = 1
                    iter_index = 0
                    iter_num = 0
                    while iters_left > 0:
                        # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                        iters_left -= 1
                        if is_first_ite:
                            is_first_ite = False
                            data_loader = iter(
                                utils.get_data_loader(per_task_training_dataset, self.batch_size, self.num_workers,
                                                      cuda=True if self.availabel_cudas else False))
                            # NOTE:  [train_dataset]  is training-set of current task
                            #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                            iter_num = iters_left = len(data_loader)
                            continue

                        #####-----CURRENT BATCH-----#####
                        try:
                            x, y = next(data_loader)  # --> sample training data of current task
                        except StopIteration:
                            raise ValueError("next(data_loader) error while read data. ")
                        x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                        if self.pre_FE_cls is not None:
                            self.FM_cls_domain.eval()
                            with torch.no_grad():
                                features = self.pre_FE_cls(x)[-2]
                                scores = self.FM_cls_domain(features)[-2]
                                scores = scores[:, :(self.classes_per_task * (task - 1))]
                        else:
                            scores = None

                        # ---> Train MAIN MODEL
                        # Train the main model with this batch
                        loss_dict = self.train_a_batch(x, y, scores, active_classes, optimizer, task)
                        iter_index += 1
                        if iter_index % args.print_interval == 0:
                            results = f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                      f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                      f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"
                            self.batch_train_logger.info(
                                results
                            )
                            print(results)
                    scheduler.step()
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes)
                    self.batch_train_logger.info(
                        f"batch train current task validate || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
                        f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    )
                    self.batch_train_logger.info(f"------------------------------------------------------------------")
                    print(f'batch train task : {task:0>3d}, epoch: {epoch}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (
                        acc1, acc5))
                if task > 1:
                    self.train_BiasLayer(per_task_valing_dataset, task, active_classes, args.print_interval,
                                         train_method=args.train_method)
            elif task == 1:
                model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                       "cifar10pretrain/LwF_EXP/LwF_MC/task2/test1/rate_/cifar100_checkpoints/" \
                                       "50_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"


                model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                       "cifar10pretrain/LwF_EXP/LwF_MC/task5/test1/rate_/cifar100_checkpoints/" \
                                       "20_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                if self.dataset_name == "CIFAR10":
                    model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                                           "cifar10_Test/LwF_EXP/Test_unbias/biasModel/task2/rate_/test_1/" \
                                           "cifar100_checkpoints/5_cifar100_resnet34_LwF_rate_1_FE_cls_task1.pth"
                    model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                                           "cifar10_Test/LwF_EXP/LwF_OD/task5/rate_/test_1/cifar100_checkpoints/" \
                                           "2_cifar100_resnet34_LwF_rate_1_FE_cls_task1.pth"

                model_path_for_task10 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                        "cifar10pretrain/LwF_EXP/LwF_MC/task10/test1/rate_/cifar100_checkpoints/" \
                                        "10_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                if self.tasks == 2:
                    model_path = model_path_for_task2
                elif self.tasks == 5:
                    model_path = model_path_for_task5
                elif self.tasks == 10:
                    model_path = model_path_for_task10

                self.FE_cls = self.load_FE_cls_model(model_path)
                # self.FE_cls = torch.load("")
                # self.FE_cls = torch.load("cifar10_preTrain_task_1_FE_cls.pth")
                # continue
                print("model for tasks{}:".format(self.tasks), self.FE_cls)
            self.batch_train_logger.info(f'##########feature extractor train task {task} End.#########')
            self.logger.info(f'#############feature extractor train task {task} End.##############')
            self.logger.info(f'#############Example handler task {task} start.##############')
            print("feature extractor train task-%d End" % (task))
            print("Example handler task-%d start." % (task))
            # EXEMPLARS: update exemplar sets
            exemplars_per_class = int(np.floor(self.memory_budget / (self.classes_per_task * task)))
            # for each new class trained on, assist_current_exemplar_set
            new_classes = list(range(self.classes_per_task * (task - 1), self.classes_per_task * task))
            if task > 1:
                self.reduce_exemplar_sets(exemplars_per_class)

            '''for each new class trained on, constructexemplar_set'''
            for class_id in new_classes:
                # create new dataset containing only all examples of this class
                class_dataset = SubDataset(original_dataset=self.original_train_datasets[task - 1],
                                           sub_labels=[class_id])
                # based on this dataset, construct new exemplar-set for this class
                print("construct_exemplar_set class_id:", class_id)
                self.construct_exemplar_set(dataset=class_dataset, n=exemplars_per_class)
            '''task validate'''
            self.compute_means = True
            self.logger.info(f'#############task: {task:0>3d} is finished Test begin. ##############')
            acc_past_tasks, acc_list = self.tasks_validate(task)
            BiC_result["task_{}_results".format(task)] = acc_past_tasks
            print("task: %d ncm acc:" % task, acc_past_tasks)
            self.logger.info(f'"task: {task} classififer:{"ncm"} fc acc: {acc_past_tasks}"')
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="fc")
            BiC_result["task_{}_fc_results".format(task)] = acc_past_tasks
            print("task: %d fc acc:" % task, acc_past_tasks)
            self.logger.info(f'"task: {task} classififer:{"fc"} fc acc: {acc_past_tasks}"')

            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="Nobias_fc")
            BiC_result["task_{}_Nobias_fc_results".format(task)] = acc_past_tasks
            print("task: %d Nobias_fc acc:" % task, acc_past_tasks)
            self.logger.info(f'"task: {task} classififer: Nobias_fc acc: {acc_past_tasks}"')

            self.pre_FE_cls = copy.deepcopy(self.FE_cls).eval()
            self.compute_means = True
            with open(self.result_file, 'w') as fw:
                json.dump(BiC_result, fw, indent=4)
        with open(self.result_file, 'w') as fw:
            json.dump(BiC_result, fw, indent=4)

    def train_BiasLayer(self, per_task_valing_dataset, task, active_classes, print_interval, train_method):
        current_classes_num = self.classes_per_task
        self.bias_layer = BiasLayer()
        if train_method == 0:
            optimizer = self.build_optimize_biaslayer(self.lr / 100)
            epochs = 45
            gap = int(epochs / 3)
            milestones = [15, 30]
        elif train_method == 1:
            optimizer = self.build_optimize_biaslayer(self.lr / 100)
            epochs = 60
            gap = int(epochs / 3)
            milestones = [gap, 2 * gap]
        elif train_method == 2:
            optimizer = self.build_optimize_biaslayer(self.lr / 10)
            epochs = 96
            gap = int(epochs / 4)
            milestones = [gap, 2 * gap, 3 * gap]
        elif train_method == 3:
            optimizer = self.build_optimize_biaslayer(self.lr / 10)
            epochs = 160
            gap = int(epochs / 4)
            milestones = [gap, 2 * gap, 3 * gap]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        for epoch in range(1, epochs + 1):
            is_first_ite = True
            iters_left = 1
            iter_index = 0
            iter_num = 0
            while iters_left > 0:
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                iters_left -= 1
                if is_first_ite:
                    is_first_ite = False
                    data_loader = iter(
                        utils.get_data_loader(per_task_valing_dataset, self.batch_size, self.num_workers,
                                              cuda=True if self.availabel_cudas else False))
                    # NOTE:  [train_dataset]  is training-set of current task
                    #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                    iter_num = iters_left = len(data_loader)
                    continue

                #####-----CURRENT BATCH-----#####
                try:
                    x, y = next(data_loader)  # --> sample training data of current task
                except StopIteration:
                    raise ValueError("next(data_loader) error while read data. ")
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                FE_cls_targets = self.FE_cls_forward(x)
                FE_cls_targets = FE_cls_targets[:, :(self.classes_per_task * task)]
                loss_dict = self.BiasLayer_train_a_batch(FE_cls_targets, y, current_classes_num, optimizer,
                                                         active_classes)
                iter_index += 1
                if iter_num > 20:
                    if iter_index % int(print_interval / 2) == 0:
                        results = f"bias layer train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                  f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                  f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"
                        self.batch_train_logger.info(
                            results
                        )
                        print(results)
                else:
                    results = f"bias layer train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                              f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                              f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"
                    self.batch_train_logger.info(
                        results
                    )
                    print(results)
            print("self.bias_layer.parameters:", self.bias_layer.params)
            scheduler.step()
            acc1, acc5, throughput = self.current_task_validate_biaslayer(task, active_classes, current_classes_num)
            self.batch_train_logger.info(
                f" bias layer validate || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
                f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
            )
            self.batch_train_logger.info(f"------------------------------------------------------------------")
            print(f'bias layer task : {task:0>3d}, epoch: {epoch}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (
                acc1, acc5))

        pass

    def train_a_batch(self, x, y, scores, active_classes, optimizer, task=1):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <tensor> batch of corresponding labels
        [scores]          None or <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x]
                            NOTE: only to be used for "BCE with distill" (only when scenario=="class")
        [active_classes]  None or (<list> of) <list> with "active" classes
        [task]            <int>, for setting task-specific mask'''
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        # Set model to training-mode
        self.train()
        # Reset optimizer
        optimizer.zero_grad()
        criterion = torch.nn.CrossEntropyLoss().cuda()
        soft_target_criterion = SoftTarget_CrossEntropy().cuda()
        if x is not None:

            # Run model
            y_hat = self(x)
            # -if needed, remove predictions for classes not in current task
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                y_hat = y_hat[:, class_entries]

            # Calculate prediction loss
            # -binary prediction loss
            y_hat_fordistill = y_hat[:, :(self.classes_per_task * (task - 1))]
            binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
            # if scores is not None:
            #    binary_targets = binary_targets[:, -self.classes_per_task:]
            #    binary_targets = torch.cat([torch.sigmoid(scores / self.KD_temp), binary_targets], dim=1)
            if scores is None:
                predL = None if y is None else Func.binary_cross_entropy_with_logits(
                    input=y_hat, target=binary_targets, reduction='none'
                ).sum(dim=1).mean()  # --> sum over classes, then average over batch
            else:
                scores_hats = torch.softmax(scores / self.KD_temp, dim=1)
                loss_distill = soft_target_criterion(y_hat_fordistill, scores_hats,
                                                     self.KD_temp) * self.KD_temp * self.KD_temp
                loss_cls = criterion(y_hat, y)
                lamda = (task - 1) / task
                predL = (1 - lamda) * loss_cls + lamda * loss_distill
            if len(active_classes) >= 5:
                acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
                top1.update(acc1.item(), x.size(0))
                top5.update(acc5.item(), x.size(0))
                losses.update(predL, x.size(0))
                # Calculate training-precision
                precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
            else:
                acc1 = accuracy(y_hat, y, topk=(1,))[0]
                top1.update(acc1.item(), x.size(0))
                losses.update(predL, x.size(0))
                # Calculate training-precision
                precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
        else:
            warnings.filterwarnings('training data is none.')
            precision = predL = None
            # -> it's possible there is only "replay" [e.g., for offline with task-incremental learning]
        loss_total = predL
        loss_total.backward()
        # Take optimization-step
        optimizer.step()
        # Return the dictionary with different training-loss split in categories
        if len(active_classes) >= 5:
            return {
                'loss_total': loss_total.item(),
                'precision': precision if precision is not None else 0.,
                "top1": top1.avg,
                "top5": top5.avg,
                "losses": losses.avg
            }
        else:
            return {
                'loss_total': loss_total.item(),
                'precision': precision if precision is not None else 0.,
                "top1": top1.avg,
                "top5": 0,
                "losses": losses.avg
            }

    def BiasLayer_train_a_batch(self, FE_cls_targets, y, current_classes_num, optimizer, active_classes=None):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

                [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
                [y]               <tensor> batch of corresponding labels
                [scores]          None or <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x]
                                    NOTE: only to be used for "BCE with distill" (only when scenario=="class")
                [active_classes]  None or (<list> of) <list> with "active" classes
                [task]            <int>, for setting task-specific mask'''
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        # Set model to training-mode
        self.bias_layer.train()
        # Reset optimizer
        optimizer.zero_grad()
        criterion = torch.nn.CrossEntropyLoss().cuda()

        # Run model
        y_hat = self.bias_layer(FE_cls_targets, current_classes_num)

        predL = criterion(y_hat, y)
        if len(active_classes) >= 5:
            acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
            top1.update(acc1.item(), FE_cls_targets.size(0))
            top5.update(acc5.item(), FE_cls_targets.size(0))
            losses.update(predL, FE_cls_targets.size(0))
            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / FE_cls_targets.size(0)
        else:
            acc1 = accuracy(y_hat, y, topk=(1,))[0]
            top1.update(acc1.item(), FE_cls_targets.size(0))
            losses.update(predL, FE_cls_targets.size(0))
            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / FE_cls_targets.size(0)
        loss_total = predL
        loss_total.backward()
        # Take optimization-step
        optimizer.step()
        # Return the dictionary with different training-loss split in categories
        if len(active_classes) >= 5:
            return {
                'loss_total': loss_total.item(),
                'precision': precision if precision is not None else 0.,
                "top1": top1.avg,
                "top5": top5.avg,
                "losses": losses.avg
            }
        else:
            return {
                'loss_total': loss_total.item(),
                'precision': precision if precision is not None else 0.,
                "top1": top1.avg,
                "top5": 0,
                "losses": losses.avg
            }
        pass

    def current_task_validate(self, task, active_classes):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        val_loader = utils.get_data_loader(self.val_datasets[task - 1], self.batch_size, self.num_workers,
                                           cuda=True if self.availabel_cudas else False)
        mode = self.training
        # switch to evaluate mode
        self.eval()
        with torch.no_grad():
            end = time.time()
            for inputs, labels in val_loader:
                data_time.update(time.time() - end)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                y_hat = self(inputs)
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    y_hat = y_hat[:, class_entries]
                if len(active_classes) >= 5:
                    acc1, acc5 = accuracy(y_hat, labels, topk=(1, 5))
                    top1.update(acc1.item(), inputs.size(0))
                    top5.update(acc5.item(), inputs.size(0))
                else:
                    acc1 = accuracy(y_hat, labels, topk=(1,))[0]
                    top1.update(acc1.item(), inputs.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
        throughput = 1.0 / (batch_time.avg / self.batch_size)
        self.train(mode=mode)
        if len(active_classes) >= 5:
            return top1.avg, top5.avg, throughput
        else:
            return top1.avg, 0, throughput

    def current_task_validate_biaslayer(self, task, active_classes, current_classes_num):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        val_loader = utils.get_data_loader(self.val_datasets[task - 1], self.batch_size, self.num_workers,
                                           cuda=True if self.availabel_cudas else False)
        mode = self.training
        # switch to evaluate mode
        self.eval()
        self.bias_layer.eval()
        with torch.no_grad():
            end = time.time()
            for inputs, labels in val_loader:
                data_time.update(time.time() - end)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                y_hat = self(inputs)
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    y_hat = y_hat[:, class_entries]
                y_hat = self.bias_layer(y_hat, current_classes_num)
                if len(active_classes) >= 5:
                    acc1, acc5 = accuracy(y_hat, labels, topk=(1, 5))
                    top1.update(acc1.item(), inputs.size(0))
                    top5.update(acc5.item(), inputs.size(0))
                else:
                    acc1 = accuracy(y_hat, labels, topk=(1,))[0]
                    top1.update(acc1.item(), inputs.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
        throughput = 1.0 / (batch_time.avg / self.batch_size)
        self.train(mode=mode)
        if len(active_classes) >= 5:
            return top1.avg, top5.avg, throughput
        else:
            return top1.avg, 0, throughput
        pass

    def per_task_validate(self, val_dataset, task, classifier="ncm"):
        # todo
        val_loader = utils.get_data_loader(val_dataset, self.batch_size, self.num_workers,
                                           cuda=True if self.availabel_cudas else False)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        correct = 0
        active_classes = list(range(self.classes_per_task * task))
        for inputs, labels in val_loader:
            correct_temp = 0
            data_time.update(time.time() - end)
            inputs, labels = inputs.cuda(), labels.cuda()
            if classifier == "ncm":
                y_hat = self.classify_with_exemplars(inputs)
            elif "fc" in classifier:
                self.eval()
                with torch.no_grad():
                    y_hat = self(inputs)
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    y_hat = y_hat[:, class_entries]
                if task > 1 and "Nobias" not in classifier:
                    self.bias_layer.eval()
                    with torch.no_grad():
                        y_hat = self.bias_layer(y_hat, self.classes_per_task)
                _, predicted = torch.max(y_hat, 1)
                y_hat = predicted
            correct_temp += y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
        throughput = 1.0 / (batch_time.avg / self.batch_size)
        return [top1.avg, throughput]
        pass

    def tasks_validate(self, task, classifier="ncm"):
        # todo
        acc_past_tasks = []
        acc = []
        for task_id in range(task):
            predict_result = self.per_task_validate(self.val_datasets[task_id], task, classifier)
            acc_past_tasks.append(predict_result)
            acc.append(predict_result[0])
        acc = np.array(acc)
        print("--------classifier{} acc_avg:{}--------".format(classifier, acc.mean()))
        self.logger.info(
            f"per task {task}, classifier{classifier} avg acc:{acc.mean()}"
            f"-------------------------------------------------------------"
        )
        return acc_past_tasks, acc
        pass
