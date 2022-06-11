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
from ImageNet.alg_model import resnetforimagenet
from public import utils
from public.data import ExemplarDataset, SubDataset, get_multitask_experiment, get_dataset, AVAILABLE_TRANSFORMS, \
    get_multitask_imagenet_experiment, inv_imagenet_normalize
from exemplars import ExemplarHandler
from public.util_models import FeatureExtractor, FE_cls

# -------------------------------------------------------------------------------------------------#

# --------------------#
# ----- iCaRL -----#
# --------------------#
from public.utils import AverageMeter, accuracy


class iCaRL(ExemplarHandler):
    def __init__(self, model_name, dataset_name, dataset_path, dataset_json_path, num_classes, hidden_size,
                 epochs, num_workers,
                 rate, tasks, logger, batch_train_logger, batch_size, result_file, memory_budget, norm_exemplars,
                 herding, lr, momentum, weight_decay, optim_type, milestones, KD_temp, gamma, availabel_cudas,
                 MLP_name=None, MLP_KD_temp=None, MLP_KD_temp_2=None,
                 MLP_lr=None, MLP_rate=None, MLP_momentum=None,
                 MLP_epochs=None, MLP_milestones=None,
                 MLP_weight_decay=None,
                 MLP_lrgamma=None, MLP_optim_type=None, MLP_distill_rate=None, seed=0):
        ExemplarHandler.__init__(self, memory_budget, batch_size, num_workers, norm_exemplars, herding,
                                 feature_dim=hidden_size, num_classes=num_classes,
                                 MLP_name=MLP_name, MLP_KD_temp=MLP_KD_temp, MLP_KD_temp_2=MLP_KD_temp_2,
                                 MLP_lr=MLP_lr, MLP_rate=MLP_rate, MLP_momentum=MLP_momentum,
                                 MLP_milestones=MLP_milestones,
                                 MLP_lrgamma=MLP_lrgamma, MLP_weight_decay=MLP_weight_decay,
                                 MLP_epochs=MLP_epochs, MLP_optim_type=MLP_optim_type,
                                 MLP_distill_rate=MLP_distill_rate,
                                 availabel_cudas=availabel_cudas)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_json_path = dataset_json_path
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.dataset_path = dataset_path
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
        self.device = "cuda" if self.availabel_cudas else "cpu"
        self.pre_FE_cls = None
        self.train_datasets, self.val_datasets, self.data_config, self.classes_per_task = \
            self.get_dataset(dataset_name)
        self.FE_cls = self.construct_model(rate)

    def forward(self, x):
        final_features, target = self.FE_cls(x)
        return target

    def get_dataset(self, dataset_name):
        (train_datasets, test_datasets), config, classes_per_task = \
            get_multitask_imagenet_experiment(dataset_name, self.tasks, data_dir=self.dataset_path,
                                              imagenet_json_path=self.dataset_json_path)
        return train_datasets, test_datasets, config, classes_per_task

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

    def construct_model(self, rate):
        if self.availabel_cudas:
            # os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
            # device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
            # model = torch.nn.DataParallel(FE_cls(resnetforimagenet.__dict__[self.model_name](rate=rate, get_feature=True),
            #                                      self.hidden_size, self.num_classes),
            #                               device_ids=device_ids).cuda()
            model = FE_cls(resnetforimagenet.__dict__[self.model_name](rate=rate, get_feature=True),
                           int(512 * self.rate),
                           self.num_classes)
            model.load_state_dict(
                torch.load("/share/home/kcli/CL_research/iCaRL_ILtFA/imagenet100Pretrain_resnet18_rate1_FEcls.pth"))
            model = torch.nn.DataParallel(model).cuda()
            # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            model = FE_cls(resnetforimagenet.__dict__[self.model_name](rate=rate, get_feature=True),
                           int(512 * self.rate),
                           self.num_classes)
            model.load_state_dict(
                torch.load("/share/home/kcli/CL_research/iCaRL_ILtFA/imagenet100Pretrain_resnet18_rate1_FEcls.pth"))

        # model = torch.load("/share/home/kcli/CL_research/iCaRL_ILtFA/pretrain_models/FEcls_imagenet100_pretrain.pth")
        print(type(model))
        print(model)
        return model

    def load_model(self, model_path):
        if self.availabel_cudas:
            # os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
            # device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
            # model = torch.nn.DataParallel(FE_cls(resnetforimagenet.__dict__[self.model_name](rate=rate, get_feature=True),
            #                                      self.hidden_size, self.num_classes),
            #                               device_ids=device_ids).cuda()
            model = FE_cls(resnetforimagenet.__dict__[self.model_name](rate=self.rate, get_feature=True),
                           int(512 * self.rate),
                           self.num_classes)
            model.load_state_dict(
                torch.load(model_path))
            model = torch.nn.DataParallel(model).cuda()
            # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
            # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            model = FE_cls(resnetforimagenet.__dict__[self.model_name](rate=self.rate, get_feature=True),
                           int(512 * self.rate),
                           self.num_classes)
            model.load_state_dict(
                torch.load(model_path))
        return model
        pass

    def load_task1_model(self, rate):
        if self.availabel_cudas:
            # os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
            # device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
            # model = torch.nn.DataParallel(FE_cls(resnetforimagenet.__dict__[self.model_name](rate=rate, get_feature=True),
            #                                      self.hidden_size, self.num_classes),
            #                               device_ids=device_ids).cuda()
            model = FE_cls(resnetforimagenet.__dict__[self.model_name](rate=rate, get_feature=True),
                           int(512 * self.rate),
                           self.num_classes)
            model.load_state_dict(
                torch.load(
                    "/share/home/kcli/CL_research/iCaRL_ILtFA/imagenetEXP_log/Imagenet100Pretrain/EFAfIL_task20_log/"
                    "alldistillcls_imgs_prefeaturedistill_CLS_log/rate_/EFAfIL_checkpoints/imagenet100Pretriain_resnet18_rate_1_FE_cls_task_1.pth"))
            model = torch.nn.DataParallel(model).cuda()
            # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            model = FE_cls(resnetforimagenet.__dict__[self.model_name](rate=rate, get_feature=True),
                           int(512 * self.rate),
                           self.num_classes)
            model.load_state_dict(
                torch.load(
                    "/share/home/kcli/CL_research/iCaRL_ILtFA/imagenetEXP_log/Imagenet100Pretrain/EFAfIL_task20_log/"
                    "alldistillcls_imgs_prefeaturedistill_CLS_log/rate_/EFAfIL_checkpoints/imagenet100Pretriain_resnet18_rate_1_FE_cls_task_1.pth"))

        # model = torch.load("/share/home/kcli/CL_research/iCaRL_ILtFA/pretrain_models/FEcls_imagenet100_pretrain.pth")
        print(type(model))
        return model

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

    def train_main(self, args):
        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''
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
        iCaRL_result = {"timestamp": str(datetime.datetime.now()), "model_rate": self.rate}
        iCaRL_result.update(self.data_config)
        img_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS["imagenet_examplar"]["train_transform"],
        ])
        for task, train_dataset in enumerate(self.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')
            self.batch_train_logger.info(f'New task {task} begin.')
            print("New task %d begin." % task)
            # Add exemplars (if available) to current dataset (if requested)
            if task > 1 and self.memory_budget > 0:
                exemplar_dataset = ExemplarDataset(self.exemplar_sets, img_transform=img_transform,
                                                   inv_transform=inv_imagenet_normalize)
                training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
            else:
                training_dataset = train_dataset

            # Find [active_classes]
            if task > 1:
                active_classes = list(range(self.classes_per_task * task))
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
                            data_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
                                                                     self.num_workers,
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
                        else:
                            scores = None

                        # ---> Train MAIN MODEL
                        # Train the main model with this batch
                        loss_dict = self.train_a_batch(x, y, scores,
                                                       active_classes, optimizer, task)
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
                if task == 1:
                    checkpoint_file = os.path.join(args.checkpoint_path,
                                                   "{}_resnet18_rate_1_FEcls_task_{}.pth".format(self.classes_per_task,
                                                                                                 task))
                    torch.save(self.FE_cls.module.state_dict(), checkpoint_file)
            else:
                model_task20_for_task1 = "/share/home/kcli/CL_research/iCaRL_ILtFA/Imagenet100EXP_July/" \
                                         "Imagenet100pretrain/iCaRL_EXP/DATAforCL_1/task20/test1/rate_1/" \
                                         "imagenet100_checkpoints/5_resnet18_rate_1_FEcls_task_1.pth"
                model_task4_for_task1 = "/share/home/kcli/CL_research/iCaRL_ILtFA/Imagenet100EXP_July/" \
                                        "Imagenet100pretrain/iCaRL_EXP/DATAforCL_1/task4/test3/rate_1/" \
                                        "imagenet100_checkpoints/25_resnet18_rate_1_FEcls_task_1.pth"
                model_task10_for_task1 = "/share/home/kcli/CL_research/iCaRL_ILtFA/Imagenet100EXP_July/" \
                                         "Imagenet100pretrain/iCaRL_EXP/DATAforCL_1/task10/test1/rate_1/" \
                                         "imagenet100_checkpoints/10_resnet18_rate_1_FEcls_task_1.pth"
                if self.tasks == 4:
                    model_path = model_task4_for_task1
                elif self.tasks == 10:
                    model_path = model_task10_for_task1
                elif self.tasks == 20:
                    model_path = model_task20_for_task1
                self.FE_cls = self.load_model(model_path)
            self.batch_train_logger.info(f'##########feature extractor train task {task} End.#########')
            self.logger.info(f'#############feature extractor train task {task} End.##############')
            self.logger.info(f'#############Example handler task {task} start.##############')
            print("feature extractor train task-%d End" % (task))
            print("Example handler task-%d start." % (task))
            # EXEMPLARS: update exemplar sets
            exemplars_per_class = int(np.floor(self.memory_budget / (self.classes_per_task * task)))
            # reduce examplar-sets
            self.reduce_exemplar_sets(exemplars_per_class)
            # for each new class trained on, construct examplar-set
            new_classes = list(range(self.classes_per_task * (task - 1), self.classes_per_task * task))
            for class_id in new_classes:
                # create new dataset containing only all examples of this class
                class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                print("construct_exemplar_set class_id:", class_id)
                # based on this dataset, construct new exemplar-set for this class
                self.construct_exemplar_set(dataset=class_dataset, n=exemplars_per_class)
            self.compute_means = True
            self.logger.info(f'#############task: {task:0>3d} is finished Test begin. ##############')
            acc_past_tasks, acc_list = self.ncm_tasks_validate(task)
            iCaRL_result["task_{}_results".format(task)] = acc_past_tasks
            print("task: %d ncm acc:" % task, acc_past_tasks)
            self.logger.info(f'"task: {task} classififer:{"ncm"} fc acc: {acc_past_tasks}"')
            acc_past_tasks, acc_list = self.ncm_tasks_validate(task, classifier="fc")
            iCaRL_result["task_{}_fc_results".format(task)] = acc_past_tasks
            print("task: %d fc acc:" % task, acc_past_tasks)
            self.logger.info(f'"task: {task} classififer:{"fc"} fc acc: {acc_past_tasks}"')
            self.pre_FE_cls = copy.deepcopy(self.FE_cls).eval()
            with open(self.result_file, 'w') as fw:
                json.dump(iCaRL_result, fw, indent=4)
        with open(self.result_file, 'w') as fw:
            json.dump(iCaRL_result, fw, indent=4)

    def feature_relearning_train_main(self, args):
        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''
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
        iCaRL_result = {"timestamp": str(datetime.datetime.now()), "model_rate": self.rate, "is_iCaRL": args.is_iCaRL}
        iCaRL_result.update(self.data_config)
        img_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS["imagenet_examplar"]["train_transform"],
        ])
        for task, train_dataset in enumerate(self.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')
            self.batch_train_logger.info(f'New task {task} begin.')
            print("New task %d begin." % task)
            # Add exemplars (if available) to current dataset (if requested)
            if task > 1 and self.memory_budget > 0:
                exemplar_sets_for_feature_relearn = copy.deepcopy(self.exemplar_sets)
                feature_relearn_exemplar_dataset = ExemplarDataset(exemplar_sets_for_feature_relearn,
                                                                   img_transform=img_transform,
                                                                   inv_transform=inv_imagenet_normalize)
                # feature_relearn_exemplar_dataset = ExemplarDataset(exemplar_sets_for_feature_relearn)
                exemplar_dataset = ExemplarDataset(self.exemplar_sets,
                                                   img_transform=img_transform,
                                                   inv_transform=inv_imagenet_normalize)
                training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
            else:
                training_dataset = train_dataset

            # Find [active_classes]
            if task > 1:
                active_classes = list(range(self.classes_per_task * task))
                if task > 1:
                    self.logger.info(f'New task {task} begin: use feature_relearn_exemplar_dataset '
                                     f'to train FA'
                                     f'use feature_relearn_exemplar_dataset to train FE_cls.')
                    assert feature_relearn_exemplar_dataset is not None
                    self.EFAfIL_split_feature_mapper_cls_domain_train(train_dataset, feature_relearn_exemplar_dataset,
                                                                      self.val_datasets,
                                                                      self.classes_per_task,
                                                                      active_classes, task,
                                                                      NewData_to_oneclass=args.NewData_to_oneclass,
                                                                      val_current_task=args.val_current_task)
                    self.pre_FE_cls = copy.deepcopy(self.FE_cls).eval()
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
                                utils.get_data_loader(training_dataset, self.batch_size, self.num_workers,
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
                            # extra_scores = None

                        # ---> Train MAIN MODEL
                        # Train the main model with this batch
                        loss_dict = self.train_a_batch(x, y, scores, active_classes, optimizer, task)
                        iter_index += 1
                        if iter_index % args.print_interval == 0:
                            rtls = f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                   f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                   f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"
                            self.batch_train_logger.info(
                                rtls
                            )
                            print(rtls)
                    scheduler.step()
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes)
                    self.batch_train_logger.info(
                        f"batch train current task validate || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
                        f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    )
                    self.batch_train_logger.info(f"------------------------------------------------------------------")
                    print(f'batch train task : {task:0>3d}, epoch: {epoch}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (
                        acc1, acc5))
                    print("------------------------------------")
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes)
                    self.batch_train_logger.info(
                        f"batch train current task validate || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
                        f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    )
                    self.batch_train_logger.info(f"------------------------------------------------------------------")
                    print(f'batch train task : {task:0>3d}, epoch: {epoch}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (
                        acc1, acc5))
                    print("------------------------------------")
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes,
                                                                        val_current_task=args.val_current_task)
                    self.batch_train_logger.info(
                        f"batch train old classes validate || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
                        f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    )
                    self.batch_train_logger.info(f"------------------------------------------------------------------")
                    print(
                        f'batch train old classes : {task:0>3d}, epoch: {epoch}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (
                            acc1, acc5))
                    print("------------------------------------")
                # if task == 1:
                #     checkpoint_file = os.path.join(args.checkpoints,
                #                                    "{}_cifar100_resnet34_iCaRL_rate_{}_FE_cls_task1.pth".format(
                #                                        self.classes_per_task, self.rate))
                #     torch.save(self.FE_cls.module.state_dict(), checkpoint_file)
            else:
                model_task20_for_task1 = "/share/home/kcli/CL_research/iCaRL_ILtFA/Imagenet100EXP_July/" \
                                         "Imagenet100pretrain/iCaRL_EXP/DATAforCL_1/task20/test1/rate_1/"\
                                              "imagenet100_checkpoints/5_resnet18_rate_1_FEcls_task_1.pth"
                model_task4_for_task1 = "/share/home/kcli/CL_research/iCaRL_ILtFA/Imagenet100EXP_July/" \
                                         "Imagenet100pretrain/iCaRL_EXP/DATAforCL_1/task4/test3/rate_1/" \
                                         "imagenet100_checkpoints/25_resnet18_rate_1_FEcls_task_1.pth"
                model_task10_for_task1 = "/share/home/kcli/CL_research/iCaRL_ILtFA/Imagenet100EXP_July/" \
                                         "Imagenet100pretrain/iCaRL_EXP/DATAforCL_1/task10/test1/rate_1/" \
                                         "imagenet100_checkpoints/10_resnet18_rate_1_FEcls_task_1.pth"
                if self.tasks == 4:
                    model_path = model_task4_for_task1
                elif self.tasks == 10:
                    model_path = model_task10_for_task1
                elif self.tasks == 20:
                    model_path = model_task20_for_task1
                self.FE_cls = self.load_model(model_path)
                # self.FE_cls = torch.load("")
                # self.FE_cls = torch.load("cifar10_preTrain_task_1_FE_cls.pth")
                # continue
                print("model:", self.FE_cls)
            self.batch_train_logger.info(f'##########feature extractor train task {task} End.#########')
            self.logger.info(f'#############feature extractor train task {task} End.##############')
            self.logger.info(f'#############Example handler task {task} start.##############')
            print("feature extractor train task-%d End" % (task))
            print("Example handler task-%d start." % (task))
            # EXEMPLARS: update exemplar sets
            exemplars_per_class = int(np.floor(self.memory_budget / (self.classes_per_task * task)))
            # reduce examplar-sets
            self.reduce_exemplar_sets(exemplars_per_class)
            # for each new class trained on, construct examplar-set
            new_classes = list(range(self.classes_per_task * (task - 1), self.classes_per_task * task))
            for class_id in new_classes:
                # create new dataset containing only all examples of this class
                print("construct_exemplar_set class_id:", class_id)
                class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                # based on this dataset, construct new exemplar-set for this class
                self.construct_exemplar_set(dataset=class_dataset, n=exemplars_per_class)
            self.compute_means = True
            self.logger.info(f'#############task: {task:0>3d} is finished Test begin. ##############')
            acc_past_tasks, acc_list = self.ncm_tasks_validate(task)
            iCaRL_result["task_{}_results".format(task)] = acc_past_tasks
            print("task: %d ncm acc:" % task, acc_past_tasks)
            self.logger.info(f'"task: {task} classififer:{"ncm"} fc acc: {acc_past_tasks}"')
            acc_past_tasks, acc_list = self.ncm_tasks_validate(task, classifier="fc")
            iCaRL_result["task_{}_fc_results".format(task)] = acc_past_tasks
            print("task: %d fc acc:" % task, acc_past_tasks)
            self.logger.info(f'"task: {task} classififer:{"fc"} fc acc: {acc_past_tasks}"')
            self.compute_means = True
            with open(self.result_file, 'w') as fw:
                json.dump(iCaRL_result, fw, indent=4)
            if "forCL_1" in self.dataset_json_path and task == self.tasks / 2:
                break
        with open(self.result_file, 'w') as fw:
            json.dump(iCaRL_result, fw, indent=4)

    def LwF_MC_feature_relearning_train_main(self, args):
        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''
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
        iCaRL_result = {"timestamp": str(datetime.datetime.now()), "model_rate": self.rate, "is_iCaRL": args.is_iCaRL}
        iCaRL_result.update(self.data_config)
        img_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS["imagenet"]["train_transform"],
        ])
        for task, train_dataset in enumerate(self.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')
            self.batch_train_logger.info(f'New task {task} begin.')
            print("New task %d begin." % task)
            # Add exemplars (if available) to current dataset (if requested)
            if task > 1 and self.memory_budget > 0:
                exemplar_sets_for_feature_relearn = copy.deepcopy(self.exemplar_sets)
                feature_relearn_exemplar_dataset = ExemplarDataset(exemplar_sets_for_feature_relearn,
                                                                   img_transform=img_transform,
                                                                   inv_transform=inv_imagenet_normalize)
                # feature_relearn_exemplar_dataset = ExemplarDataset(exemplar_sets_for_feature_relearn)
                exemplar_dataset = ExemplarDataset(self.exemplar_sets)
                training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
            else:
                training_dataset = train_dataset

            # Find [active_classes]
            if task >= 1:
                active_classes = list(range(self.classes_per_task * task))
                if task > 1:
                    self.logger.info(f'New task {task} begin: use feature_relearn_exemplar_dataset '
                                     f'to train FA'
                                     f'use exemplar_dataset to train FE_cls.')
                    assert feature_relearn_exemplar_dataset is not None
                    self.EFAfIL_split_feature_mapper_cls_domain_train(train_dataset, feature_relearn_exemplar_dataset,
                                                                      self.val_datasets,
                                                                      self.classes_per_task,
                                                                      active_classes, task)
                    self.pre_FE_cls = copy.deepcopy(self.FE_cls).eval()
                optimizer = self.build_optimize()
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=self.milestones, gamma=self.gamma)
                for epoch in range(1, self.epochs + 1):
                    is_first_ite = True
                    iters_left = 1
                    iter_index = 0
                    while iters_left > 0:
                        # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                        iters_left -= 1
                        iter_num = 0
                        if is_first_ite:
                            is_first_ite = False
                            data_loader = iter(
                                utils.get_data_loader(training_dataset, self.batch_size, self.num_workers,
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
                            # extra_scores = None

                        # ---> Train MAIN MODEL
                        # Train the main model with this batch
                        loss_dict = self.train_a_batch_MC(x, y, scores, active_classes, optimizer, task)
                        iter_index += 1
                        if iter_index % args.print_interval == 0:
                            self.batch_train_logger.info(
                                f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], "
                                f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: "
                                f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"
                                f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
                            )
                    scheduler.step()
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes)
                    self.batch_train_logger.info(
                        f"batch train current task validate || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
                        f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    )
                    self.batch_train_logger.info(f"------------------------------------------------------------------")
                    print(f'batch train task : {task:0>3d}, epoch: {epoch}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (
                        acc1, acc5))
                # if task == 1:
                #     checkpoint_file = os.path.join(args.checkpoints,
                #                                    "{}_cifar100_resnet34_iCaRL_rate_{}_FE_cls_task1.pth".format(
                #                                        self.classes_per_task, self.rate))
                #     torch.save(self.FE_cls.module.state_dict(), checkpoint_file)
            # else:
            #     self.FE_cls = self.load_FE_cls_model("/share/home/kcli/CL_research/iCaRL_ILtFA/cifar10pretrain_EFAfIL_task1.pth")
            #     # self.FE_cls = torch.load("")
            #     # self.FE_cls = torch.load("cifar10_preTrain_task_1_FE_cls.pth")
            #     # continue
            #     print("model:", self.FE_cls)
            self.batch_train_logger.info(f'##########feature extractor train task {task} End.#########')
            self.logger.info(f'#############feature extractor train task {task} End.##############')
            self.logger.info(f'#############Example handler task {task} start.##############')
            print("feature extractor train task-%d End" % (task))
            print("Example handler task-%d start." % (task))
            # EXEMPLARS: update exemplar sets
            exemplars_per_class = int(np.floor(self.memory_budget / (self.classes_per_task * task)))
            # reduce examplar-sets
            self.reduce_exemplar_sets(exemplars_per_class)
            # for each new class trained on, construct examplar-set
            new_classes = list(range(self.classes_per_task * (task - 1), self.classes_per_task * task))
            for class_id in new_classes:
                # create new dataset containing only all examples of this class
                print("construct_exemplar_set class_id:", class_id)
                class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                # based on this dataset, construct new exemplar-set for this class
                self.construct_exemplar_set(dataset=class_dataset, n=exemplars_per_class)
            self.compute_means = True
            self.logger.info(f'#############task: {task:0>3d} is finished Test begin. ##############')
            acc_past_tasks, acc_list = self.ncm_tasks_validate(task)
            iCaRL_result["task_{}_results".format(task)] = acc_past_tasks
            print("task: %d ncm acc:" % task, acc_past_tasks)
            self.logger.info(f'"task: {task} classififer:{"ncm"} fc acc: {acc_past_tasks}"')
            acc_past_tasks, acc_list = self.ncm_tasks_validate(task, classifier="fc")
            iCaRL_result["task_{}_fc_results".format(task)] = acc_past_tasks
            print("task: %d fc acc:" % task, acc_past_tasks)
            self.logger.info(f'"task: {task} classififer:{"fc"} fc acc: {acc_past_tasks}"')
            if self.compute_means:
                self.batch_train_logger.info(
                    "self.compute_means is True"
                )
            else:
                self.batch_train_logger.info(
                    "self.compute_means is False"
                )
            self.compute_means = True
            with open(self.result_file, 'w') as fw:
                json.dump(iCaRL_result, fw, indent=4)
        with open(self.result_file, 'w') as fw:
            json.dump(iCaRL_result, fw, indent=4)

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
        if x is not None:

            # Run model
            y_hat = self(x)
            # -if needed, remove predictions for classes not in current task
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                y_hat = y_hat[:, class_entries]

            # Calculate prediction loss
            # -binary prediction loss
            binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
            if scores is not None:
                binary_targets = binary_targets[:, -self.classes_per_task:]
                binary_targets = torch.cat([torch.sigmoid(scores), binary_targets], dim=1)
            predL = None if y is None else Func.binary_cross_entropy_with_logits(
                input=y_hat, target=binary_targets, reduction='none'
            ).sum(dim=1).mean()  # --> sum over classes, then average over batch
            if len(active_classes) >= 5:
                acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
                top1.update(acc1.item(), x.size(0))
                top5.update(acc5.item(), x.size(0))
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
                'task': task,
                'loss_total': loss_total.item(),
                'precision': precision if precision is not None else 0.,
                "top1": top1.avg,
                "top5": top5.avg,
                "losses": losses.avg
            }
        else:
            return {
                'task': task,
                'loss_total': loss_total.item(),
                'precision': precision if precision is not None else 0.,
                "top1": top1.avg,
                "top5": 0,
                "losses": losses.avg
            }

    def train_a_batch_MC(self, x, y, scores, active_classes, optimizer, task=1):
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
        if x is not None:

            # Run model
            y_hat = self(x)
            # -if needed, remove predictions for classes not in current task
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                y_hat = y_hat[:, class_entries]

            # Calculate prediction loss
            # -binary prediction loss
            if scores is not None:
                y_hat_fordistill = y_hat[:, :(self.classes_per_task * (task - 1))]
                binary_targets = torch.sigmoid(scores / self.KD_temp)
                loss_distill = Func.binary_cross_entropy_with_logits(
                    input=y_hat_fordistill, target=binary_targets, reduction='none'
                ).sum(dim=1).mean()  # --> sum over classes, then average over batch
                loss_cls = criterion(y_hat, y)
                predL = loss_distill + loss_cls
            else:
                binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
                predL = None if y is None else Func.binary_cross_entropy_with_logits(
                    input=y_hat, target=binary_targets, reduction='none'
                ).sum(dim=1).mean()  # --> sum over classes, then average over batch
            if len(active_classes) >= 5:
                acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
                top1.update(acc1.item(), x.size(0))
                top5.update(acc5.item(), x.size(0))
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
                'task': task,
                'loss_total': loss_total.item(),
                'precision': precision if precision is not None else 0.,
                "top1": top1.avg,
                "top5": top5.avg,
                "losses": losses.avg
            }
        else:
            return {
                'task': task,
                'loss_total': loss_total.item(),
                'precision': precision if precision is not None else 0.,
                "top1": top1.avg,
                "top5": 0,
                "losses": losses.avg
            }

    def train_a_batch_plus_extraData(self, x, y, extra_x, extra_y, scores, extra_scores, active_classes,
                                     optimizer, task):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        # Set model to training-mode
        self.train()
        # Reset optimizer
        optimizer.zero_grad()
        criterion = torch.nn.CrossEntropyLoss().cuda()
        # Run model
        y_hat = self(x)
        # -if needed, remove predictions for classes not in current task
        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
            y_hat = y_hat[:, class_entries]
        # Calculate prediction loss
        # -binary prediction loss
        binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
        extra_predL = 0
        if scores is not None:
            binary_targets = binary_targets[:, -self.classes_per_task:]
            binary_targets = torch.cat([torch.sigmoid(scores / self.KD_temp), binary_targets], dim=1)
            extra_y_hat = self(extra_x)
            extra_y_hat = extra_y_hat[:, :(self.classes_per_task * (task - 1))]
            extra_scores = torch.sigmoid(extra_scores / self.KD_temp)
            extra_predL = Func.binary_cross_entropy_with_logits(
                input=extra_y_hat, target=extra_scores, reduction='none'
            ).sum(dim=1).mean()
        predL = extra_predL + Func.binary_cross_entropy_with_logits(
            input=y_hat, target=binary_targets, reduction='none'
        ).sum(dim=1).mean()  # --> sum over classes, then average over batch
        acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
        top1.update(acc1.item(), x.size(0))
        top5.update(acc5.item(), x.size(0))
        losses.update(predL, x.size(0))
        # Calculate training-precision
        precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
        loss_total = predL
        loss_total.backward()
        # Take optimization-step
        optimizer.step()
        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'precision': precision if precision is not None else 0.,
            "top1": top1.avg,
            "top5": top5.avg,
            "losses": losses.avg
        }

    def current_task_validate(self, task, active_classes, val_current_task=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        if val_current_task:
            val_loader = utils.get_data_loader(self.val_datasets[task - 1], self.batch_size,  # task index must minus 1
                                               self.num_workers,
                                               cuda=True if self.availabel_cudas else False)
        else:
            val_dataset = self.val_datasets[0]
            for i in range(1, task - 1):
                val_dataset = ConcatDataset([val_dataset, self.val_datasets[i]])
            val_loader = utils.get_data_loader(val_dataset, self.batch_size,  # task index must minus 1
                                               self.num_workers,
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

    def ncm_task_validate(self, val_dataset, task, classifier="ncm"):
        # todo
        val_loader = utils.get_data_loader(val_dataset, self.batch_size,
                                           self.num_workers,
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
            elif classifier == "fc":
                self.eval()
                with torch.no_grad():
                    y_hat = self(inputs)
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    y_hat = y_hat[:, class_entries]
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

    def ncm_tasks_validate(self, task, classifier="ncm"):
        # todo
        acc_past_tasks = []
        acc = []
        for task_id in range(task):
            predict_result = self.ncm_task_validate(self.val_datasets[task_id], task, classifier)
            acc_past_tasks.append(predict_result)
            acc.append(predict_result[0])
        acc = np.array(acc)
        print("classifier{} acc_avg:{}".format(classifier, acc.mean()))
        self.logger.info(
            f"per task {task}, classifier{classifier} avg acc:{acc.mean()}"
            f"-------------------------------------------------------------"
        )
        return acc_past_tasks, acc
        pass
