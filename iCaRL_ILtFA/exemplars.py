import abc
import os
import random
import time
from torch.nn import functional as Func
import joblib
import torch
from sklearn import svm
from sklearn.model_selection import train_test_split
from torch import nn
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset

from public import utils, MLP
import copy
import numpy as np

from public.data import SubDataset, FeaturesDataset, ExemplarDataset
from public.util_models import MLP_cls_domain_dis, MLP_for_FM, SoftTarget_CrossEntropy, BiasLayer
from public.utils import AverageMeter, accuracy, convert_to_oneclass


class ExemplarHandler(nn.Module, metaclass=abc.ABCMeta):
    """Abstract  module for a classifier that can store and use exemplars.

    Adds a exemplar-methods to subclasses, and requires them to provide a 'feature-extractor' method."""

    def __init__(self, memory_budget, batchsize, num_workers, norm_exemplars, herding,
                 feature_dim, num_classes,
                 MLP_name, MLP_KD_temp, MLP_KD_temp_2,
                 MLP_lr, MLP_rate, MLP_momentum, MLP_milestones, MLP_lrgamma, MLP_weight_decay,
                 MLP_epochs, MLP_optim_type, MLP_distill_rate, availabel_cudas):
        super().__init__()

        # list with exemplar-sets
        self.exemplar_sets = []  # --> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.assist_current_exemplar_sets = []  # --> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_means = []
        self.compute_means = True

        # settings
        self.memory_budget = memory_budget
        self.norm_exemplars = norm_exemplars
        self.herding = herding
        self.num_workers = num_workers

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.batch_size = batchsize
        self.MLP_name = MLP_name

        self.MLP_rate = MLP_rate

        self.MLP_KD_temp = MLP_KD_temp
        self.MLP_KD_temp_2 = MLP_KD_temp_2

        self.MLP_distill_rate = MLP_distill_rate
        self.MLP_lr = MLP_lr
        self.MLP_momentum = MLP_momentum
        self.MLP_milestones = MLP_milestones
        self.MLP_gamma = MLP_lrgamma
        self.MLP_weight_decay = MLP_weight_decay
        self.MLP_epochs = MLP_epochs
        self.MLP_optim_type = MLP_optim_type
        self.availabel_cudas = availabel_cudas
        self.pre_FM_cls_domain = None
        self.MLP_device = "cuda" if self.availabel_cudas else "cpu"
        self.FM_cls_domain = None
        self.unbias_cls_of_FMcls = None

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def construct_FM_cls_domain_model(self):
        if self.availabel_cudas:
            FM_cls_domain_model = torch.nn.DataParallel(
                MLP_cls_domain_dis(MLP.__dict__[self.MLP_name](input_dim=self.feature_dim,
                                                               out_dim=self.feature_dim,
                                                               rate=self.MLP_rate),
                                   self.feature_dim, self.num_classes)).cuda()
            cudnn.benchmark = True
        else:
            FM_cls_domain_model = MLP_cls_domain_dis(MLP.__dict__[self.MLP_name](input_dim=self.feature_dim,
                                                                                 out_dim=self.feature_dim,
                                                                                 rate=self.MLP_rate),
                                                     self.feature_dim, self.num_classes)
        return FM_cls_domain_model

    def build_FM_cls_domain_optimize(self):
        # Define optimizer (only include parameters that "requires_grad")
        optim_list = [{'params': filter(lambda p: p.requires_grad, self.FM_cls_domain.parameters()), 'lr': self.MLP_lr}]
        optimizer = None
        if self.MLP_optim_type in ("adam", "adam_reset"):
            if self.MLP_weight_decay:
                optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999), weight_decay=self.MLP_weight_decay)
            else:
                optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999))
        elif self.MLP_optim_type == "sgd":
            if self.MLP_momentum:
                optimizer = torch.optim.SGD(optim_list, momentum=self.MLP_momentum, weight_decay=self.MLP_weight_decay)
            else:
                optimizer = torch.optim.SGD(optim_list)
        else:
            raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(self.MLP_optim_type))
        return optimizer

    def build_optimize_bias_cls(self, lr):
        # Define optimizer (only include parameters that "requires_grad")
        assert self.unbias_cls_of_FMcls is not None
        optim_list = [{'params': filter(lambda p: p.requires_grad, self.unbias_cls_of_FMcls.parameters()), 'lr': lr}]
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

    @abc.abstractmethod
    def feature_extractor(self, images):
        pass

    @abc.abstractmethod
    def get_FE_cls_output(self, images):
        pass

    def FM_cls_forward_nograd(self, features):
        self.FM_cls_domain.eval()
        with torch.no_grad():
            if type(self.FM_cls_domain) is torch.nn.DataParallel:
                return self.FM_cls_domain.module.cls(features)
            else:
                return self.FM_cls_domain.cls(features)

    def feature_mapper_cls_domain_train(self, training_dataset, test_datasets,
                                        classes_per_task, active_classes, task, NewData_to_oneclass=False,
                                        val_current_task=True):
        # todo Done
        self.batch_train_logger.info(f'####Task {task} EFAfIL_iCaRL feature_mapper_cls_domain_train begin.####')
        self.logger.info(f'####Task {task} EFAfIL_iCaRL  feature_mapper_cls_domain_train begin.####')
        print("Task %d , train feature mapping..." % task)
        print("Ablation NoOversample study")
        self.logger.info(f'####Task {task} Ablation NoOversample study.####')
        if self.FM_cls_domain is None:
            self.FM_cls_domain = self.construct_FM_cls_domain_model()
        mode = self.training
        self.eval()
        self.FM_cls_domain.train()
        criterion = nn.CrossEntropyLoss()
        soft_target_criterion = SoftTarget_CrossEntropy().cuda()
        optimizer = self.build_FM_cls_domain_optimize()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
        for epoch in range(self.MLP_epochs):
            train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
                                                      self.num_workers,
                                                      cuda=True if self.availabel_cudas else False))
            iter_index = 0
            iters_left = len(train_loader)
            while iters_left > 0:
                iters_left -= 1
                optimizer.zero_grad()
                iter_index += 1
                '''读取数据'''
                imgs, labels = next(train_loader)
                '''Take strategy of seeing new coming classes as one super class'''
                if NewData_to_oneclass:
                    labels = convert_to_oneclass(labels)
                imgs, labels = imgs.to(self.MLP_device), labels.to(self.MLP_device)
                '''获取图片数据在FE_cls的输出'''
                imgs_2_features, imgs_2_targets = self.get_FE_cls_output(imgs)  # todo Done
                imgs_2_targets = imgs_2_targets[:, :(classes_per_task * (task - 1))]
                '''获取在要训练的模型FM'''
                imgs_2_feature_hat, y_hat, _ = self.FM_cls_domain(imgs_2_features)
                y_hat = y_hat[:, :(classes_per_task * task)]
                '''make distill target and train data'''
                y_hat_pre_tasks = y_hat[:, :(classes_per_task * (task - 1))]

                '''softTarget loss distill'''
                # imgs_2_targets = torch.softmax(imgs_2_targets / self.MLP_KD_temp, dim=1)
                # loss_distill_current_imgs = soft_target_criterion(y_hat_pre_tasks, imgs_2_targets,
                #                                                   self.MLP_KD_temp)
                #
                # loss_distill = loss_distill_current_imgs

                '''binary loss distill'''
                imgs_2_targets = torch.sigmoid(imgs_2_targets / self.MLP_KD_temp)
                y_hat_pre_tasks /= self.MLP_KD_temp_2
                loss_distill_current_imgs = Func.binary_cross_entropy_with_logits(input=y_hat_pre_tasks,
                                                                                  target=imgs_2_targets,
                                                                                  reduction='none').sum(dim=1).mean()
                loss_distill = loss_distill_current_imgs
                '''loss classify'''
                '''cross entropy loss classify'''
                loss_cls = criterion(y_hat, labels)
                '''total loss '''
                loss_total = loss_cls + self.MLP_distill_rate * loss_distill * self.MLP_KD_temp ** 2
                loss_total.backward()
                optimizer.step()
                precision = None if y_hat is None else (labels == y_hat.max(1)[
                    1]).sum().item() / labels.size(0)
                lr = scheduler.get_last_lr()[0]
                ite_info = {
                    'task': task,
                    'epoch': epoch,
                    'lr': lr,
                    'loss_total': loss_total.item(),
                    'precision': precision if precision is not None else 0.,
                }
                self.batch_train_logger.info(
                    f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, "
                    f"lr: {lr}, loss_total: {loss_total}, acc: {precision}")
                print(ite_info)
                print("....................................")
            scheduler.step()
        print("-----------------------------------")
        acc1, acc5, throughput = self.current_task_validate_FM_cls_domain(test_datasets, task, active_classes)  # todo
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapper_cls_domain train task : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))
        print("-----------------------------------")
        acc1, acc5, throughput = self.current_task_validate_FM_cls_domain(test_datasets, task, active_classes,
                                                                          val_current_task=val_current_task)  # todo
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train old classes || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature mapper_cls_domain train old classes|| task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapper_cls_domain train old classes : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (
            acc1, acc5))
        print("-----------------------------------")
        print("Train feature mapper_cls_domain End.")
        print("Train feature mapper_cls_domain End.")
        self.pre_FM_cls_domain = copy.deepcopy(self.FM_cls_domain).eval()
        self.train(mode=mode)
        pass

    def EFAfIL_split_feature_mapper_cls_domain_train(self, train_dataset, exemplar_dataset, test_datasets,
                                                     classes_per_task, active_classes, task, NewData_to_oneclass=False,
                                                     val_current_task=True):
        # todo Done
        self.batch_train_logger.info(f'####Task {task} EFAfIL_iCaRL feature_mapper_cls_domain_train begin.####')
        self.logger.info(f'####Task {task} EFAfIL_iCaRL  feature_mapper_cls_domain_train begin.####')
        print("Task %d , train feature mapping..." % task)
        # print("Ablation NoAnchor study")
        # self.logger.info(f'####Task {task} Ablation NoAnchor study.####')
        if self.FM_cls_domain is None:
            self.FM_cls_domain = self.construct_FM_cls_domain_model()
        mode = self.training
        self.eval()
        self.FM_cls_domain.train()
        criterion = nn.CrossEntropyLoss()
        soft_target_criterion = SoftTarget_CrossEntropy().cuda()
        optimizer = self.build_FM_cls_domain_optimize()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
        train_dataset_iter_index = 0
        exemplar_datasets_iter_index = 0
        train_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                  self.num_workers,
                                                  cuda=True if self.availabel_cudas else False))
        exemplar_loader = iter(utils.get_data_loader(exemplar_dataset,
                                                     self.batch_size,
                                                     self.num_workers,
                                                     cuda=True if self.availabel_cudas else False))
        train_imgs_num = len(train_loader)
        exemplar_num = len(exemplar_loader)
        for epoch in range(self.MLP_epochs):
            if train_dataset_iter_index == train_imgs_num:
                train_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                          self.num_workers,
                                                          cuda=True if self.availabel_cudas else False))
                train_dataset_iter_index = 0
            if exemplar_datasets_iter_index == exemplar_num:
                exemplar_loader = iter(utils.get_data_loader(exemplar_dataset,
                                                             self.batch_size,
                                                             self.num_workers,
                                                             cuda=True if self.availabel_cudas else False))
                exemplar_datasets_iter_index = 0
            iter_index = 0
            iters_left = max(train_imgs_num, exemplar_num)
            while iters_left > 0:
                iters_left -= 1
                optimizer.zero_grad()
                if train_dataset_iter_index == train_imgs_num:
                    train_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                              self.num_workers,
                                                              cuda=True if self.availabel_cudas else False))
                    train_dataset_iter_index = 0
                if exemplar_datasets_iter_index == exemplar_num:
                    exemplar_loader = iter(
                        utils.get_data_loader(exemplar_dataset,
                                              self.batch_size,
                                              self.num_workers,
                                              cuda=True if self.availabel_cudas else False))
                    exemplar_datasets_iter_index = 0
                iter_index += 1
                '''读取数据'''
                imgs, labels = next(train_loader)

                '''Take strategy of seeing new coming classes as one super class'''
                if NewData_to_oneclass:
                    labels = convert_to_oneclass(labels)

                examplar_imgs, examplar_labels = next(exemplar_loader)
                train_dataset_iter_index += 1
                exemplar_datasets_iter_index += 1
                imgs, labels = imgs.to(self.MLP_device), labels.to(self.MLP_device)
                examplar_imgs, examplar_labels = examplar_imgs.to(self.MLP_device), examplar_labels.to(self.MLP_device)
                '''获取图片数据在FE_cls的输出'''
                imgs_2_features, imgs_2_targets = self.get_FE_cls_output(imgs)  # todo Done
                exemplar_imgs_2_features, exemplar_imgs_2_targets = self.get_FE_cls_output(examplar_imgs)
                imgs_2_targets = imgs_2_targets[:, :(classes_per_task * task)]
                exemplar_imgs_2_targets = exemplar_imgs_2_targets[:, :(classes_per_task * task)]
                '''获取在要训练的模型FM的输出结果'''
                imgs_2_feature_hat, y_hat, _ = self.FM_cls_domain(imgs_2_features)
                y_hat = y_hat[:, :(classes_per_task * task)]
                exemplar_imgs_2_feature_hat, exemplar_y_hat, _ = self.FM_cls_domain(exemplar_imgs_2_features)
                exemplar_y_hat = exemplar_y_hat[:, :(classes_per_task * task)]
                '''make distill target and train data'''
                y_hat_pre_tasks = y_hat[:, :(classes_per_task * (task - 1))]
                exemplar_y_hat_pre_tasks = exemplar_y_hat[:, :(classes_per_task * (task - 1))]
                imgs_2_targets = imgs_2_targets[:, :(classes_per_task * (task - 1))]
                exemplar_imgs_2_targets = exemplar_imgs_2_targets[:, :(classes_per_task * (task - 1))]
                if self.MLP_KD_temp_2 > 0:
                    '''binary loss distill'''
                    imgs_2_targets = torch.sigmoid(imgs_2_targets / self.MLP_KD_temp)
                    exemplar_imgs_2_targets = torch.sigmoid(exemplar_imgs_2_targets / self.MLP_KD_temp)
                    y_hat_pre_tasks /= self.MLP_KD_temp_2
                    exemplar_y_hat_pre_tasks /= self.MLP_KD_temp_2
                    loss_distill_current_imgs = Func.binary_cross_entropy_with_logits(input=y_hat_pre_tasks,
                                                                                      target=imgs_2_targets,
                                                                                      reduction='none').sum(
                        dim=1).mean()
                    loss_distill_exemplar_imgs = Func.binary_cross_entropy_with_logits(input=exemplar_y_hat_pre_tasks,
                                                                                       target=exemplar_imgs_2_targets,
                                                                                       reduction='none').sum(
                        dim=1).mean()
                    loss_distill = loss_distill_current_imgs + loss_distill_exemplar_imgs
                else:
                    '''softTarget loss distill'''
                    imgs_2_targets = torch.softmax(imgs_2_targets / self.MLP_KD_temp, dim=1)
                    exemplar_imgs_2_targets = torch.softmax(exemplar_imgs_2_targets / self.MLP_KD_temp, dim=1)
                    loss_distill_current_imgs = soft_target_criterion(y_hat_pre_tasks, imgs_2_targets,
                                                                      self.MLP_KD_temp)

                    loss_distill_exemplar_imgs = soft_target_criterion(exemplar_y_hat_pre_tasks,
                                                                       exemplar_imgs_2_targets,
                                                                       self.MLP_KD_temp)
                    loss_distill = loss_distill_current_imgs + loss_distill_exemplar_imgs

                '''loss classify'''
                '''cross entropy loss classify'''
                loss_cls = criterion(y_hat, labels) + criterion(exemplar_y_hat, examplar_labels)
                '''total loss '''
                loss_total = loss_cls + self.MLP_distill_rate * (loss_distill * self.MLP_KD_temp ** 2)

                '''Ablation study'''
                # loss_total = loss_cls

                loss_total.backward()
                optimizer.step()
                precision = None if y_hat is None else (labels == y_hat.max(1)[
                    1]).sum().item() / labels.size(0)
                lr = scheduler.get_last_lr()[0]
                ite_info = {
                    'task': task,
                    'epoch': epoch,
                    'lr': lr,
                    'loss_total': loss_total.item(),
                    'precision': precision if precision is not None else 0.,
                }
                self.batch_train_logger.info(
                    f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, "
                    f"lr: {lr}, loss_total: {loss_total}, acc: {precision}")
                print(ite_info)
                print("....................................")
            scheduler.step()
        print("-----------------------------------")
        acc1, acc5, throughput = self.current_task_validate_FM_cls_domain(test_datasets, task, active_classes)  # todo
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapper_cls_domain train task : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))
        print("-----------------------------------")
        acc1, acc5, throughput = self.current_task_validate_FM_cls_domain(test_datasets, task, active_classes,
                                                                          val_current_task=False)  # todo
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train old classes || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature mapper_cls_domain train old classes|| task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapper_cls_domain train old classes : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (
            acc1, acc5))
        print("-----------------------------------")
        print("Train feature mapper_cls_domain End.")
        self.pre_FM_cls_domain = copy.deepcopy(self.FM_cls_domain).eval()
        self.train(mode=mode)
        pass

    def retrain_bias_cls_of_FMcls(self, per_task_valing_dataset, val_datasets, task, active_classes, print_interval,
                                  train_method=0, bias_or_cRT=0):
        current_classes_num = self.classes_per_task
        if self.unbias_cls_of_FMcls is None:
            if bias_or_cRT == 0:
                self.unbias_cls_of_FMcls = BiasLayer()
            else:
                # self.unbias_cls_of_FMcls = nn.Linear(self.feature_dim, self.classes_per_task * task)
                if type(self.FM_cls_domain) is torch.nn.DataParallel:
                    self.unbias_cls_of_FMcls = copy.deepcopy(self.FM_cls_domain.module.cls)
                else:
                    self.unbias_cls_of_FMcls = copy.deepcopy(self.FM_cls_domain.cls)
        if train_method == 0:
            optimizer = self.build_optimize_bias_cls(0.001)
            epochs = 45
            gap = int(epochs / 3)
            milestones = [gap, 2 * gap]
        elif train_method == 1:
            optimizer = self.build_optimize_bias_cls(0.001)
            epochs = 60
            gap = int(epochs / 3)
            milestones = [gap, 2 * gap]
        elif train_method == 2:
            optimizer = self.build_optimize_bias_cls(0.01)
            epochs = 96
            gap = int(epochs / 4)
            milestones = [gap, 2 * gap, 3 * gap]
        elif train_method == 3:
            if bias_or_cRT == 0:
                optimizer = self.build_optimize_bias_cls(0.01)
            else:
                optimizer = self.build_optimize_bias_cls(0.1)
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
                x, y = x.to(self.MLP_device), y.to(self.MLP_device)  # --> transfer them to correct device
                features = self.feature_extractor(x)
                if bias_or_cRT == 0:
                    FM_cls_targets = self.FM_cls_forward_nograd(features)
                    FM_cls_targets = FM_cls_targets[:, :(self.classes_per_task * task)]
                    loss_dict = self.Bias_Cls_train_a_batch(FM_cls_targets, y, current_classes_num, optimizer,
                                                            active_classes)
                    print("self.unbias_CLs.parameters:", self.unbias_cls_of_FMcls.params)
                else:
                    loss_dict = self.cRT_cls_train_a_batch(features, y, current_classes_num, optimizer,
                                                           active_classes)
                iter_index += 1

                if iter_num > 20:
                    if iter_index % int(print_interval / 2) == 0:
                        if bias_or_cRT == 0:
                            results = f"bias_cls train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                      f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                      f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}" \
                                      f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
                        else:
                            results = f"cRT_cls train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                      f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                      f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}" \
                                      f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
                        self.batch_train_logger.info(
                            results
                        )
                        print(results)
                else:
                    if bias_or_cRT == 0:
                        results = f"bias_cls train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                  f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                  f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}" \
                                  f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
                    else:
                        results = f"cRT_cls train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                  f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                  f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}" \
                                  f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
                    self.batch_train_logger.info(
                        results
                    )
                    print(results)

            # print("self.unbias_CLs.parameters:", self.unbias_cls_of_FMcls.params)
            scheduler.step()
            if bias_or_cRT == 0:
                acc1, acc5, throughput = self.current_task_validate_bias_Cls(val_datasets, task, active_classes,
                                                                             current_classes_num)
            else:
                acc1, acc5, throughput = self.current_task_validate_cRT_Cls(val_datasets, task, active_classes)
            self.batch_train_logger.info(
                f" unbias_CLs validate || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
                f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
            )
            self.batch_train_logger.info(f"------------------------------------------------------------------")
            print(f'unbias_CLs task : {task:0>3d}, epoch: {epoch}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (
                acc1, acc5))

        pass

    def Bias_Cls_train_a_batch(self, FM_cls_targets, y, current_classes_num, optimizer, active_classes=None):
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
        self.unbias_cls_of_FMcls.train()
        # Reset optimizer
        optimizer.zero_grad()
        criterion = torch.nn.CrossEntropyLoss().cuda()

        # Run model
        y_hat = self.unbias_cls_of_FMcls(FM_cls_targets, current_classes_num)

        predL = criterion(y_hat, y)
        if len(active_classes) >= 5:
            acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
            top1.update(acc1.item(), FM_cls_targets.size(0))
            top5.update(acc5.item(), FM_cls_targets.size(0))
            losses.update(predL, FM_cls_targets.size(0))
            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / FM_cls_targets.size(0)
        else:
            acc1 = accuracy(y_hat, y, topk=(1,))[0]
            top1.update(acc1.item(), FM_cls_targets.size(0))
            losses.update(predL, FM_cls_targets.size(0))
            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / FM_cls_targets.size(0)
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

    def cRT_cls_train_a_batch(self, features, y, current_classes_num, optimizer, active_classes=None):
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
        self.unbias_cls_of_FMcls.train()
        # Reset optimizer
        optimizer.zero_grad()
        criterion = torch.nn.CrossEntropyLoss().cuda()

        # Run model
        y_hat = self.unbias_cls_of_FMcls(features)
        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
            y_hat = y_hat[:, class_entries]
        predL = criterion(y_hat, y)
        if len(active_classes) >= 5:
            acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
            top1.update(acc1.item(), y_hat.size(0))
            top5.update(acc5.item(), y_hat.size(0))
            losses.update(predL, y_hat.size(0))
            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / y_hat.size(0)
        else:
            acc1 = accuracy(y_hat, y, topk=(1,))[0]
            top1.update(acc1.item(), y_hat.size(0))
            losses.update(predL, y_hat.size(0))
            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / y_hat.size(0)
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

    def EFAfIL_split_feature_mapper_cls_domain_train_bias(self, Bias_layer, train_dataset, exemplar_dataset,
                                                          test_datasets,
                                                          classes_per_task, active_classes, task):
        # todo Done
        self.batch_train_logger.info(f'####Task {task} EFAfIL_iCaRL feature_mapper_cls_domain_train begin.####')
        self.logger.info(f'####Task {task} EFAfIL_iCaRL  feature_mapper_cls_domain_train begin.####')
        print("Task %d , train feature mapping..." % task)
        if self.FM_cls_domain is None:
            self.FM_cls_domain = self.construct_FM_cls_domain_model()
        mode = self.training
        self.eval()
        if task > 2:
            Bias_layer.eval()
        self.FM_cls_domain.train()
        criterion = nn.CrossEntropyLoss()
        soft_target_criterion = SoftTarget_CrossEntropy().cuda()
        optimizer = self.build_FM_cls_domain_optimize()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
        train_dataset_iter_index = 0
        exemplar_datasets_iter_index = 0
        train_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                  self.num_workers,
                                                  cuda=True if self.availabel_cudas else False))
        exemplar_loader = iter(utils.get_data_loader(exemplar_dataset,
                                                     self.batch_size,
                                                     self.num_workers,
                                                     cuda=True if self.availabel_cudas else False))
        train_imgs_num = len(train_loader)
        exemplar_num = len(exemplar_loader)
        for epoch in range(self.MLP_epochs):
            if train_dataset_iter_index == train_imgs_num:
                train_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                          self.num_workers,
                                                          cuda=True if self.availabel_cudas else False))
                train_dataset_iter_index = 0
            if exemplar_datasets_iter_index == exemplar_num:
                exemplar_loader = iter(utils.get_data_loader(exemplar_dataset,
                                                             self.batch_size,
                                                             self.num_workers,
                                                             cuda=True if self.availabel_cudas else False))
                exemplar_datasets_iter_index = 0
            iter_index = 0
            iters_left = max(train_imgs_num, exemplar_num)
            while iters_left > 0:
                iters_left -= 1
                optimizer.zero_grad()
                if train_dataset_iter_index == train_imgs_num:
                    train_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                              self.num_workers,
                                                              cuda=True if self.availabel_cudas else False))
                    train_dataset_iter_index = 0
                if exemplar_datasets_iter_index == exemplar_num:
                    exemplar_loader = iter(
                        utils.get_data_loader(exemplar_dataset,
                                              self.batch_size,
                                              self.num_workers,
                                              cuda=True if self.availabel_cudas else False))
                    exemplar_datasets_iter_index = 0
                iter_index += 1
                '''读取数据'''
                imgs, labels = next(train_loader)
                examplar_imgs, examplar_labels = next(exemplar_loader)
                train_dataset_iter_index += 1
                exemplar_datasets_iter_index += 1
                imgs, labels = imgs.to(self.MLP_device), labels.to(self.MLP_device)
                examplar_imgs, examplar_labels = examplar_imgs.to(self.MLP_device), examplar_labels.to(self.MLP_device)
                '''获取图片数据在FE_cls的输出'''
                imgs_2_features, imgs_2_targets = self.get_FE_cls_output(imgs)  # todo Done
                exemplar_imgs_2_features, exemplar_imgs_2_targets = self.get_FE_cls_output(examplar_imgs)
                imgs_2_targets = imgs_2_targets[:, :(classes_per_task * (task - 1))]
                exemplar_imgs_2_targets = exemplar_imgs_2_targets[:, :(classes_per_task * (task - 1))]
                '''获取在要训练的模型FM'''
                imgs_2_feature_hat, y_hat, _ = self.FM_cls_domain(imgs_2_features)
                y_hat = y_hat[:, :(classes_per_task * task)]
                exemplar_imgs_2_feature_hat, exemplar_y_hat, _ = self.FM_cls_domain(exemplar_imgs_2_features)
                exemplar_y_hat = exemplar_y_hat[:, :(classes_per_task * task)]
                '''make distill target and train data'''
                y_hat_pre_tasks = y_hat[:, :(classes_per_task * (task - 1))]
                exemplar_y_hat_pre_tasks = exemplar_y_hat[:, :(classes_per_task * (task - 1))]
                '''获取图片数据在Bias_layer的输出'''
                if task > 2:
                    assert Bias_layer is not None
                    with torch.no_grad():
                        imgs_2_targets = Bias_layer(imgs_2_targets, classes_per_task)
                        exemplar_imgs_2_targets = Bias_layer(exemplar_imgs_2_targets, classes_per_task)
                if self.MLP_KD_temp_2 > 0:
                    '''binary loss distill'''
                    imgs_2_targets = torch.sigmoid(imgs_2_targets / self.MLP_KD_temp)
                    exemplar_imgs_2_targets = torch.sigmoid(exemplar_imgs_2_targets / self.MLP_KD_temp)
                    y_hat_pre_tasks /= self.MLP_KD_temp_2
                    exemplar_y_hat_pre_tasks /= self.MLP_KD_temp_2
                    loss_distill_current_imgs = Func.binary_cross_entropy_with_logits(input=y_hat_pre_tasks,
                                                                                      target=imgs_2_targets,
                                                                                      reduction='none').sum(
                        dim=1).mean()
                    loss_distill_exemplar_imgs = Func.binary_cross_entropy_with_logits(input=exemplar_y_hat_pre_tasks,
                                                                                       target=exemplar_imgs_2_targets,
                                                                                       reduction='none').sum(
                        dim=1).mean()
                    loss_distill = loss_distill_current_imgs + loss_distill_exemplar_imgs
                else:
                    '''softTarget loss distill'''
                    imgs_2_targets = torch.softmax(imgs_2_targets / self.MLP_KD_temp, dim=1)
                    exemplar_imgs_2_targets = torch.softmax(exemplar_imgs_2_targets / self.MLP_KD_temp, dim=1)
                    loss_distill_current_imgs = soft_target_criterion(y_hat_pre_tasks, imgs_2_targets,
                                                                      self.MLP_KD_temp)

                    loss_distill_exemplar_imgs = soft_target_criterion(exemplar_y_hat_pre_tasks,
                                                                       exemplar_imgs_2_targets,
                                                                       self.MLP_KD_temp)
                    loss_distill = loss_distill_current_imgs + loss_distill_exemplar_imgs
                '''loss classify'''
                '''cross entropy loss classify'''
                loss_cls = criterion(y_hat, labels) + criterion(exemplar_y_hat, examplar_labels)
                '''total loss '''
                loss_total = loss_cls + self.MLP_distill_rate * (loss_distill * self.MLP_KD_temp ** 2)
                loss_total.backward()
                optimizer.step()
                precision = None if y_hat is None else (labels == y_hat.max(1)[
                    1]).sum().item() / labels.size(0)
                lr = scheduler.get_last_lr()[0]
                ite_info = {
                    'task': task,
                    'epoch': epoch,
                    'lr': lr,
                    'loss_total': loss_total.item(),
                    'precision': precision if precision is not None else 0.,
                }
                self.batch_train_logger.info(
                    f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, "
                    f"lr: {lr}, loss_total: {loss_total}, acc: {precision}")
                print(ite_info)
                print("....................................")
            scheduler.step()
        acc1, acc5, throughput = self.current_task_validate_FM_cls_domain(test_datasets, task, active_classes)  # todo
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapper_cls_domain train task : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))
        print("Train feature mapper_cls_domain End.")
        self.pre_FM_cls_domain = copy.deepcopy(self.FM_cls_domain).eval()
        self.train(mode=mode)
        pass

    def current_task_validate_FM_cls_domain(self, test_datasets, task, active_classes, val_current_task=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        if val_current_task:
            val_loader = utils.get_data_loader(test_datasets[task - 1], self.batch_size,  # task index must minus 1
                                               self.num_workers,
                                               cuda=True if self.availabel_cudas else False)
        else:
            val_dataset = test_datasets[0]
            for i in range(1, task - 1):
                val_dataset = ConcatDataset([val_dataset, test_datasets[i]])
            val_loader = utils.get_data_loader(val_dataset, self.batch_size,  # task index must minus 1
                                               self.num_workers,
                                               cuda=True if self.availabel_cudas else False)
        end = time.time()
        mode = self.FM_cls_domain.training
        self.FM_cls_domain.eval()
        for inputs, labels in val_loader:
            data_time.update(time.time() - end)
            inputs, labels = inputs.to(self.MLP_device), labels.to(self.MLP_device)
            with torch.no_grad():
                _, y_hat, _ = self.FM_cls_domain(self.feature_extractor(inputs))
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
        self.FM_cls_domain.train(mode=mode)
        if len(active_classes) >= 5:
            return top1.avg, top5.avg, throughput
        else:
            return top1.avg, 0, throughput
        pass

    ####----MANAGING EXEMPLAR SETS----####

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]

    def free_assist_current_exemplar_sets(self):
        self.assist_current_exemplar_sets = []

    def construct_exemplar_set(self, dataset, n):
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

        Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''

        # set model to eval()-mode
        mode = self.training
        self.eval()

        n_max = len(dataset)
        print("n_max:{}".format(n_max))
        self.logger.info("len(class_dataset):{}".format(n_max))
        exemplar_set = []

        if self.herding:
            # compute features for each example in [dataset]
            first_entry = True
            dataloader = utils.get_data_loader(dataset, self.batch_size, self.num_workers, is_shuffle=False,
                                               cuda=self._is_on_cuda())
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self._device())
                print("self.feature_extractor(image_batch)")
                with torch.no_grad():
                    feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            if self.norm_exemplars:
                features = F.normalize(features, p=2, dim=1)

            # calculate mean of all features
            class_mean = torch.mean(features, dim=0, keepdim=True)
            if self.norm_exemplars:
                class_mean = F.normalize(class_mean, p=2, dim=1)

            # one by one, select exemplar that makes mean of all exemplars as close to [class_mean] as possible
            exemplar_features = torch.zeros_like(features[:min(n, n_max)])
            list_of_selected = []
            for k in range(min(n, n_max)):
                if k > 0:
                    exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                    features_means = (features + exemplar_sum) / (k + 1)
                    features_dists = features_means - class_mean
                else:
                    features_dists = features - class_mean
                index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1))
                if index_selected in list_of_selected:
                    raise ValueError("Exemplars should not be repeated!!!!")
                list_of_selected.append(index_selected)
                exemplar_set.append(dataset[index_selected][0].numpy())
                exemplar_features[k] = copy.deepcopy(features[index_selected])

                # make sure this example won't be selected again
                features[index_selected] = features[index_selected] + 100000
        else:
            indeces_selected = np.random.choice(n_max, size=min(n, n_max), replace=False)
            for k in indeces_selected:
                exemplar_set.append(dataset[k][0].numpy())

        # add this [exemplar_set] as a [n]x[ich]x[isz]x[isz] to the list of [exemplar_sets]
        self.exemplar_sets.append(np.array(exemplar_set))

        # set mode of model back
        self.train(mode=mode)

    def construct_assist_current_exemplar_set(self, dataset, n):
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

        Note that [dataset] should be from specific class; selected sets are added to [self.assist_current_exemplar_sets] in order.'''

        # set model to eval()-mode
        mode = self.training
        self.eval()

        n_max = len(dataset)
        exemplar_set = []

        if self.herding:
            # compute features for each example in [dataset]
            first_entry = True
            dataloader = utils.get_data_loader(dataset, self.batch_size, self.num_workers, is_shuffle=False,
                                               cuda=self._is_on_cuda())
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self._device())
                print("self.feature_extractor(image_batch)")
                with torch.no_grad():
                    feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            if self.norm_exemplars:
                features = F.normalize(features, p=2, dim=1)

            # calculate mean of all features
            class_mean = torch.mean(features, dim=0, keepdim=True)
            if self.norm_exemplars:
                class_mean = F.normalize(class_mean, p=2, dim=1)

            # one by one, select exemplar that makes mean of all exemplars as close to [class_mean] as possible
            exemplar_features = torch.zeros_like(features[:min(n, n_max)])
            list_of_selected = []
            for k in range(min(n, n_max)):
                if k > 0:
                    exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                    features_means = (features + exemplar_sum) / (k + 1)
                    features_dists = features_means - class_mean
                else:
                    features_dists = features - class_mean
                index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1))
                if index_selected in list_of_selected:
                    raise ValueError("Exemplars should not be repeated!!!!")
                list_of_selected.append(index_selected)

                exemplar_set.append(dataset[index_selected][0].numpy())
                exemplar_features[k] = copy.deepcopy(features[index_selected])

                # make sure this example won't be selected again
                features[index_selected] = features[index_selected] + 100000
        else:
            indeces_selected = np.random.choice(n_max, size=min(n, n_max), replace=False)
            for k in indeces_selected:
                exemplar_set.append(dataset[k][0].numpy())

        # add this [exemplar_set] as a [n]x[ich]x[isz]x[isz] to the list of [exemplar_sets]
        self.assist_current_exemplar_sets.append(np.array(exemplar_set))

        # set mode of model back
        self.train(mode=mode)

    ####----CLASSIFICATION----####

    def classify_with_exemplars(self, x, allowed_classes=None):
        """Classify images by nearest-means-of-exemplars (after transform to feature representation)

        INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch
                    allowed_classes = None or <list> containing all "active classes" between which should be chosen

        OUTPUT:     preds = <tensor> of size (bsz,)"""

        # Set model to eval()-mode
        mode = self.training
        self.eval()

        batch_size = x.size(0)

        # Do the exemplar-means need to be recomputed?
        if self.compute_means:
            exemplar_means = []  # --> list of 1D-tensors (of size [feature_size]), list is of length [n_classes]
            for P_y in self.exemplar_sets:
                exemplars = []
                # Collect all exemplars in P_y into a <tensor> and extract their features
                for ex in P_y:
                    exemplars.append(torch.from_numpy(ex))
                exemplars = torch.stack(exemplars).to(self._device())
                with torch.no_grad():
                    features = self.feature_extractor(exemplars)
                if self.norm_exemplars:
                    features = F.normalize(features, p=2, dim=1)
                # Calculate their mean and add to list
                mu_y = features.mean(dim=0, keepdim=True)
                if self.norm_exemplars:
                    mu_y = F.normalize(mu_y, p=2, dim=1)
                exemplar_means.append(mu_y.squeeze())  # -> squeeze removes all dimensions of size 1
            # Update model's attributes
            self.exemplar_means = exemplar_means
            self.compute_means = False

        # Reorganize the [exemplar_means]-<tensor>
        exemplar_means = self.exemplar_means if allowed_classes is None else [
            self.exemplar_means[i] for i in allowed_classes
        ]
        means = torch.stack(exemplar_means)  # (n_classes, feature_size)
        means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)

        # Extract features for input data (and reorganize)
        with torch.no_grad():
            feature = self.feature_extractor(x)  # (batch_size, feature_size)
        if self.norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

        # For each data-point in [x], find which exemplar-mean is closest to its extracted features
        dists = (feature - means).pow(2).sum(dim=1).squeeze()  # (batch_size, n_classes)
        _, preds = dists.min(1)

        # Set mode of model back
        self.train(mode=mode)

        return preds

    def current_task_validate_bias_Cls(self, val_datasets, task, active_classes, current_classes_num):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        val_loader = utils.get_data_loader(val_datasets[task - 1], self.batch_size, self.num_workers,
                                           cuda=True if self.availabel_cudas else False)
        mode = self.training
        # switch to evaluate mode
        self.eval()
        self.unbias_cls_of_FMcls.eval()
        with torch.no_grad():
            end = time.time()
            for inputs, labels in val_loader:
                data_time.update(time.time() - end)
                inputs, labels = inputs.to(self.MLP_device), labels.to(self.MLP_device)
                features = self.feature_extractor(inputs)
                y_hat = self.FM_cls_forward_nograd(features)
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    y_hat = y_hat[:, class_entries]
                y_hat = self.unbias_cls_of_FMcls(y_hat, current_classes_num)
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

    def current_task_validate_cRT_Cls(self, val_datasets, task, active_classes):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        val_loader = utils.get_data_loader(val_datasets[task - 1], self.batch_size, self.num_workers,
                                           cuda=True if self.availabel_cudas else False)
        mode = self.training
        # switch to evaluate mode
        self.eval()
        self.unbias_cls_of_FMcls.eval()
        with torch.no_grad():
            end = time.time()
            for inputs, labels in val_loader:
                data_time.update(time.time() - end)
                inputs, labels = inputs.to(self.MLP_device), labels.to(self.MLP_device)
                features = self.feature_extractor(inputs)
                y_hat = self.unbias_cls_of_FMcls(features)
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
        pass


class FeaturesHandler(nn.Module, metaclass=abc.ABCMeta):
    """Abstract  module for a classifier that can store and use exemplars.

    Adds a exemplar-methods to subclasses, and requires them to provide a 'feature-extractor' method."""

    # MLP_name, num_classes, hidden_size, Exemple_memory_budget,
    # Feature_memory_budget, norm_exemplars, herding, batch_size, sim_alpha, MLP_lr,
    # MLP_momentum, MLP_milestones, MLP_lrgamma, MLP_weight_decay, MLP_epochs, optim_type,
    # svm_sample_type, svm_max_iter, availabel_cudas, logger, batch_train_logger
    def __init__(self, MLP_name, num_classes, hidden_size, Exemple_memory_budget, num_workers,
                 Feature_memory_budget, norm_exemplars, herding, batch_size, sim_alpha, MLP_lr, MLP_momentum,
                 MLP_milestones, MLP_lrgamma, MLP_weight_decay, MLP_epochs, MLP_optim_type, MLP_KD_temp,
                 svm_sample_type, svm_max_iter,
                 availabel_cudas, logger, batch_train_logger):
        super().__init__()

        # list with exemplar-sets
        self.exemplar_feature_sets = []  # --> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_feature_means = []
        self.exemplar_sets = []  # --> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_means = []
        self.logger = logger
        self.batch_train_logger = batch_train_logger
        self.compute_means = True
        self.svm_train = True
        # settings
        self.num_workers = num_workers

        self.MLP_KD_temp = MLP_KD_temp

        self.MLP_name = MLP_name
        self.num_classes = num_classes
        self.feature_dim = hidden_size
        self.Exemple_memory_budget = Exemple_memory_budget
        self.Feature_memory_budget = Feature_memory_budget
        self.norm_exemplars = norm_exemplars
        self.herding = herding
        self.batch_size = batch_size
        self.alpha = sim_alpha
        self.MLP_lr = MLP_lr
        self.MLP_momentum = MLP_momentum
        self.MLP_milestones = MLP_milestones
        self.MLP_gamma = MLP_lrgamma
        self.MLP_weight_decay = MLP_weight_decay
        self.MLP_epochs = MLP_epochs
        self.optim_type = MLP_optim_type
        self.sample_type = svm_sample_type
        self.availabel_cudas = availabel_cudas
        self.pre_FM = None
        self.svm = None
        self.svm_max_iter = svm_max_iter
        self.MLP_device = "cuda" if self.availabel_cudas else "cpu"
        self.FM = self.construct_FM_model()
        self.classifer = nn.Linear(self.feature_dim, num_classes).to(self.MLP_device)

    def construct_FM_model(self):
        if self.availabel_cudas:
            # os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
            # device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
            FM_model = torch.nn.DataParallel(MLP_for_FM(MLP.__dict__[self.MLP_name](input_dim=self.feature_dim,
                                                                                    out_dim=self.feature_dim))).cuda()
            cudnn.benchmark = True
        else:
            FM_model = MLP_for_FM(MLP.__dict__[self.MLP_name](input_dim=self.feature_dim,
                                                              out_dim=self.feature_dim))
        return FM_model

    def build_FM_optimize(self):
        # Define optimizer (only include parameters that "requires_grad")
        optim_list = [{'params': filter(lambda p: p.requires_grad, self.FM.parameters()), 'lr': self.MLP_lr}]
        optimizer = None
        if self.optim_type in ("adam", "adam_reset"):
            if self.MLP_weight_decay:
                optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999), weight_decay=self.MLP_weight_decay)
            else:
                optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999))
        elif self.optim_type == "sgd":
            if self.MLP_momentum:
                optimizer = torch.optim.SGD(optim_list, momentum=self.MLP_momentum, weight_decay=self.MLP_weight_decay)
            else:
                optimizer = torch.optim.SGD(optim_list)
        else:
            raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(self.optim_type))

        return optimizer

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def feature_extractor(self, images):
        pass

    @abc.abstractmethod
    def get_preFE_feature(self, images):
        pass

    @abc.abstractmethod
    def get_cls_target(self, prefeatures):
        pass

    @abc.abstractmethod
    def get_cls_forward_target(self, features):
        pass

    @abc.abstractmethod
    def prefeatures_2_target(self, prefeatures):
        pass

    @abc.abstractmethod
    def get_precls_target(self, prefeatures):
        pass

    @abc.abstractmethod
    def prefeatures_2_precls_target(self, prefeatures):
        pass

    def get_FM_target(self, prefeatures):
        self.FM.eval()
        with torch.no_grad():
            return self.FM(prefeatures)[-1]

    def classifier_forward(self, features):
        self.classifer.eval()
        with torch.no_grad():
            return self.classifer(features)

    def feature_mapping(self, features):
        if type(self.FM) is torch.nn.DataParallel:
            return self.FM.module.get_mapping_features(prefeatures=features)
        else:
            return self.FM.get_mapping_features(prefeatures=features)

    def get_preFE_FM_features(self, imgs):
        return self.feature_mapping(self.get_preFE_feature(imgs))

    def linearSVM_train(self, pre_tasks_features, pre_tasks_targets, current_task_features,
                        current_task_target, task, sample_type="oversample", svm_store_path=None):
        # todo
        self.batch_train_logger.info(f'#############Task {task}  linearsvm train begin.##############')
        self.logger.info(f'############ task {task} linearsvm train start.##############')
        self.batch_train_logger.info(
            f'#############Task {task}  linearsvm train sample type is {sample_type}.##############')
        trainData = []
        trainLabels = []
        if len(pre_tasks_features) > 0:
            add_num = len(current_task_features[0]) - len(pre_tasks_features[0])
            assert add_num >= 0
            if sample_type == "oversample":
                for class_id in range(len(pre_tasks_targets)):
                    temp = pre_tasks_features[class_id]
                    np.random.shuffle(temp)
                    temp = list(temp)
                    temp += temp[:add_num]
                    trainData += temp
                    trainLabels += [pre_tasks_targets[class_id]] * len(temp)
                for class_id in range(len(current_task_target)):
                    temp = current_task_features[class_id]
                    temp = list(temp)
                    trainData += temp
                    trainLabels += [current_task_target[class_id]] * len(temp)
            elif sample_type == "undersample":
                for class_id in range(len(pre_tasks_targets)):
                    temp = pre_tasks_features[class_id]
                    temp = list(temp)
                    trainData += temp
                    trainLabels += [pre_tasks_targets[class_id]] * len(temp)
                for class_id in range(len(current_task_target)):
                    temp = current_task_features[class_id]
                    np.random.shuffle(temp)
                    temp = list(temp)
                    temp = temp[:-add_num]
                    trainData += temp
                    trainLabels += [current_task_target[class_id]] * len(temp)
            else:
                for class_id in range(len(pre_tasks_targets)):
                    temp = pre_tasks_features[class_id]
                    temp = list(temp)
                    trainData += temp
                    trainLabels += [pre_tasks_targets[class_id]] * len(temp)
                for class_id in range(len(current_task_target)):
                    temp = current_task_features[class_id]
                    temp = list(temp)
                    trainData += temp
                    trainLabels += [current_task_target[class_id]] * len(temp)
        else:
            for class_id in range(len(current_task_target)):
                temp = current_task_features[class_id]
                temp = list(temp)
                trainData += temp
                trainLabels += [current_task_target[class_id]] * len(temp)
        # randnum = random.randint(0, 100)
        # random.seed(randnum)
        # random.shuffle(trainData)
        # random.seed(randnum)
        # random.shuffle(trainLabels)
        train_X, test_X, train_y, test_y = train_test_split(np.array(trainData), np.array(trainLabels), test_size=0.1,
                                                            random_state=5)
        train_X, train_y = trainData, trainLabels
        self.svm = svm.SVC(kernel='linear', probability=True, class_weight='balanced', max_iter=self.svm_max_iter)
        self.svm.fit(train_X, train_y)
        svm_start_time = time.time()
        pred_score = self.svm.score(test_X, test_y)
        print('svm testing accuracy:')
        print(pred_score)
        if svm_store_path:
            print("save svm model...")
            joblib.dump(self.svm, svm_store_path)
        self.batch_train_logger.info(
            f'#############Task {task}  linearsvm train end. acc: {pred_score:.3f}. '
            f'svm train time: {time.time() - svm_start_time}##############')
        self.logger.info(f'############ Task {task}  linearsvm train end. acc: {pred_score:.3f}. '
                         f'svm train time: {time.time() - svm_start_time}##############')
        print('svm train End.')
        pass

    def linearSVM_train_featureSet(self, task, svm_store_path=None):
        # todo
        self.batch_train_logger.info(f'#############Task {task}  linearsvm train begin.##############')
        self.logger.info(f'############ task {task} linearSVM_train_featureSet start.##############')
        trainData = []
        trainLabels = []
        for class_id, exemplar_feature_set in enumerate(self.exemplar_feature_sets):
            trainLabels += [class_id] * len(exemplar_feature_set)
            for exemplar_feature in exemplar_feature_set:
                trainData.append(exemplar_feature)
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(trainData)
        random.seed(randnum)
        random.shuffle(trainLabels)
        train_X, test_X, train_y, test_y = train_test_split(np.array(trainData), np.array(trainLabels), test_size=0.1,
                                                            random_state=5)
        train_X, train_y = trainData, trainLabels
        self.svm = svm.SVC(kernel='linear', probability=True, class_weight='balanced', max_iter=self.svm_max_iter)
        self.svm.fit(train_X, train_y)
        svm_start_time = time.time()
        pred_score = self.svm.score(test_X, test_y)
        print('svm testing accuracy:')
        print(pred_score)
        if svm_store_path:
            print("save svm model...")
            joblib.dump(self.svm, svm_store_path)
        self.batch_train_logger.info(
            f'#############Task {task}  linearsvm train end. acc: {pred_score:.3f}. '
            f'svm train time: {time.time() - svm_start_time}##############')
        self.logger.info(f'############ Task {task}  linearsvm train end. acc: {pred_score:.3f}. '
                         f'svm train time: {time.time() - svm_start_time}##############')
        print('svm train End.')
        pass

    def cifar100_feature_handle_main(self, training_dataset, test_datasets, classes_per_task, active_classes, task):
        self.batch_train_logger.info(f'#############Examples & Features handler task {task} start.##############')
        self.logger.info(f'#############Examples & Features handler task {task} start.##############')
        print("Examples & Features handler task-%d start." % (task))
        pre_tasks_features = []
        pre_tasks_targets = []
        class_nums = classes_per_task * task
        feature_memory_noUse = self.Feature_memory_budget - (class_nums * 500)
        if feature_memory_noUse <= 0 or self.Exemple_memory_budget <= 0:
            features_per_class = int(np.floor(self.Feature_memory_budget / (classes_per_task * task)))
            examplars_per_class = int(np.floor(self.Exemple_memory_budget / (classes_per_task * task)))
        else:
            features_per_class = 500
            examplars_per_class = min(500, int(((feature_memory_noUse / 24) + self.Exemple_memory_budget) / class_nums))
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, self.Feature_memory_budget: "
            f"{self.Feature_memory_budget}, self.Exemple_memory_budget: {self.Exemple_memory_budget}, features_per_class:{features_per_class}, "
            f"examplars_per_class:{examplars_per_class} "
        )
        print("Examples & Features handler task-%d exemplar-feature sample start." % (task))
        if task > 1:
            self.feature_mapper_train(training_dataset, test_datasets, active_classes, task)
            for class_id, exemplar_feature_set in enumerate(self.exemplar_feature_sets):
                pre_tasks_targets.append(class_id)
                exemplar_features = []
                for exemplar_feature in exemplar_feature_set:
                    exemplar_features.append(torch.from_numpy(exemplar_feature))
                exemplar_features = torch.stack(exemplar_features).to(self.MLP_device)
                pre_tasks_features.append(self.feature_mapping(exemplar_features).cpu().numpy())
            self.update_features_sets(pre_tasks_features, features_per_class)
        current_task_features = []
        current_task_target = []
        # for each new class trained on, construct examplar-set
        new_classes = list(range(classes_per_task * (task - 1), classes_per_task * task))
        for class_id in new_classes:
            # create new dataset containing only all examples of this class
            class_dataset = SubDataset(original_dataset=training_dataset, sub_labels=[class_id])
            current_task_target.append(class_id)
            dataloader = utils.get_data_loader(class_dataset, self.batch_size, self.num_workers, is_shuffle=False,
                                               cuda=self._is_on_cuda())
            first_entry = True
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self.MLP_device)
                feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            # current_task_features.append(features.numpy())
            current_task_features.append(features)
            self.construct_exemplar_feature_set(class_dataset, features, examplars_per_class,
                                                features_per_class)

        self.reduce_exemplar_sets(examplars_per_class)
        current_task_features = torch.stack(current_task_features)
        current_task_features = current_task_features.numpy()
        # if task > 1:
        #     self.train_classifer(pre_tasks_features, current_task_features, test_datasets, active_classes, task)
        print("Examples & Features handler task-%d exemplar-feature sample END." % (task))
        # print("Examples & Features handler task-%d svm train begin." % (task))
        # # self.linearSVM_train(pre_tasks_features, pre_tasks_targets, current_task_features,
        # #                      current_task_target, task, self.sample_type)
        # self.linearSVM_train_featureSet(task)

    def imagenet_feature_handle_main(self, training_dataset, test_datasets, classes_per_task, active_classes, task):
        self.batch_train_logger.info(f'#############Examples & Features handler task {task} start.##############')
        self.logger.info(f'#############Examples & Features handler task {task} start.##############')
        print("Examples & Features handler task-%d start." % (task))
        pre_tasks_features = []
        pre_tasks_targets = []
        class_nums = classes_per_task * task
        feature_memory_noUse = self.Feature_memory_budget - (class_nums * 1300)
        if feature_memory_noUse <= 0 or self.Exemple_memory_budget <= 0:
            features_per_class = int(np.floor(self.Feature_memory_budget / (classes_per_task * task)))
            examplars_per_class = int(np.floor(self.Exemple_memory_budget / (classes_per_task * task)))
        else:
            features_per_class = 1300
            examplars_per_class = int(((feature_memory_noUse / 294) + self.Exemple_memory_budget) / class_nums)
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, self.Feature_memory_budget: "
            f"{self.Feature_memory_budget}, self.Exemple_memory_budget: {self.Exemple_memory_budget}, features_per_class:{features_per_class}, "
            f"examplars_per_class:{examplars_per_class} "
        )
        print("Examples & Features handler task-%d exemplar-feature sample start." % (task))
        if task > 1:
            self.feature_mapper_train(training_dataset, test_datasets, active_classes, task)
            for class_id, exemplar_feature_set in enumerate(self.exemplar_feature_sets):
                pre_tasks_targets.append(class_id)
                exemplar_features = []
                for exemplar_feature in exemplar_feature_set:
                    exemplar_features.append(torch.from_numpy(exemplar_feature))
                exemplar_features = torch.stack(exemplar_features).to(self.MLP_device)
                pre_tasks_features.append(self.feature_mapping(exemplar_features).cpu().numpy())
            self.update_features_sets(pre_tasks_features, features_per_class)
        current_task_features = []
        current_task_target = []
        # for each new class trained on, construct examplar-set
        new_classes = list(range(classes_per_task * (task - 1), classes_per_task * task))
        for class_id in new_classes:
            # create new dataset containing only all examples of this class
            class_dataset = SubDataset(original_dataset=training_dataset, sub_labels=[class_id])
            current_task_target.append(class_id)
            dataloader = utils.get_data_loader(class_dataset, self.batch_size, self.num_workers, is_shuffle=False,
                                               cuda=self._is_on_cuda())
            first_entry = True
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self.MLP_device)
                feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            # current_task_features.append(features.numpy())
            current_task_features.append(features)
            self.construct_exemplar_feature_set(class_dataset, features, examplars_per_class,
                                                features_per_class)

        self.reduce_exemplar_sets(examplars_per_class)
        current_task_features = torch.stack(current_task_features)
        current_task_features = current_task_features.numpy()
        # if task > 1:
        #     self.train_classifer(pre_tasks_features, current_task_features, test_datasets, active_classes, task)
        print("Examples & Features handler task-%d exemplar-feature sample END." % (task))
        print("Examples & Features handler task-%d svm train begin." % (task))
        self.linearSVM_train(pre_tasks_features, pre_tasks_targets, current_task_features,
                             current_task_target, task, self.sample_type)

    def train_classifer(self, pre_tasks_features, current_task_features, test_datasets, active_classes, task):
        feature_dataset = ExemplarDataset(pre_tasks_features)
        # tasks_feature = np.vstack((tasks_feature, current_task_features))
        current_feature_dataset = ExemplarDataset(current_task_features)
        feature_dataset = ConcatDataset([feature_dataset, current_feature_dataset])

        optim_list = [{'params': filter(lambda p: p.requires_grad, self.classifer.parameters()), 'lr': self.MLP_lr}]
        optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999), weight_decay=self.MLP_weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
        criterion = nn.CrossEntropyLoss()
        self.classifer.train()
        for epoch in range(self.MLP_epochs):
            train_loader = iter(utils.get_data_loader(feature_dataset, self.batch_size,
                                                      self.num_workers,
                                                      cuda=True if self.availabel_cudas else False))
            iters_left = len(train_loader)
            iter_index = 0

            while iters_left > 0:
                iters_left -= 1
                iter_index += 1
                optimizer.zero_grad()
                features, labels = next(train_loader)
                labels = labels.to(self.MLP_device)
                features = features.to(self.MLP_device)
                target_hat = self.classifer(features)
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    target_hat = target_hat[:, class_entries]

                # Calculate prediction loss
                # -binary prediction lo
                loss = criterion(target_hat, labels)
                loss.backward()
                optimizer.step()
                precision = None if labels is None else (labels == target_hat.max(1)[1]).sum().item() / features.size(0)
                ite_info = {
                    'task': task,
                    "train_state": "feature classifier",
                    "iter": iter_index,
                    'lr': scheduler.get_last_lr()[0],
                    'loss_total': loss.item(),
                    'precision': precision if precision is not None else 0.,
                }
                self.batch_train_logger.info(
                    f'Task {task} features classifier train  loss_total: {loss:.2f}  acc: {precision:.2f}.')
                self.batch_train_logger.info(f'...........................................................')
                # print("Task %d || Epoch %d || batchindex %d || info:" % (task, epoch, iter_index))
                print(ite_info)
                print("....................................")
            scheduler.step()
        acc1, acc5, throughput = self.current_task_validate_classifier(test_datasets, task, active_classes)
        self.batch_train_logger.info(
            f"feature classifier batch train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature classifer train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'features classifier : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))
        print("Train features classifier End.")
        pass

    def feature_mapper_train(self, training_dataset, test_datasets, active_classes, task):
        # todo Done
        self.batch_train_logger.info(f'####Task {task}  feature_mapper_train begin.####')
        self.logger.info(f'####Task {task}  feature_mapper_train begin.####')
        mode = self.training
        self.eval()
        self.FM.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = self.build_FM_optimize()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
        print("Task %d , train feature mapping..." % task)
        for epoch in range(self.MLP_epochs):
            train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
                                                      self.num_workers,
                                                      cuda=True if self.availabel_cudas else False))
            iters_left = len(train_loader)
            iter_index = 0

            while iters_left > 0:
                iters_left -= 1
                iter_index += 1
                optimizer.zero_grad()
                imgs, labels = next(train_loader)
                labels = labels.to(self.MLP_device)
                imgs = imgs.to(self.MLP_device)
                pre_features = self.get_preFE_feature(imgs)
                features = self.feature_extractor(imgs)
                feature_hat = self.FM(pre_features)
                scores = self.get_cls_forward_target(feature_hat)
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    scores = scores[:, class_entries]
                loss_sim = 1 - torch.cosine_similarity(features, feature_hat).mean()
                # binary_targets = utils.to_one_hot(labels.cpu(), scores.size(1)).to(self.MLP_device)
                # loss_cls = Func.binary_cross_entropy_with_logits(
                #     input=scores, target=binary_targets, reduction='none'
                # ).sum(dim=1).mean()  # --> sum over classes, then average over batch
                loss_cls = criterion(scores, labels)
                loss = self.alpha * loss_sim + loss_cls
                loss.backward()
                optimizer.step()
                precision = None if labels is None else (labels == scores.max(1)[1]).sum().item() / imgs.size(0)
                ite_info = {
                    'task': task,
                    'lr': scheduler.get_last_lr()[0],
                    'loss_total': loss.item(),
                    'precision': precision if precision is not None else 0.,
                }
                self.batch_train_logger.info(f'Task {task}  loss_total: {loss:.2f}  acc: {precision:.2f}.')
                self.batch_train_logger.info(f'...........................................................')
                # print("Task %d || Epoch %d || batchindex %d || info:" % (task, epoch, iter_index))
                print(ite_info)
                print("....................................")
            scheduler.step()
        acc1, acc5, throughput = self.current_task_validate_FM(test_datasets, task, active_classes)
        self.batch_train_logger.info(
            f"feature mapping batch train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature mapping train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapping batch train task : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))
        print("Train feature mapping End.")
        self.pre_FM = copy.deepcopy(self.FM).eval()
        self.train(mode=mode)
        pass

    ####----MANAGING EXEMPLAR SETS----####

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]

    def update_features_sets(self, pre_tasks_features, features_per_class):
        # todo done!
        for index, feature_set in enumerate(pre_tasks_features):
            self.exemplar_feature_sets[index] = feature_set[: features_per_class]
        pass

    def construct_exemplar_feature_set(self, class_dataset, current_class_features, Exemplar_per_class,
                                       features_per_class):
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

        Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''
        # todo Done
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

                Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''
        if Exemplar_per_class > features_per_class:
            raise ValueError("imgs_per_class must not surpass features_per_class.")
        n_max = len(class_dataset)
        exemplar_set = []
        # features = torch.Tensor(copy.deepcopy(current_class_features))
        features = copy.deepcopy(current_class_features)
        # current_class_features = torch.Tensor(current_class_features)
        exemplar_features = torch.zeros_like(features[:min(features_per_class, n_max)])
        if self.herding:

            # calculate mean of all features
            class_mean = torch.mean(current_class_features, dim=0, keepdim=True)
            # one by one, select exemplar that makes mean of all exemplars as close to [class_mean] as possible
            list_of_selected = []
            for k in range(min(features_per_class, n_max)):
                if k > 0:
                    exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                    features_means = (features + exemplar_sum) / (k + 1)
                    features_dists = features_means - class_mean
                else:
                    features_dists = features - class_mean
                index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1))
                if index_selected in list_of_selected:
                    raise ValueError("Exemplars should not be repeated!!!!")
                list_of_selected.append(index_selected)
                if len(exemplar_set) < Exemplar_per_class:
                    exemplar_set.append(class_dataset[index_selected][0].numpy())
                exemplar_features[k] = copy.deepcopy(features[index_selected])

                # make sure this example won't be selected again
                features[index_selected] = features[index_selected] + 100000
        else:
            indeces_selected = np.random.choice(n_max, size=min(features_per_class, n_max), replace=False)
            for k in indeces_selected:
                if len(exemplar_set) < Exemplar_per_class:
                    exemplar_set.append(class_dataset[k][0].numpy())
                exemplar_features[k] = copy.deepcopy(features[k])

        # add this [exemplar_set] as a [n]x[ich]x[isz]x[isz] to the list of [exemplar_sets]
        self.exemplar_sets.append(np.array(exemplar_set))
        self.exemplar_feature_sets.append(exemplar_features.numpy())
        pass

    def classify_with_exemplars(self, x, allowed_classes=None):
        """Classify images by nearest-means-of-exemplars (after transform to feature representation)

        INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch
                    allowed_classes = None or <list> containing all "active classes" between which should be chosen

        OUTPUT:     preds = <tensor> of size (bsz,)"""

        # Set model to eval()-mode
        mode = self.training
        self.eval()

        batch_size = x.size(0)

        # Do the exemplar-means need to be recomputed?
        if self.compute_means:
            exemplar_means = []  # --> list of 1D-tensors (of size [feature_size]), list is of length [n_classes]
            for P_y in self.exemplar_sets:
                exemplars = []
                # Collect all exemplars in P_y into a <tensor> and extract their features
                for ex in P_y:
                    exemplars.append(torch.from_numpy(ex))
                exemplars = torch.stack(exemplars).to(self._device())
                with torch.no_grad():
                    features = self.feature_extractor(exemplars)
                if self.norm_exemplars:
                    features = F.normalize(features, p=2, dim=1)
                # Calculate their mean and add to list
                mu_y = features.mean(dim=0, keepdim=True)
                if self.norm_exemplars:
                    mu_y = F.normalize(mu_y, p=2, dim=1)
                exemplar_means.append(mu_y.squeeze())  # -> squeeze removes all dimensions of size 1
            # Update model's attributes
            self.exemplar_means = exemplar_means
            self.compute_means = False

        # Reorganize the [exemplar_means]-<tensor>
        exemplar_means = self.exemplar_means if allowed_classes is None else [
            self.exemplar_means[i] for i in allowed_classes
        ]
        means = torch.stack(exemplar_means)  # (n_classes, feature_size)
        means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)

        # Extract features for input data (and reorganize)
        with torch.no_grad():
            feature = self.feature_extractor(x)  # (batch_size, feature_size)
        if self.norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

        # For each data-point in [x], find which exemplar-mean is closest to its extracted features
        dists = (feature - means).pow(2).sum(dim=1).squeeze()  # (batch_size, n_classes)
        _, preds = dists.min(1)

        # Set mode of model back
        self.train(mode=mode)

        return preds

    ####----CLASSIFICATION----####

    def classify_with_features(self, x, use_FM=False, allowed_classes=None):
        """Classify images by nearest-means-of-exemplars (after transform to feature representation)

        INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch
                    allowed_classes = None or <list> containing all "active classes" between which should be chosen

        OUTPUT:     preds = <tensor> of size (bsz,)"""
        # todo
        # Set model to eval()-mode
        mode = self.training
        self.eval()

        batch_size = x.size(0)

        # Do the exemplar-means need to be recomputed?
        print("self.exemplar_feature_sets:", len(self.exemplar_feature_sets))
        print("self.exemplar_feature_sets len:", len(self.exemplar_feature_sets[0]))
        if self.compute_means:
            exemplar_means = []  # --> list of 1D-tensors (of size [feature_size]), list is of length [n_classes]
            for ef in self.exemplar_feature_sets:
                exemplar_features = torch.from_numpy(ef).to(self.MLP_device)
                if self.norm_exemplars:
                    exemplar_features = F.normalize(exemplar_features, p=2, dim=1)
                # Calculate their mean and add to list
                mu_y = exemplar_features.mean(dim=0, keepdim=True)
                if self.norm_exemplars:
                    mu_y = F.normalize(mu_y, p=2, dim=1)
                exemplar_means.append(mu_y.squeeze())  # -> squeeze removes all dimensions of size 1
            # Update model's attributes
            self.exemplar_means = exemplar_means
            self.compute_means = False

        # Reorganize the [exemplar_means]-<tensor>
        exemplar_means = self.exemplar_means if allowed_classes is None else [
            self.exemplar_means[i] for i in allowed_classes
        ]
        means = torch.stack(exemplar_means)  # (n_classes, feature_size)
        means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)

        # Extract features for input data (and reorganize)
        with torch.no_grad():
            if use_FM and self.pre_FM is not None:
                feature = self.get_preFE_FM_features(x)
            else:
                feature = self.feature_extractor(x)  # (batch_size, feature_size)
            # if self.pre_FM is not None:
            #     feature = self.feature_mapping(self.get_preFE_feature(x))
            # else:
            #     feature = self.feature_extractor(x)  # (batch_size, feature_size)
        if self.norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

        # For each data-point in [x], find which exemplar-mean is closest to its extracted features
        dists = (feature - means).pow(2).sum(dim=1).squeeze()  # (batch_size, n_classes)
        _, preds = dists.min(1)

        # Set mode of model back
        self.train(mode=mode)

        return preds
        pass

    def ILtFA_classify(self, x, classifier="linearSVM", active_classes=None, use_FM=False, allowed_classes=None):
        if classifier == "examplar_ncm":
            return self.classify_with_exemplars(x, allowed_classes)
        elif classifier == "feature_ncm":
            return self.classify_with_features(x, use_FM=use_FM)
        elif classifier == "linearSVM":
            # if self.pre_FM is not None:
            #     features = self.feature_mapping(self.get_preFE_feature(x))
            # else:
            #     features = self.feature_extractor(x)  # (batch_size, feature_size)
            features = self.feature_extractor(x)
            return torch.from_numpy(self.svm.predict(features.cpu().numpy())).to(self.MLP_device)
        elif classifier == "fc":
            # if self.pre_FM:
            #     scores = self.get_FM_target(self.get_preFE_feature(x))
            # else:
            scores = self.get_cls_target(self.feature_extractor(x))
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                scores = scores[:, class_entries]
            _, predicted = torch.max(scores, 1)
            return predicted
        elif classifier == "fcls":
            if self.pre_FM:
                scores = self.classifier_forward(self.feature_extractor(x))
            else:
                scores = self.get_cls_target(self.feature_extractor(x))
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                scores = scores[:, class_entries]
            _, predicted = torch.max(scores, 1)
            return predicted
        else:
            raise ValueError("classifier must be ncm or linearSVM.")

    def current_task_validate_FM(self, test_datasets, task, active_classes):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        val_loader = utils.get_data_loader(test_datasets[task - 1], self.batch_size,  # task index must minus 1
                                           self.num_workers,
                                           cuda=True if self.availabel_cudas else False)
        mode = self.training
        # switch to evaluate mode
        self.eval()
        self.FM.eval()
        with torch.no_grad():
            end = time.time()
            for inputs, labels in val_loader:
                data_time.update(time.time() - end)
                inputs, labels = inputs.to(self.MLP_device), labels.to(self.MLP_device)
                feature_hat = self.feature_mapping(self.get_preFE_feature(inputs))
                y_hat = self.get_cls_target(feature_hat)
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

    def current_task_validate_classifier(self, test_datasets, task, active_classes):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        val_loader = utils.get_data_loader(test_datasets[task - 1], self.batch_size,  # task index must minus 1
                                           self.num_workers,
                                           cuda=True if self.availabel_cudas else False)
        mode = self.training
        # switch to evaluate mode
        self.eval()
        self.FM.eval()
        self.classifer.eval()
        with torch.no_grad():
            end = time.time()
            for inputs, labels in val_loader:
                data_time.update(time.time() - end)
                inputs, labels = inputs.to(self.MLP_device), labels.to(self.MLP_device)
                mapped_feature = self.feature_mapping(self.get_preFE_feature(inputs))
                y_hat = self.classifer(mapped_feature)
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


class EFAfIL_FeaturesHandler(nn.Module, metaclass=abc.ABCMeta):
    """Abstract  module for a classifier that can store and use exemplars.

    Adds a exemplar-methods to subclasses, and requires them to provide a 'feature-extractor' method."""

    def __init__(self, MLP_name, num_classes, hidden_size, Exemple_memory_budget, num_workers,
                 Feature_memory_budget, norm_exemplars, herding, batch_size, sim_alpha, MLP_lr, MLP_rate,
                 MLP_momentum, MLP_milestones, MLP_lrgamma, MLP_weight_decay, MLP_epochs, MLP_optim_type,
                 MLP_KD_temp, MLP_KD_temp_2, MLP_distill_rate, svm_sample_type, svm_max_iter, availabel_cudas, logger,
                 batch_train_logger):
        super().__init__()

        # list with exemplar-sets
        self.exemplar_feature_sets = []  # --> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_feature_means = []
        self.exemplar_sets = []  # --> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_means = []
        self.logger = logger
        self.batch_train_logger = batch_train_logger
        self.compute_means = True
        self.svm_train = True
        # settings
        self.num_workers = num_workers

        self.MLP_rate = MLP_rate
        self.MLP_KD_temp = MLP_KD_temp
        self.MLP_KD_temp_2 = MLP_KD_temp_2
        self.MLP_distill_rate = MLP_distill_rate

        self.MLP_name = MLP_name
        self.num_classes = num_classes
        self.feature_dim = hidden_size
        self.Exemple_memory_budget = Exemple_memory_budget
        self.Feature_memory_budget = Feature_memory_budget
        self.norm_exemplars = norm_exemplars
        self.herding = herding
        self.batch_size = batch_size
        self.alpha = sim_alpha
        self.MLP_lr = MLP_lr
        self.MLP_momentum = MLP_momentum
        self.MLP_milestones = MLP_milestones
        self.MLP_gamma = MLP_lrgamma
        self.MLP_weight_decay = MLP_weight_decay
        self.MLP_epochs = MLP_epochs
        self.MLP_optim_type = MLP_optim_type
        self.sample_type = svm_sample_type
        self.availabel_cudas = availabel_cudas
        self.pre_FM_cls_domain = None
        self.svm = None
        self.svm_max_iter = svm_max_iter
        self.MLP_device = "cuda" if self.availabel_cudas else "cpu"
        self.FM_cls_domain = self.construct_FM_cls_domain_model()
        self.unbias_cls_of_FMcls = None
        # self.classifer = nn.Linear(self.feature_dim, num_classes).to(self.MLP_device)

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def get_FE_cls_output(self, images):
        pass

    @abc.abstractmethod
    def feature_extractor(self, images):
        pass

    @abc.abstractmethod
    def current_task_model_feature_extractor(self, images):
        pass

    @abc.abstractmethod
    def get_preFE_feature(self, images):
        pass

    @abc.abstractmethod
    def get_cls_target(self, features):
        pass

    @abc.abstractmethod
    def get_cls_forward_target(self, features):
        pass

    @abc.abstractmethod
    def get_FE_cls_target(self, x):
        pass

    def pre_FM_mapping(self, features):
        if type(self.pre_FM_cls_domain) is torch.nn.DataParallel:
            return self.pre_FM_cls_domain.module.get_mapping_features(features)
        else:
            return self.pre_FM_cls_domain.get_mapping_features(features)

    # def get_prefeatures_results(self, prefeatures):
    #     self.FM_cls_domain.eval()
    #     with torch.no_grad:
    #         return self.FM_cls_domain(prefeatures)

    def feature_mapping(self, features):
        if type(self.FM_cls_domain) is torch.nn.DataParallel:
            return self.FM_cls_domain.module.get_mapping_features(features)
        else:
            return self.FM_cls_domain.get_mapping_features(features)

    def prefeature_2_FMtarget(self, FMfeatures):
        self.FM_cls_domain.eval()
        with torch.no_grad():
            if type(self.FM_cls_domain) is torch.nn.DataParallel:
                return self.FM_cls_domain.module.cls(FMfeatures)
            else:
                return self.FM_cls_domain.cls(FMfeatures)

    def get_preFE_FM_features(self, imgs):
        return self.feature_mapping(self.get_preFE_feature(imgs))

    def FM_cls_forward_nograd(self, features):
        self.FM_cls_domain.eval()
        with torch.no_grad():
            if type(self.FM_cls_domain) is torch.nn.DataParallel:
                return self.FM_cls_domain.module.cls(features)
            else:
                return self.FM_cls_domain.cls(features)

    def construct_FM_cls_domain_model(self):
        if self.availabel_cudas:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
            device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
            FM_cls_domain_model = torch.nn.DataParallel(
                MLP_cls_domain_dis(MLP.__dict__[self.MLP_name](input_dim=self.feature_dim,
                                                               out_dim=self.feature_dim,
                                                               rate=self.MLP_rate),
                                   self.feature_dim, self.num_classes), device_ids=device_ids).cuda()
            cudnn.benchmark = True
        else:
            FM_cls_domain_model = MLP_cls_domain_dis(MLP.__dict__[self.MLP_name](input_dim=self.feature_dim,
                                                                                 out_dim=self.feature_dim,
                                                                                 rate=self.MLP_rate),
                                                     self.feature_dim, self.num_classes)
        return FM_cls_domain_model

    def build_FM_cls_domain_optimize(self):
        # Define optimizer (only include parameters that "requires_grad")
        optim_list = [{'params': filter(lambda p: p.requires_grad, self.FM_cls_domain.parameters()), 'lr': self.MLP_lr}]
        optimizer = None
        if self.MLP_optim_type in ("adam", "adam_reset"):
            if self.MLP_weight_decay:
                optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999), weight_decay=self.MLP_weight_decay)
            else:
                optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999))
        elif self.MLP_optim_type == "sgd":
            if self.MLP_momentum:
                optimizer = torch.optim.SGD(optim_list, momentum=self.MLP_momentum, weight_decay=self.MLP_weight_decay)
            else:
                optimizer = torch.optim.SGD(optim_list)
        else:
            raise ValueError(
                "Unrecognized optimizer, '{}' is not currently a valid option".format(self.MLP_optim_type))

        return optimizer

    def linearSVM_train(self, pre_tasks_features, pre_tasks_targets, current_task_features,
                        current_task_target, task, sample_type="oversample", svm_store_path=None):
        # todo
        self.batch_train_logger.info(f'#############Task {task}  linearsvm train begin.##############')
        self.logger.info(f'############ task {task} linearsvm train start.##############')
        self.batch_train_logger.info(
            f'#############Task {task}  linearsvm train sample type is {sample_type}.##############')
        trainData = []
        trainLabels = []
        if len(pre_tasks_features) > 0:
            add_num = len(current_task_features[0]) - len(pre_tasks_features[0])
            if sample_type == "oversample":
                for class_id in range(len(pre_tasks_targets)):
                    pre_ft = torch.from_numpy(pre_tasks_features[class_id])
                    if self.norm_exemplars:
                        temp = F.normalize(pre_ft, p=2, dim=1)
                    else:
                        temp = pre_ft
                    temp = temp.numpy()
                    np.random.shuffle(temp)
                    temp = list(temp)
                    temp += temp[:add_num]
                    trainData += temp
                    trainLabels += [pre_tasks_targets[class_id]] * len(temp)
                for class_id in range(len(current_task_target)):
                    current_ft = torch.from_numpy(current_task_features[class_id])
                    if self.norm_exemplars:
                        temp = F.normalize(current_ft, p=2, dim=1)
                    else:
                        temp = current_ft
                    temp = temp.numpy()
                    temp = list(temp)
                    trainData += temp
                    trainLabels += [current_task_target[class_id]] * len(temp)
            elif sample_type == "undersample":
                for class_id in range(len(pre_tasks_targets)):
                    pre_ft = torch.from_numpy(pre_tasks_features[class_id])
                    if self.norm_exemplars:
                        temp = F.normalize(pre_ft, p=2, dim=1)
                    else:
                        temp = pre_ft
                    temp = temp.numpy()
                    temp = list(temp)
                    trainData += temp
                    trainLabels += [pre_tasks_targets[class_id]] * len(temp)
                for class_id in range(len(current_task_target)):
                    current_ft = torch.from_numpy(current_task_features[class_id])
                    if self.norm_exemplars:
                        temp = F.normalize(current_ft, p=2, dim=1)
                    else:
                        temp = current_ft
                    temp = temp.numpy()
                    np.random.shuffle(temp)
                    temp = list(temp)
                    temp = temp[:-add_num]
                    trainData += temp
                    trainLabels += [current_task_target[class_id]] * len(temp)
            else:
                for class_id in range(len(pre_tasks_targets)):
                    pre_ft = torch.from_numpy(pre_tasks_features[class_id])
                    if self.norm_exemplars:
                        temp = F.normalize(pre_ft, p=2, dim=1)
                    else:
                        temp = pre_ft
                    temp = temp.numpy()
                    temp = list(temp)
                    trainData += temp
                    trainLabels += [pre_tasks_targets[class_id]] * len(temp)
                for class_id in range(len(current_task_target)):
                    current_ft = torch.from_numpy(current_task_features[class_id])
                    if self.norm_exemplars:
                        temp = F.normalize(current_ft, p=2, dim=1)
                    else:
                        temp = current_ft
                    temp = temp.numpy()
                    temp = list(temp)
                    trainData += temp
                    trainLabels += [current_task_target[class_id]] * len(temp)
        else:
            for class_id in range(len(current_task_target)):
                current_ft = torch.from_numpy(current_task_features[class_id])
                if self.norm_exemplars:
                    temp = F.normalize(current_ft, p=2, dim=1)
                else:
                    temp = current_ft
                temp = temp.numpy()
                temp = list(temp)
                trainData += temp
                trainLabels += [current_task_target[class_id]] * len(temp)
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(trainData)
        random.seed(randnum)
        random.shuffle(trainLabels)
        train_X, test_X, train_y, test_y = train_test_split(np.array(trainData), np.array(trainLabels), test_size=0.1,
                                                            random_state=5)
        self.svm = svm.SVC(kernel='linear', probability=True, class_weight='balanced', max_iter=self.svm_max_iter)
        self.svm.fit(train_X, train_y)
        svm_start_time = time.time()
        pred_score = self.svm.score(test_X, test_y)
        print('svm testing accuracy:')
        print(pred_score)
        if svm_store_path:
            print("save svm model...")
            joblib.dump(self.svm, svm_store_path)
        self.batch_train_logger.info(
            f'#############Task {task}  linearsvm train end. acc: {pred_score:.3f}. '
            f'svm train time: {time.time() - svm_start_time}##############')
        self.logger.info(f'############ Task {task}  linearsvm train end. acc: {pred_score:.3f}. '
                         f'svm train time: {time.time() - svm_start_time}##############')
        print('svm train End.')
        pass

    def linearSVM_train_featureSet(self, task, svm_store_path=None):
        # todo
        self.batch_train_logger.info(f'#############Task {task}  linearsvm train begin.##############')
        self.logger.info(f'############ task {task} linearSVM_train_featureSet start.##############')
        trainData = []
        trainLabels = []
        for class_id, exemplar_feature_set in enumerate(self.exemplar_feature_sets):
            trainLabels += [class_id] * len(exemplar_feature_set)
            for exemplar_feature in exemplar_feature_set:
                trainData.append(exemplar_feature)
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(trainData)
        random.seed(randnum)
        random.shuffle(trainLabels)
        train_X, test_X, train_y, test_y = train_test_split(np.array(trainData), np.array(trainLabels), test_size=0.1,
                                                            random_state=5)
        train_X, train_y = trainData, trainLabels
        self.svm = svm.SVC(kernel='linear', probability=True, class_weight='balanced', max_iter=self.svm_max_iter)
        self.svm.fit(train_X, train_y)
        svm_start_time = time.time()
        pred_score = self.svm.score(test_X, test_y)
        print('svm testing accuracy:')
        print(pred_score)
        if svm_store_path:
            print("save svm model...")
            joblib.dump(self.svm, svm_store_path)
        self.batch_train_logger.info(
            f'#############Task {task}  linearsvm train end. acc: {pred_score:.3f}. '
            f'svm train time: {time.time() - svm_start_time}##############')
        self.logger.info(f'############ Task {task}  linearsvm train end. acc: {pred_score:.3f}. '
                         f'svm train time: {time.time() - svm_start_time}##############')
        print('svm train End.')
        pass

    def feature_mapper_train(self, training_dataset, test_datasets, active_classes, task, use_NewfeatureSpace=False):
        # todo Done
        self.batch_train_logger.info(f'####Task {task}  feature_mapper_train begin.####')
        self.logger.info(f'####Task {task}  feature_mapper_train begin.####')
        mode = self.training
        self.eval()
        self.FM_cls_domain.train()
        criterion = nn.CrossEntropyLoss()
        MSEloss_func = torch.nn.MSELoss(reduction='mean')
        optimizer = self.build_FM_cls_domain_optimize()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
        print("Task %d , train feature mapping..." % task)
        for epoch in range(self.MLP_epochs):
            train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
                                                      self.num_workers,
                                                      cuda=True if self.availabel_cudas else False))
            iters_left = len(train_loader)
            iter_index = 0
            while iters_left > 0:
                iters_left -= 1
                iter_index += 1
                optimizer.zero_grad()
                imgs, labels = next(train_loader)
                labels = labels.to(self.MLP_device)
                imgs = imgs.to(self.MLP_device)
                pre_features = self.get_preFE_feature(imgs)
                if use_NewfeatureSpace:
                    '''map pre_features and features to new space'''
                    pre_features = self.pre_FM_mapping(pre_features)
                features = self.feature_extractor(imgs)

                feature_hat, target_hat, _ = self.FM_cls_domain(pre_features)
                scores = self.get_cls_forward_target(feature_hat)
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    scores = scores[:, class_entries]
                loss_sim = 1 - torch.cosine_similarity(features, feature_hat).mean()
                # loss_sim = MSEloss_func(features, feature_hat)
                loss_cls = criterion(scores, labels)
                loss = self.alpha * loss_sim + loss_cls
                loss.backward()
                optimizer.step()
                precision = None if labels is None else (labels == scores.max(1)[1]).sum().item() / imgs.size(0)
                ite_info = {
                    'task': task,
                    'lr': scheduler.get_last_lr()[0],
                    'loss_total': loss.item(),
                    'precision': precision if precision is not None else 0.,
                }
                self.batch_train_logger.info(f'Task {task}  loss_total: {loss:.2f}  acc: {precision:.2f}.')
                self.batch_train_logger.info(f'...........................................................')
                # print("Task %d || Epoch %d || batchindex %d || info:" % (task, epoch, iter_index))
                print(ite_info)
                print("....................................")
            scheduler.step()
        acc1, acc5, throughput = self.current_task_validate_FM_Retrain(test_datasets, task, active_classes,
                                                                       use_NewfeatureSpace)
        self.batch_train_logger.info(
            f"feature mapper train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature mapper train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapper train task : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))
        print("Train feature mapper End.")
        self.pre_FM_cls_domain = copy.deepcopy(self.FM_cls_domain).eval()
        self.train(mode=mode)
        pass

    def feature_mapper_train_oversample(self, train_dataset, exemplar_dataset, test_datasets, active_classes, task,
                                        use_NewfeatureSpace=False):
        # todo Done
        self.batch_train_logger.info(f'####Task {task}  feature_mapper_train begin.####')
        self.logger.info(f'####Task {task}  feature_mapper_train begin.####')
        mode = self.training
        self.eval()
        self.FM_cls_domain.train()
        criterion = nn.CrossEntropyLoss()
        MSEloss_func = torch.nn.MSELoss(reduction='mean')
        optimizer = self.build_FM_cls_domain_optimize()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
        print("Task %d , train feature mapping..." % task)
        train_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                  self.num_workers,
                                                  cuda=True if self.availabel_cudas else False))
        exemplar_train_loader = iter(utils.get_data_loader(exemplar_dataset,
                                                           self.batch_size,
                                                           self.num_workers,
                                                           cuda=True if self.availabel_cudas else False))
        train_imgs_num = len(train_loader)
        exemplar_num = len(exemplar_train_loader)
        exemplar_train_iter_index = 0
        train_dataset_iter_index = 0
        for epoch in range(self.MLP_epochs):
            if train_dataset_iter_index == train_imgs_num:
                train_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                          self.num_workers,
                                                          cuda=True if self.availabel_cudas else False))
                train_dataset_iter_index = 0
            if exemplar_train_iter_index == exemplar_num:
                exemplar_train_loader = iter(utils.get_data_loader(exemplar_dataset,
                                                                   self.batch_size,
                                                                   self.num_workers,
                                                                   cuda=True if self.availabel_cudas else False))
                exemplar_train_iter_index = 0
            iters_left = max(train_imgs_num, exemplar_num)
            iter_index = 0
            while iters_left > 0:
                iters_left -= 1
                iter_index += 1
                optimizer.zero_grad()
                if train_dataset_iter_index == train_imgs_num:
                    train_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                              self.num_workers,
                                                              cuda=True if self.availabel_cudas else False))
                    train_dataset_iter_index = 0
                if exemplar_train_iter_index == exemplar_num:
                    exemplar_train_loader = iter(utils.get_data_loader(exemplar_dataset,
                                                                       self.batch_size,
                                                                       self.num_workers,
                                                                       cuda=True if self.availabel_cudas else False))
                    exemplar_train_iter_index = 0
                imgs, labels = next(train_loader)
                labels = labels.to(self.MLP_device)
                imgs = imgs.to(self.MLP_device)

                exemplar_imgs, exemplar_labels = next(exemplar_train_loader)
                exemplar_labels = exemplar_labels.to(self.MLP_device)
                exemplar_imgs = exemplar_imgs.to(self.MLP_device)

                exemplar_train_iter_index += 1
                train_dataset_iter_index += 1
                '''获取旧的features'''
                pre_features = self.get_preFE_feature(imgs)
                exemplar_pre_features = self.get_preFE_feature(exemplar_imgs)

                '''获取FM_mapping后的旧的features'''
                if use_NewfeatureSpace:
                    '''map pre_features and features to new space'''
                    pre_features = self.pre_FM_mapping(pre_features)
                    exemplar_pre_features = self.pre_FM_mapping(exemplar_pre_features)

                '''获取新的features'''
                features = self.feature_extractor(imgs)
                exemplar_features = self.feature_extractor(exemplar_imgs)
                '''用FM_cls_domain来训练，先获取output'''
                feature_hat, target_hat, _ = self.FM_cls_domain(pre_features)
                exemplar_feature_hat, exemplar_target_hat, _ = self.FM_cls_domain(exemplar_pre_features)
                '''获取在FE_cls的cls的输出，必须有计算梯度'''
                scores = self.get_cls_forward_target(feature_hat)
                exemplar_scores = self.get_cls_forward_target(exemplar_feature_hat)
                assert scores is not None
                assert active_classes is not None
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    scores = scores[:, class_entries]
                    exemplar_scores = exemplar_scores[:, class_entries]

                loss_sim_imgs = 1 - torch.cosine_similarity(features, feature_hat).mean()
                loss_sim_exemplar_imgs = 1 - torch.cosine_similarity(exemplar_features, exemplar_feature_hat).mean()
                loss_sim = loss_sim_imgs + loss_sim_exemplar_imgs
                # loss_sim = MSEloss_func(features, feature_hat) + MSEloss_func(exemplar_features, exemplar_feature_hat)
                loss_cls = criterion(scores, labels) + criterion(exemplar_scores, exemplar_labels)
                loss = self.alpha * loss_sim + loss_cls
                loss.backward()
                optimizer.step()
                precision = None if labels is None else (labels == scores.max(1)[1]).sum().item() / imgs.size(0)
                ite_info = {
                    'task': task,
                    'lr': scheduler.get_last_lr()[0],
                    'loss_total': loss.item(),
                    'precision': precision if precision is not None else 0.,
                }
                self.batch_train_logger.info(f'Task {task}  loss_total: {loss:.2f}  acc: {precision:.2f}.')
                self.batch_train_logger.info(f'...........................................................')
                # print("Task %d || Epoch %d || batchindex %d || info:" % (task, epoch, iter_index))
                print(ite_info)
                print("....................................")
            scheduler.step()
        acc1, acc5, throughput = self.current_task_validate_FM_Retrain(test_datasets, task, active_classes,
                                                                       use_NewfeatureSpace)
        self.batch_train_logger.info(
            f"feature mapper train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature mapper train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapper train task : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))
        print("-----------------------------------")

        acc1, acc5, throughput = self.current_task_validate_FM_Retrain(test_datasets, task, active_classes,
                                                                       use_NewfeatureSpace, val_current_task=False)
        self.batch_train_logger.info(
            f"feature mapper train || old classes: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature mapper train || old classes: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapper train  old classes : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))

        print("Train feature mapper End.")
        self.pre_FM_cls_domain = copy.deepcopy(self.FM_cls_domain).eval()
        self.train(mode=mode)
        pass

    def cifar100_feature_handle_main(self, train_dataset, classes_per_task, task, use_FM=False):
        pre_tasks_features = []
        pre_tasks_targets = []
        class_nums = classes_per_task * task
        feature_memory_noUse = self.Feature_memory_budget - (class_nums * 500)
        if feature_memory_noUse <= 0 or self.Exemple_memory_budget <= 0:
            features_per_class = int(np.floor(self.Feature_memory_budget / (classes_per_task * task)))
            examplars_per_class = int(np.floor(self.Exemple_memory_budget / (classes_per_task * task)))
        else:
            features_per_class = 500
            examplars_per_class = min(500, int(((feature_memory_noUse / 24) + self.Exemple_memory_budget) / class_nums))
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, self.Feature_memory_budget: "
            f"{self.Feature_memory_budget}, self.Exemple_memory_budget: {self.Exemple_memory_budget}, features_per_class:{features_per_class}, "
            f"examplars_per_class:{examplars_per_class} "
        )
        if task > 1:
            for class_id, exemplar_feature_set in enumerate(self.exemplar_feature_sets):
                pre_tasks_targets.append(class_id)
                exemplars = []
                for exemplar_feature in exemplar_feature_set:
                    exemplars.append(torch.from_numpy(exemplar_feature))
                exemplars = torch.stack(exemplars).to(self._device())
                pre_tasks_features.append(self.feature_mapping(exemplars).cpu().numpy())
            self.update_features_sets(pre_tasks_features, features_per_class)
            self.reduce_exemplar_sets(examplars_per_class)
        current_task_features = []
        current_task_target = []
        # for each new class trained on, construct examplar-set
        new_classes = list(range(classes_per_task * (task - 1), classes_per_task * task))
        for class_id in new_classes:
            # create new dataset containing only all examples of this class
            class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
            current_task_target.append(class_id)
            dataloader = utils.get_data_loader(class_dataset, self.batch_size, self.num_workers, is_shuffle=False,
                                               cuda=self._is_on_cuda())
            first_entry = True
            print("construct_exemplar_feature_set class_id:", class_id)
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self._device())
                print("class_id{}: self.feature_extractor(image_batch)".format(class_id))
                if use_FM and task > 1:
                    feature_batch = self.get_preFE_FM_features(image_batch).cpu()
                else:
                    feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            current_task_features.append(features)
            self.construct_exemplar_feature_set(class_dataset, features, examplars_per_class, features_per_class)
        # current_task_features = torch.stack(current_task_features)
        # current_task_features = current_task_features.numpy()
        # self.linearSVM_train(pre_tasks_features, pre_tasks_targets, current_task_features,
        #                      current_task_target, self.sample_type)

    def cifar100_feature_handle_main_FM(self, training_dataset, exemplar_dataset, train_dataset, test_datasets,
                                        classes_per_task,
                                        active_classes, task, img_size,
                                        feature_dim, use_dynamicMem=True, FM_reTrain=False, use_FM=False,
                                        use_NewfeatureSpace=False):
        pre_tasks_features = []
        pre_tasks_targets = []
        class_nums = classes_per_task * task
        # feature_memory_noUse = self.Feature_memory_budget - self.Exemple_memory_budget - (class_nums * 500)
        feature_memory_noUse = self.Feature_memory_budget - (class_nums * 500)
        if (not use_dynamicMem) or (feature_memory_noUse <= 0 or self.Exemple_memory_budget <= 0):
            features_per_class = int(np.floor(self.Feature_memory_budget / (classes_per_task * task)))
            examplars_per_class = int(np.floor(self.Exemple_memory_budget / (classes_per_task * task)))
        else:
            features_per_class = 500
            examplars_per_class = min(500, int(
                ((feature_memory_noUse / int(img_size / feature_dim)) + self.Exemple_memory_budget) / class_nums))
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, self.Feature_memory_budget: "
            f"{self.Feature_memory_budget}, self.Exemple_memory_budget: {self.Exemple_memory_budget}, features_per_class:{features_per_class}, "
            f"examplars_per_class:{examplars_per_class} "
        )
        if task > 1:
            if FM_reTrain:
                if self.Exemple_memory_budget > 0:
                    self.feature_mapper_train_oversample(training_dataset, exemplar_dataset, test_datasets,
                                                         active_classes,
                                                         task, use_NewfeatureSpace=use_NewfeatureSpace)
                else:
                    self.feature_mapper_train(training_dataset, test_datasets, active_classes, task,
                                              use_NewfeatureSpace=use_NewfeatureSpace)
            for class_id, exemplar_feature_set in enumerate(self.exemplar_feature_sets):
                pre_tasks_targets.append(class_id)
                exemplars = []
                for exemplar_feature in exemplar_feature_set:
                    exemplars.append(torch.from_numpy(exemplar_feature))
                exemplars = torch.stack(exemplars).to(self._device())
                if FM_reTrain and use_NewfeatureSpace:
                    exemplars = self.pre_FM_mapping(exemplars)
                pre_tasks_features.append(self.feature_mapping(exemplars).cpu().numpy())
                # pre_tasks_features.append(exemplars.cpu().numpy())
            self.update_features_sets(pre_tasks_features, features_per_class)
            self.reduce_exemplar_sets(examplars_per_class)
        current_task_features = []
        current_task_target = []
        # for each new class trained on, construct examplar-set
        new_classes = list(range(classes_per_task * (task - 1), classes_per_task * task))
        for class_id in new_classes:
            # create new dataset containing only all examples of this class
            class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
            current_task_target.append(class_id)
            dataloader = utils.get_data_loader(class_dataset, self.batch_size, self.num_workers, is_shuffle=False,
                                               cuda=self._is_on_cuda())
            first_entry = True
            print("construct_exemplar_feature_set class_id:", class_id)
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self._device())
                print("class_id{}: self.feature_extractor(image_batch)".format(class_id))
                if use_FM and task > 1:
                    feature_batch = self.get_preFE_FM_features(image_batch).cpu()
                else:
                    feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            current_task_features.append(features)
            self.construct_exemplar_feature_set(class_dataset, features, examplars_per_class, features_per_class)
        print("len(self.exemplar_sets[0]):", len(self.exemplar_sets[0]))
        # current_task_features = torch.stack(current_task_features)
        # current_task_features = current_task_features.numpy()
        # self.linearSVM_train(pre_tasks_features, pre_tasks_targets, current_task_features,
        #                      current_task_target, self.sample_type)

    def cifar100_feature_handle_main_FM_hardsample(self, training_dataset, exemplar_dataset, train_dataset,
                                                   test_datasets,
                                                   classes_per_task,
                                                   active_classes, task, img_size,
                                                   feature_dim, use_dynamicMem=True, FM_reTrain=False, use_FM=False,
                                                   use_NewfeatureSpace=False,
                                                   sigmoid_softmax=1):
        pre_tasks_features = []
        pre_tasks_targets = []
        class_nums = classes_per_task * task
        # feature_memory_noUse = self.Feature_memory_budget - self.Exemple_memory_budget - (class_nums * 500)
        feature_memory_noUse = self.Feature_memory_budget - (class_nums * 500)
        if (not use_dynamicMem) or (feature_memory_noUse <= 0 or self.Exemple_memory_budget <= 0):
            features_per_class = int(np.floor(self.Feature_memory_budget / (classes_per_task * task)))
            examplars_per_class = int(np.floor(self.Exemple_memory_budget / (classes_per_task * task)))
        else:
            features_per_class = 500
            examplars_per_class = min(500, int(
                ((feature_memory_noUse / int(img_size / feature_dim)) + self.Exemple_memory_budget) / class_nums))
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, self.Feature_memory_budget: "
            f"{self.Feature_memory_budget}, self.Exemple_memory_budget: {self.Exemple_memory_budget}, features_per_class:{features_per_class}, "
            f"examplars_per_class:{examplars_per_class} "
        )
        if task > 1:
            if FM_reTrain:
                # self.feature_mapper_train(training_dataset, test_datasets, active_classes, task,
                #                           use_NewfeatureSpace=use_NewfeatureSpace)
                self.feature_mapper_train_oversample(training_dataset, exemplar_dataset, test_datasets, active_classes,
                                                     task, use_NewfeatureSpace=use_NewfeatureSpace)
            for class_id, exemplar_feature_set in enumerate(self.exemplar_feature_sets):
                pre_tasks_targets.append(class_id)
                exemplars = []
                for exemplar_feature in exemplar_feature_set:
                    exemplars.append(torch.from_numpy(exemplar_feature))
                exemplars = torch.stack(exemplars).to(self._device())
                if FM_reTrain and use_NewfeatureSpace:
                    exemplars = self.pre_FM_mapping(exemplars)
                pre_tasks_features.append(self.feature_mapping(exemplars).cpu().numpy())
                # pre_tasks_features.append(exemplars.cpu().numpy())
            self.update_features_sets(pre_tasks_features, features_per_class)
            self.reduce_exemplar_sets(examplars_per_class)
        current_task_features = []
        current_task_target = []
        current_task_outputs = []
        # for each new class trained on, construct examplar-set
        new_classes = list(range(classes_per_task * (task - 1), classes_per_task * task))
        for class_id in new_classes:
            # create new dataset containing only all examples of this class
            class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
            current_task_target.append(class_id)
            dataloader = utils.get_data_loader(class_dataset, self.batch_size, self.num_workers, is_shuffle=False,
                                               cuda=self._is_on_cuda())
            first_entry = True
            print("construct_exemplar_feature_set class_id:", class_id)
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self._device())
                print("class_id{}: self.feature_extractor(image_batch)".format(class_id))
                feature_batch, target_batch = self.get_FE_cls_output(image_batch)
                feature_batch = feature_batch.cpu()
                target_batch = target_batch[:, :class_nums]
                target_batch = torch.softmax(target_batch, dim=1)
                # if sigmoid_softmax != 0:
                #     target_batch = torch.softmax(target_batch, dim=1)
                # else:
                #     target_batch = torch.sigmoid(target_batch)
                target_batch = target_batch[:, class_id]
                target_batch = target_batch.cpu()
                if first_entry:
                    features = feature_batch
                    targets = target_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
                    targets = torch.cat([targets, target_batch], dim=0)
            current_task_features.append(features)
            self.construct_exemplar_feature_set_hardsample(class_dataset, features, targets, examplars_per_class,
                                                           features_per_class)
        print("len(self.exemplar_sets[0]):", len(self.exemplar_sets[0]))
        # current_task_features = torch.stack(current_task_features)
        # current_task_features = current_task_features.numpy()
        # self.linearSVM_train(pre_tasks_features, pre_tasks_targets, current_task_features,
        #                      current_task_target, self.sample_type)

    def imagenet_feature_handle_main(self, train_dataset, exemplar_dataset, test_datasets, classes_per_task,
                                     active_classes, task,
                                     img_size, feature_dim,
                                     FM_reTrain=False, use_FM=False, use_NewfeatureSpace=False):
        pre_tasks_features = []
        pre_tasks_targets = []
        class_nums = classes_per_task * task
        feature_memory_noUse = self.Feature_memory_budget - class_nums * 1300
        if feature_memory_noUse <= 0 or self.Exemple_memory_budget <= 0:
            features_per_class = int(np.floor(self.Feature_memory_budget / (classes_per_task * task)))
            examplars_per_class = int(np.floor(self.Exemple_memory_budget / (classes_per_task * task)))
        else:
            features_per_class = 1300
            examplars_per_class = min(1300,
                                      int(((feature_memory_noUse / int(
                                          img_size / feature_dim)) + self.Exemple_memory_budget) / class_nums))
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, self.Feature_memory_budget: "
            f"{self.Feature_memory_budget}, self.Exemple_memory_budget: {self.Exemple_memory_budget}, features_per_class:{features_per_class}, "
            f"examplars_per_class:{examplars_per_class} "
        )
        if task > 1:
            if FM_reTrain:
                # self.feature_mapper_train(train_dataset, test_datasets, active_classes, task)
                self.feature_mapper_train_oversample(train_dataset, exemplar_dataset, test_datasets, active_classes,
                                                     task, use_NewfeatureSpace=use_NewfeatureSpace)
            for class_id, exemplar_feature_set in enumerate(self.exemplar_feature_sets):
                pre_tasks_targets.append(class_id)
                exemplars = []
                for exemplar_feature in exemplar_feature_set:
                    exemplars.append(torch.from_numpy(exemplar_feature))
                exemplars = torch.stack(exemplars).to(self._device())
                if FM_reTrain and use_NewfeatureSpace:
                    exemplars = self.pre_FM_mapping(exemplars)
                pre_tasks_features.append(self.feature_mapping(exemplars).cpu().numpy())
            self.update_features_sets(pre_tasks_features, features_per_class)
            self.reduce_exemplar_sets(examplars_per_class)
        current_task_features = []
        current_task_target = []
        # for each new class trained on, construct examplar-set
        new_classes = list(range(classes_per_task * (task - 1), classes_per_task * task))
        for class_id in new_classes:
            # create new dataset containing only all examples of this class
            class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
            current_task_target.append(class_id)
            dataloader = utils.get_data_loader(class_dataset, self.batch_size, self.num_workers, is_shuffle=False,
                                               cuda=self._is_on_cuda())
            first_entry = True
            print("construct_exemplar_feature_set class_id:", class_id)
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self._device())
                print("class_id{}: self.feature_extractor(image_batch)".format(class_id))
                if use_FM and task > 1:
                    feature_batch = self.get_preFE_FM_features(image_batch).cpu()
                else:
                    feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            current_task_features.append(features)
            self.construct_exemplar_feature_set(class_dataset, features, examplars_per_class, features_per_class)
        # current_task_features = torch.stack(current_task_features)
        # current_task_features = current_task_features.numpy()
        # self.linearSVM_train(pre_tasks_features, pre_tasks_targets, current_task_features,
        #                      current_task_target, self.sample_type)

    def tiny_imagenet_feature_handle_main(self, training_dataset, train_dataset, test_datasets, classes_per_task,
                                          active_classes, task, img_size,
                                          feature_dim, FM_reTrain=False, use_FM=False,
                                          use_NewfeatureSpace=False):
        pre_tasks_features = []
        pre_tasks_targets = []
        class_nums = classes_per_task * task
        feature_memory_noUse = self.Feature_memory_budget - class_nums * 500
        if feature_memory_noUse <= 0 or self.Exemple_memory_budget <= 0:
            features_per_class = int(np.floor(self.Feature_memory_budget / (classes_per_task * task)))
            examplars_per_class = int(np.floor(self.Exemple_memory_budget / (classes_per_task * task)))
        else:
            features_per_class = 500
            examplars_per_class = min(500, int(
                ((feature_memory_noUse / int(img_size / feature_dim)) + self.Exemple_memory_budget) / class_nums))
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, self.Feature_memory_budget: "
            f"{self.Feature_memory_budget}, self.Exemple_memory_budget: {self.Exemple_memory_budget}, features_per_class:{features_per_class}, "
            f"examplars_per_class:{examplars_per_class} "
        )
        if task > 1:
            if FM_reTrain:
                self.feature_mapper_train(training_dataset, test_datasets, active_classes, task,
                                          use_NewfeatureSpace=use_NewfeatureSpace)
            for class_id, exemplar_feature_set in enumerate(self.exemplar_feature_sets):
                pre_tasks_targets.append(class_id)
                exemplars = []
                for exemplar_feature in exemplar_feature_set:
                    exemplars.append(torch.from_numpy(exemplar_feature))
                exemplars = torch.stack(exemplars).to(self._device())
                if FM_reTrain and use_NewfeatureSpace:
                    exemplars = self.pre_FM_mapping(exemplars)
                pre_tasks_features.append(self.feature_mapping(exemplars).cpu().numpy())
            self.update_features_sets(pre_tasks_features, features_per_class)
            self.reduce_exemplar_sets(examplars_per_class)
        current_task_features = []
        current_task_target = []
        # for each new class trained on, construct examplar-set
        new_classes = list(range(classes_per_task * (task - 1), classes_per_task * task))
        for class_id in new_classes:
            # create new dataset containing only all examples of this class
            class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
            current_task_target.append(class_id)
            dataloader = utils.get_data_loader(class_dataset, self.batch_size, self.num_workers, is_shuffle=False,
                                               cuda=self._is_on_cuda())
            first_entry = True
            print("construct_exemplar_feature_set class_id:", class_id)
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self._device())
                print("class_id{}: self.feature_extractor(image_batch)".format(class_id))
                if use_FM and task > 1:
                    feature_batch = self.get_preFE_FM_features(image_batch).cpu()
                else:
                    feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            current_task_features.append(features)
            self.construct_exemplar_feature_set(class_dataset, features, examplars_per_class, features_per_class)
        # current_task_features = torch.stack(current_task_features)
        # current_task_features = current_task_features.numpy()
        # self.linearSVM_train(pre_tasks_features, pre_tasks_targets, current_task_features,
        #                      current_task_target, self.sample_type)

    def get_features(self, training_dataset):
        dataloader = utils.get_data_loader(training_dataset, self.batch_size, self.num_workers,
                                           cuda=self._is_on_cuda())
        first_entry = True
        features = None
        labels = None
        for image_batch, label_batch in dataloader:
            image_batch = image_batch.to(self._device())
            feature_batch = self.feature_extractor(image_batch).cpu()
            if first_entry:
                features = feature_batch
                labels = label_batch
                first_entry = False
            else:
                features = torch.cat([features, feature_batch], dim=0)
                labels = torch.cat([labels, label_batch], dim=0)
        return features.numpy(), labels.numpy()

    def EFAfIL_split_feature_mapper_cls_domain_train(self, training_dataset, test_datasets, classes_per_task,
                                                     active_classes, task):
        # todo Done
        self.batch_train_logger.info(f'####Task {task} EFAfIL feature_mapper_cls_domain_train begin.####')
        self.logger.info(f'####Task {task} EFAfIL  feature_mapper_cls_domain_train begin.####')
        print("Task %d , train feature mapping..." % task)
        pre_features_target_hats_datasets = ExemplarDataset(self.exemplar_feature_sets)
        mode = self.training
        self.eval()
        self.FM_cls_domain.train()
        criterion = nn.CrossEntropyLoss()
        soft_target_criterion = SoftTarget_CrossEntropy().cuda()
        optimizer = self.build_FM_cls_domain_optimize()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
        train_dataset_iter_index = 0
        pre_features_target_hats_datasets_iter_index = 0
        train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
                                                  self.num_workers,
                                                  cuda=True if self.availabel_cudas else False))
        pre_features_target_hats_train_loader = iter(utils.get_data_loader(pre_features_target_hats_datasets,
                                                                           self.batch_size,
                                                                           self.num_workers,
                                                                           cuda=True if self.availabel_cudas else False))
        train_imgs_num = len(train_loader)
        pre_features_target_hats_num = len(pre_features_target_hats_train_loader)
        for epoch in range(self.MLP_epochs):
            if train_dataset_iter_index == train_imgs_num:
                train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
                                                          self.num_workers,
                                                          cuda=True if self.availabel_cudas else False))
                train_dataset_iter_index = 0
            if pre_features_target_hats_datasets_iter_index == pre_features_target_hats_num:
                pre_features_target_hats_train_loader = iter(utils.get_data_loader(pre_features_target_hats_datasets,
                                                                                   self.batch_size,
                                                                                   self.num_workers,
                                                                                   cuda=True if self.availabel_cudas else False))
                pre_features_target_hats_datasets_iter_index = 0
            iter_index = 0
            iters_left = max(train_imgs_num, pre_features_target_hats_num)
            while iters_left > 0:
                iters_left -= 1
                optimizer.zero_grad()
                if train_dataset_iter_index == train_imgs_num:
                    train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
                                                              self.num_workers,
                                                              cuda=True if self.availabel_cudas else False))
                    train_dataset_iter_index = 0
                if pre_features_target_hats_datasets_iter_index == pre_features_target_hats_num:
                    pre_features_target_hats_train_loader = iter(
                        utils.get_data_loader(pre_features_target_hats_datasets,
                                              self.batch_size,
                                              self.num_workers,
                                              cuda=True if self.availabel_cudas else False))
                    pre_features_target_hats_datasets_iter_index = 0
                iter_index += 1
                '''读取数据'''
                imgs, labels = next(train_loader)
                pre_features, pre_labels = next(pre_features_target_hats_train_loader)
                train_dataset_iter_index += 1
                pre_features_target_hats_datasets_iter_index += 1
                imgs, labels = imgs.to(self.MLP_device), labels.to(self.MLP_device)
                pre_features, pre_labels = pre_features.to(self.MLP_device), pre_labels.to(self.MLP_device)
                '''获取图片数据在FE_cls的输出'''
                imgs_2_features, imgs_2_targets = self.get_FE_cls_output(imgs)
                imgs_2_targets = imgs_2_targets[:, :(classes_per_task * task)]
                '''获取在要训练的模型FM'''
                imgs_2_feature_hat, y_hat, domain_hat = self.FM_cls_domain(imgs_2_features)
                pre_features_hat, pre_y_hat, pre_domain_hat = self.FM_cls_domain(pre_features)
                y_hat = y_hat[:, :(classes_per_task * task)]
                pre_y_hat = pre_y_hat[:, :(classes_per_task * task)]
                '''pre_features 在FE_cls的输出'''
                pre_features_targets = self.get_cls_target(pre_features)

                y_hat_pre_tasks = y_hat[:, :(classes_per_task * (task - 1))]
                pre_y_hat_tasks = pre_y_hat[:, :(classes_per_task * (task - 1))]
                '''做distill的输入'''
                imgs_2_targets = imgs_2_targets[:, :(classes_per_task * (task - 1))]
                pre_features_targets = pre_features_targets[:, :(classes_per_task * (task - 1))]
                '''binary loss distill'''
                if self.MLP_KD_temp_2 > 0:
                    '''binary loss distill'''
                    imgs_2_targets = torch.sigmoid(imgs_2_targets / self.MLP_KD_temp)
                    pre_features_targets = torch.sigmoid(pre_features_targets / self.MLP_KD_temp)
                    y_hat_pre_tasks /= self.MLP_KD_temp_2
                    pre_y_hat_tasks /= self.MLP_KD_temp_2
                    loss_distill_current_imgs = Func.binary_cross_entropy_with_logits(input=y_hat_pre_tasks,
                                                                                      target=imgs_2_targets,
                                                                                      reduction='none').sum(
                        dim=1).mean()
                    loss_distill_features = Func.binary_cross_entropy_with_logits(input=pre_y_hat_tasks,
                                                                                  target=pre_features_targets,
                                                                                  reduction='none').sum(dim=1).mean()
                    loss_distill = loss_distill_current_imgs + loss_distill_features
                else:
                    '''softTarget loss distill'''
                    imgs_2_targets = torch.softmax(imgs_2_targets / self.MLP_KD_temp, dim=1)
                    pre_features_targets = torch.softmax(pre_features_targets / self.MLP_KD_temp, dim=1)
                    loss_distill_current_imgs = soft_target_criterion(y_hat_pre_tasks, imgs_2_targets,
                                                                      self.MLP_KD_temp)

                    loss_distill_features = soft_target_criterion(pre_y_hat_tasks,
                                                                  pre_features_targets,
                                                                  self.MLP_KD_temp)
                    loss_distill = loss_distill_current_imgs + loss_distill_features
                '''loss classify'''
                '''cross entropy loss classify'''
                loss_cls = criterion(y_hat, labels) + criterion(pre_y_hat, pre_labels)
                '''total loss '''
                loss_total = loss_cls + self.MLP_distill_rate * loss_distill * self.MLP_KD_temp ** 2
                loss_total.backward()
                optimizer.step()
                precision = None if y_hat is None else (labels == y_hat.max(1)[
                    1]).sum().item() / labels.size(0)
                lr = scheduler.get_last_lr()[0]
                ite_info = {
                    'task': task,
                    'epoch': epoch,
                    'lr': lr,
                    'loss_total': loss_total.item(),
                    'precision': precision if precision is not None else 0.,
                }
                self.batch_train_logger.info(
                    f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, "
                    f"lr: {lr}, loss_total: {loss_total}, acc: {precision}")
                print(ite_info)
                print("....................................")
            scheduler.step()
        acc1, acc5, throughput = self.current_task_validate_FM_cls_domain(test_datasets, task, active_classes)
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapper_cls_domain train task : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))

        acc1, acc5, throughput = self.current_task_validate_FM_cls_domain(test_datasets, task, active_classes,
                                                                          val_current_task=False)  # todo
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train old classes || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature mapper_cls_domain train old classes|| task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapper_cls_domain train old classes : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (
            acc1, acc5))
        print("-----------------------------------")

        print("Train feature mapper_cls_domain End.")
        self.pre_FM_cls_domain = copy.deepcopy(self.FM_cls_domain).eval()
        self.train(mode=mode)
        pass

    def EFAfIL_split_feature_mapper_cls_domain_train_exemplarOverSample(self, train_dataset, exemplar_dataset,
                                                                        test_datasets, classes_per_task,
                                                                        active_classes, task):
        # todo Done
        self.batch_train_logger.info(f'####Task {task} EFAfIL feature_mapper_cls_domain_train begin.####')
        self.logger.info(f'####Task {task} EFAfIL  feature_mapper_cls_domain_train begin.####')
        print("Task %d , train feature mapping..." % task)
        pre_features_target_hats_datasets = ExemplarDataset(self.exemplar_feature_sets)
        mode = self.training
        self.eval()
        self.FM_cls_domain.train()
        criterion = nn.CrossEntropyLoss()
        soft_target_criterion = SoftTarget_CrossEntropy().cuda()
        optimizer = self.build_FM_cls_domain_optimize()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
        train_dataset_iter_index = 0
        exemplar_datasets_iter_index = 0
        pre_features_target_hats_datasets_iter_index = 0
        train_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                  self.num_workers,
                                                  cuda=True if self.availabel_cudas else False))

        exemplar_loader = iter(utils.get_data_loader(exemplar_dataset,
                                                     self.batch_size,
                                                     self.num_workers,
                                                     cuda=True if self.availabel_cudas else False))

        pre_features_target_hats_train_loader = iter(utils.get_data_loader(pre_features_target_hats_datasets,
                                                                           self.batch_size,
                                                                           self.num_workers,
                                                                           cuda=True if self.availabel_cudas else False))
        train_imgs_num = len(train_loader)
        exemplar_num = len(exemplar_loader)
        pre_features_target_hats_num = len(pre_features_target_hats_train_loader)
        for epoch in range(self.MLP_epochs):
            if train_dataset_iter_index == train_imgs_num:
                train_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                          self.num_workers,
                                                          cuda=True if self.availabel_cudas else False))
                train_dataset_iter_index = 0

            if exemplar_datasets_iter_index == exemplar_num:
                exemplar_loader = iter(
                    utils.get_data_loader(exemplar_dataset,
                                          self.batch_size,
                                          self.num_workers,
                                          cuda=True if self.availabel_cudas else False))
                exemplar_datasets_iter_index = 0

            if pre_features_target_hats_datasets_iter_index == pre_features_target_hats_num:
                pre_features_target_hats_train_loader = iter(utils.get_data_loader(pre_features_target_hats_datasets,
                                                                                   self.batch_size,
                                                                                   self.num_workers,
                                                                                   cuda=True if self.availabel_cudas else False))
                pre_features_target_hats_datasets_iter_index = 0
            iter_index = 0
            iters_left = max(train_imgs_num, exemplar_num, pre_features_target_hats_num)
            while iters_left > 0:
                iters_left -= 1
                optimizer.zero_grad()
                if train_dataset_iter_index == train_imgs_num:
                    train_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                              self.num_workers,
                                                              cuda=True if self.availabel_cudas else False))
                    train_dataset_iter_index = 0

                if exemplar_datasets_iter_index == exemplar_num:
                    exemplar_loader = iter(
                        utils.get_data_loader(exemplar_dataset,
                                              self.batch_size,
                                              self.num_workers,
                                              cuda=True if self.availabel_cudas else False))
                    exemplar_datasets_iter_index = 0

                if pre_features_target_hats_datasets_iter_index == pre_features_target_hats_num:
                    pre_features_target_hats_train_loader = iter(
                        utils.get_data_loader(pre_features_target_hats_datasets,
                                              self.batch_size,
                                              self.num_workers,
                                              cuda=True if self.availabel_cudas else False))
                    pre_features_target_hats_datasets_iter_index = 0
                iter_index += 1
                '''读取数据'''
                imgs, labels = next(train_loader)
                examplar_imgs, examplar_labels = next(exemplar_loader)
                pre_features, pre_labels = next(pre_features_target_hats_train_loader)
                train_dataset_iter_index += 1
                exemplar_datasets_iter_index += 1
                pre_features_target_hats_datasets_iter_index += 1
                imgs, labels = imgs.to(self.MLP_device), labels.to(self.MLP_device)
                examplar_imgs, examplar_labels = examplar_imgs.to(self.MLP_device), examplar_labels.to(self.MLP_device)
                pre_features, pre_labels = pre_features.to(self.MLP_device), pre_labels.to(self.MLP_device)
                '''获取图片数据在FE_cls的输出'''
                imgs_2_features, imgs_2_targets = self.get_FE_cls_output(imgs)  # todo Done
                exemplar_imgs_2_features, exemplar_imgs_2_targets = self.get_FE_cls_output(examplar_imgs)
                imgs_2_targets = imgs_2_targets[:, :(classes_per_task * (task - 1))]
                exemplar_imgs_2_targets = exemplar_imgs_2_targets[:, :(classes_per_task * (task - 1))]

                '''获取pre_features在FE_cls的输出  '''
                pre_features_targets = self.get_cls_target(pre_features)
                pre_features_targets = pre_features_targets[:, :(classes_per_task * (task - 1))]

                '''获取在要训练的模型FM的输出'''
                imgs_2_feature_hat, y_hat, _ = self.FM_cls_domain(imgs_2_features)
                y_hat = y_hat[:, :(classes_per_task * task)]
                exemplar_imgs_2_feature_hat, exemplar_y_hat, _ = self.FM_cls_domain(exemplar_imgs_2_features)
                exemplar_y_hat = exemplar_y_hat[:, :(classes_per_task * task)]
                pre_features_hat, pre_y_hat, _ = self.FM_cls_domain(pre_features)
                pre_y_hat = pre_y_hat[:, :(classes_per_task * task)]

                '''make distill train data'''
                y_hat_pre_tasks = y_hat[:, :(classes_per_task * (task - 1))]
                exemplar_y_hat_pre_tasks = exemplar_y_hat[:, :(classes_per_task * (task - 1))]
                pre_y_hat_tasks = pre_y_hat[:, :(classes_per_task * (task - 1))]

                if self.MLP_KD_temp_2 > 0:
                    '''binary loss distill'''
                    imgs_2_targets = torch.sigmoid(imgs_2_targets / self.MLP_KD_temp)
                    exemplar_imgs_2_targets = torch.sigmoid(exemplar_imgs_2_targets / self.MLP_KD_temp)
                    pre_features_targets = torch.sigmoid(pre_features_targets / self.MLP_KD_temp)
                    y_hat_pre_tasks /= self.MLP_KD_temp_2
                    exemplar_y_hat_pre_tasks /= self.MLP_KD_temp_2
                    pre_y_hat_tasks /= self.MLP_KD_temp_2
                    loss_distill_current_task = Func.binary_cross_entropy_with_logits(input=y_hat_pre_tasks,
                                                                                      target=imgs_2_targets,
                                                                                      reduction='none').sum(
                        dim=1).mean()
                    loss_distill_pre_tasks = Func.binary_cross_entropy_with_logits(input=exemplar_y_hat_pre_tasks,
                                                                                   target=exemplar_imgs_2_targets,
                                                                                   reduction='none').sum(dim=1).mean()
                    loss_distill_pre_tasks_feature_replay = Func.binary_cross_entropy_with_logits(input=pre_y_hat_tasks,
                                                                                                  target=pre_features_targets,
                                                                                                  reduction='none').sum(
                        dim=1).mean()
                    loss_distill = loss_distill_current_task + loss_distill_pre_tasks + loss_distill_pre_tasks_feature_replay
                else:
                    '''softTarget loss distill'''
                    exemplar_imgs_2_targets = torch.softmax(exemplar_imgs_2_targets / self.MLP_KD_temp, dim=1)
                    imgs_2_targets = torch.softmax(imgs_2_targets / self.MLP_KD_temp, dim=1)
                    pre_features_targets = torch.softmax(pre_features_targets / self.MLP_KD_temp, dim=1)

                    loss_distill_current_task = soft_target_criterion(y_hat_pre_tasks, imgs_2_targets,
                                                                      self.MLP_KD_temp)

                    loss_distill_pre_tasks = soft_target_criterion(exemplar_y_hat_pre_tasks, exemplar_imgs_2_targets,
                                                                   self.MLP_KD_temp)

                    loss_distill_pre_tasks_feature_replay = soft_target_criterion(pre_y_hat_tasks,
                                                                                  pre_features_targets,
                                                                                  self.MLP_KD_temp)
                    loss_distill = loss_distill_current_task + loss_distill_pre_tasks + loss_distill_pre_tasks_feature_replay

                '''loss classify'''
                '''cross entropy loss classify'''
                loss_cls = criterion(y_hat, labels) + criterion(exemplar_y_hat, examplar_labels) + \
                           criterion(pre_y_hat, pre_labels)
                '''total loss '''
                loss_total = loss_cls + self.MLP_distill_rate * loss_distill * (self.MLP_KD_temp ** 2)
                loss_total.backward()
                optimizer.step()
                precision = None if y_hat is None else (labels == y_hat.max(1)[
                    1]).sum().item() / labels.size(0)
                lr = scheduler.get_last_lr()[0]
                ite_info = {
                    'task': task,
                    'epoch': epoch,
                    'lr': lr,
                    'loss_total': loss_total.item(),
                    'precision': precision if precision is not None else 0.,
                }
                self.batch_train_logger.info(
                    f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, "
                    f"lr: {lr}, loss_total: {loss_total}, acc: {precision}")
                print(ite_info)
                print("....................................")
            scheduler.step()
        acc1, acc5, throughput = self.current_task_validate_FM_cls_domain(test_datasets, task, active_classes)
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapper_cls_domain train task : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))

        acc1, acc5, throughput = self.current_task_validate_FM_cls_domain(test_datasets, task, active_classes,
                                                                          val_current_task=False)  # todo
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train old classes || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature mapper_cls_domain train old classes|| task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapper_cls_domain train old classes : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (
            acc1, acc5))
        print("-----------------------------------")
        print("Train feature mapper_cls_domain End.")
        self.pre_FM_cls_domain = copy.deepcopy(self.FM_cls_domain).eval()
        self.train(mode=mode)
        pass

    def build_optimize_bias_cls(self, lr):
        # Define optimizer (only include parameters that "requires_grad")
        assert self.unbias_cls_of_FMcls is not None
        optim_list = [{'params': filter(lambda p: p.requires_grad, self.unbias_cls_of_FMcls.parameters()), 'lr': lr}]
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

    def retrain_bias_cls_of_FMcls(self, per_task_valing_dataset, val_datasets, task, active_classes, print_interval,
                                  train_method=0, bias_or_cRT=0):
        current_classes_num = self.classes_per_task
        if self.unbias_cls_of_FMcls is None:
            if bias_or_cRT == 0:
                self.unbias_cls_of_FMcls = BiasLayer()
            else:
                # self.unbias_cls_of_FMcls = nn.Linear(self.feature_dim, self.classes_per_task * task)
                if type(self.FM_cls_domain) is torch.nn.DataParallel:
                    self.unbias_cls_of_FMcls = copy.deepcopy(self.FM_cls_domain.module.cls)
                else:
                    self.unbias_cls_of_FMcls = copy.deepcopy(self.FM_cls_domain.cls)
        if train_method == 0:
            optimizer = self.build_optimize_bias_cls(0.001)
            epochs = 45
            gap = int(epochs / 3)
            milestones = [gap, 2 * gap]
        elif train_method == 1:
            optimizer = self.build_optimize_bias_cls(0.001)
            epochs = 60
            gap = int(epochs / 3)
            milestones = [gap, 2 * gap]
        elif train_method == 2:
            optimizer = self.build_optimize_bias_cls(0.01)
            epochs = 96
            gap = int(epochs / 4)
            milestones = [gap, 2 * gap, 3 * gap]
        elif train_method == 3:
            if bias_or_cRT == 0:
                optimizer = self.build_optimize_bias_cls(0.01)
            else:
                optimizer = self.build_optimize_bias_cls(0.1)
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
                x, y = x.to(self.MLP_device), y.to(self.MLP_device)  # --> transfer them to correct device
                features = self.feature_extractor(x)
                if bias_or_cRT == 0:
                    FM_cls_targets = self.FM_cls_forward_nograd(features)
                    FM_cls_targets = FM_cls_targets[:, :(self.classes_per_task * task)]
                    loss_dict = self.Bias_Cls_train_a_batch(FM_cls_targets, y, current_classes_num, optimizer,
                                                            active_classes)
                    print("self.unbias_CLs.parameters:", self.unbias_cls_of_FMcls.params)
                else:
                    loss_dict = self.cRT_cls_train_a_batch(features, y, current_classes_num, optimizer,
                                                           active_classes)
                iter_index += 1
                if iter_index % print_interval == 0:

                    if bias_or_cRT == 0:
                        results = f"bias_cls train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                  f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                  f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}" \
                                  f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
                    else:
                        results = f"cRT_cls train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                  f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                  f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}" \
                                  f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
                    self.batch_train_logger.info(
                        results
                    )
                    print(results)
            # print("self.unbias_CLs.parameters:", self.unbias_cls_of_FMcls.params)
            scheduler.step()
            if bias_or_cRT == 0:
                acc1, acc5, throughput = self.current_task_validate_bias_Cls(val_datasets, task, active_classes,
                                                                             current_classes_num)
            else:
                acc1, acc5, throughput = self.current_task_validate_cRT_Cls(val_datasets, task, active_classes)
            self.batch_train_logger.info(
                f" unbias_CLs validate || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
                f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
            )
            self.batch_train_logger.info(f"------------------------------------------------------------------")
            print(f'unbias_CLs task : {task:0>3d}, epoch: {epoch}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (
                acc1, acc5))

        pass

    def Bias_Cls_train_a_batch(self, FM_cls_targets, y, current_classes_num, optimizer, active_classes=None):
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
        self.unbias_cls_of_FMcls.train()
        # Reset optimizer
        optimizer.zero_grad()
        criterion = torch.nn.CrossEntropyLoss().cuda()

        # Run model
        y_hat = self.unbias_cls_of_FMcls(FM_cls_targets, current_classes_num)

        predL = criterion(y_hat, y)
        if len(active_classes) >= 5:
            acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
            top1.update(acc1.item(), FM_cls_targets.size(0))
            top5.update(acc5.item(), FM_cls_targets.size(0))
            losses.update(predL, FM_cls_targets.size(0))
            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / FM_cls_targets.size(0)
        else:
            acc1 = accuracy(y_hat, y, topk=(1,))[0]
            top1.update(acc1.item(), FM_cls_targets.size(0))
            losses.update(predL, FM_cls_targets.size(0))
            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / FM_cls_targets.size(0)
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

    def cRT_cls_train_a_batch(self, features, y, current_classes_num, optimizer, active_classes=None):
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
        self.unbias_cls_of_FMcls.train()
        # Reset optimizer
        optimizer.zero_grad()
        criterion = torch.nn.CrossEntropyLoss().cuda()

        # Run model
        y_hat = self.unbias_cls_of_FMcls(features)
        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
            y_hat = y_hat[:, class_entries]
        predL = criterion(y_hat, y)
        if len(active_classes) >= 5:
            acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
            top1.update(acc1.item(), y_hat.size(0))
            top5.update(acc5.item(), y_hat.size(0))
            losses.update(predL, y_hat.size(0))
            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / y_hat.size(0)
        else:
            acc1 = accuracy(y_hat, y, topk=(1,))[0]
            top1.update(acc1.item(), y_hat.size(0))
            losses.update(predL, y_hat.size(0))
            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / y_hat.size(0)
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

    def EFAfIL_split_feature_mapper_cls_domain_train_bias(self, Bias_layer, training_dataset, test_datasets,
                                                          classes_per_task, active_classes, task):
        # todo Done
        self.batch_train_logger.info(f'####Task {task} EFAfIL_iCaRL feature_mapper_cls_domain_train begin.####')
        self.logger.info(f'####Task {task} EFAfIL_iCaRL  feature_mapper_cls_domain_train begin.####')
        print("Task %d , train feature mapping..." % task)
        if self.FM_cls_domain is None:
            self.FM_cls_domain = self.construct_FM_cls_domain_model()
        mode = self.training
        self.eval()
        if task > 2:
            Bias_layer.eval()
        self.FM_cls_domain.train()
        pre_features_target_hats_datasets = ExemplarDataset(self.exemplar_feature_sets)
        criterion = nn.CrossEntropyLoss()
        soft_target_criterion = SoftTarget_CrossEntropy().cuda()
        optimizer = self.build_FM_cls_domain_optimize()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
        train_dataset_iter_index = 0
        pre_features_target_hats_datasets_iter_index = 0
        train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
                                                  self.num_workers,
                                                  cuda=True if self.availabel_cudas else False))
        pre_features_target_hats_train_loader = iter(utils.get_data_loader(pre_features_target_hats_datasets,
                                                                           self.batch_size,
                                                                           self.num_workers,
                                                                           cuda=True if self.availabel_cudas else False))
        train_imgs_num = len(train_loader)
        pre_features_target_hats_num = len(pre_features_target_hats_train_loader)
        for epoch in range(self.MLP_epochs):
            if train_dataset_iter_index == train_imgs_num:
                train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
                                                          self.num_workers,
                                                          cuda=True if self.availabel_cudas else False))
                train_dataset_iter_index = 0
            if pre_features_target_hats_datasets_iter_index == pre_features_target_hats_num:
                pre_features_target_hats_train_loader = iter(utils.get_data_loader(pre_features_target_hats_datasets,
                                                                                   self.batch_size,
                                                                                   self.num_workers,
                                                                                   cuda=True if self.availabel_cudas else False))
                pre_features_target_hats_datasets_iter_index = 0
            iter_index = 0
            iters_left = max(train_imgs_num, pre_features_target_hats_num)
            while iters_left > 0:
                iters_left -= 1
                optimizer.zero_grad()
                if train_dataset_iter_index == train_imgs_num:
                    train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
                                                              self.num_workers,
                                                              cuda=True if self.availabel_cudas else False))
                    train_dataset_iter_index = 0
                if pre_features_target_hats_datasets_iter_index == pre_features_target_hats_num:
                    pre_features_target_hats_train_loader = iter(
                        utils.get_data_loader(pre_features_target_hats_datasets,
                                              self.batch_size,
                                              self.num_workers,
                                              cuda=True if self.availabel_cudas else False))
                    pre_features_target_hats_datasets_iter_index = 0
                iter_index += 1
                '''读取数据'''
                imgs, labels = next(train_loader)
                pre_features, pre_labels = next(pre_features_target_hats_train_loader)
                train_dataset_iter_index += 1
                pre_features_target_hats_datasets_iter_index += 1
                imgs, labels = imgs.to(self.MLP_device), labels.to(self.MLP_device)
                pre_features, pre_labels = pre_features.to(self.MLP_device), pre_labels.to(self.MLP_device)
                '''获取图片数据在FE_cls的输出'''
                imgs_2_features, imgs_2_targets = self.get_FE_cls_output(imgs)
                imgs_2_targets = imgs_2_targets[:, :(classes_per_task * task)]
                '''获取在要训练的模型FM'''
                imgs_2_feature_hat, y_hat, _ = self.FM_cls_domain(imgs_2_features)
                pre_features_hat, pre_y_hat, _ = self.FM_cls_domain(pre_features)
                y_hat = y_hat[:, :(classes_per_task * task)]
                pre_y_hat = pre_y_hat[:, :(classes_per_task * task)]

                y_hat_pre_tasks = y_hat[:, :(classes_per_task * (task - 1))]
                pre_y_hat_pre_tasks = pre_y_hat[:, :(classes_per_task * (task - 1))]
                pre_features_targets = self.get_cls_target(pre_features)
                imgs_2_targets = imgs_2_targets[:, :(classes_per_task * (task - 1))]
                pre_features_targets = pre_features_targets[:, :(classes_per_task * (task - 1))]

                '''获取图片数据在Bias_layer的输出'''
                if task > 2:
                    assert Bias_layer is not None
                    with torch.no_grad():
                        imgs_2_targets = Bias_layer(imgs_2_targets, classes_per_task)
                        pre_features_targets = Bias_layer(pre_features_targets, classes_per_task)
                if self.MLP_KD_temp_2 > 0:
                    '''binary loss distill'''
                    imgs_2_targets = torch.sigmoid(imgs_2_targets / self.MLP_KD_temp)
                    pre_features_targets = torch.sigmoid(pre_features_targets / self.MLP_KD_temp)
                    y_hat_pre_tasks /= self.MLP_KD_temp_2
                    pre_y_hat_pre_tasks /= self.MLP_KD_temp_2
                    loss_distill_current_imgs = Func.binary_cross_entropy_with_logits(input=y_hat_pre_tasks,
                                                                                      target=imgs_2_targets,
                                                                                      reduction='none').sum(
                        dim=1).mean()
                    loss_distill_features = Func.binary_cross_entropy_with_logits(input=pre_y_hat_pre_tasks,
                                                                                  target=pre_features_targets,
                                                                                  reduction='none').sum(dim=1).mean()
                    loss_distill = loss_distill_current_imgs + loss_distill_features
                else:
                    '''softTarget loss distill'''
                    imgs_2_targets = torch.softmax(imgs_2_targets / self.MLP_KD_temp, dim=1)
                    pre_features_targets = torch.softmax(pre_features_targets / self.MLP_KD_temp, dim=1)
                    loss_distill_current_imgs = soft_target_criterion(y_hat_pre_tasks, imgs_2_targets,
                                                                      self.MLP_KD_temp)

                    loss_distill_features = soft_target_criterion(pre_y_hat_pre_tasks,
                                                                  pre_features_targets,
                                                                  self.MLP_KD_temp)
                    loss_distill = loss_distill_current_imgs + loss_distill_features
                '''loss classify'''
                '''cross entropy loss classify'''
                loss_cls = criterion(y_hat, labels) + criterion(pre_y_hat, pre_labels)
                '''total loss '''
                loss_total = loss_cls + self.MLP_distill_rate * loss_distill * (self.MLP_KD_temp ** 2)
                loss_total.backward()
                optimizer.step()
                precision = None if y_hat is None else (labels == y_hat.max(1)[
                    1]).sum().item() / labels.size(0)
                lr = scheduler.get_last_lr()[0]
                ite_info = {
                    'task': task,
                    'epoch': epoch,
                    'lr': lr,
                    'loss_total': loss_total.item(),
                    'precision': precision if precision is not None else 0.,
                }
                self.batch_train_logger.info(
                    f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, "
                    f"lr: {lr}, loss_total: {loss_total}, acc: {precision}")
                print(ite_info)
                print("....................................")
            scheduler.step()
        acc1, acc5, throughput = self.current_task_validate_FM_cls_domain(test_datasets, task, active_classes)  # todo
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapper_cls_domain train task : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))
        print("Train feature mapper_cls_domain End.")
        self.pre_FM_cls_domain = copy.deepcopy(self.FM_cls_domain).eval()
        self.train(mode=mode)
        pass

    def EFAfIL_split_feature_mapper_cls_domain_train_bias_ovarsample(self, Bias_layer, train_dataset, exemplar_set,
                                                                     test_datasets, classes_per_task, active_classes,
                                                                     task):
        # todo Done
        self.batch_train_logger.info(f'####Task {task} EFAfIL_iCaRL feature_mapper_cls_domain_train begin.####')
        self.logger.info(f'####Task {task} EFAfIL_iCaRL  feature_mapper_cls_domain_train begin.####')
        print("Task %d , train feature mapping..." % task)
        if self.FM_cls_domain is None:
            self.FM_cls_domain = self.construct_FM_cls_domain_model()
        mode = self.training
        self.eval()
        if task > 2:
            Bias_layer.eval()
        self.FM_cls_domain.train()
        pre_features_target_hats_datasets = ExemplarDataset(self.exemplar_feature_sets)
        criterion = nn.CrossEntropyLoss()
        soft_target_criterion = SoftTarget_CrossEntropy().cuda()
        optimizer = self.build_FM_cls_domain_optimize()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
        train_dataset_iter_index = 0
        exemplar_dataset_iter_index = 0
        pre_features_target_hats_datasets_iter_index = 0
        train_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                  self.num_workers,
                                                  cuda=True if self.availabel_cudas else False))
        exemplar_train_loader = iter(utils.get_data_loader(exemplar_set, self.batch_size,
                                                           self.num_workers,
                                                           cuda=True if self.availabel_cudas else False))
        pre_features_target_hats_train_loader = iter(utils.get_data_loader(pre_features_target_hats_datasets,
                                                                           self.batch_size,
                                                                           self.num_workers,
                                                                           cuda=True if self.availabel_cudas else False))
        train_imgs_num = len(train_loader)
        exemplar_train_imgs_num = len(exemplar_train_loader)
        pre_features_target_hats_num = len(pre_features_target_hats_train_loader)
        for epoch in range(self.MLP_epochs):
            if train_dataset_iter_index == train_imgs_num:
                train_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                          self.num_workers,
                                                          cuda=True if self.availabel_cudas else False))
                train_dataset_iter_index = 0
            if exemplar_dataset_iter_index == exemplar_train_imgs_num:
                exemplar_train_loader = iter(utils.get_data_loader(exemplar_set, self.batch_size,
                                                                   self.num_workers,
                                                                   cuda=True if self.availabel_cudas else False))
                exemplar_dataset_iter_index = 0
            if pre_features_target_hats_datasets_iter_index == pre_features_target_hats_num:
                pre_features_target_hats_train_loader = iter(utils.get_data_loader(pre_features_target_hats_datasets,
                                                                                   self.batch_size,
                                                                                   self.num_workers,
                                                                                   cuda=True if self.availabel_cudas else False))
                pre_features_target_hats_datasets_iter_index = 0
            iter_index = 0
            iters_left = max(train_imgs_num, pre_features_target_hats_num, exemplar_train_imgs_num)
            while iters_left > 0:
                iters_left -= 1
                optimizer.zero_grad()
                if train_dataset_iter_index == train_imgs_num:
                    train_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                              self.num_workers,
                                                              cuda=True if self.availabel_cudas else False))
                    train_dataset_iter_index = 0
                if exemplar_dataset_iter_index == exemplar_train_imgs_num:
                    exemplar_train_loader = iter(utils.get_data_loader(exemplar_set, self.batch_size,
                                                                       self.num_workers,
                                                                       cuda=True if self.availabel_cudas else False))
                    exemplar_dataset_iter_index = 0
                if pre_features_target_hats_datasets_iter_index == pre_features_target_hats_num:
                    pre_features_target_hats_train_loader = iter(
                        utils.get_data_loader(pre_features_target_hats_datasets,
                                              self.batch_size,
                                              self.num_workers,
                                              cuda=True if self.availabel_cudas else False))
                    pre_features_target_hats_datasets_iter_index = 0
                iter_index += 1
                '''读取数据'''
                imgs, labels = next(train_loader)
                exemplar_imgs, exemplar_labels = next(exemplar_train_loader)
                pre_features, pre_labels = next(pre_features_target_hats_train_loader)
                train_dataset_iter_index += 1
                exemplar_dataset_iter_index += 1
                pre_features_target_hats_datasets_iter_index += 1
                imgs, labels = imgs.to(self.MLP_device), labels.to(self.MLP_device)
                exemplar_imgs, exemplar_labels = exemplar_imgs.to(self.MLP_device), exemplar_labels.to(self.MLP_device)
                pre_features, pre_labels = pre_features.to(self.MLP_device), pre_labels.to(self.MLP_device)
                '''获取图片数据在FE_cls的输出'''
                imgs_2_features, imgs_2_targets = self.get_FE_cls_output(imgs)
                imgs_2_targets = imgs_2_targets[:, :(classes_per_task * task)]
                exemplar_imgs_2_features, exemplar_imgs_2_targets = self.get_FE_cls_output(exemplar_imgs)
                exemplar_imgs_2_targets = exemplar_imgs_2_targets[:, :(classes_per_task * task)]
                '''获取在要训练的模型FM的输出'''
                imgs_2_feature_hat, y_hat, _ = self.FM_cls_domain(imgs_2_features)
                pre_features_hat, pre_y_hat, _ = self.FM_cls_domain(pre_features)
                y_hat = y_hat[:, :(classes_per_task * task)]
                pre_y_hat = pre_y_hat[:, :(classes_per_task * task)]

                exemplar_imgs_2_feature_hat, exemplar_y_hat, _ = self.FM_cls_domain(exemplar_imgs_2_features)
                exemplar_y_hat = exemplar_y_hat[:, :(classes_per_task * task)]

                '''获取特定位置的输出'''
                y_hat_pre_tasks = y_hat[:, :(classes_per_task * (task - 1))]
                exemplar_y_hat_pre_tasks = exemplar_y_hat[:, :(classes_per_task * (task - 1))]

                pre_y_hat_pre_tasks = pre_y_hat[:, :(classes_per_task * (task - 1))]
                pre_features_targets = self.get_cls_target(pre_features)

                imgs_2_targets = imgs_2_targets[:, :(classes_per_task * (task - 1))]
                exemplar_imgs_2_targets = exemplar_imgs_2_targets[:, :(classes_per_task * (task - 1))]

                pre_features_targets = pre_features_targets[:, :(classes_per_task * (task - 1))]

                '''获取图片数据在Bias_layer的输出'''
                if task > 2:
                    assert Bias_layer is not None
                    with torch.no_grad():
                        imgs_2_targets = Bias_layer(imgs_2_targets, classes_per_task)
                        exemplar_imgs_2_targets = Bias_layer(exemplar_imgs_2_targets, classes_per_task)
                        pre_features_targets = Bias_layer(pre_features_targets, classes_per_task)
                if self.MLP_KD_temp_2 > 0:
                    '''binary loss distill'''
                    imgs_2_targets = torch.sigmoid(imgs_2_targets / self.MLP_KD_temp)
                    exemplar_imgs_2_targets = torch.sigmoid(exemplar_imgs_2_targets / self.MLP_KD_temp)
                    pre_features_targets = torch.sigmoid(pre_features_targets / self.MLP_KD_temp)
                    y_hat_pre_tasks /= self.MLP_KD_temp_2
                    exemplar_y_hat_pre_tasks /= self.MLP_KD_temp_2
                    pre_y_hat_pre_tasks /= self.MLP_KD_temp_2
                    loss_distill_current_imgs = Func.binary_cross_entropy_with_logits(input=y_hat_pre_tasks,
                                                                                      target=imgs_2_targets,
                                                                                      reduction='none').sum(
                        dim=1).mean()
                    loss_distill_current_exemplar_imgs = Func.binary_cross_entropy_with_logits(
                        input=exemplar_y_hat_pre_tasks,
                        target=exemplar_imgs_2_targets,
                        reduction='none').sum(
                        dim=1).mean()
                    loss_distill_features = Func.binary_cross_entropy_with_logits(input=pre_y_hat_pre_tasks,
                                                                                  target=pre_features_targets,
                                                                                  reduction='none').sum(dim=1).mean()
                    loss_distill = loss_distill_current_imgs + loss_distill_current_exemplar_imgs + \
                                   loss_distill_features
                else:
                    '''softTarget loss distill'''
                    imgs_2_targets = torch.softmax(imgs_2_targets / self.MLP_KD_temp, dim=1)
                    exemplar_imgs_2_targets = torch.softmax(exemplar_imgs_2_targets / self.MLP_KD_temp, dim=1)
                    pre_features_targets = torch.softmax(pre_features_targets / self.MLP_KD_temp, dim=1)
                    loss_distill_current_imgs = soft_target_criterion(y_hat_pre_tasks, imgs_2_targets,
                                                                      self.MLP_KD_temp)

                    loss_distill_current_exemplar_imgs = soft_target_criterion(exemplar_y_hat_pre_tasks,
                                                                               exemplar_imgs_2_targets,
                                                                               self.MLP_KD_temp)

                    loss_distill_features = soft_target_criterion(pre_y_hat_pre_tasks,
                                                                  pre_features_targets,
                                                                  self.MLP_KD_temp)
                    loss_distill = loss_distill_current_imgs + loss_distill_current_exemplar_imgs + \
                                   loss_distill_features
                '''loss classify'''
                '''cross entropy loss classify'''
                loss_cls = criterion(y_hat, labels) + criterion(exemplar_y_hat, exemplar_labels) + \
                           criterion(pre_y_hat, pre_labels)
                '''total loss '''
                loss_total = loss_cls + self.MLP_distill_rate * loss_distill * (self.MLP_KD_temp ** 2)
                loss_total.backward()
                optimizer.step()
                precision = None if y_hat is None else (labels == y_hat.max(1)[
                    1]).sum().item() / labels.size(0)
                lr = scheduler.get_last_lr()[0]
                ite_info = {
                    'task': task,
                    'epoch': epoch,
                    'lr': lr,
                    'loss_total': loss_total.item(),
                    'precision': precision if precision is not None else 0.,
                }
                self.batch_train_logger.info(
                    f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, "
                    f"lr: {lr}, loss_total: {loss_total}, acc: {precision}")
                print(ite_info)
                print("....................................")
            scheduler.step()
        acc1, acc5, throughput = self.current_task_validate_FM_cls_domain(test_datasets, task, active_classes)  # todo
        self.batch_train_logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.logger.info(
            f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
            f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapper_cls_domain train task : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))
        print("Train feature mapper_cls_domain End.")
        self.pre_FM_cls_domain = copy.deepcopy(self.FM_cls_domain).eval()
        self.train(mode=mode)
        pass

    def get_current_task_model_features(self, training_dataset):
        dataloader = utils.get_data_loader(training_dataset, self.batch_size, self.num_workers,
                                           cuda=self._is_on_cuda())
        first_entry = True
        features = None
        labels = None
        for image_batch, label_batch in dataloader:
            image_batch = image_batch.to(self._device())
            feature_batch = self.current_task_model_feature_extractor(image_batch).cpu()
            if first_entry:
                features = feature_batch
                labels = label_batch
                first_entry = False
            else:
                features = torch.cat([features, feature_batch], dim=0)
                labels = torch.cat([labels, label_batch], dim=0)
        return features.numpy(), labels.numpy()

    ####----MANAGING EXEMPLAR SETS----####

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]

    def reduce_features_sets(self, m):
        for y, P_y in enumerate(self.exemplar_feature_sets):
            self.exemplar_feature_sets[y] = P_y[:m]

    def update_features_sets(self, pre_tasks_features, features_per_class):
        # todo done!
        for index, feature_set in enumerate(pre_tasks_features):
            self.exemplar_feature_sets[index] = feature_set[: features_per_class]
        pass

    def construct_exemplar_feature_set(self, class_dataset, current_class_features, Exemplar_per_class,
                                       features_per_class):
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

        Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''
        # todo Done
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

                Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''
        # if Exemplar_per_class > features_per_class:
        #     raise ValueError("imgs_per_class must not surpass features_per_class.")
        n_max = len(class_dataset)
        print("n_max:{}".format(n_max))
        # self.logger.info("len(class_dataset):{}".format(n_max))
        reserve_num = min(features_per_class + Exemplar_per_class, n_max)
        exemplar_set = []
        # features = torch.Tensor(copy.deepcopy(current_class_features))
        features = copy.deepcopy(current_class_features)
        # current_class_features = torch.Tensor(current_class_features)
        exemplar_features = torch.zeros_like(features[:reserve_num])
        if self.herding:

            # calculate mean of all features
            class_mean = torch.mean(current_class_features, dim=0, keepdim=True)
            # one by one, select exemplar that makes mean of all exemplars as close to [class_mean] as possible
            list_of_selected = []
            for k in range(reserve_num):
                if k > 0:
                    exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                    features_means = (features + exemplar_sum) / (k + 1)
                    features_dists = features_means - class_mean
                else:
                    features_dists = features - class_mean
                index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1))
                if index_selected in list_of_selected:
                    raise ValueError("Exemplars should not be repeated!!!!")
                list_of_selected.append(index_selected)
                if len(exemplar_set) < Exemplar_per_class:
                    exemplar_set.append(class_dataset[index_selected][0].numpy())
                exemplar_features[k] = copy.deepcopy(features[index_selected])

                # make sure this example won't be selected again
                features[index_selected] = features[index_selected] + 100000
        else:
            indeces_selected = np.random.choice(n_max, size=reserve_num, replace=False)
            for k in range(min(features_per_class, n_max)):
                if len(exemplar_set) < Exemplar_per_class:
                    exemplar_set.append(class_dataset[k][0].numpy())
                exemplar_features[k] = copy.deepcopy(features[indeces_selected[k]])

        # add this [exemplar_set] as a [n]x[ich]x[isz]x[isz] to the list of [exemplar_sets]
        self.exemplar_sets.append(np.array(exemplar_set))
        if (features_per_class + Exemplar_per_class) <= n_max:
            self.exemplar_feature_sets.append(exemplar_features[Exemplar_per_class:reserve_num].numpy())
        else:
            self.exemplar_feature_sets.append(exemplar_features[-features_per_class:].numpy())
        pass

    def construct_exemplar_feature_set_hardsample(self, class_dataset, current_class_features, current_class_output,
                                                  Exemplar_per_class, features_per_class):
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

        Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''
        # todo Done
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

                Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''
        # if Exemplar_per_class > features_per_class:
        #     raise ValueError("imgs_per_class must not surpass features_per_class.")
        n_max = len(class_dataset)
        print("n_max:{}".format(n_max))
        # self.logger.info("len(class_dataset):{}".format(n_max))
        reserve_num = min(features_per_class + Exemplar_per_class, n_max)
        exemplar_set = []
        # features = torch.Tensor(copy.deepcopy(current_class_features))
        features = copy.deepcopy(current_class_features)
        # current_class_features = torch.Tensor(current_class_features)
        exemplar_features = torch.zeros_like(features[:reserve_num])
        if self.herding:

            # calculate mean of all features
            class_mean = torch.mean(current_class_features, dim=0, keepdim=True)
            # one by one, select exemplar that makes mean of all exemplars as close to [class_mean] as possible
            list_of_selected = []
            for k in range(reserve_num):
                if k > 0:
                    exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                    features_means = (features + exemplar_sum) / (k + 1)
                    features_dists = features_means - class_mean
                else:
                    features_dists = features - class_mean
                index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1))
                if index_selected in list_of_selected:
                    raise ValueError("Exemplars should not be repeated!!!!")
                list_of_selected.append(index_selected)
                if len(exemplar_set) < Exemplar_per_class:
                    exemplar_set.append(class_dataset[index_selected][0].numpy())
                exemplar_features[k] = copy.deepcopy(features[index_selected])

                # make sure this example won't be selected again
                features[index_selected] = features[index_selected] + 100000
        else:
            indeces_selected = np.random.choice(n_max, size=reserve_num, replace=False)
            for k in range(min(features_per_class, n_max)):
                if len(exemplar_set) < Exemplar_per_class:
                    exemplar_set.append(class_dataset[k][0].numpy())
                exemplar_features[k] = copy.deepcopy(features[indeces_selected[k]])

        # add this [exemplar_set] as a [n]x[ich]x[isz]x[isz] to the list of [exemplar_sets]
        self.exemplar_sets.append(np.array(exemplar_set))
        g = zip(current_class_output, current_class_features)
        g = sorted(g, reverse=True)
        current_class_features_new, current_class_output_new = zip(*g)
        self.exemplar_feature_sets.append(current_class_features_new[-features_per_class:].numpy())
        # if (features_per_class + Exemplar_per_class) <= n_max:
        #     self.exemplar_feature_sets.append(exemplar_features[Exemplar_per_class:reserve_num].numpy())
        # else:
        #     self.exemplar_feature_sets.append(exemplar_features[-features_per_class:].numpy())
        pass

    ####----CLASSIFICATION----####

    def classify_with_features(self, x, use_FM=False, allowed_classes=None):
        """Classify images by nearest-means-of-exemplars (after transform to feature representation)

        INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch
                    allowed_classes = None or <list> containing all "active classes" between which should be chosen

        OUTPUT:     preds = <tensor> of size (bsz,)"""
        # todo
        # Set model to eval()-mode
        mode = self.training
        self.eval()

        batch_size = x.size(0)

        # Do the exemplar-means need to be recomputed?
        # print("self.exemplar_feature_sets:", len(self.exemplar_feature_sets))
        # print("self.exemplar_feature_sets len:", len(self.exemplar_feature_sets[0]))
        if self.compute_means:
            exemplar_means = []  # --> list of 1D-tensors (of size [feature_size]), list is of length [n_classes]
            for ef in self.exemplar_feature_sets:
                exemplar_features = torch.from_numpy(ef).to(self.MLP_device)
                if self.norm_exemplars:
                    exemplar_features = F.normalize(exemplar_features, p=2, dim=1)
                # Calculate their mean and add to list
                mu_y = exemplar_features.mean(dim=0, keepdim=True)
                if self.norm_exemplars:
                    mu_y = F.normalize(mu_y, p=2, dim=1)
                exemplar_means.append(mu_y.squeeze())  # -> squeeze removes all dimensions of size 1
            # Update model's attributes
            self.exemplar_means = exemplar_means
            self.compute_means = False

        # Reorganize the [exemplar_means]-<tensor>
        exemplar_means = self.exemplar_means if allowed_classes is None else [
            self.exemplar_means[i] for i in allowed_classes
        ]
        means = torch.stack(exemplar_means)  # (n_classes, feature_size)
        means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)

        # Extract features for input data (and reorganize)
        # with torch.no_grad():
        # feature = self.feature_extractor(x)
        # if self.pre_FM_cls_domain is not None:
        #     feature = self.get_preFE_FM_features(x)
        # else:
        if use_FM and self.pre_FM_cls_domain is not None:
            feature = self.get_preFE_FM_features(x)
        else:
            feature = self.feature_extractor(x)  # (batch_size, feature_size)
        if self.norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

        # For each data-point in [x], find which exemplar-mean is closest to its extracted features
        dists = (feature - means).pow(2).sum(dim=1).squeeze()  # (batch_size, n_classes)
        _, preds = dists.min(1)

        # Set mode of model back
        self.train(mode=mode)

        return preds
        pass

    def classify_with_exemplars(self, x, allowed_classes=None):
        """Classify images by nearest-means-of-exemplars (after transform to feature representation)

        INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch
                    allowed_classes = None or <list> containing all "active classes" between which should be chosen

        OUTPUT:     preds = <tensor> of size (bsz,)"""

        # Set model to eval()-mode
        mode = self.training
        self.eval()

        batch_size = x.size(0)

        # Do the exemplar-means need to be recomputed?
        if self.compute_means:
            exemplar_means = []  # --> list of 1D-tensors (of size [feature_size]), list is of length [n_classes]
            for P_y in self.exemplar_sets:
                exemplars = []
                # Collect all exemplars in P_y into a <tensor> and extract their features
                for ex in P_y:
                    exemplars.append(torch.from_numpy(ex))
                exemplars = torch.stack(exemplars).to(self._device())
                with torch.no_grad():
                    features = self.feature_extractor(exemplars)
                if self.norm_exemplars:
                    features = F.normalize(features, p=2, dim=1)
                # Calculate their mean and add to list
                mu_y = features.mean(dim=0, keepdim=True)
                if self.norm_exemplars:
                    mu_y = F.normalize(mu_y, p=2, dim=1)
                exemplar_means.append(mu_y.squeeze())  # -> squeeze removes all dimensions of size 1
            # Update model's attributes
            self.exemplar_means = exemplar_means
            self.compute_means = False

        # Reorganize the [exemplar_means]-<tensor>
        exemplar_means = self.exemplar_means if allowed_classes is None else [
            self.exemplar_means[i] for i in allowed_classes
        ]
        means = torch.stack(exemplar_means)  # (n_classes, feature_size)
        means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)

        # Extract features for input data (and reorganize)
        with torch.no_grad():
            feature = self.feature_extractor(x)  # (batch_size, feature_size)
        if self.norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

        # For each data-point in [x], find which exemplar-mean is closest to its extracted features
        dists = (feature - means).pow(2).sum(dim=1).squeeze()  # (batch_size, n_classes)
        _, preds = dists.min(1)

        # Set mode of model back
        self.train(mode=mode)

        return preds

    def classify_with_exemplars_plus_features(self, x, allowed_classes=None):
        """Classify images by nearest-means-of-exemplars (after transform to feature representation)

        INPUT:      x = <tensor> of size (bsz,ich,isz,isz) with input image batch
                    allowed_classes = None or <list> containing all "active classes" between which should be chosen

        OUTPUT:     preds = <tensor> of size (bsz,)"""

        # Set model to eval()-mode
        mode = self.training
        self.eval()

        batch_size = x.size(0)

        # Do the exemplar-means need to be recomputed?
        assert len(self.exemplar_sets) == len(self.exemplar_feature_sets)
        length = len(self.exemplar_sets)
        if self.compute_means:
            exemplar_means = []  # --> list of 1D-tensors (of size [feature_size]), list is of length [n_classes]
            for i in range(length):
                # for P_y in self.exemplar_sets:
                exemplars = []
                # Collect all exemplars in P_y into a <tensor> and extract their features
                for ex in self.exemplar_sets[i]:
                    exemplars.append(torch.from_numpy(ex))
                exemplars = torch.stack(exemplars).to(self._device())
                with torch.no_grad():
                    features = self.feature_extractor(exemplars)
                features = torch.cat((features, torch.from_numpy(self.exemplar_feature_sets[i]).to(self._device())),
                                     dim=0)
                if self.norm_exemplars:
                    features = F.normalize(features, p=2, dim=1)
                # Calculate their mean and add to list
                mu_y = features.mean(dim=0, keepdim=True)
                if self.norm_exemplars:
                    mu_y = F.normalize(mu_y, p=2, dim=1)
                exemplar_means.append(mu_y.squeeze())  # -> squeeze removes all dimensions of size 1
            # Update model's attributes
            self.exemplar_means = exemplar_means
            self.compute_means = False

        # Reorganize the [exemplar_means]-<tensor>
        exemplar_means = self.exemplar_means if allowed_classes is None else [
            self.exemplar_means[i] for i in allowed_classes
        ]
        means = torch.stack(exemplar_means)  # (n_classes, feature_size)
        means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)

        # Extract features for input data (and reorganize)
        with torch.no_grad():
            feature = self.feature_extractor(x)  # (batch_size, feature_size)
        if self.norm_exemplars:
            feature = F.normalize(feature, p=2, dim=1)
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

        # For each data-point in [x], find which exemplar-mean is closest to its extracted features
        dists = (feature - means).pow(2).sum(dim=1).squeeze()  # (batch_size, n_classes)
        _, preds = dists.min(1)

        # Set mode of model back
        self.train(mode=mode)

        return preds

    def EFAfIL_classify(self, x, classifier, active_classes, task, use_FM=False, allowed_classes=None):
        if classifier == "feature_ncm":
            return self.classify_with_features(x, use_FM=use_FM, allowed_classes=allowed_classes)
        elif classifier == "examplar_ncm":
            return self.classify_with_exemplars(x, allowed_classes=allowed_classes)
        elif classifier == "examplar_feature_ncm":
            return self.classify_with_exemplars_plus_features(x, allowed_classes=allowed_classes)
        elif classifier == "fc":
            self.FM_cls_domain.eval()
            with torch.no_grad():
                if task > 1:
                    targets = self.FM_cls_domain(self.get_preFE_feature(x))[-2]
                else:
                    targets = self.get_FE_cls_target(x)
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    targets = targets[:, class_entries]
            _, predicted = torch.max(targets, 1)
            return predicted
        elif classifier == "linearSVM":
            # if task > 1:
            #     features = self.get_preFE_FM_features(x)
            # else:
            if use_FM and task > 1:
                features = self.get_preFE_FM_features(x)
            else:
                features = self.feature_extractor(x)
            if self.norm_exemplars:
                features = F.normalize(features, p=2, dim=1)
            return torch.from_numpy(self.svm.predict(features.cpu().numpy())).to(self.MLP_device)
        elif classifier == "fcls":
            self.eval()
            with torch.no_grad():
                _, targets = self(x)
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                targets = targets[:, class_entries]
            _, predicted = torch.max(targets, 1)
            return predicted
        else:
            raise ValueError("classifier must be ncm/linearSVM/fc/fcls.")

    def current_task_validate_FM_cls_domain(self, test_datasets, task, active_classes, val_current_task=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        if val_current_task:
            val_loader = utils.get_data_loader(test_datasets[task - 1], self.batch_size,  # task index must minus 1
                                               self.num_workers,
                                               cuda=True if self.availabel_cudas else False)
        else:
            val_dataset = test_datasets[0]
            for i in range(1, task - 1):
                val_dataset = ConcatDataset([val_dataset, test_datasets[i]])
            val_loader = utils.get_data_loader(val_dataset, self.batch_size,  # task index must minus 1
                                               self.num_workers,
                                               cuda=True if self.availabel_cudas else False)
        end = time.time()
        mode = self.FM_cls_domain.training
        self.FM_cls_domain.eval()
        for inputs, labels in val_loader:
            data_time.update(time.time() - end)
            inputs, labels = inputs.to(self.MLP_device), labels.to(self.MLP_device)
            with torch.no_grad():
                _, y_hat, _ = self.FM_cls_domain(self.feature_extractor(inputs))
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
        self.FM_cls_domain.train(mode=mode)
        if len(active_classes) >= 5:
            return top1.avg, top5.avg, throughput
        else:
            return top1.avg, 0, throughput
        pass

    def current_task_validate_FM_Retrain(self, test_datasets, task, active_classes, use_NewfeatureSpace,
                                         val_current_task=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        if val_current_task:
            val_loader = utils.get_data_loader(test_datasets[task - 1], self.batch_size,  # task index must minus 1
                                               self.num_workers,
                                               cuda=True if self.availabel_cudas else False)
        else:
            val_dataset = test_datasets[0]
            for i in range(1, task - 1):
                val_dataset = ConcatDataset([val_dataset, test_datasets[i]])
            val_loader = utils.get_data_loader(val_dataset, self.batch_size,  # task index must minus 1
                                               self.num_workers,
                                               cuda=True if self.availabel_cudas else False)
        # val_loader = utils.get_data_loader(test_datasets[task - 1], self.batch_size,  # task index must minus 1
        #                                    self.num_workers,
        #                                    cuda=True if self.availabel_cudas else False)
        end = time.time()
        self.FM_cls_domain.eval()
        for inputs, labels in val_loader:
            data_time.update(time.time() - end)
            inputs, labels = inputs.to(self.MLP_device), labels.to(self.MLP_device)
            with torch.no_grad():
                if use_NewfeatureSpace:
                    features = self.feature_mapping(self.pre_FM_mapping(self.get_preFE_feature(inputs)))
                else:
                    features = self.feature_mapping(self.get_preFE_feature(inputs))
                y_hat = self.get_cls_target(features)
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
        if len(active_classes) >= 5:
            return top1.avg, top5.avg, throughput
        else:
            return top1.avg, 0, throughput
        pass

    def current_task_validate_bias_Cls(self, val_datasets, task, active_classes, current_classes_num):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        val_loader = utils.get_data_loader(val_datasets[task - 1], self.batch_size, self.num_workers,
                                           cuda=True if self.availabel_cudas else False)
        mode = self.training
        # switch to evaluate mode
        self.eval()
        self.unbias_cls_of_FMcls.eval()
        with torch.no_grad():
            end = time.time()
            for inputs, labels in val_loader:
                data_time.update(time.time() - end)
                inputs, labels = inputs.to(self.MLP_device), labels.to(self.MLP_device)
                features = self.feature_extractor(inputs)
                y_hat = self.FM_cls_forward_nograd(features)
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    y_hat = y_hat[:, class_entries]
                y_hat = self.unbias_cls_of_FMcls(y_hat, current_classes_num)
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

    def current_task_validate_cRT_Cls(self, val_datasets, task, active_classes):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        val_loader = utils.get_data_loader(val_datasets[task - 1], self.batch_size, self.num_workers,
                                           cuda=True if self.availabel_cudas else False)
        mode = self.training
        # switch to evaluate mode
        self.eval()
        self.unbias_cls_of_FMcls.eval()
        with torch.no_grad():
            end = time.time()
            for inputs, labels in val_loader:
                data_time.update(time.time() - end)
                inputs, labels = inputs.to(self.MLP_device), labels.to(self.MLP_device)
                features = self.feature_extractor(inputs)
                y_hat = self.unbias_cls_of_FMcls(features)
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
        pass
