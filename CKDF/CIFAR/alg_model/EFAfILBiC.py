"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
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
from public import utils
from public.data import ExemplarDataset, get_multitask_experiment, get_dataset, AVAILABLE_TRANSFORMS
from exemplars import EFAfIL_FeaturesHandler, FeaturesHandler, ExemplarHandler
from public.util_models import FE_cls, SoftTarget_CrossEntropy, BiasLayer

# -------------------------------------------------------------------------------------------------#

# --------------------#
# ----- EFAfIL -----#
# --------------------
from public.utils import AverageMeter, accuracy


class EFAfILBiC(EFAfIL_FeaturesHandler):
    def __init__(self, model_name, MLP_name, dataset_name,
                 dataset_path, num_classes, rate, tasks,
                 logger, batch_train_logger, result_file, use_exemplars,
                 hidden_size, Exemple_memory_budget,
                 Feature_memory_budget, optim_type, MLP_optim_type,
                 norm_exemplars, herding, FM_reTrain, use_NewfeatureSpace, batch_size,
                 num_workers, seed, availabel_cudas,

                 epochs, CNN_lr, CNN_momentum,
                 CNN_weight_decay, CNN_milestones,
                 kd_lamb, fd_gamma, lrgamma, KD_temp,

                 MLP_lr, MLP_momentum,
                 MLP_epochs, MLP_milestones,
                 svm_sample_type,
                 MLP_weight_decay,
                 MLP_lrgamma, sim_alpha, svm_max_iter, use_FM, oversample):
        EFAfIL_FeaturesHandler.__init__(self, MLP_name, num_classes, hidden_size, Exemple_memory_budget, num_workers,
                                        Feature_memory_budget, norm_exemplars, herding, batch_size, sim_alpha, MLP_lr,
                                        MLP_momentum, MLP_milestones, MLP_lrgamma, MLP_weight_decay, MLP_epochs,
                                        MLP_optim_type,
                                        KD_temp, svm_sample_type, svm_max_iter, availabel_cudas, logger,
                                        batch_train_logger)
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.rate = rate
        self.tasks = tasks
        self.logger = logger
        self.batch_train_logger = batch_train_logger
        self.result_file = result_file
        self.use_exemplars = use_exemplars
        self.hidden_size = hidden_size
        self.optim_type = optim_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.availabel_cudas = availabel_cudas
        self.device = "cuda" if self.availabel_cudas else "cpu"
        self.FM_reTrain = FM_reTrain
        self.use_FM = use_FM
        self.use_NewfeatureSpace = use_NewfeatureSpace
        self.oversample = oversample

        self.epochs = epochs
        self.lr = CNN_lr
        self.momentum = CNN_momentum
        self.weight_decay = CNN_weight_decay
        self.milestones = CNN_milestones
        self.kd_lamb = kd_lamb
        self.fd_gamma = fd_gamma
        self.gamma = lrgamma
        self.KD_temp = KD_temp

        self.train_datasets, self.val_datasets, self.data_config, self.classes_per_task = \
            self.get_dataset(dataset_name)
        self.pre_FE_cls = None
        self.pre_bias_layer = None
        # self.FE_cls = None
        self.FE_cls = self.construct_model(rate)
        self.bias_layer = None

    def forward(self, x):
        final_features, target = self.FE_cls(x)
        return final_features, target

    def get_dataset(self, dataset_name):
        (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
            name=dataset_name, tasks=self.tasks, data_dir=self.dataset_path,
            exception=True if self.seed == 0 else False,
        )
        return train_datasets, test_datasets, config, classes_per_task

    def construct_model(self, rate):
        # if self.availabel_cudas:
        #     # os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
        #     # device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
        #     model = FE_cls(resnetforcifar.__dict__[self.model_name](rate=rate, get_feature=True), self.hidden_size,
        #                    self.num_classes)
        #     # model.load_state_dict(
        #     #     torch.load("/share/home/kcli/CL_research/iCaRL_ILtFA/pretrain_models/cifar10_pretrain_1_4.pth"))
        #     model = torch.nn.DataParallel(model).cuda()
        # else:
        #     model = FE_cls(resnetforcifar.__dict__[self.model_name](rate=rate, get_feature=True), self.hidden_size,
        #                    self.class_num)
        # model.load_state_dict(
        #     torch.load("/share/home/kcli/CL_research/iCaRL_ILtFA/pretrain_models/cifar10_pretrain_1_4.pth"))
        model = torch.load("/share/home/kcli/CL_research/iCaRL_ILtFA/pretrain_models/cifar10_pretrain_1_4.pth")
        print(type(model))
        print(model)
        return model

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

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False

    def get_preFE_feature(self, images):
        if self.pre_FE_cls is not None:
            with torch.no_grad():
                features = self.pre_FE_cls(images)[-2]
                return features
        else:
            raise ValueError("The pre feature extractor is None.")

    def get_FE_cls_output(self, images):
        mode = self.training
        self.eval()
        with torch.no_grad():
            features, targets = self.FE_cls(images)
        self.train(mode=mode)
        return features, targets

    def feature_extractor(self, images):
        mode = self.training
        self.eval()
        with torch.no_grad():
            features = self.FE_cls(images)[-2]
        self.train(mode=mode)
        return features

    def get_cls_target(self, features):
        if type(self.FE_cls) is torch.nn.DataParallel:
            return self.FE_cls.module.get_cls_results(features)
        else:
            return self.FE_cls.get_cls_results(features)

    def get_FE_cls_target(self, x):
        mode = self.training
        self.eval()
        with torch.no_grad():
            _, targets = self.FE_cls(x)
        self.train(mode=mode)
        return targets

    def FE_cls_forward(self, images):
        mode = self.FE_cls.training
        self.FE_cls.eval()
        with torch.no_grad():
            targets = self.FE_cls(images)[-1]
        self.FE_cls.train(mode=mode)
        return targets

    def get_preFE_FM_feature_target(self, x):
        prefeatures = self.get_preFE_feature(x)
        self.FM_cls_domain.eval()
        with torch.no_grad():
            FM_features, FM_targets, domain = self.FM_cls_domain(prefeatures)
        return FM_features, FM_targets

    def train_FE_cls(self, args, train_dataset, active_classes):
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
                    data_loader = iter(utils.get_data_loader(train_dataset, self.batch_size, self.num_workers,
                                                             cuda=True if self.availabel_cudas else False))
                    iter_num = iters_left = len(data_loader)
                    continue

                # -----------------Collect data------------------#

                #####-----CURRENT BATCH-----#####
                x, y = next(data_loader)  # --> sample training data of current task
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                loss_dict = self.train_FE_cls_a_batch(x, y, optimizer, active_classes)
                iter_index += 1
                if iter_index % args.print_interval == 0:
                    self.batch_train_logger.info(
                        f"Task {1}, batch train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], "
                        f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: "
                        f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"
                        f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
                    )
                    print(loss_dict)
            scheduler.step()
            acc1, acc5, throughput = self.current_task_validate(1, active_classes)
            self.batch_train_logger.info(
                f"batch train FE_cls || task: {1}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
                f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
            )
            self.batch_train_logger.info(f"---------------------------------------------------------------")
            print(f'batch train task  FE_cls: {1}, epoch: {epoch} 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))

    def train_FE_cls_a_batch(self, x, y, optimizer, active_classes):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <tensor> batch of corresponding labels
        [scores]          None or <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x]
                            NOTE: only to be used for "BCE with distill" (only when scenario=="class")
        [rnt]             <number> in [0,1], relative importance of new task
        [active_classes]  None or (<list> of) <list> with "active" classes
        [task]            <int>, for setting task-specific mask'''

        # Set model to training-mode
        self.train()
        # Reset optimizer
        optimizer.zero_grad()
        loss_total = None
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        # Run model
        features, y_hat = self(x)
        # -if needed, remove predictions for classes not in current task
        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
            y_hat = y_hat[:, class_entries]
        binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
        predL = None if y is None else Func.binary_cross_entropy_with_logits(
            input=y_hat, target=binary_targets, reduction='none'
        ).sum(dim=1).mean()  # --> sum over classes, then average over batch
        loss_total = predL
        if len(active_classes) >= 5:
            acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
            top1.update(acc1.item(), x.size(0))
            top5.update(acc5.item(), x.size(0))
        else:
            acc1 = accuracy(y_hat, y, topk=(1,))[0]
            top1.update(acc1.item(), x.size(0))
        losses.update(loss_total, x.size(0))
        # Calculate training-precision
        precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
        loss_total.backward()
        # Take optimization-step
        optimizer.step()
        # Return the dictionary with different training-loss split in categories
        if len(active_classes) >= 5:
            return {
                'task': 1,
                'loss_total': loss_total.item(),
                'precision': precision if precision is not None else 0.,
                "top1": top1.avg,
                "top5": top5.avg,
                "losses": losses.avg
            }
        else:
            return {
                'task': 1,
                'loss_total': loss_total.item(),
                'precision': precision if precision is not None else 0.,
                "top1": top1.avg,
                "top5": 0,
                "losses": losses.avg
            }

    def train_main(self, args):
        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''
        print("seed:", self.seed)
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            cudnn.deterministic = True

        gpus = torch.cuda.device_count()
        cudnn.benchmark = True
        cudnn.enabled = True
        # Set model in training-mode
        # Loop over all tasks.
        self.logger.info(f'use {gpus} gpus')
        self.logger.info(f"args: {args}")
        EFAfIL_result = {"timestamp": str(datetime.datetime.now()), "model_rate": self.rate}
        EFAfIL_result.update(self.data_config)
        for task, train_dataset in enumerate(self.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')
            self.batch_train_logger.info(f'New task {task} begin.')
            print("New task %d begin." % task)
            # Add exemplars (if available) to current dataset (if requested)
            if self.use_exemplars and task > 1:
                exemplar_dataset = ExemplarDataset(self.exemplar_sets)
                training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
            else:
                training_dataset = train_dataset

            active_classes = list(range(self.classes_per_task * task))
            if task > 1:
                self.EFAfIL_split_feature_mapper_cls_domain_train(training_dataset, self.val_datasets,
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
                                                      cuda=True if self.availabel_cudas else False,
                                                      drop_last=True))
                            iter_num = iters_left = len(data_loader)
                            continue

                        # -----------------Collect data------------------#

                        #####-----CURRENT BATCH-----#####
                        x, y = next(data_loader)  # --> sample training data of current task
                        x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                        FM_features, FM_targets = self.get_preFE_FM_feature_target(x)
                        # ---> Train MAIN MODEL
                        # Train the main model with this batch
                        loss_dict = self.train_a_batch(x, y, FM_targets=FM_targets, FM_features=FM_features,
                                                       optimizer=optimizer, active_classes=active_classes, task=task)
                        iter_index += 1
                        if iter_index % args.print_interval == 0:
                            self.batch_train_logger.info(
                                f"Task {task:0>3d}, batch train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], "
                                f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: "
                                f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"
                                f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
                            )
                    scheduler.step()
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes)
                    results = f"batch train FE_cls || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, " \
                              f"top5 acc:  {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    self.batch_train_logger.info(
                        results
                    )
                    print(results)
                examplars_per_class = int(np.floor(self.Exemple_memory_budget / (self.classes_per_task * task)))
                self.reduce_exemplar_sets(examplars_per_class)
            elif task == 1:
                # self.FE_cls = torch.load("task_1_FE_cls.pth")
                self.FE_cls = torch.load("cifar10_preTrain_task_1_FE_cls.pth")
                # # continue
                print("model:", self.FE_cls)
                # self.train_FE_cls(args, train_dataset, active_classes)
                # torch.save(self.FE_cls, "./imagenet100_preTrain_task_1_FE_cls.pth")
            # else:
            #     break
            self.feature_handle_main(train_dataset, self.classes_per_task, task)  # todo Done
            self.batch_train_logger.info(f'##########feature extractor train task {task} End.#########')
            self.logger.info(f'#############feature extractor train task {task} End.##############')
            self.logger.info(f'#############task {task} validate  begin.##############')
            print("feature extractor train task-%d End" % (task))
            acc_past_tasks, acc_list = self.tasks_validate(task)
            EFAfIL_result["task_{}_results".format(task)] = acc_past_tasks
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="ncm")
            EFAfIL_result["task_{}_ncm_results".format(task)] = acc_past_tasks
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="fc", active_classes=active_classes)
            EFAfIL_result["task_{}_fc_results".format(task)] = acc_past_tasks
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="fcls", active_classes=active_classes)
            EFAfIL_result["task_{}_FE_cls_results".format(task)] = acc_past_tasks
            # print(ILtFA_result_temp)
            # if task == 2 and self.pre_FE_cls:
            #     torch.save(self.pre_FE_cls, "./pre_FE_cls_examplar_0_back.pth")
            #     torch.save(self.FE_cls, "./FE_cls_examplar_0_back.pth")
        with open(self.result_file, 'w') as fw:
            json.dump(EFAfIL_result, fw, indent=4)

    def train_a_batch(self, x, y, FM_targets, FM_features, optimizer, active_classes, task):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <tensor> batch of corresponding labels
        [scores]          None or <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x]
                            NOTE: only to be used for "BCE with distill" (only when scenario=="class")
        [rnt]             <number> in [0,1], relative importance of new task
        [active_classes]  None or (<list> of) <list> with "active" classes
        [task]            <int>, for setting task-specific mask'''

        # Set model to training-mode
        self.train()
        # Reset optimizer
        optimizer.zero_grad()
        loss_total = None
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        loss_func = torch.nn.MSELoss(reduction='mean')
        criteria = torch.nn.CrossEntropyLoss()
        # Run model
        features, y_hat = self(x)
        # -if needed, remove predictions for classes not in current task
        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
            y_hat = y_hat[:, class_entries]
            # FM_targets = FM_targets[:, class_entries]
        scores_hats = FM_targets[:, :(self.classes_per_task * (task - 1))]
        scores_hats = torch.sigmoid(scores_hats / self.KD_temp)
        # scores_hats = torch.sigmoid(FM_targets / self.KD_temp)
        binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
        binary_targets = binary_targets[:, -self.classes_per_task:]
        binary_targets = torch.cat([scores_hats, binary_targets], dim=1)
        loss_cls_distill = None if y is None else Func.binary_cross_entropy_with_logits(
            input=y_hat, target=binary_targets, reduction='none'
        ).sum(dim=1).mean()
        # loss_sim = loss_func(features, FM_features)
        loss_sim = 1 - torch.cosine_similarity(features, FM_features).mean()
        # loss_total = loss_cls_distill + self.fd_gamma * loss_sim
        loss_total = loss_cls_distill + loss_sim
        loss_total.backward()
        # Take optimization-step
        optimizer.step()
        # loss_total = self.kd_lamb * loss_distill + self.fd_gamma * loss_sim + loss_cls
        if len(active_classes) >= 5:
            acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
            top1.update(acc1.item(), x.size(0))
            top5.update(acc5.item(), x.size(0))
        else:
            acc1 = accuracy(y_hat, y, topk=(1,))[0]
            top1.update(acc1.item(), x.size(0))
        losses.update(loss_total, x.size(0))
        # Calculate training-precision
        precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
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

    def feature_replay_train_main(self, args):
        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''
        print("seed:", self.seed)
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            cudnn.deterministic = True

        gpus = torch.cuda.device_count()
        cudnn.benchmark = True
        cudnn.enabled = True
        # Set model in training-mode
        # Loop over all tasks.
        self.logger.info(f'use {gpus} gpus')
        self.logger.info(f"args: {args}")
        EFAfILBiC_result = {"timestamp": str(datetime.datetime.now()), "model_rate": self.rate}
        EFAfILBiC_result.update(self.data_config)
        for task, train_dataset in enumerate(self.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')
            self.batch_train_logger.info(f'New task {task} begin.')
            print("New task %d begin." % task)
            # Add exemplars (if available) to current dataset (if requested)
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            per_task_train_dataset, per_task_val_dataset = torch.utils.data.random_split(train_dataset,
                                                                                         [train_size,
                                                                                          val_size])
            if task > 1 and self.Exemple_memory_budget > 0:
                copy_exemplar_sets = copy.deepcopy(self.exemplar_sets)
                exemplar_dataset = ExemplarDataset(copy_exemplar_sets)
                exemplar_train_size = int(0.9 * len(exemplar_dataset))
                exemplar_val_size = len(exemplar_dataset) - exemplar_train_size
                examplar_train_dataset, examplar_val_dataset = torch.utils.data.random_split(exemplar_dataset,
                                                                                             [exemplar_train_size,
                                                                                              exemplar_val_size])
                per_task_training_dataset = ConcatDataset([per_task_train_dataset, examplar_train_dataset])
                training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
                examplar_val_sample_num_per_class = exemplar_val_size / (self.classes_per_task * (task - 1))
                current_val_sample_num_per_class = val_size / self.classes_per_task
                dif_rate = int(current_val_sample_num_per_class / examplar_val_sample_num_per_class)
                if self.oversample:
                    per_task_valing_dataset = ConcatDataset([per_task_val_dataset, examplar_val_dataset])
                    for i in range(1, dif_rate):
                        per_task_valing_dataset = ConcatDataset([per_task_valing_dataset, examplar_val_dataset])
                else:
                    current_val_samples_size = int(examplar_val_sample_num_per_class * self.classes_per_task)
                    giveup_size = len(per_task_val_dataset) - current_val_samples_size
                    per_task_valing_dataset, _ = torch.utils.data.random_split(per_task_val_dataset,
                                                                               [current_val_samples_size,
                                                                                giveup_size])
                    per_task_valing_dataset = ConcatDataset([per_task_valing_dataset, examplar_val_dataset])

            else:
                per_task_training_dataset = per_task_train_dataset
                training_dataset = train_dataset
                per_task_valing_dataset = per_task_val_dataset

            active_classes = list(range(self.classes_per_task * task))
            if task > 1:
                self.EFAfIL_split_feature_mapper_cls_domain_train(training_dataset, self.val_datasets,
                                                                  self.classes_per_task,
                                                                  active_classes, task)
                self.pre_FE_cls = copy.deepcopy(self.FE_cls).eval()
                optimizer = self.build_optimize()
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=self.milestones, gamma=self.gamma)
                '''get feature_replay data loader'''
                feature_replay_datasets = ExemplarDataset(self.exemplar_feature_sets)
                feature_replay_dataset_index = 0
                feature_replay_data_loader = iter(utils.get_data_loader(feature_replay_datasets, self.batch_size,
                                                                        self.num_workers,
                                                                        cuda=True if self.availabel_cudas else False))
                feature_replay_dataset_num = len(feature_replay_data_loader)
                extra_index = 0
                print("feature_replay_dataset_num:", feature_replay_dataset_num)

                for epoch in range(1, self.epochs + 1):
                    is_first_ite = True
                    iters_left = 1
                    iter_index = 0

                    '''if feature_replay end, get new feature_replay data loader'''
                    if feature_replay_dataset_index == feature_replay_dataset_num:
                        feature_replay_data_loader = iter(
                            utils.get_data_loader(feature_replay_datasets, self.batch_size, self.num_workers,
                                                  cuda=True if self.availabel_cudas else False))
                        feature_replay_dataset_index = 0

                    while iters_left > 0:
                        # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                        iters_left -= 1
                        iter_num = 0
                        if is_first_ite:
                            is_first_ite = False
                            data_loader = iter(
                                utils.get_data_loader(per_task_training_dataset, self.batch_size, self.num_workers,
                                                      cuda=True if self.availabel_cudas else False))
                            iter_num = iters_left = len(data_loader)
                            continue

                        # -----------------Collect data------------------#

                        #####-----CURRENT BATCH-----#####
                        x, y = next(data_loader)  # --> sample training data of current task

                        '''if feature_replay end, get new feature_replay data loader'''
                        if feature_replay_dataset_index == feature_replay_dataset_num:
                            feature_replay_data_loader = iter(
                                utils.get_data_loader(feature_replay_datasets, self.batch_size, self.num_workers,
                                                      cuda=True if self.availabel_cudas else False))
                            feature_replay_dataset_index = 0
                        feature_replay_features, feature_replay_labels = next(feature_replay_data_loader)
                        feature_replay_dataset_index += 1
                        x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                        feature_replay_features, feature_replay_labels = feature_replay_features.to(
                            self.device), feature_replay_labels.to(self.device)
                        '''获取图片在pre_FEcls跟FM模型的输出'''
                        FM_features, FM_targets = self.get_preFE_FM_feature_target(x)
                        '''获取保存的features在FM模型的输出'''
                        feature_replay_FM_features = self.feature_mapping(feature_replay_features)
                        feature_replay_FM_targets = self.prefeature_2_FMtarget(feature_replay_FM_features)
                        # ---> Train MAIN MODEL
                        # Train the main model with this batch
                        loss_dict = self.train_a_batch_feature_replay(x, y, feature_replay_features,
                                                                      feature_replay_labels, FM_targets=FM_targets,
                                                                      FM_features=FM_features,
                                                                      feature_replay_FM_features=feature_replay_FM_features,
                                                                      feature_replay_FM_targets=feature_replay_FM_targets,
                                                                      optimizer=optimizer,
                                                                      active_classes=active_classes, task=task)
                        iter_index += 1
                        if iter_index % args.print_interval == 0:
                            self.batch_train_logger.info(
                                f"Task {task:0>3d}, batch train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], "
                                f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: "
                                f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"
                                f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
                            )
                    scheduler.step()
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes)
                    results = f"batch train FE_cls || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, " \
                              f"top5 acc:  {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    self.batch_train_logger.info(
                        results
                    )
                    print(results)
            elif task == 1:
                self.train_FE_cls(args, training_dataset, active_classes)

            # self.cifar100_feature_handle_main(train_dataset, self.classes_per_task, task)  # todo Done
            self.cifar100_feature_handle_main_FM(training_dataset, train_dataset, self.val_datasets,
                                                 self.classes_per_task,
                                                 active_classes, task, use_dynamicMem=True, FM_reTrain=True,
                                                 use_NewfeatureSpace=self.use_NewfeatureSpace)
            if task > 1:
                self.train_BiasLayer_feature_replay(per_task_valing_dataset, task, active_classes, args.print_interval)
            self.batch_train_logger.info(f'##########feature extractor train task {task} End.#########')
            self.logger.info(f'#############feature extractor train task {task} End.##############')
            self.logger.info(f'#############task {task} validate  begin.##############')
            print("feature extractor train task-%d End" % (task))
            if self.Exemple_memory_budget > 0:
                acc_past_tasks, acc_list = self.tasks_validate(task, classifier="ncm")
                EFAfILBiC_result["task_{}_examplar_ncm_results".format(task)] = acc_past_tasks
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="fc")
            EFAfILBiC_result["task_{}_FE_cls_results".format(task)] = acc_past_tasks
            if self.compute_means:
                self.batch_train_logger.info(
                    "self.compute_means is True"
                )
            else:
                self.batch_train_logger.info(
                    "self.compute_means is False"
                )
            with open(self.result_file, 'w') as fw:
                json.dump(EFAfILBiC_result, fw, indent=4)
        with open(self.result_file, 'w') as fw:
            json.dump(EFAfILBiC_result, fw, indent=4)

    def feature_replay_train_main_softTargetCrossEntropy(self, args):
        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''
        print("seed:", self.seed)
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            cudnn.deterministic = True

        gpus = torch.cuda.device_count()
        cudnn.benchmark = True
        cudnn.enabled = True
        # Set model in training-mode
        # Loop over all tasks.
        self.logger.info(f'use {gpus} gpus')
        self.logger.info(f"args: {args}")
        EFAfIL_result = {"timestamp": str(datetime.datetime.now()), "model_rate": self.rate}
        EFAfIL_result.update(self.data_config)
        for task, train_dataset in enumerate(self.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')
            self.batch_train_logger.info(f'len(train_datasets):{len(self.train_datasets)} New task {task} begin.')
            print("New task %d begin." % task)
            # Add exemplars (if available) to current dataset (if requested)
            if self.use_exemplars and task > 1:
                exemplar_dataset = ExemplarDataset(self.exemplar_sets)
                training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
            else:
                training_dataset = train_dataset

            active_classes = list(range(self.classes_per_task * task))
            if task > 1:
                self.EFAfIL_split_feature_mapper_cls_domain_train_softTargetCrossEntropy(training_dataset,
                                                                                         self.val_datasets,
                                                                                         self.classes_per_task,
                                                                                         active_classes, task)
                self.pre_FE_cls = copy.deepcopy(self.FE_cls).eval()
                optimizer = self.build_optimize()
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=self.milestones, gamma=self.gamma)
                '''get feature_replay data loader'''
                feature_replay_datasets = ExemplarDataset(self.exemplar_feature_sets)
                feature_replay_dataset_index = 0
                feature_replay_data_loader = iter(utils.get_data_loader(feature_replay_datasets, self.batch_size,
                                                                        self.num_workers,
                                                                        cuda=True if self.availabel_cudas else False))
                feature_replay_dataset_num = len(feature_replay_data_loader)
                extra_index = 0
                print("feature_replay_dataset_num:", feature_replay_dataset_num)

                for epoch in range(1, self.epochs + 1):
                    is_first_ite = True
                    iters_left = 1
                    iter_index = 0

                    '''if feature_replay end, get new feature_replay data loader'''
                    if feature_replay_dataset_index == feature_replay_dataset_num:
                        feature_replay_data_loader = iter(
                            utils.get_data_loader(feature_replay_datasets, self.batch_size, self.num_workers,
                                                  cuda=True if self.availabel_cudas else False))
                        feature_replay_dataset_index = 0

                    while iters_left > 0:
                        # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                        iters_left -= 1
                        iter_num = 0
                        if is_first_ite:
                            is_first_ite = False
                            data_loader = iter(
                                utils.get_data_loader(training_dataset, self.batch_size, self.num_workers,
                                                      cuda=True if self.availabel_cudas else False,
                                                      drop_last=True))
                            iter_num = iters_left = len(data_loader)
                            continue

                        # -----------------Collect data------------------#

                        #####-----CURRENT BATCH-----#####
                        x, y = next(data_loader)  # --> sample training data of current task

                        '''if feature_replay end, get new feature_replay data loader'''
                        if feature_replay_dataset_index == feature_replay_dataset_num:
                            feature_replay_data_loader = iter(
                                utils.get_data_loader(feature_replay_datasets, self.batch_size, self.num_workers,
                                                      cuda=True if self.availabel_cudas else False))
                            feature_replay_dataset_index = 0
                        feature_replay_features, feature_replay_labels = next(feature_replay_data_loader)
                        feature_replay_dataset_index += 1
                        x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                        feature_replay_features, feature_replay_labels = feature_replay_features.to(
                            self.device), feature_replay_labels.to(self.device)
                        '''获取图片在pre_FEcls跟FM模型的输出'''
                        FM_features, FM_targets = self.get_preFE_FM_feature_target(x)
                        '''获取保存的features在FM模型的输出'''
                        feature_replay_FM_features = self.feature_mapping(feature_replay_features)
                        feature_replay_FM_targets = self.prefeature_2_FMtarget(feature_replay_FM_features)
                        # ---> Train MAIN MODEL
                        # Train the main model with this batch
                        loss_dict = self.train_a_batch_feature_replay_softTargetCrossEntropy(x, y,
                                                                                             feature_replay_features,
                                                                                             feature_replay_labels,
                                                                                             FM_targets=FM_targets,
                                                                                             FM_features=FM_features,
                                                                                             feature_replay_FM_features=feature_replay_FM_features,
                                                                                             feature_replay_FM_targets=feature_replay_FM_targets,
                                                                                             optimizer=optimizer,
                                                                                             active_classes=active_classes,
                                                                                             task=task)
                        iter_index += 1
                        if iter_index % args.print_interval == 0:
                            self.batch_train_logger.info(
                                f"Task {task:0>3d}, batch train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], "
                                f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: "
                                f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"
                                f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
                            )
                    scheduler.step()
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes)
                    results = f"batch train FE_cls || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, " \
                              f"top5 acc:  {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    self.batch_train_logger.info(
                        results
                    )
                    print(results)
                examplars_per_class = int(np.floor(self.Exemple_memory_budget / (self.classes_per_task * task)))
                self.reduce_exemplar_sets(examplars_per_class)
            elif task == 1:
                # self.FE_cls = torch.load("task_1_FE_cls.pth")
                # self.FE_cls = torch.load("cifar10_preTrain_task_1_FE_cls.pth")
                # # continue
                self.train_FE_cls(args, train_dataset, active_classes)
                # print("model:", self.FE_cls)
                # torch.save(self.FE_cls, "./cifar10_preTrain_task_1_FE_cls_1_4.pth")
            # else:
            #     break
            self.cifar100_feature_handle_main(train_dataset, self.classes_per_task, task)  # todo Done
            # self.cifar100_feature_handle_main_FM(training_dataset, self.val_datasets, self.classes_per_task,
            #                                      active_classes, task, use_dynamicMem=True, FM_reTrain=True)
            self.batch_train_logger.info(f'##########feature extractor train task {task} End.#########')
            self.logger.info(f'#############feature extractor train task {task} End.##############')
            self.logger.info(f'#############task {task} validate  begin.##############')
            print("feature extractor train task-%d End" % (task))
            # acc_past_tasks, acc_list = self.tasks_validate(task)
            # EFAfIL_result["task_{}_svm_results".format(task)] = acc_past_tasks
            if self.Exemple_memory_budget > 0:
                acc_past_tasks, acc_list = self.tasks_validate(task, classifier="examplar_ncm", use_FM=self.use_FM)
                EFAfIL_result["task_{}_examplar_ncm_results".format(task)] = acc_past_tasks
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="feature_ncm", use_FM=self.use_FM)
            EFAfIL_result["task_{}_feature_ncm_results".format(task)] = acc_past_tasks
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="feature_ncm", use_FM=True)
            EFAfIL_result["task_{}_featureNCM_useFM_results".format(task)] = acc_past_tasks
            # acc_past_tasks, acc_list = self.tasks_validate(task, classifier="fc", active_classes=active_classes)
            # EFAfIL_result["task_{}_fc_results".format(task)] = acc_past_tasks
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="fcls", active_classes=active_classes)
            EFAfIL_result["task_{}_FE_cls_results".format(task)] = acc_past_tasks
            if self.compute_means:
                self.batch_train_logger.info(
                    "self.compute_means is True"
                )
            else:
                self.batch_train_logger.info(
                    "self.compute_means is False"
                )
            with open(self.result_file, 'w') as fw:
                json.dump(EFAfIL_result, fw, indent=4)
            # print(ILtFA_result_temp)
            # if task == 2 and self.pre_FE_cls:
            #     torch.save(self.pre_FE_cls, "./pre_FE_cls_examplar_0_back.pth")
            #     torch.save(self.FE_cls, "./FE_cls_examplar_0_back.pth")
        with open(self.result_file, 'w') as fw:
            json.dump(EFAfIL_result, fw, indent=4)

    def train_a_batch_feature_replay(self, x, y, feature_replay_features,
                                     feature_replay_labels, FM_targets,
                                     FM_features,
                                     feature_replay_FM_features,
                                     feature_replay_FM_targets,
                                     optimizer,
                                     active_classes, task):
        self.train()
        # Reset optimizer
        optimizer.zero_grad()
        loss_total = None
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        # Run model
        features, y_hat = self(x)
        if type(self.FE_cls) is torch.nn.DataParallel:
            feature_replay_y_hat = self.FE_cls.module.cls(feature_replay_features)
        else:
            feature_replay_y_hat = self.FE_cls.cls(feature_replay_features)
        # -if needed, remove predictions for classes not in current task
        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
            y_hat = y_hat[:, class_entries]
        '''binary cross entropy loss of distill && cross entropy loss of cls'''
        loss_total = self.BinaryCE_distill_CE_cls(y, y_hat, FM_targets, feature_replay_y_hat, feature_replay_FM_targets,
                                                  task)
        # '''BiC_loss'''
        # loss_total = self.BiC_loss_distill_cls(y, y_hat, FM_targets, feature_replay_y_hat, feature_replay_FM_targets,
        #                                        task)

        loss_total.backward()
        # Take optimization-step
        optimizer.step()
        # loss_total = self.kd_lamb * loss_distill + self.fd_gamma * loss_sim + loss_cls
        if len(active_classes) >= 5:
            acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
            top1.update(acc1.item(), x.size(0))
            top5.update(acc5.item(), x.size(0))
        else:
            acc1 = accuracy(y_hat, y, topk=(1,))[0]
            top1.update(acc1.item(), x.size(0))
        losses.update(loss_total, x.size(0))
        # Calculate training-precision
        precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
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
        pass

    def BinaryCE_distill_CE_cls(self, y, y_hat, FM_targets, feature_replay_y_hat, feature_replay_FM_targets, task):
        '''获取图片数据在FE_cls跟FM的对应于前n-1个任务的输出，获取保存的feature数据在FM以及FE_cls的对应于前n-1个任务的输出，制作蒸馏数据和分类数据'''
        scores_hats = FM_targets[:, :(self.classes_per_task * (task - 1))]
        scores_hats = torch.sigmoid(scores_hats / self.KD_temp)
        feature_replay_y_hat_fordistill = feature_replay_y_hat[:, :(self.classes_per_task * (task - 1))]
        feature_replay_scores_hats = feature_replay_FM_targets[:, :(self.classes_per_task * (task - 1))]
        feature_replay_scores_hats = torch.sigmoid(feature_replay_scores_hats / self.KD_temp)

        binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
        binary_targets = binary_targets[:, -self.classes_per_task:]
        binary_targets = torch.cat([scores_hats, binary_targets], dim=1)
        '''loss BiC'''
        # loss_cls_distill = Func.binary_cross_entropy_with_logits(input=y_hat, target=binary_targets, reduction='none')
        # loss_cls_current = loss_cls_distill[:, -self.classes_per_task:].sum(dim=1).mean()
        # loss_distill_current = loss_cls_distill[:, :-self.classes_per_task].sum(dim=1).mean()
        # loss_distill_feature_replay = Func.binary_cross_entropy_with_logits(
        #     input=feature_replay_y_hat_fordistill, target=feature_replay_scores_hats, reduction='none'
        # ).sum(dim=1).mean()
        # lamda = (task - 1) / task
        # loss_total = lamda * loss_cls_current + (1 - lamda) * (loss_distill_current + loss_distill_feature_replay)
        '''loss EFAfIL'''
        loss_cls_distill = Func.binary_cross_entropy_with_logits(input=y_hat, target=binary_targets, reduction='none'
                                                                 ).sum(dim=1).mean()
        loss_distill_feature_replay = Func.binary_cross_entropy_with_logits(
            input=feature_replay_y_hat_fordistill, target=feature_replay_scores_hats, reduction='none'
        ).sum(dim=1).mean()
        return loss_cls_distill + loss_distill_feature_replay

    def BiC_loss_distill_cls(self, y, y_hat, FM_targets, feature_replay_y_hat, feature_replay_FM_targets, task):
        criteria = torch.nn.CrossEntropyLoss()
        soft_target_criterion = SoftTarget_CrossEntropy().cuda()
        '''获取图片数据在FE_cls跟FM的对应于前n-1个任务的输出，获取保存的feature数据在FM以及FE_cls的对应于前n-1个任务的输出，制作蒸馏数据和分类数据'''
        y_hat_fordistill = y_hat[:, :(self.classes_per_task * (task - 1))]
        scores_hats = FM_targets[:, :(self.classes_per_task * (task - 1))]
        scores_hats = torch.softmax(scores_hats / self.KD_temp, dim=1)
        feature_replay_y_hat_fordistill = feature_replay_y_hat[:, :(self.classes_per_task * (task - 1))]
        feature_replay_scores_hats = feature_replay_FM_targets[:, :(self.classes_per_task * (task - 1))]
        feature_replay_scores_hats = torch.softmax(feature_replay_scores_hats / self.KD_temp, dim=1)
        '''loss distill'''
        loss_distill_current = soft_target_criterion(y_hat_fordistill, scores_hats,
                                                     self.KD_temp) * self.KD_temp * self.KD_temp
        loss_distill_feature_replay = soft_target_criterion(feature_replay_y_hat_fordistill, feature_replay_scores_hats,
                                                            self.KD_temp) * self.KD_temp * self.KD_temp

        loss_cls_current = criteria(y_hat, y)
        lamda = (task - 1) / task
        loss_total = (loss_distill_current + loss_distill_feature_replay) * lamda + (1 - lamda) * loss_cls_current
        return loss_total
        pass

    def train_BiasLayer_feature_replay(self, per_task_valing_dataset, task, active_classes, print_interval):
        current_classes_num = self.classes_per_task
        self.bias_layer = BiasLayer()
        optimizer = self.build_optimize_biaslayer(self.lr / 100)
        epochs = int(self.epochs / 3)
        gap = int(epochs / 3)
        milestones = [gap, 2 * gap]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        '''get feature_replay data loader'''
        feature_replay_datasets = ExemplarDataset(self.exemplar_feature_sets)
        feature_replay_dataset_index = 0
        feature_replay_data_loader = iter(utils.get_data_loader(feature_replay_datasets, self.batch_size,
                                                                self.num_workers,
                                                                cuda=True if self.availabel_cudas else False))
        feature_replay_dataset_num = len(feature_replay_data_loader)
        for epoch in range(1, epochs + 1):
            is_first_ite = True
            iters_left = 1
            iter_index = 0
            '''if feature_replay end, get new feature_replay data loader'''
            if feature_replay_dataset_index == feature_replay_dataset_num:
                feature_replay_data_loader = iter(
                    utils.get_data_loader(feature_replay_datasets, self.batch_size, self.num_workers,
                                          cuda=True if self.availabel_cudas else False))
                feature_replay_dataset_index = 0
            while iters_left > 0:
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                iters_left -= 1
                iter_num = 0
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

                '''if feature_replay end, get new feature_replay data loader'''
                if feature_replay_dataset_index == feature_replay_dataset_num:
                    feature_replay_data_loader = iter(
                        utils.get_data_loader(feature_replay_datasets, self.batch_size, self.num_workers,
                                              cuda=True if self.availabel_cudas else False))
                    feature_replay_dataset_index = 0
                feature_replay_features, feature_replay_labels = next(feature_replay_data_loader)
                feature_replay_dataset_index += 1
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                feature_replay_features, feature_replay_labels = feature_replay_features.to(
                    self.device), feature_replay_labels.to(self.device)

                FE_cls_targets = self.FE_cls_forward(x)
                FE_cls_targets = FE_cls_targets[:, :(self.classes_per_task * task)]
                if type(self.FE_cls) is torch.nn.DataParallel:
                    FE_cls_targets_feature_replay = self.FE_cls.module.get_cls_results(feature_replay_features)
                else:
                    FE_cls_targets_feature_replay = self.FE_cls.get_cls_results(feature_replay_features)
                FE_cls_targets_feature_replay = FE_cls_targets_feature_replay[:, :(self.classes_per_task * task)]
                loss_dict = self.BiasLayer_train_a_batch_feature_replay(FE_cls_targets, y,
                                                                        FE_cls_targets_feature_replay,
                                                                        feature_replay_labels, current_classes_num,
                                                                        optimizer)
                iter_index += 1
                if iter_index % print_interval == 0:
                    self.batch_train_logger.info(
                        f"bias layer train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], "
                        f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: "
                        f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"
                        f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
                    )
                    # print(loss_dict)
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

    def BiasLayer_train_a_batch_feature_replay(self, FE_cls_targets, y, FE_cls_targets_feature_replay,
                                               feature_replay_labels, current_classes_num, optimizer):
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
        y_hat_feature_replay = self.bias_layer(FE_cls_targets_feature_replay, current_classes_num)
        predL = criterion(y_hat, y) + criterion(y_hat_feature_replay, feature_replay_labels)
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
                _, y_hat = self(inputs)
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

    def train_a_batch_feature_replay_softTargetCrossEntropy(self, x, y, feature_replay_features,
                                                            feature_replay_labels, FM_targets,
                                                            FM_features,
                                                            feature_replay_FM_features,
                                                            feature_replay_FM_targets,
                                                            optimizer,
                                                            active_classes, task):
        self.train()
        # Reset optimizer
        optimizer.zero_grad()
        loss_total = None
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        loss_func = torch.nn.MSELoss(reduction='mean')
        criteria = torch.nn.CrossEntropyLoss()
        soft_target_criterion = SoftTarget_CrossEntropy().cuda()
        # Run model
        features, y_hat = self(x)
        if type(self.FE_cls) is torch.nn.DataParallel:
            feature_replay_y_hat = self.FE_cls.module.cls(feature_replay_features)
        else:
            feature_replay_y_hat = self.FE_cls.cls(feature_replay_features)
        # -if needed, remove predictions for classes not in current task
        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
            y_hat = y_hat[:, class_entries]
            # FM_targets = FM_targets[:, class_entries]
        '''获取图片数据在FE_cls跟FM的对应于前n-1个任务的输出，获取保存的feature数据在FM以及FE_cls的对应于前n-1个任务的输出，制作蒸馏数据和分类数据'''
        y_hat_fordistill = y_hat[:, :(self.classes_per_task * (task - 1))]
        scores_hats = FM_targets[:, :(self.classes_per_task * (task - 1))]
        scores_hats = torch.softmax(scores_hats / self.KD_temp, dim=1)
        feature_replay_y_hat_fordistill = feature_replay_y_hat[:, :(self.classes_per_task * (task - 1))]
        feature_replay_y_hat_forcls = feature_replay_y_hat[:, :(self.classes_per_task * task)]
        feature_replay_scores_hats = feature_replay_FM_targets[:, :(self.classes_per_task * (task - 1))]
        feature_replay_scores_hats = torch.softmax(feature_replay_scores_hats / self.KD_temp, dim=1)

        # scores_hats = torch.sigmoid(FM_targets / self.KD_temp)
        # binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
        # binary_targets = binary_targets[:, -self.classes_per_task:]
        # binary_targets = torch.cat([scores_hats, binary_targets], dim=1)
        '''loss distill'''
        loss_distill_current = soft_target_criterion(y_hat_fordistill, scores_hats,
                                                     self.KD_temp) * self.KD_temp * self.KD_temp
        loss_distill_feature_replay = soft_target_criterion(feature_replay_y_hat_fordistill, feature_replay_scores_hats,
                                                            self.KD_temp) * self.KD_temp * self.KD_temp

        loss_featue_replay_cls = criteria(feature_replay_y_hat_forcls, feature_replay_labels)
        loss_cls_current = criteria(y_hat, y)
        # loss_sim = loss_func(features, FM_features)
        loss_sim = 1 - torch.cosine_similarity(features, FM_features).mean()
        # loss_total = loss_cls_distill + self.fd_gamma * loss_sim
        distill_current_rate = 1
        cls_current_rate = 1
        distill_feature_replay_rate = 1
        featue_replay_cls_rate = 0
        sim_rate = 0
        loss_total = loss_distill_current * distill_current_rate + loss_cls_current * cls_current_rate \
                     + loss_featue_replay_cls * featue_replay_cls_rate \
                     + loss_distill_feature_replay * distill_feature_replay_rate \
                     + loss_sim * sim_rate
        loss_total.backward()
        # Take optimization-step
        optimizer.step()
        # loss_total = self.kd_lamb * loss_distill + self.fd_gamma * loss_sim + loss_cls
        if len(active_classes) >= 5:
            acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
            top1.update(acc1.item(), x.size(0))
            top5.update(acc5.item(), x.size(0))
        else:
            acc1 = accuracy(y_hat, y, topk=(1,))[0]
            top1.update(acc1.item(), x.size(0))
        losses.update(loss_total, x.size(0))
        # Calculate training-precision
        precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
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
        pass

    def extra_data_train_main(self, args):
        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''
        print("seed:", self.seed)
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            cudnn.deterministic = True

        gpus = torch.cuda.device_count()
        cudnn.benchmark = True
        cudnn.enabled = True
        # Set model in training-mode
        # Loop over all tasks.
        self.logger.info(f'use {gpus} gpus')
        self.logger.info(f"args: {args}")
        EFAfIL_result = {"timestamp": str(datetime.datetime.now()), "model_rate": self.rate}
        EFAfIL_result.update(self.data_config)
        for task, train_dataset in enumerate(self.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')
            self.batch_train_logger.info(f'New task {task} begin.')
            print("New task %d begin." % task)
            # Add exemplars (if available) to current dataset (if requested)
            if self.use_exemplars and task > 1:
                exemplar_dataset = ExemplarDataset(self.exemplar_sets)
                training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
            else:
                training_dataset = train_dataset

            active_classes = list(range(self.classes_per_task * task))
            if task > 1:
                # self.EFAfIL_feature_mapper_cls_domain_train(training_dataset,
                #                                             self.val_datasets, self.classes_per_task,
                #                                             active_classes, task)
                self.EFAfIL_split_feature_mapper_cls_domain_train_extraData(training_dataset, self.extra_train_datasets,
                                                                            self.val_datasets, self.classes_per_task,
                                                                            active_classes, task)
                self.pre_FE_cls = copy.deepcopy(self.FE_cls).eval()
                optimizer = self.build_optimize()
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=self.milestones, gamma=self.gamma)

                '''get extra data loader'''
                extra_train_dataset_index = 0
                extra_data_loader = iter(
                    utils.get_data_loader(self.extra_train_datasets, self.batch_size, self.num_workers,
                                          cuda=True if self.availabel_cudas else False))
                extra_train_dataset_num = len(extra_data_loader)
                extra_index = 0
                print("extra_train_dataset_num:", extra_train_dataset_num)

                '''get feature_replay data loader'''
                feature_replay_datasets = ExemplarDataset(self.exemplar_feature_sets)
                feature_replay_dataset_index = 0
                feature_replay_data_loader = iter(utils.get_data_loader(feature_replay_datasets, self.batch_size,
                                                                        self.num_workers,
                                                                        cuda=True if self.availabel_cudas else False))
                feature_replay_dataset_num = len(feature_replay_data_loader)
                extra_index = 0
                print("feature_replay_dataset_num:", feature_replay_dataset_num)

                for epoch in range(1, self.epochs + 1):
                    is_first_ite = True
                    iters_left = 1
                    iter_index = 0
                    extra_index = 0

                    '''if extra data end, get new extra data loader'''
                    if extra_train_dataset_index == extra_train_dataset_num:
                        extra_data_loader = iter(utils.get_data_loader(self.extra_train_datasets, self.batch_size,
                                                                       self.num_workers,
                                                                       cuda=True if self.availabel_cudas else False))
                        extra_train_dataset_index = 0
                    '''if feature_replay end, get new feature_replay data loader'''
                    if feature_replay_dataset_index == feature_replay_dataset_num:
                        feature_replay_data_loader = iter(
                            utils.get_data_loader(feature_replay_datasets, self.batch_size, self.num_workers,
                                                  cuda=True if self.availabel_cudas else False))
                        feature_replay_dataset_index = 0
                    while iters_left > 0:
                        # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                        iters_left -= 1
                        iter_num = 0
                        if is_first_ite:
                            is_first_ite = False
                            data_loader = iter(
                                utils.get_data_loader(training_dataset, self.batch_size, self.num_workers,
                                                      cuda=True if self.availabel_cudas else False,
                                                      drop_last=True))
                            iter_num = iters_left = len(data_loader)
                            continue

                        # -----------------Collect data------------------#

                        #####-----CURRENT BATCH-----#####
                        x, y = next(data_loader)  # --> sample training data of current task
                        x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                        '''if extra data end, get new extra data loader'''
                        if extra_train_dataset_index == extra_train_dataset_num:
                            extra_data_loader = iter(utils.get_data_loader(self.extra_train_datasets, self.batch_size,
                                                                           self.num_workers,
                                                                           cuda=True if self.availabel_cudas else False))
                            extra_train_dataset_index = 0
                        extra_x, extra_y = next(extra_data_loader)
                        extra_x, extra_y = extra_x.to(self.device), extra_y.to(self.device)
                        extra_train_dataset_index += 1
                        '''if feature_replay end, get new feature_replay data loader'''
                        if feature_replay_dataset_index == feature_replay_dataset_num:
                            feature_replay_data_loader = iter(
                                utils.get_data_loader(feature_replay_datasets, self.batch_size, self.num_workers,
                                                      cuda=True if self.availabel_cudas else False))
                            feature_replay_dataset_index = 0
                        feature_replay_features, feature_replay_labels = next(feature_replay_data_loader)
                        feature_replay_dataset_index += 1
                        feature_replay_features, feature_replay_labels = feature_replay_features.to(
                            self.device), feature_replay_labels.to(self.device)
                        FM_features, FM_targets = self.get_preFE_FM_feature_target(x)
                        extra_FM_features, extra_FM_targets = self.get_preFE_FM_feature_target(extra_x)
                        # ---> Train MAIN MODEL
                        # Train the main model with this batch
                        '''获取保存的features在FM模型的输出'''
                        feature_replay_FM_features = self.feature_mapping(feature_replay_features)
                        feature_replay_FM_targets = self.prefeature_2_FMtarget(feature_replay_FM_features)
                        # ---> Train MAIN MODEL
                        # Train the main model with this batch
                        loss_dict = self.train_a_batch_plus_extraData(x, y, feature_replay_features,
                                                                      feature_replay_labels, extra_x, extra_y,
                                                                      FM_targets=FM_targets,
                                                                      FM_features=FM_features,
                                                                      feature_replay_FM_features=feature_replay_FM_features,
                                                                      feature_replay_FM_targets=feature_replay_FM_targets,
                                                                      extra_FM_features=extra_FM_features,
                                                                      extra_FM_targets=extra_FM_targets,
                                                                      optimizer=optimizer,
                                                                      active_classes=active_classes, task=task)
                        iter_index += 1
                        if iter_index % args.print_interval == 0:
                            self.batch_train_logger.info(
                                f"Task {task:0>3d}, batch train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], "
                                f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: "
                                f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"
                                f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
                            )
                    scheduler.step()
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes)
                    results = f"batch train FE_cls || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, " \
                              f"top5 acc:  {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    self.batch_train_logger.info(
                        results
                    )
                    print(results)
                examplars_per_class = int(np.floor(self.Exemple_memory_budget / (self.classes_per_task * task)))
                self.reduce_exemplar_sets(examplars_per_class)
            elif task == 1:
                # self.FE_cls = torch.load("task_1_FE_cls.pth")
                # self.FE_cls = torch.load("cifar10_preTrain_task_1_FE_cls.pth")
                # # continue
                # print("model:", self.FE_cls)
                self.train_FE_cls(args, train_dataset, active_classes)
                # torch.save(self.FE_cls, "./imagenet100_preTrain_task_1_FE_cls.pth")
            # else:
            #     break
            self.feature_handle_main(train_dataset, self.classes_per_task, task, self.use_FM)  # todo Done
            self.batch_train_logger.info(f'##########feature extractor train task {task} End.#########')
            self.logger.info(f'#############feature extractor train task {task} End.##############')
            self.logger.info(f'#############task {task} validate  begin.##############')
            print("feature extractor train task-%d End" % (task))
            # acc_past_tasks, acc_list = self.tasks_validate(task, use_FM=self.use_FM)
            # EFAfIL_result["task_{}_results".format(task)] = acc_past_tasks
            if self.Exemple_memory_budget > 0:
                acc_past_tasks, acc_list = self.tasks_validate(task, classifier="examplar_ncm", use_FM=self.use_FM)
                EFAfIL_result["task_{}_examplar_ncm_results".format(task)] = acc_past_tasks
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="feature_ncm", use_FM=self.use_FM)
            EFAfIL_result["task_{}_feature_ncm_results".format(task)] = acc_past_tasks
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="feature_ncm", use_FM=True)
            EFAfIL_result["task_{}_featureNCM_useFM_results".format(task)] = acc_past_tasks
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="fc", active_classes=active_classes)
            EFAfIL_result["task_{}_fc_results".format(task)] = acc_past_tasks
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="fcls", active_classes=active_classes)
            EFAfIL_result["task_{}_FE_cls_results".format(task)] = acc_past_tasks
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
                json.dump(EFAfIL_result, fw, indent=4)
            # print(ILtFA_result_temp)
            # if task == 2 and self.pre_FE_cls:
            #     torch.save(self.pre_FE_cls, "./pre_FE_cls_examplar_0_back.pth")
            #     torch.save(self.FE_cls, "./FE_cls_examplar_0_back.pth")
        with open(self.result_file, 'w') as fw:
            json.dump(EFAfIL_result, fw, indent=4)

    def train_a_batch_plus_extraData(self, x, y, feature_replay_features,
                                     feature_replay_labels, extra_x, extra_y, FM_targets,
                                     FM_features,
                                     feature_replay_FM_features,
                                     feature_replay_FM_targets,
                                     extra_FM_features,
                                     extra_FM_targets,
                                     optimizer,
                                     active_classes, task):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

                [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
                [y]               <tensor> batch of corresponding labels
                [scores]          None or <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x]
                                    NOTE: only to be used for "BCE with distill" (only when scenario=="class")
                [rnt]             <number> in [0,1], relative importance of new task
                [active_classes]  None or (<list> of) <list> with "active" classes
                [task]            <int>, for setting task-specific mask'''

        # Set model to training-mode
        self.train()
        # Reset optimizer
        optimizer.zero_grad()
        loss_total = None
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        loss_func = torch.nn.MSELoss(reduction='mean')
        criteria = torch.nn.CrossEntropyLoss()
        # Run model
        features, y_hat = self(x)
        extra_features, extra_y_hat = self(extra_x)
        if type(self.FE_cls) is torch.nn.DataParallel:
            feature_replay_y_hat = self.FE_cls.module.cls(feature_replay_features)
        else:
            feature_replay_y_hat = self.FE_cls.cls(feature_replay_features)
        # -if needed, remove predictions for classes not in current task
        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
            y_hat = y_hat[:, class_entries]
            # FM_targets = FM_targets[:, class_entries]
        extra_y_hat = extra_y_hat[:, :(self.classes_per_task * (task - 1))]

        scores_hats = FM_targets[:, :(self.classes_per_task * (task - 1))]
        extra_scores_hats = extra_FM_targets[:, :(self.classes_per_task * (task - 1))]

        scores_hats = torch.sigmoid(scores_hats / self.KD_temp)
        extra_scores_hats = torch.sigmoid(extra_scores_hats / self.KD_temp)

        # scores_hats = torch.sigmoid(FM_targets / self.KD_temp)
        binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
        binary_targets = binary_targets[:, -self.classes_per_task:]
        binary_targets = torch.cat([scores_hats, binary_targets], dim=1)
        loss_cls_distill = None if y is None else Func.binary_cross_entropy_with_logits(
            input=y_hat, target=binary_targets, reduction='none'
        ).sum(dim=1).mean()

        loss_extra_distill = Func.binary_cross_entropy_with_logits(
            input=extra_y_hat, target=extra_scores_hats, reduction='none'
        ).sum(dim=1).mean()
        # loss_sim = loss_func(features, FM_features)
        loss_sim = 1 - torch.cosine_similarity(features, FM_features).mean()
        loss_extra_sim = 1 - torch.cosine_similarity(extra_features, extra_FM_features).mean()

        feature_replay_y_hat_fordistill = feature_replay_y_hat[:, :(self.classes_per_task * (task - 1))]
        feature_replay_y_hat_forcls = feature_replay_y_hat[:, :(self.classes_per_task * task)]
        feature_replay_scores_hats = feature_replay_FM_targets[:, :(self.classes_per_task * (task - 1))]
        feature_replay_scores_hats = torch.sigmoid(feature_replay_scores_hats / self.KD_temp)

        loss_distill_feature_replay = Func.binary_cross_entropy_with_logits(
            input=feature_replay_y_hat_fordistill, target=feature_replay_scores_hats, reduction='none'
        ).sum(dim=1).mean()
        loss_featue_replay_cls = criteria(feature_replay_y_hat_forcls, feature_replay_labels)

        # loss_total = loss_cls_distill + self.fd_gamma * loss_sim
        loss_total = loss_cls_distill + loss_distill_feature_replay + loss_extra_distill
        loss_total.backward()
        # Take optimization-step
        optimizer.step()
        # loss_total = self.kd_lamb * loss_distill + self.fd_gamma * loss_sim + loss_cls
        if len(active_classes) >= 5:
            acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
            top1.update(acc1.item(), x.size(0))
            top5.update(acc5.item(), x.size(0))
        else:
            acc1 = accuracy(y_hat, y, topk=(1,))[0]
            top1.update(acc1.item(), x.size(0))
        losses.update(loss_total, x.size(0))
        # Calculate training-precision
        precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
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

    def per_task_validate(self, val_dataset, task, classifier="ncm"):
        # todo
        val_loader = utils.get_data_loader(val_dataset, self.batch_size, self.num_workers,
                                           cuda=True if self.availabel_cudas else False)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        correct = 0
        for inputs, labels in val_loader:
            correct_temp = 0
            data_time.update(time.time() - end)
            inputs, labels = inputs.cuda(), labels.cuda()
            if classifier == "ncm":
                y_hat = self.classify_with_exemplars(inputs)
            elif classifier == "fc":
                y_hat = self.FE_cls_forward(inputs)
                y_hat = y_hat[:, :self.classes_per_task * task]
                if task > 1:
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

    def tasks_validate(self, task, classifier="fc"):
        # todo
        acc_past_tasks = []
        acc = []
        for task_id in range(task):
            predict_result = self.per_task_validate(self.val_datasets[task_id], task, classifier)
            acc_past_tasks.append(predict_result)
            acc.append(predict_result[0])
            print("{} per task acc:".format(classifier), predict_result)
            self.logger.info(
                f"per task {task}, classifier{classifier} acc:{predict_result[0]}"
            )
        self.compute_means = True
        self.svm_train = True
        return acc_past_tasks, np.array(acc)
        pass

    def current_task_validate(self, task, active_classes):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        val_loader = utils.get_data_loader(self.val_datasets[task - 1], self.batch_size,  # task index must minus 1
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
                _, y_hat = self(inputs)
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
