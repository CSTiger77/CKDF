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

from CIFAR.alg_model import resnetforcifar
from public import utils
from public.data import ExemplarDataset, get_multitask_experiment, get_dataset
from exemplars import EFAfIL_FeaturesHandler, FeaturesHandler, ExemplarHandler
from public.util_models import FE_cls

# -------------------------------------------------------------------------------------------------#

# --------------------#
# ----- EFAfIL -----#
# --------------------
from public.utils import AverageMeter, accuracy


class EFAfIL(EFAfIL_FeaturesHandler):
    def __init__(self, model_name, MLP_name, dataset_name,
                 dataset_path, num_classes, rate, tasks,
                 logger, batch_train_logger, result_file, use_exemplars,
                 hidden_size, Exemple_memory_budget,
                 Feature_memory_budget, optim_type, MLP_optim_type,
                 norm_exemplars, herding, batch_size,
                 num_workers, seed, availabel_cudas,

                 epochs, CNN_lr, CNN_momentum,
                 CNN_weight_decay, CNN_milestones,
                 kd_lamb, fd_gamma, lrgamma, KD_temp,

                 MLP_lr, MLP_momentum,
                 MLP_epochs, MLP_milestones,
                 svm_sample_type,
                 MLP_weight_decay,
                 MLP_lrgamma, sim_alpha, svm_max_iter):
        EFAfIL_FeaturesHandler.__init__(self, MLP_name, num_classes, hidden_size, Exemple_memory_budget,
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
        # self.extra_train_dataset = get_dataset("CIFAR10", 'train', dir=self.dataset_path)
        self.pre_FE_cls = None
        self.FE_cls = None
        # self.FE_cls = self.construct_model(rate)

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
        if self.availabel_cudas:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
            device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
            model = torch.nn.DataParallel(FE_cls(resnetforcifar.__dict__[self.model_name](rate=rate, get_feature=True),
                                                 self.hidden_size, self.num_classes),
                                          device_ids=device_ids).cuda()
        else:
            model = FE_cls(resnetforcifar.__dict__[self.model_name](rate=rate, get_feature=True), self.hidden_size,
                           self.class_num)
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

    def feature_extractor(self, images):
        mode = self.training
        self.eval()
        with torch.no_grad():
            features = self.FE_cls(images)[-2]
        self.train(mode=mode)
        return features

    def get_cls_target(self, prefeatures):
        if type(self.FE_cls) is torch.nn.DataParallel:
            return self.FE_cls.module.get_cls_results(prefeatures)
        else:
            return self.FE_cls.get_cls_results(prefeatures)

    def get_FE_cls_target(self, x):
        mode = self.training
        self.eval()
        with torch.no_grad():
            _, targets = self.FE_cls(x)
        self.train(mode=mode)
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
                    data_loader = iter(utils.get_data_loader(train_dataset, self.batch_size,
                                                             cuda=True if self.availabel_cudas else False,
                                                             drop_last=True))
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
        loss_total = criterion(y_hat, y)
        acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
        top1.update(acc1.item(), x.size(0))
        top5.update(acc5.item(), x.size(0))
        losses.update(loss_total, x.size(0))
        # Calculate training-precision
        precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
        loss_total.backward()
        # Take optimization-step
        optimizer.step()
        # Return the dictionary with different training-loss split in categories
        return {
            'task': 1,
            'loss_total': loss_total.item(),
            'precision': precision if precision is not None else 0.,
            "top1": top1.avg,
            "top5": top5.avg,
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
                self.EFAfIL_feature_mapper_cls_domain_train(train_dataset, self.val_datasets, self.classes_per_task,
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
                            data_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
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
                                                       optimizer=optimizer, active_classes=active_classes,
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
                self.FE_cls = torch.load("cifar10_preTrain_task_1_FE_cls.pth")
                # continue
                print("model:", self.FE_cls)
                # self.train_FE_cls(args, train_dataset, active_classes)
                # torch.save(self.FE_cls, "./cifar10_preTrain_task_1_FE_cls.pth")
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
        scores_hats = FM_targets[:, :(self.classes_per_task * task)]
        scores_hats = torch.sigmoid(scores_hats / self.KD_temp)
        loss_distill = None if y is None else Func.binary_cross_entropy_with_logits(
            input=y_hat, target=scores_hats, reduction='none'
        ).sum(dim=1).mean()
        # loss_sim = loss_func(features, FM_features)
        loss_sim = 1 - torch.cosine_similarity(features, FM_features).mean()
        loss_cls = criteria(y_hat, y)
        loss_total = loss_cls + self.kd_lamb * loss_distill + self.fd_gamma * loss_sim
        # loss_total = self.kd_lamb * loss_distill + self.fd_gamma * loss_sim + loss_cls
        acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
        top1.update(acc1.item(), x.size(0))
        top5.update(acc5.item(), x.size(0))
        losses.update(loss_total, x.size(0))
        # Calculate training-precision
        precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
        loss_total.backward()
        # Take optimization-step
        optimizer.step()
        # Return the dictionary with different training-loss split in categories
        return {
            'task': task,
            'loss_total': loss_total.item(),
            'precision': precision if precision is not None else 0.,
            "top1": top1.avg,
            "top5": top5.avg,
            "losses": losses.avg
        }

    def per_task_validate(self, task, classifier="linearSVM", active_classes=None):
        # todo
        val_dataset = self.val_datasets[task - 1]
        val_loader = utils.get_data_loader(val_dataset, self.batch_size,
                                           cuda=True if self.availabel_cudas else False)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        correct = 0
        for inputs, labels in val_loader:
            correct_temp = 0
            data_time.update(time.time() - end)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            y_hat = self.EFAfIL_classify(inputs, classifier, active_classes, task)
            correct_temp += y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
        throughput = 1.0 / (batch_time.avg / self.batch_size)
        return [top1.avg, throughput]
        pass

    def tasks_validate(self, task, classifier="linearSVM", active_classes=None):
        # todo
        acc_past_tasks = []
        acc = []
        for task_id in range(1, task + 1):
            predict_result = self.per_task_validate(task_id, classifier, active_classes)
            acc_past_tasks.append(predict_result)
            acc.append(predict_result[0])
            print("per task acc:", predict_result)
            self.logger.info(
                f"per task {task}, classifier{classifier} acc:{predict_result[0]}"
            )
        return acc_past_tasks, np.array(acc)
        pass

    def current_task_validate(self, task, active_classes):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        val_loader = utils.get_data_loader(self.val_datasets[task - 1], self.batch_size,  # task index must minus 1
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
                acc1, acc5 = accuracy(y_hat, labels, topk=(1, 5))
                top1.update(acc1.item(), inputs.size(0))
                top5.update(acc5.item(), inputs.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
        throughput = 1.0 / (batch_time.avg / self.batch_size)
        self.train(mode=mode)
        return top1.avg, top5.avg, throughput
