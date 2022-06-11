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
from torch.autograd import Variable
from torch.nn import functional as Func
import torch
from torch.backends import cudnn
from torch.utils.data import ConcatDataset
from torchvision import transforms

from CIFAR.alg_model import resnetforcifar
from public import utils, MLP
from public.data import ExemplarDataset, get_multitask_experiment, get_dataset, AVAILABLE_TRANSFORMS, \
    inv_CIFAR_100_normalize
from exemplars import EFAfIL_FeaturesHandler, FeaturesHandler, ExemplarHandler
from public.util_models import FE_cls, SoftTarget_CrossEntropy, FE_2fc_cls, FE_3fc_cls, BiasLayer

# -------------------------------------------------------------------------------------------------#

# --------------------#
# ----- EFAfIL -----#
# --------------------
from public.utils import AverageMeter, accuracy, split_array


class EFAfIL(EFAfIL_FeaturesHandler):
    def __init__(self, model_name, MLP_name, discriminate_name, dataset_name,
                 dataset_path, num_classes, img_size, rate, discriminate_note_rate, tasks,
                 logger, batch_train_logger, result_file, use_exemplars,
                 hidden_size, Exemple_memory_budget,
                 Feature_memory_budget, optim_type, MLP_optim_type,
                 norm_exemplars, herding, FM_reTrain, use_NewfeatureSpace, batch_size,
                 num_workers, seed, availabel_cudas,

                 epochs, CNN_lr, CNN_momentum,
                 CNN_weight_decay, CNN_milestones,
                 kd_lamb, fd_gamma, lrgamma, KD_temp,

                 MLP_lr, MLP_rate, MLP_momentum, MLP_KD_temp, MLP_KD_temp_2, MLP_distill_rate,
                 MLP_epochs, MLP_milestones,
                 svm_sample_type,
                 MLP_weight_decay,
                 MLP_lrgamma, sim_alpha, svm_max_iter, use_FM, use_discriminate, note, oversample):
        EFAfIL_FeaturesHandler.__init__(self, MLP_name=MLP_name, num_classes=num_classes, hidden_size=hidden_size,
                                        Exemple_memory_budget=Exemple_memory_budget, num_workers=num_workers,
                                        Feature_memory_budget=Feature_memory_budget, norm_exemplars=norm_exemplars,
                                        herding=herding, batch_size=batch_size, sim_alpha=sim_alpha, MLP_lr=MLP_lr,
                                        MLP_rate=MLP_rate,
                                        MLP_momentum=MLP_momentum, MLP_milestones=MLP_milestones,
                                        MLP_lrgamma=MLP_lrgamma, MLP_weight_decay=MLP_weight_decay,
                                        MLP_epochs=MLP_epochs, MLP_optim_type=MLP_optim_type,
                                        MLP_KD_temp=MLP_KD_temp, MLP_KD_temp_2=MLP_KD_temp_2,
                                        MLP_distill_rate=MLP_distill_rate,
                                        svm_sample_type=svm_sample_type,
                                        svm_max_iter=svm_max_iter, availabel_cudas=availabel_cudas, logger=logger,
                                        batch_train_logger=batch_train_logger)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.discriminate_name = discriminate_name
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.img_size = img_size
        self.rate = rate
        self.discriminate_note_rate = discriminate_note_rate
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
        self.use_discriminate = use_discriminate
        self.note = note
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

        self.original_train_datasets, self.train_datasets, self.val_datasets, self.data_config, \
        self.classes_per_task = self.get_dataset(dataset_name)
        self.dis_model = None
        self.pre_FE_cls = None
        self.bias_layer = None
        self.FE_cls = self.construct_model(rate)
        self.category_dataset = self.get_dataset_category(dataset_name)
        # self.FE_cls = self.load_FE_cls_model(
        #     "/share/home/kcli/CL_research/iCaRL_ILtFA/log/FE_cls_cifar10_preTrain_log/resnet34/rate_/"
        #     "cifar10_resnet34_rate_1_FE_cls.pth")
        # self.FE_cls = self.load_FE_cls_model(
        #     "/share/home/kcli/CL_research/iCaRL_ILtFA/pretrain_models/cifar10_pretrain_1_4.pth")
        # self.FE_cls = self.load_FE_2fc_cls_model(
        #     "/share/home/kcli/CL_research/iCaRL_ILtFA/checkpoints/cifar10_resnet34_rate_1_FE_2fc_cls_64.pth")
        # self.FE_cls = self.load_FE_3fc_cls_model("/share/home/kcli/CL_research/iCaRL_ILtFA/checkpoints/"
        #                                          "cifar10_resnet34_rate_1_FE_3fc_cls_64.pth")

    def forward(self, x):
        final_features, target = self.FE_cls(x)
        return final_features, target

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

    def build_discriminate(self):
        if self.availabel_cudas:
            model = MLP.__dict__[self.discriminate_name](input_dim=self.hidden_size, out_dim=1,
                                                         rate=self.discriminate_note_rate)
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = MLP.__dict__[self.discriminate_name](input_dim=self.hidden_size, out_dim=1,
                                                         rate=self.discriminate_note_rate)
        print(type(model))
        print(model)
        return model

    def construct_model(self, rate):
        if self.availabel_cudas:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
            device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
            model = FE_cls(resnetforcifar.__dict__[self.model_name](rate=rate, get_feature=True), self.hidden_size,
                           self.num_classes)
            # model = torch.nn.DataParallel(model).cuda()
            model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            model = FE_cls(resnetforcifar.__dict__[self.model_name](rate=rate, get_feature=True), self.hidden_size,
                           self.class_num)
        # model = torch.load("/share/home/kcli/CL_research/iCaRL_ILtFA/pretrain_models/cifar10_pretrain_1_4.pth")
        print(type(model))
        # print(model)
        return model

    def load_FE_cls_model(self, model_path):
        if self.availabel_cudas:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
            device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
            # model = torch.nn.DataParallel(FE_cls(resnetforimagenet.__dict__[self.model_name](rate=rate, get_feature=True),
            #                                      self.hidden_size, self.num_classes),
            #                               device_ids=device_ids).cuda()
            model = FE_cls(resnetforcifar.__dict__[self.model_name](rate=self.rate, get_feature=True),
                           int(512 * self.rate),
                           self.num_classes)
            model.load_state_dict(
                torch.load(model_path))
            # model = torch.nn.DataParallel(model).cuda()
            model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
            # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            model = FE_cls(resnetforcifar.__dict__[self.model_name](rate=self.rate, get_feature=True),
                           int(512 * self.rate),
                           self.num_classes)
            model.load_state_dict(
                torch.load(model_path))
        print(model)
        return model
        pass

    def load_FE_2fc_cls_model(self, model_path):
        if self.availabel_cudas:
            # os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
            # device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
            # model = torch.nn.DataParallel(FE_cls(resnetforimagenet.__dict__[self.model_name](rate=rate, get_feature=True),
            #                                      self.hidden_size, self.num_classes),
            #                               device_ids=device_ids).cuda()
            model = FE_2fc_cls(resnetforcifar.__dict__[self.model_name](rate=self.rate, get_feature=True),
                               int(512 * self.rate),
                               self.hidden_size, self.num_classes)
            model.load_state_dict(
                torch.load(model_path))
            model = torch.nn.DataParallel(model).cuda()
            # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
            # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            model = FE_2fc_cls(resnetforcifar.__dict__[self.model_name](rate=self.rate, get_feature=True),
                               int(512 * self.rate),
                               self.hidden_size, self.num_classes)
            model.load_state_dict(
                torch.load(model_path))
        print(type(model))
        print(model)
        return model
        pass

    def load_FE_3fc_cls_model(self, model_path):
        if self.availabel_cudas:
            # os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
            # device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
            # model = torch.nn.DataParallel(FE_cls(resnetforimagenet.__dict__[self.model_name](rate=rate, get_feature=True),
            #                                      self.hidden_size, self.num_classes),
            #                               device_ids=device_ids).cuda()
            model = FE_3fc_cls(resnetforcifar.__dict__[self.model_name](rate=self.rate, get_feature=True),
                               int(512 * self.rate),
                               self.hidden_size, self.num_classes)
            model.load_state_dict(
                torch.load(model_path))
            model = torch.nn.DataParallel(model).cuda()
            # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
            # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            model = FE_3fc_cls(resnetforcifar.__dict__[self.model_name](rate=self.rate, get_feature=True),
                               int(512 * self.rate),
                               self.hidden_size, self.num_classes)
            model.load_state_dict(
                torch.load(model_path))
        print(type(model))
        print(model)
        return model
        pass

    def build_dis_model_optimize(self):
        # Define optimizer (only include parameters that "requires_grad")
        optim_list = [{'params': filter(lambda p: p.requires_grad, self.dis_model.parameters()), 'lr': self.MLP_lr}]
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

    def get_discriminate_output(self, features):
        mode = self.dis_model.training
        self.dis_model.eval()
        with torch.no_grad():
            output = self.dis_model(features)
        self.dis_model.train(mode=mode)
        return output

    def current_task_model_feature_extractor(self, images):
        pass

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

    def get_cls_forward_target(self, features):
        if type(self.FE_cls) is torch.nn.DataParallel:
            return self.FE_cls.module.cls_forward(features)
        else:
            return self.FE_cls.cls_forward(features)

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
            iter_num = 0
            while iters_left > 0:
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                iters_left -= 1
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
                    rlts = f"Task {1}, batch train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                           f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                           f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"
                    self.batch_train_logger.info(rlts)
                    print(rlts)
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
        img_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS["CIFAR100_examplar"]["train_transform"],
        ])
        for task, train_dataset in enumerate(self.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')
            self.batch_train_logger.info(f'New task {task} begin.')
            print("New task %d begin." % task)
            # Add exemplars (if available) to current dataset (if requested)
            if self.use_exemplars and task > 1:
                exemplar_dataset = ExemplarDataset(self.exemplar_sets, img_transform=img_transform,
                                                   inv_transform=inv_CIFAR_100_normalize)
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
                    iter_num = 0
                    while iters_left > 0:
                        # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                        iters_left -= 1
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
        EFAfIL_result = {"timestamp": str(datetime.datetime.now()), "model_rate": self.rate, "EXP_note": self.note}
        EFAfIL_result.update(self.data_config)
        img_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS[self.dataset_name]["train_transform"],
        ])
        if self.dataset_name == "CIFAR10":
            img_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS["CIFAR100"]["train_transform"],
            ])
        for task, train_dataset in enumerate(self.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')
            self.batch_train_logger.info(f'New task {task} begin.')
            print("New task %d begin." % task)
            # Add exemplars (if available) to current dataset (if requested)
            if self.Exemple_memory_budget > 0 and task > 1:
                copy_exemplar_sets = copy.deepcopy(self.exemplar_sets)
                exemplar_dataset = ExemplarDataset(copy_exemplar_sets, img_transform=img_transform,
                                                   inv_transform=inv_CIFAR_100_normalize)
                training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
            else:
                training_dataset = train_dataset

            active_classes = list(range(self.classes_per_task * task))
            if task > 1 and self.Exemple_memory_budget > 0:
                self.logger.info(f'New task {task} begin: use feature_exemplar_dataset '
                                 f'to train FA'
                                 f'use feature_exemplar_dataset to train FE_cls.')

                if self.Exemple_memory_budget > 0 and args.FM_oversample:
                    self.EFAfIL_split_feature_mapper_cls_domain_train_exemplarOverSample(train_dataset,
                                                                                         exemplar_dataset,
                                                                                         self.val_datasets,
                                                                                         self.classes_per_task,
                                                                                         active_classes, task)


                else:
                    self.EFAfIL_split_feature_mapper_cls_domain_train(training_dataset, self.val_datasets,
                                                                      self.classes_per_task,
                                                                      active_classes, task)
                '''修正FM_cls的有偏差的cls'''
                if args.train_bias_cls_of_FMcls:
                    num_exemplar_per_class = int(self.Exemple_memory_budget / ((task - 1) * self.classes_per_task))
                    begin_class = int((task - 1) * self.classes_per_task)
                    end_class_plus_1 = int(begin_class + self.classes_per_task)
                    train_dataset_bias_cls_of_FMcls = copy.deepcopy(exemplar_dataset)
                    for class_index in range(begin_class, end_class_plus_1):
                        # print("category_dataset len:", len(self.category_dataset), "||", "class_index:", class_index)
                        split_for_train_temp, _ = torch.utils.data.random_split(self.category_dataset[class_index],
                                                                                [num_exemplar_per_class,
                                                                                 len(self.category_dataset[
                                                                                         class_index]) -
                                                                                 num_exemplar_per_class],
                                                                                generator=torch.Generator().manual_seed(
                                                                                    0))

                        train_dataset_bias_cls_of_FMcls = ConcatDataset([train_dataset_bias_cls_of_FMcls,
                                                                         split_for_train_temp])
                    self.retrain_bias_cls_of_FMcls(train_dataset_bias_cls_of_FMcls, self.val_datasets, task,
                                                   active_classes, args.print_interval,
                                                   train_method=args.train_method, bias_or_cRT=args.bias_or_cRT)

                self.pre_FE_cls = copy.deepcopy(self.FE_cls).eval()
                optimizer = self.build_optimize()
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=self.milestones, gamma=self.gamma)
                dis_model_optimizer = None
                if self.use_discriminate:
                    self.dis_model = self.build_discriminate()
                    dis_model_optimizer = self.build_dis_model_optimize()
                    dis_model_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        dis_model_optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
                '''get feature_replay data loader'''
                feature_replay_datasets = ExemplarDataset(self.exemplar_feature_sets)
                feature_replay_dataset_index = 0
                feature_batch_size = int(self.batch_size * args.feature_BatchSize_rate)
                feature_replay_data_loader = iter(utils.get_data_loader(feature_replay_datasets, feature_batch_size,
                                                                        self.num_workers,
                                                                        cuda=True if self.availabel_cudas else False))
                feature_replay_dataset_num = len(feature_replay_data_loader)
                print("feature_replay_dataset_num:", feature_replay_dataset_num)

                for epoch in range(1, self.epochs + 1):
                    is_first_ite = True
                    iters_left = 1
                    iter_index = 0
                    iter_num = 0
                    '''if feature_replay end, get new feature_replay data loader'''
                    if feature_replay_dataset_index == feature_replay_dataset_num:
                        feature_replay_data_loader = iter(
                            utils.get_data_loader(feature_replay_datasets, feature_batch_size, self.num_workers,
                                                  cuda=True if self.availabel_cudas else False))
                        feature_replay_dataset_index = 0

                    while iters_left > 0:
                        # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                        iters_left -= 1
                        if is_first_ite:
                            is_first_ite = False
                            data_loader = iter(
                                utils.get_data_loader(training_dataset, self.batch_size, self.num_workers,
                                                      cuda=True if self.availabel_cudas else False))
                            iter_num = iters_left = len(data_loader)
                            continue

                        # -----------------Collect data------------------#

                        #####-----CURRENT BATCH-----#####
                        x, y = next(data_loader)  # --> sample training data of current task

                        '''if feature_replay end, get new feature_replay data loader'''
                        if feature_replay_dataset_index == feature_replay_dataset_num:
                            feature_replay_data_loader = iter(
                                utils.get_data_loader(feature_replay_datasets, feature_batch_size, self.num_workers,
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
                        if args.train_bias_cls_of_FMcls:
                            assert self.unbias_cls_of_FMcls is not None
                            self.unbias_cls_of_FMcls.eval()
                            with torch.no_grad():
                                if args.bias_or_cRT == 0:
                                    if active_classes is not None:
                                        class_entries = active_classes[-1] if type(
                                            active_classes[0]) == list else active_classes
                                        FM_targets = FM_targets[:, class_entries]
                                        feature_replay_FM_targets = feature_replay_FM_targets[:, class_entries]
                                    FM_targets = self.unbias_cls_of_FMcls(FM_targets, self.classes_per_task)
                                    feature_replay_FM_targets = self.unbias_cls_of_FMcls(feature_replay_FM_targets,
                                                                                         self.classes_per_task)
                                else:
                                    FM_targets = self.unbias_cls_of_FMcls(FM_features)
                                    feature_replay_FM_targets = self.unbias_cls_of_FMcls(feature_replay_FM_features)
                                    if active_classes is not None:
                                        class_entries = active_classes[-1] if type(
                                            active_classes[0]) == list else active_classes
                                        FM_targets = FM_targets[:, class_entries]
                                        feature_replay_FM_targets = feature_replay_FM_targets[:, class_entries]
                        # ---> Train MAIN MODEL
                        # Train the main model with this batch
                        loss_dict = self.train_a_batch_feature_replay(x, y, feature_replay_features,
                                                                      feature_replay_labels, FM_targets=FM_targets,
                                                                      FM_features=FM_features,
                                                                      feature_replay_FM_features=feature_replay_FM_features,
                                                                      feature_replay_FM_targets=feature_replay_FM_targets,
                                                                      optimizer=optimizer,
                                                                      active_classes=active_classes, task=task,
                                                                      dis_model_optimizer=dis_model_optimizer,
                                                                      epoch=epoch,
                                                                      is_iCaRL_LwF_BiC=args.is_iCaRL_LwF_BiC,
                                                                      feature_batch_size=feature_batch_size,
                                                                      test_feature_cls_sim=args.test_feature_cls_sim,
                                                                      use_feature_replay_in_new_model=
                                                                      args.use_feature_replay_in_new_model)
                        iter_index += 1
                        if iter_index % args.print_interval == 0:
                            if self.use_discriminate:
                                rlt = f"Task {task:0>3d}, batch train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                      f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                      f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}" \
                                      f"GAN train:, {dis_model_optimizer_scheduler.get_last_lr()[0]:.6f}, {loss_dict['GAN_train']}"
                            else:
                                rlt = f"Task {task:0>3d}, batch train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                      f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                      f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"
                            self.batch_train_logger.info(
                                rlt
                            )
                            print(rlt)
                    scheduler.step()
                    if self.use_discriminate:
                        dis_model_optimizer_scheduler.step()
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes)
                    results = f"batch train FE_cls || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, " \
                              f"top5 acc:  {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    self.batch_train_logger.info(
                        results
                    )
                    print(results)
                    print("-----------------------------")
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes,
                                                                        val_current_task=False)
                    self.batch_train_logger.info(
                        f"batch train old classes validate || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
                        f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    )
                    self.batch_train_logger.info(f"------------------------------------------------------------------")
                    print(
                        f'batch train old classes : {task:0>3d}, epoch: {epoch}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (
                            acc1, acc5))
                    print("------------------------------------")
            else:
                if self.rate == 1:
                    if args.is_iCaRL_LwF_BiC == 0:

                        if self.lr == 1:
                            model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                   "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task2/test2/rate_/" \
                                                   "cifar100_checkpoints/50_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                            model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                   "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task5/test2/rate_/" \
                                                   "cifar100_checkpoints/20_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                            model_path_for_task10 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                    "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task10/test2/rate_/" \
                                                    "cifar100_checkpoints/10_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                            model_path_for_task20 = ""
                        elif self.lr == 0.1:
                            model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                   "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task2/test3/rate_/" \
                                                   "cifar100_checkpoints/50_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                            if self.dataset_name == "CIFAR10":
                                model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                                                       "cifar10_Test/_Test_iCaRL_original/Test_unbias/biasModel/task2/rate_/" \
                                                       "test1/cifar100_checkpoints/" \
                                                       "5_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                            model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                   "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task5/test3/rate_/" \
                                                   "cifar100_checkpoints/20_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                            if self.dataset_name == "CIFAR10":
                                model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                                                       "cifar10_Test/_Test_iCaRL_original/iCaRL/task5/rate_/test1/" \
                                                       "cifar100_checkpoints/2_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                            model_path_for_task10 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                    "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task10/test3/rate_/" \
                                                    "cifar100_checkpoints/10_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                            model_path_for_task20 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                    "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task20/test2/rate_/" \
                                                    "cifar100_checkpoints/5_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                        elif self.lr == 2:
                            model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                   "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task2/test1/rate_/" \
                                                   "cifar100_checkpoints/50_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                            model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                   "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task5/test1/rate_/" \
                                                   "cifar100_checkpoints/20_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                            model_path_for_task10 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                    "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task10/test1/rate_/" \
                                                    "cifar100_checkpoints/10_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                            model_path_for_task20 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                    "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task20/test1/rate_/" \
                                                    "cifar100_checkpoints/5_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                    else:
                        model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                               "cifar10pretrain/LwF_EXP/LwF_MC/task2/test1/rate_/cifar100_checkpoints/" \
                                               "50_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                        if self.dataset_name == "CIFAR10":
                            model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                                                   "cifar10_Test/LwF_EXP/Test_unbias/biasModel/task2/rate_/test_1/" \
                                                   "cifar100_checkpoints/5_cifar100_resnet34_LwF_rate_1_FE_cls_task1.pth"

                        model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                               "cifar10pretrain/LwF_EXP/LwF_MC/task5/test1/rate_/cifar100_checkpoints/" \
                                               "20_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                        model_path_for_task10 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                "cifar10pretrain/LwF_EXP/LwF_MC/task10/test1/rate_/cifar100_checkpoints/" \
                                                "10_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                        model_path_for_task20 = ""
                else:
                    if self.lr == 0.1:
                        model_path_for_task2 = ""

                        model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                                               "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task5/test1/rate_1_4/" \
                                               "cifar100_checkpoints/20_cifar100_resnet34_iCaRL_rate_0.25_FE_cls_task1.pth"

                        model_path_for_task10 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                                                "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task10/rate_1_4/test1/" \
                                                "cifar100_checkpoints/10_cifar100_resnet34_iCaRL_rate_0.25_FE_cls_task1.pth"

                        model_path_for_task20 = ""

                if self.tasks == 2:
                    model_path = model_path_for_task2
                elif self.tasks == 5:
                    model_path = model_path_for_task5
                elif self.tasks == 10:
                    model_path = model_path_for_task10
                elif self.tasks == 20:
                    model_path = model_path_for_task20

                self.FE_cls = self.load_FE_cls_model(model_path)
                # print("model for tasks{}:".format(self.tasks), self.FE_cls)
                # self.train_FE_cls(args, training_dataset, active_classes)
            if task > 1 and self.Exemple_memory_budget > 0:
                self.cifar100_feature_handle_main_FM(train_dataset, exemplar_dataset,
                                                     self.original_train_datasets[task - 1],
                                                     self.val_datasets,
                                                     self.classes_per_task,
                                                     active_classes, task, self.img_size, self.hidden_size,
                                                     use_dynamicMem=True,
                                                     FM_reTrain=self.FM_reTrain,
                                                     use_NewfeatureSpace=self.use_NewfeatureSpace)
            else:
                self.cifar100_feature_handle_main_FM(train_dataset, None,
                                                     self.original_train_datasets[task - 1],
                                                     self.val_datasets,
                                                     self.classes_per_task,
                                                     active_classes, task, self.img_size, self.hidden_size,
                                                     use_dynamicMem=True,
                                                     FM_reTrain=self.FM_reTrain,
                                                     use_NewfeatureSpace=self.use_NewfeatureSpace)
            self.batch_train_logger.info(f'##########feature extractor train task {task} End.#########')
            self.logger.info(f'#############feature extractor train task {task} End.##############')
            self.logger.info(f'#############task {task} validate  begin.##############')
            print("feature extractor train task-%d End" % (task))
            # acc_past_tasks, acc_list = self.tasks_validate(task)
            # EFAfIL_result["task_{}_svm_results".format(task)] = acc_past_tasks
            self.compute_means = True
            if self.Exemple_memory_budget > 0:
                acc_past_tasks, acc_list = self.tasks_validate(task, classifier="examplar_ncm", use_FM=self.use_FM)
                EFAfIL_result["task_{}_examplar_ncm_results".format(task)] = acc_past_tasks
                acc_past_tasks, acc_list = self.tasks_validate(task, classifier="examplar_feature_ncm",
                                                               use_FM=self.use_FM)
                EFAfIL_result["task_{}_examplar_feature_ncm_results".format(task)] = acc_past_tasks
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="fcls", active_classes=active_classes)
            EFAfIL_result["task_{}_FE_cls_results".format(task)] = acc_past_tasks
            with open(self.result_file, 'w') as fw:
                json.dump(EFAfIL_result, fw, indent=4)
        with open(self.result_file, 'w') as fw:
            json.dump(EFAfIL_result, fw, indent=4)

    def feature_replay_train_main_exemplar_oversample(self, args):
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
        EFAfIL_result = {"timestamp": str(datetime.datetime.now()), "model_rate": self.rate, "EXP_note": self.note}
        EFAfIL_result.update(self.data_config)
        img_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS["CIFAR100_examplar"]["train_transform"],
        ])
        for task, train_dataset in enumerate(self.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')
            self.batch_train_logger.info(f'New task {task} begin.')
            print("New task %d begin." % task)
            # Add exemplars (if available) to current dataset (if requested)
            if self.use_exemplars and task > 1:
                copy_exemplar_sets = copy.deepcopy(self.exemplar_sets)
                exemplar_dataset = ExemplarDataset(copy_exemplar_sets, img_transform=img_transform,
                                                   inv_transform=inv_CIFAR_100_normalize)
                training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
            else:
                training_dataset = train_dataset

            active_classes = list(range(self.classes_per_task * task))
            if task > 1:
                self.logger.info(f'New task {task} begin: use feature_exemplar_dataset '
                                 f'to train FA'
                                 f'use feature_exemplar_dataset to train FE_cls.')
                if args.FM_oversample:
                    self.EFAfIL_split_feature_mapper_cls_domain_train_exemplarOverSample(train_dataset,
                                                                                         exemplar_dataset,
                                                                                         self.val_datasets,
                                                                                         self.classes_per_task,
                                                                                         active_classes, task)
                else:
                    self.EFAfIL_split_feature_mapper_cls_domain_train(training_dataset, self.val_datasets,
                                                                      self.classes_per_task,
                                                                      active_classes, task)

                self.pre_FE_cls = copy.deepcopy(self.FE_cls).eval()
                optimizer = self.build_optimize()
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=self.milestones, gamma=self.gamma)
                dis_model_optimizer = None
                if self.use_discriminate:
                    self.dis_model = self.build_discriminate()
                    dis_model_optimizer = self.build_dis_model_optimize()
                    dis_model_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        dis_model_optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
                '''get feature_replay data loader'''
                feature_replay_datasets = ExemplarDataset(self.exemplar_feature_sets)
                feature_replay_dataset_index = 0
                feature_replay_data_loader = iter(utils.get_data_loader(feature_replay_datasets, self.batch_size,
                                                                        self.num_workers,
                                                                        cuda=True if self.availabel_cudas else False))
                feature_replay_dataset_num = len(feature_replay_data_loader)
                print("feature_replay_dataset_num:", feature_replay_dataset_num)

                '''get exemplar_replay data loader'''
                exemplar_replay_dataset_index = 0
                exemplar_replay_data_loader = iter(utils.get_data_loader(exemplar_dataset, self.batch_size,
                                                                         self.num_workers,
                                                                         cuda=True if self.availabel_cudas else False))
                exemplar_replay_dataset_num = len(exemplar_replay_data_loader)
                print("exemplar_replay_dataset_num:", exemplar_replay_dataset_num)

                for epoch in range(1, self.epochs + 1):
                    is_first_ite = True
                    iters_left = 1
                    iter_index = 0
                    iter_num = 0
                    '''if feature_replay end, get new feature_replay data loader'''
                    if feature_replay_dataset_index == feature_replay_dataset_num:
                        feature_replay_data_loader = iter(
                            utils.get_data_loader(feature_replay_datasets, self.batch_size, self.num_workers,
                                                  cuda=True if self.availabel_cudas else False))
                        feature_replay_dataset_index = 0

                    '''if exemplar_replay end, get new feature_replay data loader'''
                    if exemplar_replay_dataset_index == exemplar_replay_dataset_num:
                        exemplar_replay_data_loader = iter(utils.get_data_loader(exemplar_dataset, self.batch_size,
                                                                                 self.num_workers,
                                                                                 cuda=True if self.availabel_cudas else False))
                        exemplar_replay_dataset_index = 0

                    while iters_left > 0:
                        # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                        iters_left -= 1
                        if is_first_ite:
                            is_first_ite = False
                            data_loader = iter(
                                utils.get_data_loader(train_dataset, self.batch_size, self.num_workers,
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

                        '''if exemplar_replay end, get new feature_replay data loader'''
                        if exemplar_replay_dataset_index == exemplar_replay_dataset_num:
                            exemplar_replay_data_loader = iter(utils.get_data_loader(exemplar_dataset, self.batch_size,
                                                                                     self.num_workers,
                                                                                     cuda=True if self.availabel_cudas else False))
                            exemplar_replay_dataset_index = 0

                        feature_replay_features, feature_replay_labels = next(feature_replay_data_loader)
                        feature_replay_dataset_index += 1

                        exemplar_replay_imgs, exemplar_replay_labels = next(exemplar_replay_data_loader)
                        exemplar_replay_dataset_index += 1

                        x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                        feature_replay_features, feature_replay_labels = feature_replay_features.to(
                            self.device), feature_replay_labels.to(self.device)
                        exemplar_replay_imgs, exemplar_replay_labels = exemplar_replay_imgs.to(
                            self.device), exemplar_replay_labels.to(self.device)
                        '''获取图片在pre_FEcls跟FM模型的输出'''
                        FM_features, FM_targets = self.get_preFE_FM_feature_target(x)
                        FM_exemplar_features, FM_exemplar_targets = self.get_preFE_FM_feature_target(
                            exemplar_replay_imgs)
                        '''获取保存的features在FM模型的输出'''
                        feature_replay_FM_features = self.feature_mapping(feature_replay_features)
                        feature_replay_FM_targets = self.prefeature_2_FMtarget(feature_replay_FM_features)
                        # ---> Train MAIN MODEL
                        # Train the main model with this batch
                        loss_dict = self.train_a_batch_feature_replay_exemplar_balance(x, y, exemplar_replay_imgs,
                                                                                       exemplar_replay_labels,
                                                                                       feature_replay_features,
                                                                                       feature_replay_labels,
                                                                                       FM_targets=FM_targets,
                                                                                       FM_features=FM_features,
                                                                                       FM_exemplar_targets=FM_exemplar_targets,
                                                                                       FM_exemplar_features=FM_exemplar_features,
                                                                                       feature_replay_FM_features=feature_replay_FM_features,
                                                                                       feature_replay_FM_targets=feature_replay_FM_targets,
                                                                                       optimizer=optimizer,
                                                                                       active_classes=active_classes,
                                                                                       task=task,
                                                                                       dis_model_optimizer=dis_model_optimizer,
                                                                                       epoch=epoch,
                                                                                       is_iCaRL_LwF_BiC=args.is_iCaRL_LwF_BiC,
                                                                                       test_feature_cls_sim=args.test_feature_cls_sim,
                                                                                       use_feature_replay_in_new_model=
                                                                                       args.use_feature_replay_in_new_model)
                        iter_index += 1
                        if iter_index % args.print_interval == 0:
                            if self.use_discriminate:
                                rlt = f"Task {task:0>3d}, batch train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                      f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                      f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}" \
                                      f"GAN train:, {dis_model_optimizer_scheduler.get_last_lr()[0]:.6f}, {loss_dict['GAN_train']}"
                            else:
                                rlt = f"Task {task:0>3d}, batch train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                      f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                      f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"
                            self.batch_train_logger.info(
                                rlt
                            )
                            print(rlt)
                    scheduler.step()
                    if self.use_discriminate:
                        dis_model_optimizer_scheduler.step()
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes)
                    results = f"batch train FE_cls || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, " \
                              f"top5 acc:  {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    self.batch_train_logger.info(
                        results
                    )
                    print(results)
                    print("-----------------------------")
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes,
                                                                        val_current_task=False)
                    self.batch_train_logger.info(
                        f"batch train old classes validate || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
                        f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    )
                    self.batch_train_logger.info(f"------------------------------------------------------------------")
                    print(
                        f'batch train old classes : {task:0>3d}, epoch: {epoch}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (
                            acc1, acc5))
                    print("------------------------------------")
            else:
                if self.rate == 1:
                    if args.is_iCaRL_LwF_BiC == 0:

                        if self.lr == 1:
                            model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                   "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task2/test2/rate_/" \
                                                   "cifar100_checkpoints/50_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                            model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                   "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task5/test2/rate_/" \
                                                   "cifar100_checkpoints/20_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                            model_path_for_task10 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                    "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task10/test2/rate_/" \
                                                    "cifar100_checkpoints/10_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                            model_path_for_task20 = ""
                        elif self.lr == 0.1:
                            model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                   "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task2/test3/rate_/" \
                                                   "cifar100_checkpoints/50_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                            model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                   "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task5/test3/rate_/" \
                                                   "cifar100_checkpoints/20_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                            model_path_for_task10 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                    "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task10/test3/rate_/" \
                                                    "cifar100_checkpoints/10_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                            model_path_for_task20 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                    "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task20/test2/rate_/" \
                                                    "cifar100_checkpoints/5_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                        elif self.lr == 2:
                            model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                   "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task2/test1/rate_/" \
                                                   "cifar100_checkpoints/50_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                            model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                   "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task5/test1/rate_/" \
                                                   "cifar100_checkpoints/20_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                            model_path_for_task10 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                    "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task10/test1/rate_/" \
                                                    "cifar100_checkpoints/10_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                            model_path_for_task20 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                    "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task20/test1/rate_/" \
                                                    "cifar100_checkpoints/5_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                    else:
                        model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                               "cifar10pretrain/LwF_EXP/LwF_MC/task2/test1/rate_/cifar100_checkpoints/" \
                                               "50_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                        model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                               "cifar10pretrain/LwF_EXP/LwF_MC/task5/test1/rate_/cifar100_checkpoints/" \
                                               "20_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                        model_path_for_task10 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                                "cifar10pretrain/LwF_EXP/LwF_MC/task10/test1/rate_/cifar100_checkpoints/" \
                                                "10_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                        model_path_for_task20 = ""
                else:
                    if self.lr == 0.1:
                        model_path_for_task2 = ""

                        model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                                               "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task5/test1/rate_1_4/" \
                                               "cifar100_checkpoints/20_cifar100_resnet34_iCaRL_rate_0.25_FE_cls_task1.pth"

                        model_path_for_task10 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                                                "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task10/rate_1_4/test1/" \
                                                "cifar100_checkpoints/10_cifar100_resnet34_iCaRL_rate_0.25_FE_cls_task1.pth"

                        model_path_for_task20 = ""

                if self.tasks == 2:
                    model_path = model_path_for_task2
                elif self.tasks == 5:
                    model_path = model_path_for_task5
                elif self.tasks == 10:
                    model_path = model_path_for_task10
                elif self.tasks == 20:
                    model_path = model_path_for_task20

                self.FE_cls = self.load_FE_cls_model(model_path)
                print("model for tasks{}:".format(self.tasks), self.FE_cls)
                # self.train_FE_cls(args, training_dataset, active_classes)
            if task > 1:
                self.cifar100_feature_handle_main_FM(train_dataset, exemplar_dataset,
                                                     self.original_train_datasets[task - 1],
                                                     self.val_datasets,
                                                     self.classes_per_task,
                                                     active_classes, task, self.img_size, self.hidden_size,
                                                     use_dynamicMem=True,
                                                     FM_reTrain=self.FM_reTrain,
                                                     use_NewfeatureSpace=self.use_NewfeatureSpace)
            else:
                self.cifar100_feature_handle_main_FM(train_dataset, None,
                                                     self.original_train_datasets[task - 1],
                                                     self.val_datasets,
                                                     self.classes_per_task,
                                                     active_classes, task, self.img_size, self.hidden_size,
                                                     use_dynamicMem=True,
                                                     FM_reTrain=self.FM_reTrain,
                                                     use_NewfeatureSpace=self.use_NewfeatureSpace)
            self.batch_train_logger.info(f'##########feature extractor train task {task} End.#########')
            self.logger.info(f'#############feature extractor train task {task} End.##############')
            self.logger.info(f'#############task {task} validate  begin.##############')
            print("feature extractor train task-%d End" % (task))
            # acc_past_tasks, acc_list = self.tasks_validate(task)
            # EFAfIL_result["task_{}_svm_results".format(task)] = acc_past_tasks
            self.compute_means = True
            if self.Exemple_memory_budget > 0:
                acc_past_tasks, acc_list = self.tasks_validate(task, classifier="examplar_ncm", use_FM=self.use_FM)
                EFAfIL_result["task_{}_examplar_ncm_results".format(task)] = acc_past_tasks
                acc_past_tasks, acc_list = self.tasks_validate(task, classifier="examplar_feature_ncm",
                                                               use_FM=self.use_FM)
                EFAfIL_result["task_{}_examplar_feature_ncm_results".format(task)] = acc_past_tasks
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="fcls", active_classes=active_classes)
            EFAfIL_result["task_{}_FE_cls_results".format(task)] = acc_past_tasks
            with open(self.result_file, 'w') as fw:
                json.dump(EFAfIL_result, fw, indent=4)
        with open(self.result_file, 'w') as fw:
            json.dump(EFAfIL_result, fw, indent=4)

    def feature_replay_train_main_BiC(self, args):
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
        sample_seed = args.sample_seed
        EFAfIL_result = {"timestamp": str(datetime.datetime.now()), "model_rate": self.rate, "EXP_note": self.note,
                         "sample_seed": sample_seed}
        EFAfIL_result.update(self.data_config)
        img_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS["CIFAR100_examplar"]["train_transform"],
        ])
        for task, train_dataset in enumerate(self.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')
            self.batch_train_logger.info(f'New task {task} begin.')
            print("New task %d begin." % task)
            # Add exemplars (if available) to current dataset (if requested)
            if task > 1 and self.Exemple_memory_budget > 0:
                examplar_train_data_array, examplar_val_data_array = split_array(self.exemplar_sets,
                                                                                 sample_seed=sample_seed)
                examplar_train_dataset = ExemplarDataset(examplar_train_data_array, img_transform=img_transform,
                                                         inv_transform=inv_CIFAR_100_normalize)
                examplar_val_dataset = ExemplarDataset(examplar_val_data_array, img_transform=img_transform,
                                                       inv_transform=inv_CIFAR_100_normalize)
                per_task_training_dataset = None
                per_task_train_dataset = None
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

                    if per_task_train_dataset is None:
                        per_task_train_dataset = train_dataset_temp
                    else:
                        per_task_train_dataset = ConcatDataset([per_task_train_dataset, train_dataset_temp])

                    per_task_valing_dataset = ConcatDataset([per_task_valing_dataset, val_dataset_temp])
                per_task_training_dataset = ConcatDataset([per_task_train_dataset, examplar_train_dataset])
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
                EFAfIL_train_dataset = ConcatDataset([train_dataset, feature_relearn_exemplar_dataset])

            else:
                per_task_training_dataset = train_dataset
                per_task_valing_dataset = None

                # Find [active_classes]
            active_classes = list(range(self.classes_per_task * task))
            if task > 1:
                self.logger.info(f'New task {task} begin: use feature_relearn_exemplar_dataset '
                                 f'to train FA'
                                 f'use feature_relearn_exemplar_dataset to train FE_cls.')
                assert feature_relearn_exemplar_dataset is not None
                if task <= 2:
                    self.EFAfIL_split_feature_mapper_cls_domain_train_exemplarOverSample(train_dataset,
                                                                                         feature_relearn_exemplar_dataset,
                                                                                         self.val_datasets,
                                                                                         self.classes_per_task,
                                                                                         active_classes, task)
                    # self.EFAfIL_split_feature_mapper_cls_domain_train(EFAfIL_train_dataset,
                    #                                                   self.val_datasets,
                    #                                                   self.classes_per_task,
                    #                                                   active_classes, task)
                else:
                    if args.FM_oversample:
                        self.EFAfIL_split_feature_mapper_cls_domain_train_bias_ovarsample(self.bias_layer,
                                                                                          train_dataset,
                                                                                          feature_relearn_exemplar_dataset,
                                                                                          self.val_datasets,
                                                                                          self.classes_per_task,
                                                                                          active_classes, task)
                    else:
                        self.EFAfIL_split_feature_mapper_cls_domain_train_bias(self.bias_layer, EFAfIL_train_dataset,
                                                                               self.val_datasets,
                                                                               self.classes_per_task,
                                                                               active_classes, task)
                self.pre_FE_cls = copy.deepcopy(self.FE_cls).eval()
                optimizer = self.build_optimize()
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=self.milestones, gamma=self.gamma)
                dis_model_optimizer = None
                if self.use_discriminate:
                    self.dis_model = self.build_discriminate()
                    dis_model_optimizer = self.build_dis_model_optimize()
                    dis_model_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        dis_model_optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
                '''get feature_replay data loader'''
                feature_replay_datasets = ExemplarDataset(self.exemplar_feature_sets)
                feature_replay_dataset_index = 0
                feature_replay_data_loader = iter(utils.get_data_loader(feature_replay_datasets, self.batch_size,
                                                                        self.num_workers,
                                                                        cuda=True if self.availabel_cudas else False))
                feature_replay_dataset_num = len(feature_replay_data_loader)
                print("feature_replay_dataset_num:", feature_replay_dataset_num)

                for epoch in range(1, self.epochs + 1):
                    is_first_ite = True
                    iters_left = 1
                    iter_index = 0
                    iter_num = 0
                    '''if feature_replay end, get new feature_replay data loader'''
                    if feature_replay_dataset_index == feature_replay_dataset_num:
                        feature_replay_data_loader = iter(
                            utils.get_data_loader(feature_replay_datasets, self.batch_size, self.num_workers,
                                                  cuda=True if self.availabel_cudas else False))
                        feature_replay_dataset_index = 0

                    while iters_left > 0:
                        # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                        iters_left -= 1
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
                                                                      active_classes=active_classes, task=task,
                                                                      dis_model_optimizer=dis_model_optimizer,
                                                                      epoch=epoch,
                                                                      is_iCaRL_LwF_BiC=args.is_iCaRL_LwF_BiC)
                        iter_index += 1
                        if iter_index % args.print_interval == 0:
                            if self.use_discriminate:
                                rlt = f"Task {task:0>3d}, batch train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                      f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                      f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}" \
                                      f"GAN train:, {dis_model_optimizer_scheduler.get_last_lr()[0]:.6f}, {loss_dict['GAN_train']}"
                            else:
                                rlt = f"Task {task:0>3d}, batch train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                                      f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                      f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"

                            self.batch_train_logger.info(
                                rlt
                            )
                            print(rlt)
                    scheduler.step()
                    if self.use_discriminate:
                        dis_model_optimizer_scheduler.step()
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes)
                    results = f"batch train FE_cls || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, " \
                              f"top5 acc:  {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    self.batch_train_logger.info(
                        results
                    )
                    print(results)


            elif task == 1:
                if self.rate == 1:
                    model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                           "cifar10pretrain/LwF_EXP/LwF_MC/task2/test1/rate_/cifar100_checkpoints/" \
                                           "50_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                    if self.dataset_name == "CIFAR10":
                        model_path_for_task2 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                                               "cifar10_Test/LwF_EXP/Test_unbias/biasModel/task2/rate_/test_1/" \
                                               "cifar100_checkpoints/5_cifar100_resnet34_LwF_rate_1_FE_cls_task1.pth"

                    model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                           "cifar10pretrain/LwF_EXP/LwF_MC/task5/test1/rate_/cifar100_checkpoints/" \
                                           "20_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                    if self.dataset_name == "CIFAR10":
                        model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                                               "cifar10_Test/LwF_EXP/LwF_OD/task5/rate_/test_1/cifar100_checkpoints/" \
                                               "2_cifar100_resnet34_LwF_rate_1_FE_cls_task1.pth"

                    model_path_for_task10 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_July/" \
                                            "cifar10pretrain/LwF_EXP/LwF_MC/task10/test1/rate_/cifar100_checkpoints/" \
                                            "10_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"
                else:
                    if self.lr == 0.1:
                        model_path_for_task2 = ""

                        model_path_for_task5 = "/share/home/kcli/CL_research/iCaRL_ILtFA/cifar100EXP_resnet34_Aug/" \
                                               "cifar10pretrain/_Test_iCaRL_original/iCaRL_cmp/task5/test1/rate_1_4/" \
                                               "cifar100_checkpoints/20_cifar100_resnet34_iCaRL_rate_1_FE_cls_task1.pth"

                        model_path_for_task10 = ""

                        model_path_for_task20 = ""


                if self.tasks == 2:
                    model_path = model_path_for_task2
                elif self.tasks == 5:
                    model_path = model_path_for_task5
                elif self.tasks == 10:
                    model_path = model_path_for_task10

                self.FE_cls = self.load_FE_cls_model(model_path)
                print("model for tasks{}:".format(self.tasks), self.FE_cls)
            if task <= len(self.train_datasets):
                if task > 1:
                    self.cifar100_feature_handle_main_FM(per_task_train_dataset, examplar_train_dataset,
                                                         self.original_train_datasets[task - 1],
                                                         self.val_datasets,
                                                         self.classes_per_task,
                                                         active_classes, task, self.img_size, self.hidden_size,
                                                         use_dynamicMem=True,
                                                         FM_reTrain=self.FM_reTrain,
                                                         use_NewfeatureSpace=self.use_NewfeatureSpace)
                else:
                    self.cifar100_feature_handle_main_FM(per_task_training_dataset, None,
                                                         self.original_train_datasets[task - 1],
                                                         self.val_datasets,
                                                         self.classes_per_task,
                                                         active_classes, task, self.img_size, self.hidden_size,
                                                         use_dynamicMem=True,
                                                         FM_reTrain=self.FM_reTrain,
                                                         use_NewfeatureSpace=self.use_NewfeatureSpace)
                # self.cifar100_feature_handle_main_FM(per_task_train_dataset, self.original_train_datasets[task - 1],
                #                                      self.val_datasets,
                #                                      self.classes_per_task,
                #                                      active_classes, task, self.img_size, self.hidden_size,
                #                                      use_dynamicMem=True,
                #                                      FM_reTrain=self.FM_reTrain,
                #                                      use_NewfeatureSpace=self.use_NewfeatureSpace)
                self.batch_train_logger.info(f'##########feature extractor train task {task} End.#########')
                self.logger.info(f'#############feature extractor train task {task} End.##############')
                self.logger.info(f'#############task {task} validate  begin.##############')
                print("feature extractor train task-%d End" % (task))
                if task > 1:
                    '''train bias layer'''
                    print("active_classes:", type(active_classes))
                    print(active_classes)
                    ''' add feature_replay_datasets to train bias_layer'''
                    feature_replay_datasets_for_bias_train = ExemplarDataset(self.exemplar_feature_sets)
                    self.logger.info(f"use train_BiasLayer_feature_replay to train bias_layers")
                    # self.train_BiasLayer_feature_replay(per_task_valing_dataset, feature_replay_datasets_for_bias_train,
                    #                                     task, active_classes, args.print_interval,
                    #                                     train_method=args.train_method)
                    self.train_BiasLayer(per_task_valing_dataset, task, active_classes, args.print_interval,
                                         train_method=args.train_method)
                if self.Exemple_memory_budget > 0:
                    acc_past_tasks, acc_list = self.tasks_validate(task, classifier="examplar_ncm", use_FM=self.use_FM)
                    EFAfIL_result["task_{}_examplar_ncm_results".format(task)] = acc_past_tasks
            acc_past_tasks, acc_list = self.tasks_validate(task, classifier="BiC_fc", active_classes=active_classes)
            EFAfIL_result["task_{}_FE_cls_results".format(task)] = acc_past_tasks
            with open(self.result_file, 'w') as fw:
                json.dump(EFAfIL_result, fw, indent=4)
        with open(self.result_file, 'w') as fw:
            json.dump(EFAfIL_result, fw, indent=4)

    def train_BiasLayer(self, per_task_valing_dataset, task, active_classes, print_interval, train_method=0):
        current_classes_num = self.classes_per_task
        self.bias_layer = BiasLayer()
        if train_method == 0:
            optimizer = self.build_optimize_biaslayer(self.lr / 100)
            epochs = 45
            gap = int(epochs / 3)
            milestones = [gap, 2 * gap]
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
                if iter_index % print_interval == 0:
                    results = f"bias layer train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                              f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                              f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}" \
                              f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
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

    def train_BiasLayer_feature_replay(self, per_task_valing_dataset, feature_replay_datasets, task, active_classes,
                                       print_interval, train_method=0):
        current_classes_num = self.classes_per_task
        self.bias_layer = BiasLayer()
        if train_method == 0:
            optimizer = self.build_optimize_biaslayer(self.lr / 100)
            epochs = 45
            gap = int(epochs / 3)
            milestones = [gap, 2 * gap]
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
        '''get new feature_replay data loader'''
        feature_replay_dataset_index = 0
        feature_replay_data_loader = iter(utils.get_data_loader(feature_replay_datasets, self.batch_size,
                                                                self.num_workers,
                                                                cuda=True if self.availabel_cudas else False))
        feature_replay_dataset_num = len(feature_replay_data_loader)

        for epoch in range(1, epochs + 1):
            is_first_ite = True
            iters_left = 1
            iter_index = 0
            iter_num = 0
            '''if feature_replay end, get new feature_replay data loader'''
            if feature_replay_dataset_index == feature_replay_dataset_num:
                feature_replay_data_loader = iter(
                    utils.get_data_loader(feature_replay_datasets, self.batch_size, self.num_workers,
                                          cuda=True if self.availabel_cudas else False))
                feature_replay_dataset_index = 0
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

                '''if feature_replay end, get new feature_replay data loader'''
                if feature_replay_dataset_index == feature_replay_dataset_num:
                    feature_replay_data_loader = iter(
                        utils.get_data_loader(feature_replay_datasets, self.batch_size, self.num_workers,
                                              cuda=True if self.availabel_cudas else False))
                    feature_replay_dataset_index = 0
                feature_replay_features, feature_replay_labels = next(feature_replay_data_loader)
                feature_replay_dataset_index += 1
                feature_replay_features, feature_replay_labels = feature_replay_features.to(self.device), \
                                                                 feature_replay_labels.to(
                                                                     self.device)  # --> transfer them to correct device

                FE_cls_targets = self.FE_cls_forward(x)
                FE_cls_targets = FE_cls_targets[:, :(self.classes_per_task * task)]
                '''计算在当前FE_cls的分类输出'''
                feature_FE_cls_target = self.get_cls_target(feature_replay_features)
                '''获取当前任务及以前任务的对应的输出'''
                feature_FE_cls_target = feature_FE_cls_target[:, :(self.classes_per_task * task)]
                ''''''
                FE_cls_targets = torch.cat([FE_cls_targets, feature_FE_cls_target], dim=0)
                y = torch.cat([y, feature_replay_labels], dim=0)

                loss_dict = self.BiasLayer_train_a_batch(FE_cls_targets, y, current_classes_num, optimizer,
                                                         active_classes)
                iter_index += 1
                if iter_index % print_interval == 0:
                    results = f"bias layer train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], " \
                              f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                              f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}" \
                              f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
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

    def feature_replay_train_main_max(self, args):
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
        EFAfIL_result = {"timestamp": str(datetime.datetime.now()), "model_rate": self.rate, "EXP_note": self.note}
        EFAfIL_result.update(self.data_config)
        img_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS["CIFAR100"]["train_transform"],
        ])
        for task, train_dataset in enumerate(self.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')
            self.batch_train_logger.info(f'New task {task} begin.')
            print("New task %d begin." % task)
            # Add exemplars (if available) to current dataset (if requested)
            if self.use_exemplars and task > 1:
                copy_exemplar_sets = copy.deepcopy(self.exemplar_sets)
                exemplar_dataset = ExemplarDataset(copy_exemplar_sets, img_transform=img_transform,
                                                   inv_transform=inv_CIFAR_100_normalize)
                # exemplar_dataset = ExemplarDataset(copy_exemplar_sets)
                training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
            else:
                training_dataset = train_dataset

            active_classes = list(range(self.classes_per_task * task))
            if task > 1:
                self.EFAfIL_split_feature_mapper_cls_domain_train(training_dataset, self.val_datasets,
                                                                  self.classes_per_task,
                                                                  active_classes, task)
                # self.EFAfIL_split_feature_mapper_cls_domain_train_improve(training_dataset, self.val_datasets,
                #                                                           self.classes_per_task,
                #                                                           active_classes, task)

                # self.EFAfIL_feature_mapper_cls_domain_train_improve(training_dataset, self.val_datasets,
                #                                                     self.classes_per_task,
                #                                                     active_classes, task)

                self.pre_FE_cls = copy.deepcopy(self.FE_cls).eval()
                optimizer = self.build_optimize()
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=self.milestones, gamma=self.gamma)
                dis_model_optimizer = None
                if self.use_discriminate:
                    self.dis_model = self.build_discriminate()
                    dis_model_optimizer = self.build_dis_model_optimize()
                    dis_model_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        dis_model_optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)

                train_dataset_iter_index = 0
                feature_replay_dataset_index = 0

                '''get current imgs data loader'''
                train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
                                                          self.num_workers,
                                                          cuda=True if self.availabel_cudas else False))

                '''get feature_replay data loader'''
                feature_replay_datasets = ExemplarDataset(self.exemplar_feature_sets)
                feature_replay_data_loader = iter(utils.get_data_loader(feature_replay_datasets, self.batch_size,
                                                                        self.num_workers,
                                                                        cuda=True if self.availabel_cudas else False))

                train_imgs_num = len(train_loader)
                feature_replay_dataset_num = len(feature_replay_data_loader)

                for epoch in range(1, self.epochs + 1):
                    iter_index = 0
                    '''if imgs loader end, get new imgs loader'''
                    if train_dataset_iter_index == train_imgs_num:
                        train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
                                                                  self.num_workers,
                                                                  cuda=True if self.availabel_cudas else False))
                        train_dataset_iter_index = 0

                    '''if feature_replay end, get new feature_replay data loader'''
                    if feature_replay_dataset_index == feature_replay_dataset_num:
                        feature_replay_data_loader = iter(
                            utils.get_data_loader(feature_replay_datasets, self.batch_size, self.num_workers,
                                                  cuda=True if self.availabel_cudas else False))
                        feature_replay_dataset_index = 0
                    iters_left = max(train_imgs_num, feature_replay_dataset_num)
                    while iters_left > 0:
                        # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                        iters_left -= 1

                        '''if imgs loader end, get new imgs loader'''
                        if train_dataset_iter_index == train_imgs_num:
                            train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
                                                                      self.num_workers,
                                                                      cuda=True if self.availabel_cudas else False))
                            train_dataset_iter_index = 0

                        '''if feature_replay end, get new feature_replay data loader'''
                        if feature_replay_dataset_index == feature_replay_dataset_num:
                            feature_replay_data_loader = iter(
                                utils.get_data_loader(feature_replay_datasets, self.batch_size, self.num_workers,
                                                      cuda=True if self.availabel_cudas else False))
                            feature_replay_dataset_index = 0

                        # -----------------Collect data------------------#

                        #####-----CURRENT BATCH-----#####
                        x, y = next(train_loader)  # --> sample training data of current task
                        train_dataset_iter_index += 1
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
                                                                      active_classes=active_classes, task=task,
                                                                      dis_model_optimizer=dis_model_optimizer,
                                                                      epoch=epoch)
                        iter_index += 1
                        if iter_index % args.print_interval == 0:
                            rlt = f"Task {task:0>3d}, batch train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_index:0>4d}], " \
                                  f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: " \
                                  f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}, " \
                                  f"GAN train:,{loss_dict['GAN_train']}"
                            self.batch_train_logger.info(
                                rlt
                            )
                            print(rlt)
                    scheduler.step()
                    if self.use_discriminate:
                        dis_model_optimizer_scheduler.step()
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes)
                    results = f"batch train FE_cls || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, " \
                              f"top5 acc:  {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    self.batch_train_logger.info(
                        results
                    )
                    print(results)
            elif task == 1:
                self.train_FE_cls(args, training_dataset, active_classes)
                # self.FE_cls = None
                # self.FE_cls = self.load_FE_cls_model("/share/home/kcli/CL_research/iCaRL_ILtFA/"
                #                                      "cifar10pretrain_EFAfIL_task1.pth")
                print(type(self.FE_cls))
                print(self.FE_cls)

            self.cifar100_feature_handle_main_FM(training_dataset, train_dataset, self.val_datasets,
                                                 self.classes_per_task, active_classes, task, self.img_size,
                                                 self.hidden_size, use_dynamicMem=True,
                                                 FM_reTrain=self.FM_reTrain,
                                                 use_NewfeatureSpace=self.use_NewfeatureSpace)
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
            # acc_past_tasks, acc_list = self.tasks_validate(task, classifier="feature_ncm", use_FM=True)
            # EFAfIL_result["task_{}_featureNCM_useFM_results".format(task)] = acc_past_tasks
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
        with open(self.result_file, 'w') as fw:
            json.dump(EFAfIL_result, fw, indent=4)

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
                                                                                             task=task, epoch=epoch)
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

    def train_GAN(self, x, feature_replay_FM_features, optimizer, dis_model_optimizer, task, epoch):
        '''train generator && discriminate'''
        # self.train()
        # self.dis_model.eval()
        # # Reset optimizer
        # optimizer.zero_grad()
        # '''train generator'''
        # BCE_criterion = torch.nn.BCELoss().cuda()  # 是单目标二分类交叉熵函数
        # features, _ = self(x)
        # # output = self.get_discriminate_output(features)  # 经过判别器得到的结果
        # output = self.dis_model(features)  # 经过判别器得到的结果
        # num_features = features.size(0)
        # features_real_label = torch.Tensor(num_features, 1).fill_(1.0).cuda()  # 定义真实的图片label为1
        # g_loss = BCE_criterion(output, features_real_label)  # 得到的假的图片与真实的图片的label的loss
        # g_loss.backward()  # 将误差反向传播
        # optimizer.step()  # 更新参数

        self.dis_model.train()
        dis_model_optimizer.zero_grad()
        BCE_criterion = torch.nn.BCELoss().cuda()  # 是单目标二分类交叉熵函数
        num_feature_replay = feature_replay_FM_features.size(0)
        real_label = Variable(torch.Tensor(num_feature_replay, 1).fill_(1.0), ).cuda()  # 定义真实的图片label为1
        real_score = self.dis_model(feature_replay_FM_features)  # 将真实图片放入判别器中
        d_loss_real = BCE_criterion(real_score, real_label)  # 得到真实图片的loss

        # 计算假的图片的损失
        features = self.feature_extractor(x)  # 随机噪声放入生成网络中，生成一张假的图片
        num_features = features.size(0)
        fake_label = torch.Tensor(num_features, 1).fill_(0.0).cuda()  # 定义假的图片label为0
        fake_score = self.dis_model(features)  # 判别器判断假的图片
        d_loss_fake = BCE_criterion(fake_score, fake_label)  # 得到假的图片的loss

        # 损失函数和优化
        d_loss = (d_loss_real + d_loss_fake) / 4  # 损失包括判真损失和判假损失
        d_loss.backward()  # 将误差反向传播
        dis_model_optimizer.step()  # 更新参数

        self.train()
        # Reset optimizer
        optimizer.zero_grad()
        '''train generator'''
        BCE_criterion = torch.nn.BCELoss().cuda()  # 是单目标二分类交叉熵函数
        features, _ = self(x)
        output = self.dis_model(features)  # 经过判别器得到的结果
        num_features = features.size(0)
        features_real_label = torch.Tensor(num_features, 1).fill_(1.0).cuda()  # 定义真实的图片label为1
        g_loss = BCE_criterion(output, features_real_label)  # 得到的假的图片与真实的图片的label的loss
        g_loss.backward()  # 将误差反向传播
        optimizer.step()  # 更新参数
        results = f"batch train discriminator & generator || task: {task:0>3d}, val: epoch {epoch:0>3d}, " \
                  f"d_loss: {d_loss.item():.2f}, g_loss: {g_loss.item():.2f}"

        self.batch_train_logger.info(
            results
        )
        return results

    def train_a_batch_feature_replay(self, x, y, feature_replay_features,
                                     feature_replay_labels, FM_targets,
                                     FM_features,
                                     feature_replay_FM_features,
                                     feature_replay_FM_targets,
                                     optimizer,
                                     active_classes, task, dis_model_optimizer=None, epoch=1, is_iCaRL_LwF_BiC=0,
                                     feature_batch_size=0,
                                     test_feature_cls_sim=False, use_feature_replay_in_new_model=True):
        results = ""
        if use_feature_replay_in_new_model:
            image_rate = self.batch_size / (feature_batch_size + self.batch_size)
        if self.use_discriminate:
            results = self.train_GAN(x, FM_features, optimizer, dis_model_optimizer, task, epoch)
        self.train()
        # Reset optimizer
        optimizer.zero_grad()
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        loss_func = torch.nn.MSELoss(reduction='mean')
        criteria = torch.nn.CrossEntropyLoss()
        soft_target_criterion = SoftTarget_CrossEntropy().cuda()
        features, y_hat = self(x)
        if use_feature_replay_in_new_model:
            if type(self.FE_cls) is torch.nn.DataParallel:
                feature_replay_y_hat = self.FE_cls.module.cls(feature_replay_FM_features)
            else:
                feature_replay_y_hat = self.FE_cls.cls(feature_replay_FM_features)
            # -if needed, remove predictions for classes not in current task
            feature_replay_y_hat_forcls = feature_replay_y_hat[:, :(self.classes_per_task * task)]
            feature_replay_y_hat_fordistill = feature_replay_y_hat[:, :(self.classes_per_task * (task - 1))]
            feature_replay_scores_hats = feature_replay_FM_targets[:, :(self.classes_per_task * (task - 1))]
        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
            y_hat = y_hat[:, class_entries]

        '''获取图片数据在FE_cls跟FM的对应于前n-1个任务的输出，获取保存的feature数据在FM以及FE_cls的对应于前n-1个任务的输出，制作蒸馏数据和分类数据'''
        scores_hats = FM_targets[:, :(self.classes_per_task * (task - 1))]
        y_hat_fordistill = y_hat[:, :(self.classes_per_task * (task - 1))]

        if is_iCaRL_LwF_BiC == 0:
            '''获取图片数据在FE_cls跟FM的对应于前n-1个任务的输出，获取保存的feature数据在FM以及FE_cls的对应于前n-1个任务的输出，制作蒸馏数据和分类数据'''
            scores_hats = torch.sigmoid(scores_hats)
            if use_feature_replay_in_new_model:
                feature_replay_scores_hats = torch.sigmoid(feature_replay_scores_hats)
                loss_distill_feature_replay = Func.binary_cross_entropy_with_logits(
                    input=feature_replay_y_hat_fordistill, target=feature_replay_scores_hats, reduction='none'
                ).sum(dim=1).mean()

            binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
            binary_targets = binary_targets[:, -self.classes_per_task:]
            binary_targets = torch.cat([scores_hats, binary_targets], dim=1)

            loss_cls_distill = Func.binary_cross_entropy_with_logits(input=y_hat, target=binary_targets,
                                                                     reduction='none'
                                                                     ).sum(dim=1).mean()

            if test_feature_cls_sim:
                # loss_sim = 1 - torch.cosine_similarity(features, FM_features).mean()
                loss_cls_feature_replay = criteria(feature_replay_y_hat_forcls, feature_replay_labels)
                loss_total = loss_cls_distill + loss_distill_feature_replay + loss_cls_feature_replay
            else:
                if use_feature_replay_in_new_model:
                    loss_total = image_rate * loss_cls_distill + (1 - image_rate) * loss_distill_feature_replay
                else:
                    loss_total = loss_cls_distill


        elif is_iCaRL_LwF_BiC == 1:
            scores_hats = torch.softmax(scores_hats / self.KD_temp, dim=1)
            loss_distill_current_task = soft_target_criterion(y_hat_fordistill, scores_hats,
                                                              self.KD_temp) * self.KD_temp * self.KD_temp

            if use_feature_replay_in_new_model:
                feature_replay_scores_hats = torch.softmax(feature_replay_scores_hats / self.KD_temp, dim=1)
                loss_distill_feature_replay = soft_target_criterion(feature_replay_y_hat_fordistill,
                                                                    feature_replay_scores_hats,
                                                                    self.KD_temp) * self.KD_temp * self.KD_temp

                loss_distill = image_rate * loss_distill_current_task + (1 - image_rate) * loss_distill_feature_replay
                loss_cls = image_rate * criteria(y_hat, y) + (1 - image_rate) * criteria(feature_replay_y_hat_forcls,
                                                                                         feature_replay_labels)
            else:
                loss_distill = loss_distill_current_task
                loss_cls = criteria(y_hat, y)
            # feature_replay_scores_hats = torch.sigmoid(feature_replay_scores_hats)
            # loss_distill_feature_replay = Func.binary_cross_entropy_with_logits(
            #     input=feature_replay_y_hat_fordistill, target=feature_replay_scores_hats, reduction='none'
            # ).sum(dim=1).mean()
            # loss_distill = loss_distill_current_task
            # loss_cls = criteria(y_hat, y)
            # lamda = (task - 1) / task
            # loss_total = lamda * loss_distill + (1 - lamda) * loss_cls
            loss_total = loss_distill + loss_cls
            # loss_sim = 1 - torch.cosine_similarity(features, FM_features).mean()
            # loss_total = loss_distill + loss_cls + loss_sim
        elif is_iCaRL_LwF_BiC == 2:
            scores_hats = torch.softmax(scores_hats / self.KD_temp, dim=1)
            loss_distill_current_task = soft_target_criterion(y_hat_fordistill, scores_hats,
                                                              self.KD_temp) * self.KD_temp * self.KD_temp

            if use_feature_replay_in_new_model:
                feature_replay_scores_hats = torch.softmax(feature_replay_scores_hats / self.KD_temp, dim=1)
                loss_distill_feature_replay = soft_target_criterion(feature_replay_y_hat_fordistill,
                                                                    feature_replay_scores_hats,
                                                                    self.KD_temp) * self.KD_temp * self.KD_temp
                loss_distill = image_rate * loss_distill_current_task + (1 - image_rate) * loss_distill_feature_replay
                loss_cls = image_rate * criteria(y_hat, y) + (1 - image_rate) * criteria(feature_replay_y_hat_forcls,
                                                                                         feature_replay_labels)
            else:
                loss_distill = loss_distill_current_task
                loss_cls = criteria(y_hat, y)
            # feature_replay_scores_hats = torch.sigmoid(feature_replay_scores_hats)
            # loss_distill_feature_replay = Func.binary_cross_entropy_with_logits(
            #     input=feature_replay_y_hat_fordistill, target=feature_replay_scores_hats, reduction='none'
            # ).sum(dim=1).mean()
            # loss_cls = criteria(y_hat, y)
            lamda = (task - 1) / task
            loss_total = (1 - lamda) * loss_cls + lamda * loss_distill
            # loss_sim = 1 - torch.cosine_similarity(features, FM_features).mean()
            # loss_total = (1 - lamda) * loss_cls + lamda * loss_distill + (1 - lamda) * loss_sim
        loss_total.backward()
        # Take optimization-step
        optimizer.step()
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
                "losses": losses.avg,
                "GAN_train": results
            }
        else:
            return {
                'task': task,
                'loss_total': loss_total.item(),
                'precision': precision if precision is not None else 0.,
                "top1": top1.avg,
                "top5": 0,
                "losses": losses.avg,
                "GAN_train": results
            }
        pass

    def train_a_batch_feature_replay_exemplar_balance(self, x, y, exemplar_replay_imgs, exemplar_replay_labels,
                                                      feature_replay_features, feature_replay_labels,
                                                      FM_targets, FM_features,
                                                      FM_exemplar_targets, FM_exemplar_features,
                                                      feature_replay_FM_features, feature_replay_FM_targets,
                                                      optimizer,
                                                      active_classes, task, dis_model_optimizer=None, epoch=1,
                                                      is_iCaRL_LwF_BiC=0,
                                                      test_feature_cls_sim=False, use_feature_replay_in_new_model=True):
        results = ""
        self.train()
        # Reset optimizer
        optimizer.zero_grad()
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        loss_func = torch.nn.MSELoss(reduction='mean')
        criteria = torch.nn.CrossEntropyLoss()
        soft_target_criterion = SoftTarget_CrossEntropy().cuda()
        features, y_hat = self(x)
        exemplar_features, exemplar_y_hat = self(exemplar_replay_imgs)
        if use_feature_replay_in_new_model:
            if type(self.FE_cls) is torch.nn.DataParallel:
                feature_replay_y_hat = self.FE_cls.module.cls(feature_replay_FM_features)
            else:
                feature_replay_y_hat = self.FE_cls.cls(feature_replay_FM_features)
            # -if needed, remove predictions for classes not in current task
            feature_replay_y_hat_forcls = feature_replay_y_hat[:, :(self.classes_per_task * task)]
            feature_replay_y_hat_fordistill = feature_replay_y_hat[:, :(self.classes_per_task * (task - 1))]
            feature_replay_scores_hats = feature_replay_FM_targets[:, :(self.classes_per_task * (task - 1))]
        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
            y_hat = y_hat[:, class_entries]
            exemplar_y_hat = exemplar_y_hat[:, class_entries]

        '''获取图片数据在FE_cls跟FM的对应于前n-1个任务的输出，获取保存的feature数据在FM以及FE_cls的对应于前n-1个任务的输出，制作蒸馏数据和分类数据'''
        scores_hats = FM_targets[:, :(self.classes_per_task * (task - 1))]
        y_hat_fordistill = y_hat[:, :(self.classes_per_task * (task - 1))]

        exemplar_scores_hats = FM_exemplar_targets[:, :(self.classes_per_task * (task - 1))]
        exemplar_y_hat_fordistill = exemplar_y_hat[:, :(self.classes_per_task * (task - 1))]

        if is_iCaRL_LwF_BiC == 0:
            '''获取图片数据在FE_cls跟FM的对应于前n-1个任务的输出，获取保存的feature数据在FM以及FE_cls的对应于前n-1个任务的输出，制作蒸馏数据和分类数据'''
            scores_hats = torch.sigmoid(scores_hats)
            exemplar_scores_hats = torch.sigmoid(exemplar_scores_hats)
            if use_feature_replay_in_new_model:
                feature_replay_scores_hats = torch.sigmoid(feature_replay_scores_hats)
                loss_distill_feature_replay = Func.binary_cross_entropy_with_logits(
                    input=feature_replay_y_hat_fordistill, target=feature_replay_scores_hats, reduction='none'
                ).sum(dim=1).mean()

            binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
            binary_targets = binary_targets[:, -self.classes_per_task:]
            binary_targets = torch.cat([scores_hats, binary_targets], dim=1)

            loss_current_cls_distill = Func.binary_cross_entropy_with_logits(input=y_hat, target=binary_targets,
                                                                             reduction='none'
                                                                             ).sum(dim=1).mean()

            loss_exemplar_distill = Func.binary_cross_entropy_with_logits(input=exemplar_y_hat_fordistill,
                                                                          target=exemplar_scores_hats,
                                                                          reduction='none'
                                                                          ).sum(dim=1).mean()
            loss_cls_distill = loss_current_cls_distill + loss_exemplar_distill
            if test_feature_cls_sim:
                # loss_sim = 1 - torch.cosine_similarity(features, FM_features).mean()
                loss_cls_feature_replay = criteria(feature_replay_y_hat_forcls, feature_replay_labels)
                loss_total = loss_cls_distill + loss_distill_feature_replay + loss_cls_feature_replay
            else:
                if use_feature_replay_in_new_model:
                    loss_total = loss_cls_distill + loss_distill_feature_replay
                else:
                    loss_total = loss_cls_distill


        elif is_iCaRL_LwF_BiC == 1:
            scores_hats = torch.softmax(scores_hats / self.KD_temp, dim=1)
            exemplar_scores_hats = torch.softmax(exemplar_scores_hats / self.KD_temp, dim=1)

            loss_distill_current_task_imgs = soft_target_criterion(y_hat_fordistill, scores_hats,
                                                                   self.KD_temp) * self.KD_temp * self.KD_temp
            loss_distill_current_task_exemplars = soft_target_criterion(exemplar_y_hat_fordistill, exemplar_scores_hats,
                                                                        self.KD_temp) * self.KD_temp * self.KD_temp
            loss_distill_current_task = loss_distill_current_task_imgs + loss_distill_current_task_exemplars
            if use_feature_replay_in_new_model:
                feature_replay_scores_hats = torch.softmax(feature_replay_scores_hats / self.KD_temp, dim=1)
                loss_distill_feature_replay = soft_target_criterion(feature_replay_y_hat_fordistill,
                                                                    feature_replay_scores_hats,
                                                                    self.KD_temp) * self.KD_temp * self.KD_temp

                loss_distill = loss_distill_current_task + loss_distill_feature_replay
                loss_cls = criteria(y_hat, y) + criteria(exemplar_y_hat, exemplar_replay_labels) + \
                           criteria(feature_replay_y_hat_forcls, feature_replay_labels)
            else:
                loss_distill = loss_distill_current_task
                loss_cls = criteria(y_hat, y) + criteria(exemplar_y_hat, exemplar_replay_labels)
            # feature_replay_scores_hats = torch.sigmoid(feature_replay_scores_hats)
            # loss_distill_feature_replay = Func.binary_cross_entropy_with_logits(
            #     input=feature_replay_y_hat_fordistill, target=feature_replay_scores_hats, reduction='none'
            # ).sum(dim=1).mean()
            # loss_distill = loss_distill_current_task
            # loss_cls = criteria(y_hat, y)
            loss_total = loss_distill + loss_cls
            # loss_sim = 1 - torch.cosine_similarity(features, FM_features).mean()
            # loss_total = loss_distill + loss_cls + loss_sim
        elif is_iCaRL_LwF_BiC == 2:
            scores_hats = torch.softmax(scores_hats / self.KD_temp, dim=1)
            exemplar_scores_hats = torch.softmax(exemplar_scores_hats / self.KD_temp, dim=1)

            loss_distill_current_task_imgs = soft_target_criterion(y_hat_fordistill, scores_hats,
                                                                   self.KD_temp) * self.KD_temp * self.KD_temp
            loss_distill_current_task_exemplars = soft_target_criterion(exemplar_y_hat_fordistill, exemplar_scores_hats,
                                                                        self.KD_temp) * self.KD_temp * self.KD_temp
            loss_distill_current_task = loss_distill_current_task_imgs + loss_distill_current_task_exemplars
            if use_feature_replay_in_new_model:
                feature_replay_scores_hats = torch.softmax(feature_replay_scores_hats / self.KD_temp, dim=1)
                loss_distill_feature_replay = soft_target_criterion(feature_replay_y_hat_fordistill,
                                                                    feature_replay_scores_hats,
                                                                    self.KD_temp) * self.KD_temp * self.KD_temp
                loss_distill = loss_distill_current_task + loss_distill_feature_replay
                loss_cls = criteria(y_hat, y) + criteria(exemplar_y_hat, exemplar_replay_labels) + \
                           criteria(feature_replay_y_hat_forcls, feature_replay_labels)
            else:
                loss_distill = loss_distill_current_task
                loss_cls = criteria(y_hat, y) + criteria(exemplar_y_hat, exemplar_replay_labels)
            # feature_replay_scores_hats = torch.sigmoid(feature_replay_scores_hats)
            # loss_distill_feature_replay = Func.binary_cross_entropy_with_logits(
            #     input=feature_replay_y_hat_fordistill, target=feature_replay_scores_hats, reduction='none'
            # ).sum(dim=1).mean()
            # loss_cls = criteria(y_hat, y)
            lamda = (task - 1) / task
            loss_total = (1 - lamda) * loss_cls + lamda * loss_distill
            # loss_sim = 1 - torch.cosine_similarity(features, FM_features).mean()
            # loss_total = (1 - lamda) * loss_cls + lamda * loss_distill + (1 - lamda) * loss_sim
        loss_total.backward()
        # Take optimization-step
        optimizer.step()
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
                "losses": losses.avg,
                "GAN_train": results
            }
        else:
            return {
                'task': task,
                'loss_total': loss_total.item(),
                'precision': precision if precision is not None else 0.,
                "top1": top1.avg,
                "top5": 0,
                "losses": losses.avg,
                "GAN_train": results
            }
        pass

    def train_a_batch_feature_replay_softTargetCrossEntropy(self, x, y, feature_replay_features,
                                                            feature_replay_labels, FM_targets,
                                                            FM_features,
                                                            feature_replay_FM_features,
                                                            feature_replay_FM_targets,
                                                            optimizer,
                                                            active_classes, task, epoch):
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

    def per_task_validate(self, val_dataset, task, classifier="linearSVM", active_classes=None, use_FM=False):
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
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if "BiC" in classifier:
                self.eval()
                with torch.no_grad():
                    _, y_hat = self(inputs)
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    y_hat = y_hat[:, class_entries]
                if task > 1:
                    self.bias_layer.eval()
                    with torch.no_grad():
                        y_hat = self.bias_layer(y_hat, self.classes_per_task)
                _, y_hat = torch.max(y_hat, 1)
            else:
                y_hat = self.EFAfIL_classify(inputs, classifier, active_classes, task, use_FM=use_FM)
            correct_temp += y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
        throughput = 1.0 / (batch_time.avg / self.batch_size)
        return [top1.avg, throughput]
        pass

    def tasks_validate(self, task, classifier="linearSVM", active_classes=None, use_FM=False):
        # todo
        acc_past_tasks = []
        acc = []
        if classifier == "linearSVM" and self.svm_train:
            self.linearSVM_train_featureSet(task)
            self.svm_train = False
        for task_id in range(task):
            predict_result = self.per_task_validate(self.val_datasets[task_id], task, classifier, active_classes,
                                                    use_FM=use_FM)
            acc_past_tasks.append(predict_result)
            acc.append(predict_result[0])
            print("clser: {}, per task acc:".format(classifier), predict_result)
            self.logger.info(
                f"per task {task}, classifier{classifier} acc:{predict_result[0]}"
            )
        self.compute_means = True
        self.svm_train = True
        acc = np.array(acc)
        print("classifier{} acc_avg:{}".format(classifier, acc.mean()))
        self.logger.info(
            f"per task {task}, classifier{classifier} avg acc:{acc.mean()}"
            f"-------------------------------------------------------------"
        )
        return acc_past_tasks, acc
        pass

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
        # val_loader = utils.get_data_loader(self.val_datasets[task - 1], self.batch_size,  # task index must minus 1
        #                                    self.num_workers,
        #                                    cuda=True if self.availabel_cudas else False)
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
