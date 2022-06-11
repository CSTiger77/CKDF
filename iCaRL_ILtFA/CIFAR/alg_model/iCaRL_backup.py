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
from public.data import ExemplarDataset, SubDataset, get_multitask_experiment, get_dataset, AVAILABLE_TRANSFORMS
from exemplars import ExemplarHandler
from public.util_models import FeatureExtractor

# -------------------------------------------------------------------------------------------------#

# --------------------#
# ----- iCaRL -----#
# --------------------#
from public.utils import AverageMeter, accuracy


class iCaRL(ExemplarHandler):
    def __init__(self, model_name, dataset_name, dataset_path, num_classes, epochs, extracted_layers, rate, tasks,
                 logger,
                 batch_train_logger, batch_size, result_file, memory_budget, norm_exemplars, herding, lr, momentum,
                 weight_decay, optim_type, milestones, KD_temp, gamma, availabel_cudas, seed=0):
        ExemplarHandler.__init__(self, memory_budget, norm_exemplars, herding)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.epochs = epochs
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
        self.device = "cuda" if self.availabel_cudas else "cpu"

        self.pre_FE = None
        self.train_datasets, self.val_datasets, self.data_config, self.classes_per_task = \
            self.get_dataset(dataset_name)
        # self.extra_train_datasets = get_dataset("CIFAR10", 'train', dir=self.dataset_path)
        self.FE = self.construct_model(rate)
        # self.FE = None
        # train_dataset_transform = transforms.Compose([
        #     *AVAILABLE_TRANSFORMS["imagenet_32"]['train_transform']
        # ])
        # self.extra_train_datasets = get_dataset("ImageNet100", type='train',
        #                                         dir="/n02dat01/public_resource/dataset/ImageNet/",
        #                                         imagenet_json_path="/share/home/kcli/Chore/datapreprocess/imagenet100.json",
        #                                         verbose=False,
        #                                         train_data_transform=train_dataset_transform)
        # self.FE = None

    def forward(self, x):
        final_features, target = self.FE(x)
        return target

    def get_dataset(self, dataset_name):
        (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
            name=dataset_name, tasks=self.tasks, data_dir=self.dataset_path,
            exception=True if self.seed == 0 else False,
        )
        return train_datasets, test_datasets, config, classes_per_task

    def build_optimize(self):
        # Define optimizer (only include parameters that "requires_grad")
        optim_list = [{'params': filter(lambda p: p.requires_grad, self.FE.parameters()), 'lr': self.lr}]
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
        # if self.availabel_cudas:
        #     os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
        #     device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
        #     model = torch.nn.DataParallel(FeatureExtractor(resnetforcifar.__dict__[self.model_name](rate=rate),
        #                                                    self.extracted_layers), device_ids=device_ids).cuda()
        # else:
        #     model = FeatureExtractor(resnetforcifar.__dict__[self.model_name](rate=rate), self.extracted_layers)
        model = FeatureExtractor(resnetforcifar.__dict__[self.model_name](rate=rate), self.extracted_layers)
        model.load_state_dict(torch.load('featureExtractor_imagenet100_pretrain.pth'))
        model = model.to(self.device)
        return model

    def feature_extractor(self, images):
        mode = self.FE.training
        self.FE.eval()
        with torch.no_grad():
            features = self.FE(images)[-2]
        self.FE.train(mode=mode)
        return features

    def train_main(self, args):
        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''
        print("seed:", self.seed)
        # print("model:", self.FE)
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
        for task, train_dataset in enumerate(self.train_datasets, 1):
            self.logger.info(f'New task {task} begin.')
            self.batch_train_logger.info(f'New task {task} begin.')
            print("New task %d begin." % task)
            # Add exemplars (if available) to current dataset (if requested)
            if task > 1 and self.memory_budget > 0:
                exemplar_dataset = ExemplarDataset(self.exemplar_sets)
                training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
            else:
                training_dataset = train_dataset

            # Find [active_classes]
            if task >= 1:
                active_classes = list(range(self.classes_per_task * task))
                optimizer = self.build_optimize()
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=self.milestones, gamma=self.gamma)
                # extra_train_dataset_index = 0
                # extra_data_loader = iter(utils.get_data_loader(self.extra_train_datasets, self.batch_size,
                #                                                cuda=True if self.availabel_cudas else False))
                # extra_train_dataset_num = len(extra_data_loader)
                # extra_index = 0
                # print("extra_train_dataset_num:", extra_train_dataset_num)
                for epoch in range(1, self.epochs + 1):
                    is_first_ite = True
                    iters_left = 1
                    iter_index = 0
                    extra_index = 0
                    # if extra_train_dataset_index == extra_train_dataset_num:
                    #     extra_data_loader = iter(utils.get_data_loader(self.extra_train_datasets, self.batch_size,
                    #                                                    cuda=True if self.availabel_cudas else False))
                    #     extra_train_dataset_index = 0
                    while iters_left > 0:
                        # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                        iters_left -= 1
                        iter_num = 0
                        if is_first_ite:
                            is_first_ite = False
                            data_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
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
                        # if extra_train_dataset_index == extra_train_dataset_num:
                        #     extra_data_loader = iter(utils.get_data_loader(self.extra_train_datasets, self.batch_size,
                        #                                                    cuda=True if self.availabel_cudas else False))
                        #     extra_train_dataset_index = 0
                        # extra_x, extra_y = next(extra_data_loader)
                        # extra_train_dataset_index += 1
                        x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                        # extra_x, extra_y = extra_x.to(self.device), extra_y.to(self.device)
                        if self.pre_FE is not None:
                            with torch.no_grad():
                                scores = self.pre_FE(x)[-1][:, :(self.classes_per_task * (task - 1))]
                                # extra_scores = self.pre_FE(extra_x)[-1][:, :(self.classes_per_task * (task - 1))]
                        else:
                            scores = None
                            # extra_scores = None

                        # ---> Train MAIN MODEL
                        # Train the main model with this batch
                        loss_dict = self.train_a_batch(x, y, scores,
                                                       active_classes, optimizer, task)
                        iter_index += 1
                        if iter_index % args.print_interval == 0:
                            self.batch_train_logger.info(
                                f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iter_num:0>4d}], "
                                f"lr: {scheduler.get_last_lr()[0]:.6f}, top1 acc: {loss_dict['top1']:.2f}%, top5 acc: "
                                f"{loss_dict['top5']:.2f}%, loss_total: {loss_dict['losses']:.2f}"
                                f"'precision': {loss_dict['precision']:.2f}%, loss_total: {loss_dict['loss_total']:.2f}"
                            )
                        print(loss_dict)
                    print("iters_left: ", iters_left)
                    scheduler.step()
                    acc1, acc5, throughput = self.current_task_validate(task, active_classes)
                    self.batch_train_logger.info(
                        f"batch train current task validate || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: "
                        f"{acc5:.2f}%, throughput: {throughput:.2f}sample/s"
                    )
                    self.batch_train_logger.info(f"------------------------------------------------------------------")
                    print(f'batch train task : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))
                # if task == 1:
                #     torch.save(self.FE, "featureExtractor_task_1.pth")
            # else:
            #     # self.FE = torch.load("featureExtractor_task_1.pth")
            #     self.FE = torch.load("featureExtractor_cifar10Pretrain_task_1.pth")
            #     # continue
            #     print("model:", self.FE)
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
            self.pre_FE = copy.deepcopy(self.FE).eval()
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
                binary_targets = torch.cat([torch.sigmoid(scores / self.KD_temp), binary_targets], dim=1)
            predL = None if y is None else Func.binary_cross_entropy_with_logits(
                input=y_hat, target=binary_targets, reduction='none'
            ).sum(dim=1).mean()  # --> sum over classes, then average over batch
            acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))
            top1.update(acc1.item(), x.size(0))
            top5.update(acc5.item(), x.size(0))
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
        return {
            'loss_total': loss_total.item(),
            'precision': precision if precision is not None else 0.,
            "top1": top1.avg,
            "top5": top5.avg,
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


    def current_task_validate(self, task, active_classes):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        val_loader = utils.get_data_loader(self.val_datasets[task - 1], self.batch_size,
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
                acc1, acc5 = accuracy(y_hat, labels, topk=(1, 5))
                top1.update(acc1.item(), inputs.size(0))
                top5.update(acc5.item(), inputs.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
        throughput = 1.0 / (batch_time.avg / self.batch_size)
        self.train(mode=mode)
        return top1.avg, top5.avg, throughput

    def ncm_task_validate(self, val_dataset, task, classifier="ncm"):
        # todo
        val_loader = utils.get_data_loader(val_dataset, self.batch_size,
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
        return acc_past_tasks, np.array(acc)
        pass
