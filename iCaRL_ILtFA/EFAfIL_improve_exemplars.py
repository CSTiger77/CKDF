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
from public.util_models import MLP_cls_domain_dis, MLP_for_FM
from public.utils import AverageMeter, accuracy


class EFAfIL_improve_FeaturesHandler(nn.Module, metaclass=abc.ABCMeta):
    """Abstract  module for a classifier that can store and use exemplars.

    Adds a exemplar-methods to subclasses, and requires them to provide a 'feature-extractor' method."""

    def __init__(self, MLP_name, num_classes, hidden_size, Exemple_memory_budget, num_workers,
                 Feature_memory_budget, norm_exemplars, herding, batch_size, sim_alpha, MLP_lr,
                 MLP_momentum, MLP_milestones, MLP_lrgamma, MLP_weight_decay, MLP_epochs, MLP_optim_type,
                 KD_temp, svm_sample_type, svm_max_iter, availabel_cudas, logger, batch_train_logger,
                 CNN_lr, CNN_momentum, CNN_weight_decay, CNN_milestones, lrgamma, CNN_epochs, optim_type):
        super().__init__()

        # list with exemplar-sets
        self.exemplar_feature_sets = []  # --> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_feature_means = []
        self.exemplar_sets = []  # --> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_means = []
        self.logger = logger
        self.batch_train_logger = batch_train_logger
        self.compute_means = True
        # settings
        self.num_workers = num_workers
        self.KD_temp = KD_temp
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
        self.CNN_lr = CNN_lr
        self.CNN_momentum = CNN_momentum
        self.CNN_weight_decay = CNN_weight_decay
        self.CNN_milestones = CNN_milestones
        self.CNN_lrgamma = lrgamma
        self.CNN_optim_type = optim_type
        self.CNN_epochs = CNN_epochs
        self.optim_type = MLP_optim_type
        self.sample_type = svm_sample_type
        self.availabel_cudas = availabel_cudas
        self.pre_FM_cls_domain = None
        self.svm = None
        self.svm_max_iter = svm_max_iter
        self.MLP_device = "cuda" if self.availabel_cudas else "cpu"
        self.FM_cls_domain = self.construct_FM_cls_domain_model()
        self.featureHandler_FE_cls = None
        # self.classifer = nn.Linear(self.feature_dim, num_classes).to(self.MLP_device)

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def featureHandler_FEcls_result(self, imgs):
        assert self.featureHandler_FE_cls is not None
        self.featureHandler_FE_cls.eval()
        with torch.no_grad():
            features, target = self.featureHandler_FE_cls(imgs)
        return features, target

    @abc.abstractmethod
    def get_FE_cls_output(self, images):
        pass

    @abc.abstractmethod
    def feature_extractor(self, images):
        pass

    @abc.abstractmethod
    def get_preFE_feature(self, images):
        pass

    @abc.abstractmethod
    def copy_from_EFAfIL_model(self):
        pass

    @abc.abstractmethod
    def get_cls_target(self, prefeatures):
        pass

    @abc.abstractmethod
    def get_FE_cls_target(self, x):
        pass

    # def get_prefeatures_results(self, prefeatures):
    #     self.FM_cls_domain.eval()
    #     with torch.no_grad:
    #         return self.FM_cls_domain(prefeatures)

    def feature_mapping(self, features):
        if type(self.FM_cls_domain) is torch.nn.DataParallel:
            return self.FM_cls_domain.module.get_mapping_features(features)
        else:
            return self.FM_cls_domain.get_mapping_features(features)

    def FM_cls_domain_result(self, features):
        self.FM_cls_domain.eval()
        with torch.no_grad():
            return self.FM_cls_domain(features)

    def prefeature_2_FMtarget(self, FMfeatures):
        self.FM_cls_domain.eval()
        with torch.no_grad():
            if type(self.FM_cls_domain) is torch.nn.DataParallel:
                return self.FM_cls_domain.module.cls(FMfeatures)
            else:
                return self.FM_cls_domain.cls(FMfeatures)

    def get_preFE_FM_features(self, imgs):
        return self.feature_mapping(self.get_preFE_feature(imgs))

    def construct_FM_cls_domain_model(self):
        if self.availabel_cudas:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
            device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
            FM_cls_domain_model = torch.nn.DataParallel(
                MLP_cls_domain_dis(MLP.__dict__[self.MLP_name](input_dim=self.feature_dim,
                                                               out_dim=self.feature_dim),
                                   self.feature_dim, self.num_classes),
                device_ids=device_ids).cuda()
            cudnn.benchmark = True
        else:
            FM_cls_domain_model = MLP_cls_domain_dis(MLP.__dict__[self.MLP_name](input_dim=self.feature_dim,
                                                                                 out_dim=self.feature_dim),
                                                     self.feature_dim, self.num_classes)
        return FM_cls_domain_model

    def build_feature_FEcls_optimize(self):
        # Define optimizer (only include parameters that "requires_grad")
        optim_list = [{'params': filter(lambda p: p.requires_grad, self.featureHandler_FE_cls.parameters()), 'lr': self.CNN_lr}]
        optimizer = None
        if self.CNN_optim_type in ("adam", "adam_reset"):
            if self.CNN_weight_decay:
                optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999), weight_decay=self.CNN_weight_decay)
            else:
                optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999))
        elif self.CNN_optim_type == "sgd":
            if self.CNN_momentum:
                optimizer = torch.optim.SGD(optim_list, momentum=self.CNN_momentum, weight_decay=self.CNN_weight_decay)
            else:
                optimizer = torch.optim.SGD(optim_list)
        else:
            raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(self.CNN_optim_type))

        return optimizer

    def build_FM_cls_domain_optimize(self):
        # Define optimizer (only include parameters that "requires_grad")
        optim_list = [{'params': filter(lambda p: p.requires_grad, self.FM_cls_domain.parameters()), 'lr': self.MLP_lr}]
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

    def feature_handle_main(self, train_dataset, classes_per_task, task):
        pre_tasks_features = []
        pre_tasks_targets = []
        features_per_class = int(np.floor(self.Feature_memory_budget / (classes_per_task * task)))
        examplars_per_class = int(np.floor(self.Exemple_memory_budget / (classes_per_task * task)))
        if task > 1:
            for class_id, exemplar_feature_set in enumerate(self.exemplar_feature_sets):
                pre_tasks_targets.append(class_id)
                exemplars = []
                for exemplar_feature in exemplar_feature_set:
                    exemplars.append(torch.from_numpy(exemplar_feature))
                exemplars = torch.stack(exemplars).to(self._device())
                pre_tasks_features.append(self.feature_mapping(exemplars).cpu().numpy())
            self.update_features_sets(pre_tasks_features, features_per_class)
        current_task_features = []
        current_task_target = []
        # for each new class trained on, construct examplar-set
        new_classes = list(range(classes_per_task * (task - 1), classes_per_task * task))
        for class_id in new_classes:
            # create new dataset containing only all examples of this class
            class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
            current_task_target.append(class_id)
            dataloader = utils.get_data_loader(class_dataset, self.batch_size, self.num_workers, cuda=self._is_on_cuda())
            first_entry = True
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self._device())
                feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            current_task_features.append(features)
            self.construct_exemplar_feature_set(class_dataset, features, examplars_per_class, features_per_class)
        current_task_features = torch.stack(current_task_features)
        current_task_features = current_task_features.numpy()
        self.linearSVM_train(pre_tasks_features, pre_tasks_targets, current_task_features,
                             current_task_target, self.sample_type)

    def EFAfIL_feature_mapper_cls_domain_train(self, training_dataset, test_datasets, classes_per_task,
                                               active_classes, task):
        # todo Done
        self.batch_train_logger.info(f'####Task {task} EFAfIL feature_mapper_cls_domain_train begin.####')
        self.logger.info(f'####Task {task} EFAfIL  feature_mapper_cls_domain_train begin.####')
        print("Task %d , train featureHandler_FE_cls & feature mapping..." % task)
        pre_features_target_hats_datasets = ExemplarDataset(self.exemplar_feature_sets)
        mode = self.training
        self.featureHandler_FE_cls = self.copy_from_EFAfIL_model()
        print(self.featureHandler_FE_cls)
        self.eval()
        self.FM_cls_domain.train()
        self.featureHandler_FE_cls.train()
        criterion = nn.CrossEntropyLoss()
        FM_optimizer = self.build_FM_cls_domain_optimize()
        feature_FE_cls_optimizer = self.build_feature_FEcls_optimize()
        FM_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            FM_optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
        feature_FE_cls_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            feature_FE_cls_optimizer, milestones=self.CNN_milestones, gamma=self.CNN_lrgamma)
        train_dataset_iter_index = 0
        pre_features_target_hats_datasets_iter_index = 0
        for epoch in range(self.CNN_epochs):
            train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size, self.num_workers,
                                                      cuda=True if self.availabel_cudas else False))
            pre_features_target_hats_train_loader = iter(utils.get_data_loader(pre_features_target_hats_datasets,
                                                                               self.batch_size, self.num_workers,
                                                                               cuda=True if self.availabel_cudas else False))
            train_imgs_num = len(train_loader)
            pre_features_target_hats_num = len(pre_features_target_hats_train_loader)
            iter_index = 0
            while train_dataset_iter_index > iter_index:
                _, _ = next(train_loader)
                iter_index += 1
            iter_index = 0
            while pre_features_target_hats_datasets_iter_index > iter_index:
                _, _ = next(pre_features_target_hats_train_loader)
                iter_index += 1
            iter_index = 0
            iters_left = max(train_imgs_num, pre_features_target_hats_num)
            while iters_left > 0:
                FM_optimizer.zero_grad()
                feature_FE_cls_optimizer.zero_grad()
                iters_left -= 1
                if train_dataset_iter_index == train_imgs_num:
                    train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size, self.num_workers,
                                                              cuda=True if self.availabel_cudas else False))
                    train_dataset_iter_index = 0
                if pre_features_target_hats_datasets_iter_index == pre_features_target_hats_num:
                    pre_features_target_hats_train_loader = iter(
                        utils.get_data_loader(pre_features_target_hats_datasets,
                                              self.batch_size, self.num_workers,
                                              cuda=True if self.availabel_cudas else False))
                    pre_features_target_hats_datasets_iter_index = 0
                iter_index += 1
                imgs, labels = next(train_loader)
                pre_features, pre_labels = next(pre_features_target_hats_train_loader)
                train_dataset_iter_index += 1
                pre_features_target_hats_datasets_iter_index += 1
                imgs, labels = imgs.to(self.MLP_device), labels.to(self.MLP_device)
                pre_features, pre_labels = pre_features.to(self.MLP_device), pre_labels.to(self.MLP_device)
                # target_hats = self.get_cls_target(pre_features)
                imgs_2_features, imgs_2_targets = self.featureHandler_FE_cls(imgs)
                '''train featureHandler_FE_cls'''
                features, score = self.get_FE_cls_output(imgs)
                # -if needed, remove predictions for classes not in current task
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    imgs_2_targets = imgs_2_targets[:, class_entries]
                scores_hats = score[:, :(classes_per_task * (task - 1))]
                scores_hats = torch.sigmoid(scores_hats / self.KD_temp)
                binary_targets = utils.to_one_hot(labels.cpu(), imgs_2_targets.size(1)).to(self.MLP_device)
                binary_targets = binary_targets[:, -classes_per_task:]
                binary_targets = torch.cat([scores_hats, binary_targets], dim=1)
                imgs_loss_cls_distill = Func.binary_cross_entropy_with_logits(
                    input=imgs_2_targets, target=binary_targets, reduction='none'
                ).sum(dim=1).mean()
                imgs_loss_sim = 1 - torch.cosine_similarity(imgs_2_features, features).mean()
                # loss_total = imgs_loss_cls_distill + imgs_loss_sim
                # loss_total.backward(retain_graph=True)
                '''train FM_cls_domain'''
                all_features = torch.cat((imgs_2_features, pre_features), dim=0)
                all_labels = torch.cat((labels, pre_labels), dim=0)
                all_feature_hats, all_target_hats, domain_hats = self.FM_cls_domain(all_features)
                all_scores_hats = self.get_cls_target(all_feature_hats)

                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    all_target_hats = all_target_hats[:, class_entries]
                all_scores_hats = all_scores_hats[:, :(classes_per_task * (task - 1))]
                all_scores_hats = torch.sigmoid(all_scores_hats / self.KD_temp)
                binary_targets = utils.to_one_hot(all_labels.cpu(), all_target_hats.size(1)).to(self.MLP_device)
                binary_targets = binary_targets[:, -classes_per_task:]
                binary_targets = torch.cat([all_scores_hats, binary_targets], dim=1)

                all_loss_cls_distill = Func.binary_cross_entropy_with_logits(
                    input=all_target_hats, target=binary_targets, reduction='none'
                ).sum(dim=1).mean()
                all_feature_loss_cls = criterion(all_target_hats, all_labels)
                all_feature_sim = 1 - torch.cosine_similarity(all_features, all_feature_hats).mean()
                loss_total = imgs_loss_cls_distill + imgs_loss_sim + all_loss_cls_distill + all_feature_loss_cls


                loss_total.backward()
                FM_optimizer.step()
                feature_FE_cls_optimizer.step()
                precision_1 = None if imgs_2_targets is None else (labels == imgs_2_targets.max(1)[
                    1]).sum().item() / labels.size(0)
                precision_2 = None if all_target_hats is None else (all_labels == all_target_hats.max(1)[
                    1]).sum().item() / all_labels.size(0)
                ite_info = {
                    'task': task,
                    'FM_lr': FM_scheduler.get_last_lr()[0],
                    'feature_FEcls_lr': feature_FE_cls_scheduler.get_last_lr()[0],
                    'loss_total': loss_total.item(),
                    'precision_1': precision_1 if precision_1 is not None else 0.,
                    'precision_2': precision_2 if precision_2 is not None else 0.,
                }
                print(ite_info)
                print("....................................")
            FM_scheduler.step()
            feature_FE_cls_scheduler.step()
        acc1, acc5, FM_acc1, FM_acc5 = self.current_task_validate_FM_cls_domain(test_datasets, task, active_classes)
        result = f"feature mapper_cls_domain train || task: {task:0>3d}, val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: " \
            f"{acc5:.2f}%, acc1: {acc1:.2f}, acc5: {acc5:.2f}, FM_acc1: {FM_acc1:.2f}, FM_acc5: {FM_acc5:.2f}"
        self.batch_train_logger.info(
            result
        )
        self.logger.info(
            result
        )
        self.batch_train_logger.info(f"------------------------------------------------------------------")
        self.logger.info(f"------------------------------------------------------------------")
        print(f'feature mapper_cls_domain train task : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%, '
              f'FM_acc1：%.3f%%, FM_acc5: %.3f%%' % (acc1, acc5, FM_acc1, FM_acc5))
        print("Train feature mapper_cls_domain End.")
        self.pre_FM_cls_domain = copy.deepcopy(self.FM_cls_domain).eval()
        self.train(mode=mode)
        pass

    def EFAfIL_splitfeature_mapper_cls_domain_train(self, training_dataset, test_datasets, classes_per_task,
                                                    active_classes, task):
        # todo Done
        self.batch_train_logger.info(f'####Task {task} EFAfIL feature_mapper_cls_domain_train begin.####')
        self.logger.info(f'####Task {task} EFAfIL  feature_mapper_cls_domain_train begin.####')
        print("Task %d , train feature mapping..." % task)
        pre_features_target_hats_datasets = ExemplarDataset(self.exemplar_feature_sets)
        mode = self.training
        self.featureHandler_FE_cls = self.copy_from_EFAfIL_model()
        self.eval()
        self.FM_cls_domain.train()
        self.featureHandler_FE_cls.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = self.build_FM_cls_domain_optimize()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)
        train_dataset_iter_index = 0
        pre_features_target_hats_datasets_iter_index = 0
        for epoch in range(self.MLP_epochs):
            train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size, self.num_workers,
                                                      cuda=True if self.availabel_cudas else False))
            pre_features_target_hats_train_loader = iter(utils.get_data_loader(pre_features_target_hats_datasets,
                                                                               self.batch_size, self.num_workers,
                                                                               cuda=True if self.availabel_cudas else False))
            train_imgs_num = len(train_loader)
            pre_features_target_hats_num = len(pre_features_target_hats_train_loader)
            iter_index = 0
            while train_dataset_iter_index > iter_index:
                _, _ = next(train_loader)
                iter_index += 1
            iter_index = 0
            while pre_features_target_hats_datasets_iter_index > iter_index:
                _, _ = next(pre_features_target_hats_train_loader)
                iter_index += 1
            iter_index = 0
            iters_left = max(train_imgs_num, pre_features_target_hats_num)
            while iters_left > 0:
                iters_left -= 1
                if train_dataset_iter_index == train_imgs_num:
                    train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size, self.num_workers,
                                                              cuda=True if self.availabel_cudas else False))
                    train_dataset_iter_index = 0
                if pre_features_target_hats_datasets_iter_index == pre_features_target_hats_num:
                    pre_features_target_hats_train_loader = iter(
                        utils.get_data_loader(pre_features_target_hats_datasets,
                                              self.batch_size, self.num_workers,
                                              cuda=True if self.availabel_cudas else False))
                    pre_features_target_hats_datasets_iter_index = 0
                iter_index += 1
                imgs, labels = next(train_loader)
                pre_features, pre_labels = next(pre_features_target_hats_train_loader)
                train_dataset_iter_index += 1
                pre_features_target_hats_datasets_iter_index += 1
                imgs, labels = imgs.to(self.MLP_device), labels.to(self.MLP_device)
                pre_features, pre_labels = pre_features.to(self.MLP_device), pre_labels.to(self.MLP_device)
                # target_hats = self.get_cls_target(pre_features)
                imgs_2_features, imgs_2_targets = self.featureHandler_FE_cls(imgs)
                '''train featureHandler_FE_cls'''
                with torch.no_grad():
                    features, score = self(imgs)
                # -if needed, remove predictions for classes not in current task
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    imgs_2_targets = imgs_2_targets[:, class_entries]
                scores_hats = score[:, :(classes_per_task * (task - 1))]
                scores_hats = torch.sigmoid(scores_hats / self.KD_temp)
                binary_targets = utils.to_one_hot(labels.cpu(), imgs_2_targets.size(1)).to(self.MLP_device)
                binary_targets = binary_targets[:, -classes_per_task:]
                binary_targets = torch.cat([scores_hats, binary_targets], dim=1)
                imgs_loss_cls_distill = Func.binary_cross_entropy_with_logits(
                    input=imgs_2_targets, target=binary_targets, reduction='none'
                ).sum(dim=1).mean()
                imgs_loss_sim = 1 - torch.cosine_similarity(imgs_2_features, features).mean()

                '''train FM_cls_domain'''
                all_features = torch.cat((imgs_2_features, pre_features), dim=0)
                all_labels = torch.cat((labels, pre_labels), dim=0)
                all_feature_hats, all_target_hats, domain_hats = self.FM_cls_domain(all_features)
                all_scores_hats = self.get_cls_target(all_feature_hats)
                imgs_feature_hats = all_feature_hats[:imgs_2_features.size(0), :]
                pre_task_feature_hats = all_feature_hats[imgs_2_features.size(0):, :]
                imgs_target_hats = all_target_hats[:imgs_2_features.size(0)]
                pre_task_target_hats = all_target_hats[imgs_2_features.size(0):]
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    all_target_hats = all_target_hats[:, class_entries]
                all_scores_hats = all_scores_hats[:, :(classes_per_task * (task - 1))]
                all_scores_hats = torch.sigmoid(all_scores_hats / self.KD_temp)
                # print(all_target_hats.shape, all_target_hats.size(1), all_labels.shape)
                binary_targets = utils.to_one_hot(all_labels.cpu(), all_target_hats.size(1)).to(self.MLP_device)
                binary_targets = binary_targets[:, -classes_per_task:]
                binary_targets = torch.cat([all_scores_hats, binary_targets], dim=1)
                all_loss_cls_distill = Func.binary_cross_entropy_with_logits(
                    input=all_target_hats, target=binary_targets, reduction='none'
                ).sum(dim=1).mean()
                all_feature_loss_cls = criterion(all_target_hats, all_labels)
                all_feature_sim = 1 - torch.cosine_similarity(all_features, all_feature_hats).mean()
                loss_total = imgs_loss_cls_distill + imgs_loss_sim + all_feature_loss_cls + all_feature_sim
                loss_total.backward()
                optimizer.step()
                optimizer.zero_grad()
                precision = None if all_target_hats is None else (all_labels == all_target_hats.max(1)[
                    1]).sum().item() / all_labels.size(0)
                ite_info = {
                    'task': task,
                    'lr': scheduler.get_last_lr()[0],
                    'loss_total': loss_total.item(),
                    'precision': precision if precision is not None else 0.,
                }
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
        print("Train feature mapper_cls_domain End.")
        self.pre_FM_cls_domain = copy.deepcopy(self.FM_cls_domain).eval()
        self.train(mode=mode)
        pass

    def EFAfIL_feature_mapper_cls_domain_train_extraData(self, training_dataset, extra_train_datasets,
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
        optimizer = self.build_FM_cls_domain_optimize()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.MLP_milestones, gamma=self.MLP_gamma)

        train_dataset_iter_index = 0
        pre_features_target_hats_datasets_iter_index = 0
        '''get extra dataset loader'''
        extra_train_dataset_index = 0
        extra_data_loader = iter(utils.get_data_loader(extra_train_datasets, self.batch_size, self.num_workers,
                                                       cuda=True if self.availabel_cudas else False))
        extra_train_dataset_num = len(extra_data_loader)
        print("extra_train_dataset_num:", extra_train_dataset_num)

        for epoch in range(self.MLP_epochs):
            train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size, self.num_workers,
                                                      cuda=True if self.availabel_cudas else False))
            pre_features_target_hats_train_loader = iter(utils.get_data_loader(pre_features_target_hats_datasets,
                                                                               self.batch_size, self.num_workers,
                                                                               cuda=True if self.availabel_cudas else False))
            '''if read extra data end, newly get extra dataset loader'''
            if extra_train_dataset_index == extra_train_dataset_num:
                extra_data_loader = iter(utils.get_data_loader(self.extra_train_datasets, self.batch_size, self.num_workers,
                                                               cuda=True if self.availabel_cudas else False))
                extra_train_dataset_index = 0

            train_imgs_num = len(train_loader)
            pre_features_target_hats_num = len(pre_features_target_hats_train_loader)
            iter_index = 0
            while train_dataset_iter_index > iter_index:
                _, _ = next(train_loader)
                iter_index += 1
            iter_index = 0
            while pre_features_target_hats_datasets_iter_index > iter_index:
                _, _ = next(pre_features_target_hats_train_loader)
                iter_index += 1
            iter_index = 0
            iters_left = max(train_imgs_num, pre_features_target_hats_num)

            while iters_left > 0:
                iters_left -= 1
                optimizer.zero_grad()
                if train_dataset_iter_index == train_imgs_num:
                    train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size, self.num_workers,
                                                              cuda=True if self.availabel_cudas else False))
                    train_dataset_iter_index = 0
                if pre_features_target_hats_datasets_iter_index == pre_features_target_hats_num:
                    pre_features_target_hats_train_loader = iter(
                        utils.get_data_loader(pre_features_target_hats_datasets,
                                              self.batch_size, self.num_workers,
                                              cuda=True if self.availabel_cudas else False))
                    pre_features_target_hats_datasets_iter_index = 0
                iter_index += 1
                imgs, labels = next(train_loader)
                pre_features, pre_labels = next(pre_features_target_hats_train_loader)
                train_dataset_iter_index += 1
                pre_features_target_hats_datasets_iter_index += 1

                '''if read extra data end, newly get extra dataset loader'''
                if extra_train_dataset_index == extra_train_dataset_num:
                    extra_data_loader = iter(utils.get_data_loader(self.extra_train_datasets, self.batch_size, self.num_workers,
                                                                   cuda=True if self.availabel_cudas else False))
                    extra_train_dataset_index = 0
                extra_x, extra_y = next(extra_data_loader)
                extra_train_dataset_index += 1

                imgs, labels = imgs.to(self.MLP_device), labels.to(self.MLP_device)
                pre_features, pre_labels = pre_features.to(self.MLP_device), pre_labels.to(self.MLP_device)
                extra_x, extra_y = extra_x.to(self.MLP_device), extra_y.to(self.MLP_device)

                # target_hats = self.get_cls_target(pre_features)
                with torch.no_grad():
                    imgs_2_features, imgs_2_targets = self(imgs)
                    extra_imgs_2_features, extra_imgs_2_targets = self(extra_x)

                imgs_2_feature_hat, y_hat, domain_hat = self.FM_cls_domain(imgs_2_features)
                extra_imgs_2_feature_hat, extra_y_hat, extra_domain_hat = self.FM_cls_domain(extra_imgs_2_features)
                pre_features_hat, pre_y_hat, pre_domain_hat = self.FM_cls_domain(pre_features)

                y_hat_pre_tasks = y_hat[:, :(classes_per_task * (task - 1))]
                extra_y_hat_pre_tasks = extra_y_hat[:, :(classes_per_task * (task - 1))]
                pre_y_hat_tasks = pre_y_hat[:, :(classes_per_task * (task - 1))]
                y_hat = y_hat[:, :(classes_per_task * task)]
                pre_y_hat = pre_y_hat[:, :(classes_per_task * task)]

                pre_features_targets = self.get_cls_target(pre_features)
                imgs_2_targets = imgs_2_targets[:, :(classes_per_task * (task - 1))]
                imgs_2_targets = torch.sigmoid(imgs_2_targets / self.KD_temp)

                extra_imgs_2_targets = extra_imgs_2_targets[:, :(classes_per_task * (task - 1))]
                extra_imgs_2_targets = torch.sigmoid(extra_imgs_2_targets / self.KD_temp)

                pre_features_targets = pre_features_targets[:, :(classes_per_task * (task - 1))]
                pre_features_targets = torch.sigmoid(pre_features_targets / self.KD_temp)
                current_domain = classes_per_task * (task - 1)
                domain_targets = []
                for class_id in labels:
                    if class_id >= current_domain:
                        domain_targets.append(1)
                    else:
                        domain_targets.append(0)
                pre_domain_targets = torch.Tensor([0] * len(pre_labels)).to(self.MLP_device, dtype=torch.int64)
                domain_targets = torch.Tensor(domain_targets).to(self.MLP_device, dtype=torch.int64)

                loss_distill_current_task = Func.binary_cross_entropy_with_logits(input=y_hat_pre_tasks,
                                                                                  target=imgs_2_targets,
                                                                                  reduction='none').sum(dim=1).mean()

                loss_distill_extra_data = Func.binary_cross_entropy_with_logits(input=extra_y_hat_pre_tasks,
                                                                                target=extra_imgs_2_targets,
                                                                                reduction='none').sum(dim=1).mean()

                loss_distill_pre_tasks = Func.binary_cross_entropy_with_logits(input=pre_y_hat_tasks,
                                                                               target=pre_features_targets,
                                                                               reduction='none').sum(dim=1).mean()
                loss_distill = loss_distill_pre_tasks + loss_distill_extra_data + loss_distill_current_task
                loss_cls = criterion(y_hat, labels) + criterion(pre_y_hat, pre_labels)
                # loss_cls = criterion(y_hat, labels)

                loss_similar_currentTask = 1 - torch.cosine_similarity(imgs_2_feature_hat, imgs_2_features).mean()
                loss_similar_pre_tasks = 1 - torch.cosine_similarity(pre_features_hat, pre_features).mean()
                loss_similar_extra_data = 1 - torch.cosine_similarity(extra_imgs_2_feature_hat,
                                                                      extra_imgs_2_features).mean()

                loss_sim = loss_similar_pre_tasks + loss_similar_extra_data + loss_similar_currentTask

                loss_domain = criterion(domain_hat, domain_targets) + criterion(pre_domain_hat, pre_domain_targets)
                # loss_total = self.alpha * loss_domain + loss_cls + loss_similar
                loss_total = loss_cls + loss_sim + loss_distill
                # loss_total = loss_cls + loss_similar
                loss_total.backward()
                optimizer.step()
                precision = None if y_hat is None else (labels == y_hat.max(1)[
                    1]).sum().item() / labels.size(0)
                ite_info = {
                    'task': task,
                    'lr': scheduler.get_last_lr()[0],
                    'loss_total': loss_total.item(),
                    'precision': precision if precision is not None else 0.,
                }
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
        print("Train feature mapper_cls_domain End.")
        self.pre_FM_cls_domain = copy.deepcopy(self.FM_cls_domain).eval()
        self.train(mode=mode)
        pass

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

    ####----CLASSIFICATION----####

    def classify_with_features(self, x, allowed_classes=None):
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

    def EFAfIL_classify(self, x, classifier, active_classes, task, allowed_classes=None):
        if classifier == "ncm":
            return self.classify_with_features(x, allowed_classes)
        elif classifier == "FH_fc":
            self.FM_cls_domain.eval()
            with torch.no_grad():
                if task > 1:
                    targets = self.featureHandler_FEcls_result(x)[-1]
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
            features = self.feature_extractor(x)
            if self.norm_exemplars:
                features = F.normalize(features, p=2, dim=1)
            return torch.from_numpy(self.svm.predict(features.cpu().numpy())).to(self.MLP_device)
        elif classifier == "FH_FM_cls":
            self.FM_cls_domain.eval()
            with torch.no_grad():
                if task > 1:
                    FH_features = self.featureHandler_FEcls_result(x)[-2]
                    _, targets, _ = self.FM_cls_domain(FH_features)
                else:
                    targets = self.get_FE_cls_target(x)
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                targets = targets[:, class_entries]
            _, predicted = torch.max(targets, 1)
            return predicted
        elif classifier == "FE_cls":
            self.eval()
            with torch.no_grad():
                targets = self.get_FE_cls_target(x)
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                targets = targets[:, class_entries]
            _, predicted = torch.max(targets, 1)
            return predicted
        else:
            raise ValueError("classifier must be ncm/linearSVM/fc/fcls.")

    def current_task_validate_FM_cls_domain(self, test_datasets, task, active_classes):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        FM_top1 = AverageMeter()
        FM_top5 = AverageMeter()
        val_loader = utils.get_data_loader(test_datasets[task - 1], self.batch_size, self.num_workers, # task index must minus 1
                                           cuda=True if self.availabel_cudas else False)
        end = time.time()
        self.FM_cls_domain.eval()
        for inputs, labels in val_loader:
            data_time.update(time.time() - end)
            inputs, labels = inputs.to(self.MLP_device), labels.to(self.MLP_device)
            with torch.no_grad():
                feature_hat, y_hat = self.featureHandler_FEcls_result(inputs)
                _, FM_y_hat, _ = self.FM_cls_domain_result(feature_hat)
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                y_hat = y_hat[:, class_entries]
                FM_y_hat = FM_y_hat[:, class_entries]
            acc1, acc5 = accuracy(y_hat, labels, topk=(1, 5))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            FM_acc1, FM_acc5 = accuracy(FM_y_hat, labels, topk=(1, 5))
            FM_top1.update(FM_acc1.item(), inputs.size(0))
            FM_top5.update(FM_acc5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
        throughput = 1.0 / (batch_time.avg / self.batch_size)
        return top1.avg, top5.avg, FM_top1.avg, FM_top5.avg
        pass
