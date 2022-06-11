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
from public import utils, MLP
import copy
import numpy as np

from public.data import SubDataset, FeaturesDataset
from public.util_models import MLP_cls_domain_dis, MLP_for_FM
from public.utils import AverageMeter, accuracy


class ExemplarHandler(nn.Module, metaclass=abc.ABCMeta):
    """Abstract  module for a classifier that can store and use exemplars.

    Adds a exemplar-methods to subclasses, and requires them to provide a 'feature-extractor' method."""

    def __init__(self, memory_budget, norm_exemplars, herding):
        super().__init__()

        # list with exemplar-sets
        self.exemplar_sets = []  # --> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_means = []
        self.compute_means = True

        # settings
        self.memory_budget = memory_budget
        self.norm_exemplars = norm_exemplars
        self.herding = herding

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def feature_extractor(self, images):
        pass

    ####----MANAGING EXEMPLAR SETS----####

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]

    def construct_exemplar_set(self, dataset, n):
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

        Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''

        # set model to eval()-mode
        mode = self.training
        self.eval()

        n_max = len(dataset)
        exemplar_set = []

        if self.herding:
            # compute features for each example in [dataset]
            first_entry = True
            dataloader = utils.get_data_loader(dataset, 128, cuda=self._is_on_cuda())
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self._device())
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
                features[index_selected] = features[index_selected] + 10000
        else:
            indeces_selected = np.random.choice(n_max, size=min(n, n_max), replace=False)
            for k in indeces_selected:
                exemplar_set.append(dataset[k][0].numpy())

        # add this [exemplar_set] as a [n]x[ich]x[isz]x[isz] to the list of [exemplar_sets]
        self.exemplar_sets.append(np.array(exemplar_set))

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


class FeaturesHandler(nn.Module, metaclass=abc.ABCMeta):
    """Abstract  module for a classifier that can store and use exemplars.

    Adds a exemplar-methods to subclasses, and requires them to provide a 'feature-extractor' method."""
    # MLP_name, num_classes, hidden_size, Exemple_memory_budget,
    # Feature_memory_budget, norm_exemplars, herding, batch_size, sim_alpha, MLP_lr,
    # MLP_momentum, MLP_milestones, MLP_lrgamma, MLP_weight_decay, MLP_epochs, optim_type,
    # svm_sample_type, svm_max_iter, availabel_cudas, logger, batch_train_logger
    def __init__(self, MLP_name, num_classes, hidden_size, Exemple_memory_budget,
                 Feature_memory_budget, norm_exemplars, herding, batch_size, sim_alpha, MLP_lr, MLP_momentum,
                 MLP_milestones, MLP_lrgamma, MLP_weight_decay, MLP_epochs, MLP_optim_type,
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
        # settings
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

    def construct_FM_model(self):
        if self.availabel_cudas:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
            device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
            FM_model = torch.nn.DataParallel(MLP_for_FM(MLP.__dict__[self.MLP_name](input_dim=self.feature_dim,
                                                                                    out_dim=self.feature_dim)),
                                             device_ids=device_ids).cuda()
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
    def prefeatures_2_target(self, prefeatures):
        pass

    @abc.abstractmethod
    def get_precls_target(self, prefeatures):
        pass

    @abc.abstractmethod
    def prefeatures_2_precls_target(self, prefeatures):
        pass

    def feature_mapping(self, features):
        if type(self.FM) is torch.nn.DataParallel:
            return self.FM.module.get_mapping_features(prefeatures=features)
        else:
            return self.FM.get_mapping_features(prefeatures=features)

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

    def feature_handle_main(self, training_dataset, test_datasets, classes_per_task, active_classes, task):
        self.batch_train_logger.info(f'#############Examples & Features handler task {task} start.##############')
        self.logger.info(f'#############Examples & Features handler task {task} start.##############')
        print("Examples & Features handler task-%d start." % (task))
        pre_tasks_features = []
        pre_tasks_targets = []
        features_per_class = int(np.floor(self.Feature_memory_budget / (classes_per_task * task)))
        Exemplar_per_class = int(np.floor(self.Exemple_memory_budget / (classes_per_task * task)))
        print("features_per_class:", features_per_class, "Exemplar_per_class:", Exemplar_per_class)
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
            dataloader = utils.get_data_loader(class_dataset, self.batch_size, cuda=self._is_on_cuda())
            first_entry = True
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self.MLP_device)
                if task > 1:
                    # feature_batch = self.feature_extractor(image_batch).cpu()
                    feature_batch = self.feature_mapping(self.get_preFE_feature(image_batch)).cpu()
                else:
                    feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            # current_task_features.append(features.numpy())
            current_task_features.append(features)
            self.construct_exemplar_feature_set(class_dataset, features, Exemplar_per_class,
                                                features_per_class)
        self.reduce_exemplar_sets(Exemplar_per_class)
        current_task_features = torch.stack(current_task_features)
        print("Examples & Features handler task-%d exemplar-feature sample END." % (task))
        print("Examples & Features handler task-%d svm train begin." % (task))
        self.linearSVM_train(pre_tasks_features, pre_tasks_targets, current_task_features.numpy(),
                             current_task_target, task, self.sample_type)

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
        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
        for epoch in range(self.MLP_epochs):
            train_loader = iter(utils.get_data_loader(training_dataset, self.batch_size,
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
                target_hat = self.get_cls_target(feature_hat)
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    target_hat = target_hat[:, class_entries]

                # Calculate prediction loss
                # -binary prediction loss
                binary_targets = utils.to_one_hot(labels.cpu(), target_hat.size(1)).to(self.device)
                loss_sim = 1 - torch.cosine_similarity(features, feature_hat).mean()
                loss_cls = Func.binary_cross_entropy_with_logits(
                    input=target_hat, target=binary_targets, reduction='none'
                ).sum(dim=1).mean()
                # loss_cls = criterion(target_hat, labels)
                # loss = self.alpha * loss_sim + loss_cls
                loss = loss_sim + loss_cls
                loss.backward()
                optimizer.step()
                precision = None if labels is None else (labels == target_hat.max(1)[1]).sum().item() / imgs.size(0)
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
        print(f'batch train task : {task:0>3d}, 测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))
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
            # feature = self.feature_extractor(x)
            if self.pre_FM is not None:
                feature = self.feature_mapping(self.get_preFE_feature(x))
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

    def ILtFA_classify(self, x, classifier="linearSVM", allowed_classes=None):
        if classifier == "ncm":
            return self.classify_with_features(x, allowed_classes)
        elif classifier == "linearSVM":
            if self.pre_FM is not None:
                print("get features through feature mapping.")
                features = self.feature_mapping(self.get_preFE_feature(x))
            else:
                features = self.feature_extractor(x)  # (batch_size, feature_size)
            # features = self.feature_extractor(x)
            return torch.from_numpy(self.svm.predict(features.cpu().numpy())).to(self.MLP_device)
        else:
            raise ValueError("classifier must be ncm or linearSVM.")

    def current_task_validate_FM(self, test_datasets, task, active_classes):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        val_loader = utils.get_data_loader(test_datasets[task - 1], self.batch_size,  # task index must minus 1
                                           cuda=True if self.availabel_cudas else False)
        mode = self.training
        # switch to evaluate mode
        self.eval()
        with torch.no_grad():
            end = time.time()
            for inputs, labels in val_loader:
                data_time.update(time.time() - end)
                inputs, labels = inputs.to(self.MLP_device), labels.to(self.MLP_device)
                y_hat = self.prefeatures_2_target(self.get_preFE_feature(inputs))
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


class EFAfIL_FeaturesHandler(nn.Module, metaclass=abc.ABCMeta):
    """Abstract  module for a classifier that can store and use exemplars.

    Adds a exemplar-methods to subclasses, and requires them to provide a 'feature-extractor' method."""

    def __init__(self, MLP_cls_domain, pre_MLP_cls_domain, feature_dim, num_classes, memory_budget, herding, lamb,
                 gamma, alpha,
                 MLP_lr, momentum,
                 weight_decay, optim_type, imgs_store_size, availabel_cudas):
        super().__init__()

        # list with exemplar-sets
        self.exemplar_feature_sets = []  # --> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_feature_means = []
        self.exemplar_sets = []  # --> each exemplar_set is an <np.array> of N images with shape (N, Ch, H, W)
        self.exemplar_means = []
        self.compute_means = True
        self.pre_FM_cls_domain = pre_MLP_cls_domain
        self.svm = None

        # settings
        self.imgs_store_size = imgs_store_size
        self.memory_budget = memory_budget
        self.herding = herding
        self.MLP_lr = MLP_lr
        self.lamb = lamb
        self.gamma = gamma
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optim_type = optim_type
        self.alpha = alpha
        self.lamb = lamb
        self.gamma = gamma
        if availabel_cudas:
            os.environ['CUDA_VISIBLE_DEVICES'] = availabel_cudas
            device_ids = [i for i in range(len(availabel_cudas.stirp().split(',')))]
            self.FM_cls_domain = torch.nn.DataParallel(MLP_cls_domain_dis(MLP_cls_domain, feature_dim, num_classes),
                                                       device_ids=device_ids).cuda()
            cudnn.benchmark = True
        else:
            self.FM_cls_domain = MLP_cls_domain_dis(MLP_cls_domain, feature_dim, num_classes)

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

    def get_prefeatures_results(self, prefeatures):
        self.FM_cls_domain.eval()
        with torch.no_grad():
            return self.FM_cls_domain(prefeatures)

    def feature_mapping(self, features):
        return self.FM_cls_domain.get_mapping_features(features)

    def get_preFE_FM_features(self, imgs):
        return self.feature_mapping(self.get_preFE_feature(imgs))

    def linearSVM_train(self, pre_tasks_features, pre_tasks_targets, current_task_features,
                        current_task_target, svm_store_path, sample_type="oversample"):
        # todo
        trainData = []
        trainLabels = []
        add_num = len(current_task_features[0]) - len(pre_tasks_features[0])
        for class_id in range(len(pre_tasks_targets)):
            if sample_type == "oversample":
                temp = pre_tasks_features[class_id]
                np.random.shuffle(temp)
                temp = list(temp)
                temp += temp[:add_num]
                trainData += temp
                trainLabels += [pre_tasks_targets[class_id]] * len(temp)
            else:
                temp = pre_tasks_features[class_id]
                temp = list(temp)
                trainData += temp
                trainLabels += [pre_tasks_targets[class_id]] * len(temp)

        for class_id in range(len(current_task_target)):
            if sample_type == "undersample":
                temp = current_task_features[class_id]
                np.random.shuffle(temp)
                temp = list(temp)
                temp = temp[:-add_num]
                trainData += temp
                trainLabels += [current_task_target[class_id]] * len(temp)
            else:
                temp = current_task_features[class_id]
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
        self.svm = svm.SVC(kernel='linear', probability=True, class_weight='balanced')
        self.svm.fit(train_X, train_y)
        print('svm testing accuracy:')
        print(self.svm.score(test_X, test_y))
        print("save svm model...")
        joblib.dump(self.svm, svm_store_path)
        print("save svm model done.")
        pass

    def feature_handle_main(self, training_dataset, test_datasets, classes_per_task, epochs, batch_size,
                            active_classes, task, sample_type="over_sample"):
        self.EFAfIL_feature_mapper_train(training_dataset, test_datasets, classes_per_task, epochs, batch_size,
                                         active_classes, task)
        pre_tasks_features = []
        pre_tasks_targets = []
        features_per_class = int(np.floor(self.memory_budget / (classes_per_task * task)))
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
            class_dataset = SubDataset(original_dataset=training_dataset, sub_labels=[class_id])
            current_task_target.append(class_id)
            exemplars = []
            dataloader = utils.get_data_loader(class_dataset, 128, cuda=self._is_on_cuda())
            first_entry = True
            for (image_batch, _) in dataloader:
                image_batch = image_batch.to(self._device())
                feature_batch = self.feature_extractor(image_batch).cpu()
                if first_entry:
                    features = feature_batch
                    first_entry = False
                else:
                    features = torch.cat([features, feature_batch], dim=0)
            current_task_features.append(features.numpy())
            self.construct_exemplar_feature_set(class_dataset, current_task_features[-1], features_per_class)
        self.linearSVM_train(pre_tasks_features, pre_tasks_targets, current_task_features,
                             current_task_target, sample_type)

    def EFAfIL_feature_mapper_train(self, training_dataset, test_datasets, classes_per_task, epochs, batch_size,
                                    active_classes, task):
        # todo Done
        mode = self.training
        self.eval()
        train_loader = utils.get_data_loader(training_dataset, batch_size,
                                             cuda=True if self.availabel_cudas else False,
                                             drop_last=True)
        self.FM_cls_domain.train()
        for (image_batch, labels) in train_loader:
            image_batch = image_batch.to(self._device())
            with torch.no_grad():
                feature_batch = self.get_preFE_feature(image_batch).cpu()
                labels_batch = labels_batch.cpu()
            if first_entry:
                features = feature_batch
                labels = labels_batch
                first_entry = False
            else:
                features = torch.cat([features, feature_batch], dim=0)
                labels = torch.cat([labels, labels_batch], dim=0)
        num_per_class = int(len(labels) / classes_per_task)
        for class_id, feature_exemplar_set in enumerate(self.exemplar_feature_sets):
            temp = copy.deepcopy(feature_exemplar_set)
            np.random.shuffle(temp)
            if num_per_class > len(temp):
                temp = torch.Tensor(np.vstack((temp, temp[:num_per_class])))
            features = torch.cat([features, temp], dim=0)
            labels = torch.cat([labels, torch.Tensor([class_id] * len(temp))], dim=0)
        feature_datasets = FeaturesDataset(features, labels)
        train_loader = utils.get_data_loader(feature_datasets, batch_size,
                                             cuda=True if self.availabel_cudas else False,
                                             drop_last=True)
        optim_list = [{'params': filter(lambda p: p.requires_grad, self.FM.parameters()), 'lr': self.MLP_lr}]
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
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for batch_index, pre_features, targets in enumerate(train_loader):
                optimizer.zero_grad()
                targets = targets.to(self._device())
                pre_features = pre_features.to(self._device())
                features_hat, target_hat, domain_hat = self.FM_cls_domain(pre_features)
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                    target_hat = target_hat[:, class_entries]
                domain_targets = []
                current_domain = classes_per_task * (task - 1)
                if self.domain:
                    for class_id in targets:
                        if class_id >= current_domain:
                            domain_targets.append(1)
                        else:
                            domain_targets.append(0)

                domain_ids = torch.Tensor(domain_targets).to(self._device())
                loss_domain = criterion(domain_hat, domain_targets)
                loss_cls = criterion(target_hat, targets)
                loss = self.alpha * loss_domain + loss_cls
                loss.backward()
                optimizer.step()
                precision = None if targets is None else (targets == target_hat.max(1)[
                    1]).sum().item() / pre_features.size(0)
                ite_info = {
                    'loss_total': loss.item(),
                    'precision': precision if precision is not None else 0.,
                }
                print("Task %d || Epoch %d || batchindex %d || info:" % task, epoch, batch_index)
                print(ite_info)
                print("....................................")
        print("Train feature transformer End.")
        self.pre_FM_cls_domain = copy.deepcopy(self.FM_cls_domain).eval()
        self.train(mode=mode)
        pass

    ####----MANAGING EXEMPLAR SETS----####

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_feature_sets):
            self.exemplar_feature_sets[y] = P_y[:m]

    def update_features_sets(self, pre_tasks_features, features_per_class):
        # todo done!
        for index, feature_set in enumerate(pre_tasks_features):
            self.exemplar_feature_sets[index] = feature_set[: features_per_class]
        pass

    def construct_exemplar_feature_set(self, class_dataset, current_class_features, features_per_class):
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

        Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''
        # todo Done
        '''Construct set of [n] exemplars from [dataset] using 'herding'.

                Note that [dataset] should be from specific class; selected sets are added to [self.exemplar_sets] in order.'''
        if self.imgs_store_size > features_per_class:
            raise ValueError("imgs_store_size must not surpass features_store_size.")
        n_max = len(class_dataset)
        exemplar_set = []
        features = copy.deepcopy(current_class_features)
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
                if len(exemplar_set) < self.imgs_store_size:
                    exemplar_set.append(class_dataset[index_selected][0].numpy())
                exemplar_features[k] = copy.deepcopy(features[index_selected])

                # make sure this example won't be selected again
                features[index_selected] = features[index_selected] + 100000
        else:
            indeces_selected = np.random.choice(n_max, size=min(features_per_class, n_max), replace=False)
            for k in indeces_selected:
                if len(exemplar_set) < self.imgs_store_size:
                    exemplar_set.append(class_dataset[k][0].numpy())
                exemplar_features[k] = copy.deepcopy(features[k])

        # add this [exemplar_set] as a [n]x[ich]x[isz]x[isz] to the list of [exemplar_sets]
        self.exemplar_sets.append(np.array(exemplar_set))
        self.exemplar_feature_sets.append(np.array(exemplar_features))
        pass

    ####----CLASSIFICATION----####

    def classify_with_exemplars(self, x, allowed_classes=None):
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
        pass

    def classify_main(self, x, classifier="ncm", allowed_classes=None):
        if classifier == "ncm":
            return self.classify_with_exemplars(x, allowed_classes)
        elif classifier == "fc":
            self.FM_cls_domain.eval()
            with torch.no_grad():
                targets = self.FM_cls_domain(self.get_preFE_feature(x))[-2]
            _, predicted = torch.max(targets, 1)
            return predicted.numpy()
        elif classifier == "linearSVM":
            features = self.get_preFE_FM_features(x)
            return self.svm.predict(features)


