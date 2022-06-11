import os
from logging.handlers import TimedRotatingFileHandler

import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn import functional as F
from torchvision import transforms
import copy
from public import data
import logging


###################
## Loss function ##
###################

def loss_fn_kd(scores, target_scores, T=2.):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    n = scores.size(1)
    if n > target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n - target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)  # --> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()  # --> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T ** 2

    return KD_loss


def loss_fn_kd_binary(scores, target_scores, T=2.):
    """Compute binary knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    scores_norm = torch.sigmoid(scores / T)
    targets_norm = torch.sigmoid(target_scores / T)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    n = scores.size(1)
    if n > target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n - target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm, zeros_to_add], dim=1)

    # Calculate distillation loss
    KD_loss_unnorm = -(targets_norm * torch.log(scores_norm) + (1 - targets_norm) * torch.log(1 - scores_norm))
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)  # --> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()  # --> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T ** 2

    return KD_loss


##-------------------------------------------------------------------------------------------------------------------##


#############################
## Data-handling functions ##
#############################

def get_data_loader(dataset, batch_size, num_workers, is_shuffle=True, cuda=False, collate_fn=None, drop_last=False,
                    augment=False):
    '''Return <DataLoader>-object for the provided <DataSet>-object [dataset].'''

    # If requested, make copy of original dataset to add augmenting transform (without altering original dataset)
    if augment:
        dataset_ = copy.deepcopy(dataset)
        dataset_.transform = transforms.Compose([dataset.transform, *data.AVAILABLE_TRANSFORMS['augment']])
    else:
        dataset_ = dataset

    # Create and return the <DataLoader>-object
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=is_shuffle)
    # return DataLoader(
    #     dataset_, batch_size=batch_size, shuffle=True,
    #     collate_fn=(collate_fn or default_collate), drop_last=drop_last,
    #     **({'num_workers': num_workers, 'pin_memory': True} if cuda else {})
    # )


def get_enumerate_loader(dataset, batch_size, num_workers, cuda=False, collate_fn=None, drop_last=False):
    '''Return <DataLoader>-object for the provided <DataSet>-object [dataset].'''

    # If requested, make copy of original dataset to add augmenting transform (without altering original dataset)
    # if augment:
    #     dataset_ = copy.deepcopy(dataset)
    #     dataset_.transform = transforms.Compose([dataset.transform, *data.AVAILABLE_TRANSFORMS['augment']])
    # else:
    #     dataset_ = dataset

    # Create and return the <DataLoader>-object
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    # return DataLoader(
    #     dataset_, batch_size=batch_size, shuffle=True,
    #     collate_fn=(collate_fn or default_collate), drop_last=drop_last,
    #     **({'num_workers': 4, 'pin_memory': True} if cuda else {})
    # )


def label_squeezing_collate_fn(batch):
    x, y = default_collate(batch)
    return x, y.long().squeeze()


def to_one_hot(y, classes):
    '''Convert a nd-array with integers [y] to a 2D "one-hot" tensor.'''
    c = np.zeros(shape=[len(y), classes], dtype='float32')
    c[range(len(y)), y] = 1.
    c = torch.from_numpy(c)
    return c


##-------------------------------------------------------------------------------------------------------------------##

##########################################
## Object-saving and -loading functions ##
##########################################

def save_object(object, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)


def load_object(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)


##-------------------------------------------------------------------------------------------------------------------##

################################
## Model-inspection functions ##
################################

def count_parameters(model, verbose=True):
    '''Count number of parameters, print to screen.'''
    total_params = learnable_params = fixed_params = 0
    for param in model.parameters():
        n_params = index_dims = 0
        for dim in param.size():
            n_params = dim if index_dims == 0 else n_params * dim
            index_dims += 1
        total_params += n_params
        if param.requires_grad:
            learnable_params += n_params
        else:
            fixed_params += n_params
    if verbose:
        print("--> this network has {} parameters (~{} million)"
              .format(total_params, round(total_params / 1000000, 1)))
        print("      of which: - learnable: {} (~{} million)".format(learnable_params,
                                                                     round(learnable_params / 1000000, 1)))
        print("                - fixed: {} (~{} million)".format(fixed_params, round(fixed_params / 1000000, 1)))
    return total_params, learnable_params, fixed_params


def print_model_info(model, title="MODEL"):
    '''Print information on [model] onto the screen.'''
    print("\n" + 40 * "-" + title + 40 * "-")
    print(model)
    print(90 * "-")
    _ = count_parameters(model)
    print(90 * "-")


##-------------------------------------------------------------------------------------------------------------------##

#################################
## Custom-written "nn-Modules" ##
#################################


class Identity(nn.Module):
    '''A nn-module to simply pass on the input data.'''

    def forward(self, x):
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


class Reshape(nn.Module):
    '''A nn-module to reshape a tensor to a 4-dim "image"-tensor with [image_channels] channels.'''

    def __init__(self, image_channels):
        super().__init__()
        self.image_channels = image_channels

    def forward(self, x):
        batch_size = x.size(0)  # first dimenstion should be batch-dimension.
        image_size = int(np.sqrt(x.nelement() / (batch_size * self.image_channels)))
        return x.view(batch_size, self.image_channels, image_size, image_size)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(channels = {})'.format(self.image_channels)
        return tmpstr


class ToImage(nn.Module):
    '''Reshape input units to image with pixel-values between 0 and 1.

    Input:  [batch_size] x [in_units] tensor
    Output: [batch_size] x [image_channels] x [image_size] x [image_size] tensor'''

    def __init__(self, image_channels=1):
        super().__init__()
        # reshape to 4D-tensor
        self.reshape = Reshape(image_channels=image_channels)
        # put through sigmoid-nonlinearity
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.reshape(x)
        x = self.sigmoid(x)
        return x

    def image_size(self, in_units):
        '''Given the number of units fed in, return the size of the target image.'''
        image_size = np.sqrt(in_units / self.image_channels)
        return image_size


class Flatten(nn.Module):
    '''A nn-module to flatten a multi-dimensional tensor to 2-dim tensor.'''

    def forward(self, x):
        batch_size = x.size(0)  # first dimenstion should be batch-dimension.
        return x.view(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


##-------------------------------------------------------------------------------------------------------------------##

def get_logger(log_path='log'):
    """
    Args:
        name(str): name of logger
        log_dir(str): path of log
    """
    log_dir = log_path.split('/')
    name = log_dir[-1]
    log_dir = log_dir[:-1]
    log_dir = "/".join(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_name = log_path
    info_handler = TimedRotatingFileHandler(info_name,
                                            when='D',
                                            encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    # error_name = os.path.join(log_dir, '{}.error.log'.format(name))
    # error_handler = TimedRotatingFileHandler(error_name,
    #                                          when='D',
    #                                          encoding='utf-8')
    # error_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    info_handler.setFormatter(formatter)
    # error_handler.setFormatter(formatter)

    logger.addHandler(info_handler)
    # logger.addHandler(error_handler)

    return logger


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            sample = next(self.loader)
            self.next_input, self.next_target = sample
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def adjust_loss_alpha(alpha,
                      epoch,
                      factor=0.9,
                      loss_type="ce_family",
                      loss_rate_decay="lrdv1"):
    """动态调整蒸馏的比例

    loss_type: 损失函数的类型
        "ce_family": loss输入为student的pred以及label
        "kd_family": loss输入为student的pred、teacher的pred
        "gkd_family": loss输入为student的pred、teacher的pred以及label
        "fd_family": loss输入为student的feature、teacher的feature
    loss_rate_decay: 衰减策略
        "lrdv1": 一开始就有ce或者kd
        "lrdv2": 前30epoch没有ce或者kd
    """
    if loss_rate_decay not in [
        "lrdv0", "lrdv1", "lrdv2", "lrdv3", "lrdv4", "lrdv5"
    ]:
        raise Exception("loss_rate_decay error")

    if loss_type not in ["ce_family", "kd_family", "gkd_family", "fd_family"]:
        raise Exception("loss type error")
    if loss_rate_decay == "lrdv0":
        return alpha

    elif loss_rate_decay == "lrdv1":
        return alpha * (factor ** (epoch // 30))
    elif loss_rate_decay == "lrdv2":
        if "ce" in loss_type or "kd" in loss_type:
            return 0 if epoch <= 30 else alpha * (factor ** (epoch // 30))
        else:
            return alpha * (factor ** (epoch // 30))
    elif loss_rate_decay == "lrdv3":
        if epoch >= 160:
            exponent = 2
        elif epoch >= 60:
            exponent = 1
        else:
            exponent = 0
        if "ce" in loss_type or "kd" in loss_type:
            return 0 if epoch <= 60 else alpha * (factor ** exponent)
        else:
            return alpha * (factor ** exponent)
    elif loss_rate_decay == "lrdv5":
        if "ce" in loss_type or "kd" in loss_type:
            return 0 if epoch <= 60 else alpha
        else:
            if epoch >= 160:
                return alpha * (factor ** 3)
            elif epoch >= 120:
                return alpha * (factor ** 2)
            elif epoch >= 60:
                return alpha * (factor ** 1)
            else:
                return alpha


def strore_features(FE_cls, training_dataset, class_per_task, group, file_dir,
                    pre_features_file="val_features.npy", label_file='val_labels.npy'):  # todo Done!
    FE_cls_model = copy.deepcopy(FE_cls).eval()
    val_loader = get_data_loader(training_dataset, 128, 4, cuda=True)
    with torch.no_grad():
        first_entry = True
        for image_batch, label_batch in val_loader:
            image_batch = image_batch.to('cuda')
            pre_feature_batch = FE_cls_model(image_batch)[-2]
            label_batch = label_batch.cpu()
            pre_feature_batch = pre_feature_batch.cpu()
            if first_entry:
                pre_features = pre_feature_batch
                labels = label_batch
                first_entry = False
            else:
                pre_features = torch.cat([pre_features, pre_feature_batch], dim=0)
                labels = torch.cat([labels, label_batch], dim=0)
    assert pre_features.size(0) == labels.size(0)
    pre_features_file = "{}/{}_group_{}_".format(file_dir, class_per_task, group) + pre_features_file
    label_file = '{}/{}_group_{}_'.format(file_dir, class_per_task, group) + label_file
    pre_features = pre_features.numpy()
    labels = labels.numpy()
    np.save(pre_features_file, pre_features)
    np.save(label_file, labels)
    pass


def FC_strore_features(FE_cls, FM_cls_domain, training_dataset, class_per_task, task, file_dir,
                       pre_features_file="pre_features.npy", features_file="features.npy",
                       label_file='labels.npy'):  # todo Done!
    FE_cls_model = copy.deepcopy(FE_cls).eval()
    FM_cls_domain_model = copy.deepcopy(FM_cls_domain).eval()
    val_loader = get_data_loader(training_dataset, 128, 4, cuda=True)
    with torch.no_grad():
        first_entry = True
        for image_batch, label_batch in val_loader:
            image_batch = image_batch.to('cuda')
            pre_feature_batch = FE_cls_model(image_batch)[-2]
            feature_batch = FM_cls_domain_model(pre_feature_batch)[-3].cpu()
            label_batch = label_batch.cpu()
            pre_feature_batch = pre_feature_batch.cpu()
            if first_entry:
                pre_features = pre_feature_batch
                features = feature_batch
                labels = label_batch
                first_entry = False
            else:
                pre_features = torch.cat([pre_features, pre_feature_batch], dim=0)
                features = torch.cat([features, feature_batch], dim=0)
                labels = torch.cat([labels, label_batch], dim=0)
    assert features.size(0) == labels.size(0)
    pre_features_file = "{}/{}_group_{}_".format(file_dir, class_per_task, task) + pre_features_file
    features_file = "{}/{}_group_{}_".format(file_dir, class_per_task, task) + features_file
    label_file = '{}/{}_group_{}_'.format(file_dir, class_per_task, task) + label_file
    pre_features = pre_features.numpy()
    features = features.numpy()
    labels = labels.numpy()
    np.save(pre_features_file, pre_features)
    np.save(features_file, features)
    np.save(label_file, labels)
    pass


def split_array(exemplar_sets, val_rate=0.1, sample_seed=0):
    assert len(exemplar_sets[0]) > 0
    exemplar_dataset = copy.deepcopy(exemplar_sets)
    examplar_train_sets = []
    examplar_val_sets = []
    val_last_index = int(val_rate * len(exemplar_dataset[0]))
    for examplar_dataset_per_class in exemplar_dataset:
        np.random.seed(sample_seed)
        np.random.shuffle(examplar_dataset_per_class)
        examplar_val_sets.append(examplar_dataset_per_class[:val_last_index])
        examplar_train_sets.append(examplar_dataset_per_class[val_last_index:])
    return examplar_train_sets, examplar_val_sets


def convert_to_oneclass(labels):
    for i in range(len(labels)):
        labels[i] = labels[0]
    return labels
