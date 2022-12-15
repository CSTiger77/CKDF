import copy
import json
import logging
import math
import random
import time
import os
from logging.handlers import TimedRotatingFileHandler
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from .lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR
from ..dataset import TransformedDataset, TransformedDataset_for_exemplars


def create_logger(cfg, file_suffix, net_type=None):
    dataset = cfg.DATASET.dataset_name
    log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "{}_{}_{}.{}".format(dataset, net_type, time_str, file_suffix)
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger(name=file_suffix)
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file


def get_logger(cfg, file_suffix):
    """
    Args:
        name(str): name of logger
        log_dir(str): path of log
    """
    dataset = cfg.DATASET.dataset_name
    net_type = cfg.BACKBONE.TYPE
    module_type = cfg.MODULE.TYPE
    log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "{}_{}_{}_{}.{}".format(dataset, net_type, module_type, time_str, file_suffix)
    logger = logging.getLogger(file_suffix)
    logger.setLevel(logging.INFO)
    log_file = os.path.join(log_dir, log_name)
    info_handler = TimedRotatingFileHandler(log_file,
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

    return logger, log_file


def create_valid_logger(cfg):
    dataset = cfg.DATASET.DATASET
    net_type = cfg.BACKBONE.TYPE
    module_type = cfg.MODULE.TYPE

    test_model_path = os.path.join(*cfg.TEST.MODEL_FILE.split('/')[:-2])
    log_dir = os.path.join(test_model_path, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "Test_{}_{}_{}_{}.log".format(dataset, net_type, module_type, time_str)
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Start Testing--------------------")
    logger.info("Test model: {}".format(cfg.TEST.MODEL_FILE))

    return logger, log_file


def get_optimizer(model, BASE_LR=None, optimizer_type=None, momentum=None, weight_decay=None, **kwargs):
    base_lr = BASE_LR
    params = []
    optimizer = None
    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append({"params": p})

    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif optimizer_type == 'SGDWithExtraWeightDecay':
        optimizer = SGDWithExtraWeightDecay(
            params,
            kwargs['num_class_list'],
            kwargs['classifier_shape'],
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif optimizer_type == "ADAM" or optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
    return optimizer


def get_scheduler(optimizer, lr_type="warmup", lr_step=None, lr_factor=None, warmup_epochs=5, MAX_EPOCH=200):
    LR_STEP = lr_step
    LR_gamma = lr_factor
    lr_scheduler_type = lr_type
    if lr_scheduler_type == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            LR_STEP,
            gamma=LR_gamma,
        )
    elif lr_scheduler_type == "warmup":
        scheduler = WarmupMultiStepLR(
            optimizer,
            LR_STEP,
            gamma=LR_gamma,
            warmup_epochs=warmup_epochs,
        )
    elif "CosineAnnealing" == lr_scheduler_type:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=MAX_EPOCH)
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(lr_scheduler_type))

    return scheduler


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class SGDWithExtraWeightDecay(torch.optim.Optimizer):

    def __init__(self, params, num_class_list, classifier_shape, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        self.extra_weight_decay = weight_decay / num_class_list[:, None].repeat(1, classifier_shape[-1])
        self.classifier_shape = classifier_shape
        self.first = True

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDWithExtraWeightDecay, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDWithExtraWeightDecay, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                    if self.classifier_shape == d_p.shape:
                        if self.first:
                            self.first = False
                        else:
                            d_p.add_(self.extra_weight_decay * p.data)
                            self.first = True

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss


def to_one_hot(y, classes):
    '''Convert a nd-array with integers [y] to a 2D "one-hot" tensor.'''
    c = np.zeros(shape=[len(y), classes], dtype='float32')
    c[range(len(y)), y] = 1.
    c = torch.from_numpy(c)
    return c


def read_json(json_file):
    with open(json_file, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict


def adjust_learning_rate(base_lr, lr_factor, optimizer, epoch, max_epoch, cosine=True):
    lr = base_lr
    if cosine:
        eta_min = lr * (lr_factor ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / max_epoch)) / 2
    else:
        steps = np.sum(epoch > np.asarray(lr_factor))
        if steps > 0:
            lr = lr * (lr_factor ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def strore_features(model, training_dataset, file_dir, features_file="features.npy",
                    label_file='labels.npy'):  # todo Done!
    model_temp = copy.deepcopy(model).eval()
    val_loader = DataLoader(dataset=training_dataset, batch_size=128,
                            num_workers=4, shuffle=False, drop_last=False)
    features = None
    labels = None
    for image_batch, label_batch in val_loader:
        image_batch = image_batch.to('cuda')
        feature_batch = model_temp(image_batch, is_nograd=True,
                                   feature_flag=True)
        label_batch = label_batch.cpu()
        if features is None:
            features = feature_batch
            labels = label_batch
        else:
            features = torch.cat([features, feature_batch], dim=0)
            labels = torch.cat([labels, label_batch], dim=0)
    features_file = "{}/".format(file_dir) + features_file
    label_file = '{}/'.format(file_dir) + label_file
    labels = labels.numpy()
    np.save(label_file, labels)
    features = features.cpu().numpy()
    np.save(features_file, features)
    pass


def construct_dataset_concat(original_imgs_train_dataset, transform=None, mode="train"):
    dataset = None
    if "train" in mode:
        if type(original_imgs_train_dataset) is list:
            for dataset_item in original_imgs_train_dataset:
                transformed_dataset_item = TransformedDataset(dataset_item, transform=transform)
                if dataset is None:
                    dataset = transformed_dataset_item
                else:
                    dataset = ConcatDataset([dataset, transformed_dataset_item])
        else:
            dataset = TransformedDataset(original_imgs_train_dataset, transform=transform)
    else:
        if type(original_imgs_train_dataset) is list:
            for dataset_item in original_imgs_train_dataset:
                if dataset is None:
                    dataset = dataset_item
                else:
                    dataset = ConcatDataset([dataset, dataset_item])
        else:
            dataset = original_imgs_train_dataset
    return dataset
    pass


def construct_dataset_for_exemplar_concat(original_imgs_train_dataset, transform=None):
    dataset = None
    if type(original_imgs_train_dataset) is list:
        for dataset_item in original_imgs_train_dataset:
            transformed_dataset_item = TransformedDataset_for_exemplars(dataset_item, transform=transform)
            if dataset is None:
                dataset = transformed_dataset_item
            else:
                dataset = ConcatDataset([dataset, transformed_dataset_item])
    else:
        dataset = TransformedDataset_for_exemplars(original_imgs_train_dataset, transform=transform)
    return dataset
    pass

