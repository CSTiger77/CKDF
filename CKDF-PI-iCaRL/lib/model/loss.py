import torch
from torch.nn import functional as F
import torch.nn as nn
from lib.utils import to_one_hot
from torch.nn.modules.loss import _Loss


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, output, label, reduction='mean'):
        loss = F.cross_entropy(output, label, reduction=reduction)
        return loss


class CrossEntropy_binary(nn.Module):
    def __init__(self):
        super(CrossEntropy_binary, self).__init__()

    def forward(self, output, label, reduction='mean'):
        binary_targets = to_one_hot(label.cpu(), output.size(1)).to(label.device)
        loss_cls = F.binary_cross_entropy_with_logits(input=output, target=binary_targets,
                                                      reduction='none'
                                                      ).sum(dim=1)
        if "mean" in reduction:
            loss_cls = loss_cls.mean()
        return loss_cls



# 软目标交叉熵
class SoftTarget_CrossEntropy(nn.Module):
    def __init__(self, mean=True):
        super().__init__()
        self.mean = mean

    def forward(self, output, soft_target, kd_temp):
        assert len(output) == len(soft_target)
        log_prob = torch.nn.functional.log_softmax(output / kd_temp, dim=1)
        if self.mean:
            loss = -torch.sum(log_prob * soft_target) / len(soft_target)
        else:
            loss = -torch.sum(log_prob * soft_target)
        return loss


class loss_fn_kd_KL(nn.Module):
    def __init__(self):
        super(loss_fn_kd_KL, self).__init__()

    def forward(self, scores, target_scores, T=2., reduction='mean'):
        log_scores = F.log_softmax(scores / T, dim=1)
        targets = F.softmax(target_scores / T, dim=1)
        criterion = torch.nn.KLDivLoss(reduction="none")
        loss_cls = criterion(log_scores, targets).sum(dim=1)
        if 'mean' in reduction:
            loss_cls = loss_cls.mean()
        return loss_cls


def loss_fn_kd(scores, target_scores, T=2., reduction="mean"):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    # n = scores.size(1)
    assert len(scores) == len(target_scores) and scores.size(1) == target_scores.size(1)
    # if n > target_scores.size(1):
    #     n_batch = scores.size(0)
    #     zeros_to_add = torch.zeros(n_batch, n - target_scores.size(1))
    #     zeros_to_add = zeros_to_add.to(device)
    #     targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)  # --> sum over classes
    if reduction == "mean":
        KD_loss_unnorm = KD_loss_unnorm.mean()  # --> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T ** 2

    return KD_loss


def loss_fn_kd_binary(scores, target_scores, T=2., reduction="mean"):
    """Compute binary knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    scores_norm = torch.sigmoid(scores / T)
    targets_norm = torch.sigmoid(target_scores / T)
    assert len(scores) == len(target_scores) and scores.size(1) == target_scores.size(1)
    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    # n = scores.size(1)
    # if n > target_scores.size(1):
    #     n_batch = scores.size(0)
    #     zeros_to_add = torch.zeros(n_batch, n - target_scores.size(1))
    #     zeros_to_add = zeros_to_add.to(device)
    #     targets_norm = torch.cat([targets_norm, zeros_to_add], dim=1)

    # Calculate distillation loss
    KD_loss_unnorm = -(targets_norm * torch.log(scores_norm) + (1 - targets_norm) * torch.log(1 - scores_norm))
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)  # --> sum over classes
    if reduction == "mean":
        KD_loss_unnorm = KD_loss_unnorm.mean()  # --> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T ** 2

    return KD_loss


def compute_cls_distill_binary_loss(labels, output, classes_per_task,
                                    pre_model_output_for_distill):
    binary_targets = to_one_hot(labels.cpu(), output.size(1)).to(labels.device)
    if pre_model_output_for_distill is not None:
        binary_targets = binary_targets[:, -classes_per_task:]
        binary_targets = torch.cat([torch.sigmoid(pre_model_output_for_distill), binary_targets], dim=1)
    predL = None if labels is None else F.binary_cross_entropy_with_logits(
        input=output, target=binary_targets, reduction='none'
    ).sum(dim=1).mean()  # --> sum over classes, then average over batch
    return predL


def compute_distill_binary_loss(output_for_distill, pre_model_output_for_distill):
    binary_targets = torch.sigmoid(pre_model_output_for_distill)
    predL = F.binary_cross_entropy_with_logits(input=output_for_distill, target=binary_targets, reduction='none'
                                               ).sum(dim=1).mean()  # --> sum over classes, then average over batch
    return predL


def compute_cls_binary_loss(labels, output, classes_per_task, reduction="mean"):
    binary_targets = to_one_hot(labels.cpu(), output.size(1)).to(labels.device)
    binary_targets = binary_targets[:, -classes_per_task:]
    output_for_newclass_cls = output[:, -classes_per_task:]
    predL = F.binary_cross_entropy_with_logits(
        input=output_for_newclass_cls, target=binary_targets, reduction='none'
    ).sum(dim=1)  # --> sum over classes, then average over batch
    if "mean" == reduction:
        predL = predL.mean()
    return predL


def compute_distill_loss(output_for_distill, previous_task_model_output, temp=1., reduction='mean'):
    # distill_previous_task_active_classes_num: dpt_active_classes_num
    distill_loss = loss_fn_kd(output_for_distill, previous_task_model_output, temp,
                              reduction=reduction) * (temp ** 2)
    '''if self.cfg.TRAIN.DISTILL.softmax_sigmoid == 0:
        distill_loss = loss_fn_kd(output_for_distill, previous_task_model_output, temp,
                                  reduction=reduction) * (temp ** 2)
    elif self.cfg.TRAIN.DISTILL.softmax_sigmoid == 1:
        distill_loss = loss_fn_kd_binary(output_for_distill, previous_task_model_output,
                                         temp,
                                         reduction=reduction) * (temp ** 2)
    else:
        loss_fn_kd_KL_forward = loss_fn_kd_KL()
        distill_loss = loss_fn_kd_KL_forward(output_for_distill, previous_task_model_output,
                                             T=temp, reduction=reduction) * (temp ** 2)'''
    return distill_loss
