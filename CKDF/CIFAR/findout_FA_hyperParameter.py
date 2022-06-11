import os
import time

import torch
from torch.nn import functional as Func
from CIFAR.alg_model import resnetforcifar
from public import utils, MLP
from public.data import get_multitask_experiment
from public.util_models import MLP_for_FM, FE_cls
from public.utils import AverageMeter, accuracy


def get_dataset():
    (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
        name="CIFAR100", tasks=10, data_dir=r"C:\Users\likunchi\work\dataset",
        exception=True
    )
    return train_datasets, test_datasets, config, classes_per_task


def current_task_validate(model, val_datasets, active_classes):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    val_loader = utils.get_data_loader(val_datasets, 128,  # task index must minus 1
                                       cuda=True)
    # mode = self.training
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for inputs, labels in val_loader:
            data_time.update(time.time() - end)
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            _, y_hat = model(inputs)
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                y_hat = y_hat[:, class_entries]
        acc1, acc5 = accuracy(y_hat, labels, topk=(1, 5))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
    throughput = 1.0 / (batch_time.avg / 128)
    return top1.avg, top5.avg, throughput


def build_FM_optimize(FM_model, FM_lr, optim_type, MLP_weight_decay, MLP_momentum):
    # Define optimizer (only include parameters that "requires_grad")
    optim_list = [{'params': filter(lambda p: p.requires_grad, FM_model.parameters()), 'lr': FM_lr}]
    optimizer = None
    if optim_type in ("adam", "adam_reset"):
        if MLP_weight_decay:
            optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999), weight_decay=MLP_weight_decay)
        else:
            optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999))
    elif optim_type == "sgd":
        if MLP_momentum:
            optimizer = torch.optim.SGD(optim_list, momentum=MLP_momentum, weight_decay=MLP_weight_decay)
        else:
            optimizer = torch.optim.SGD(optim_list)
    else:
        raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(optim_type))

    return optimizer


def current_task_validate_FM(test_dataset, FM_model, pre_FE_cls, FE_cls, active_classes):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    val_loader = utils.get_data_loader(test_dataset, 128,  # task index must minus 1
                                       cuda=True)
    # switch to evaluate mode
    FM_model.eval()
    pre_FE_cls.eval()
    FE_cls.eval()
    with torch.no_grad():
        end = time.time()
        for inputs, labels in val_loader:
            data_time.update(time.time() - end)
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            # _, y_hat = FM_model(pre_FE_cls(inputs)[-2])
            y_hat = FE_cls.module.get_cls_results(FM_model(pre_FE_cls(inputs)[-2])[-2])
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                y_hat = y_hat[:, class_entries]
            acc1, acc5 = accuracy(y_hat, labels, topk=(1, 5))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
    throughput = 1.0 / (batch_time.avg / 128)
    return top1.avg, top5.avg, throughput


# MLP_name, pre_FE_model, FE_model, currenttask_train_data, currenttask_val_data, active_classes,
#                          MLP_milestones, MLP_gamma, MLP_epochs

def feature_mapper_train(MLP_name, pre_FE_cls, FE_cls, training_dataset, test_dataset, active_classes,
                         MLP_milestones, MLP_gamma, MLP_epochs, FM_lr, optim_type, MLP_weight_decay,
                         MLP_momentum, availabel_cudas, KD_temp, pre_task_testdata):
    # todo Done
    os.environ['CUDA_VISIBLE_DEVICES'] = availabel_cudas
    device_ids = [i for i in range(len(availabel_cudas.strip().split(',')))]
    FM_model = torch.nn.DataParallel(MLP_for_FM(MLP.__dict__[MLP_name](input_dim=128, out_dim=128), 128, 100),
                                     device_ids=device_ids).cuda()
    pre_FE_cls.eval()
    FE_cls.eval()
    FM_model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_FM_optimize(FM_model, FM_lr, optim_type, MLP_weight_decay, MLP_momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=MLP_milestones, gamma=MLP_gamma)
    acc_max = 0
    acc_max_epoch = -1
    for epoch in range(MLP_epochs):
        train_loader = iter(utils.get_data_loader(training_dataset, 128,
                                                  cuda=True))
        iters_left = len(train_loader)
        iter_index = 0
        while iters_left > 0:
            iters_left -= 1
            iter_index += 1
            optimizer.zero_grad()
            imgs, labels = next(train_loader)
            labels = labels.to("cuda")
            imgs = imgs.to("cuda")
            with torch.no_grad():
                pre_features = pre_FE_cls(imgs)[-2]
                features, scores = FE_cls(imgs)
            feature_hat, target_hat = FM_model(pre_features)
            y_hat = FE_cls.module.get_cls_results(feature_hat)
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                target_hat = target_hat[:, class_entries]
                scores = scores[:, class_entries]
            loss_sim = 1 - torch.cosine_similarity(features, feature_hat).mean()
            loss_cls = criterion(target_hat, labels)
            loss_cls_y_hat = criterion(y_hat, labels)
            binary_targets = torch.sigmoid(scores / KD_temp)
            loss_distill = Func.binary_cross_entropy_with_logits(
                input=target_hat, target=binary_targets, reduction='none'
            ).sum(dim=1).mean()
            loss = loss_sim + loss_cls + loss_distill
            loss_2 = loss_sim + loss_cls_y_hat
            loss.backward()
            optimizer.step()
            precision = None if labels is None else (labels == target_hat.max(1)[1]).sum().item() / imgs.size(0)
            ite_info = {
                'Epoch': epoch,
                'lr': scheduler.get_last_lr()[0],
                'loss_total': loss.item(),
                'precision': precision if precision is not None else 0.,
            }
            # print("Task %d || Epoch %d || batchindex %d || info:" % (task, epoch, iter_index))
            print(ite_info)
            print("....................................")
        scheduler.step()
        acc1, acc5, throughput = current_task_validate_FM(test_dataset, FM_model, pre_FE_cls, FE_cls, active_classes)
        if acc_max < acc1:
            acc_max_epoch = epoch
            acc_max = acc1
    print(f'batch train task :  测试分类准确率为 acc1：%.3f%%, acc5: %.3f%%' % (acc1, acc5))
    print("Train feature mapping End.")
    print("acc_max_epoch:", acc_max_epoch)
    print("acc_max:", acc_max)
    preacc1, preacc5, prethroughput = current_task_validate_FM(pre_task_testdata, FM_model, pre_FE_cls, FE_cls,
                                                               active_classes)
    print(f'pre task validate results :  测试分类准确率为 preacc1：%.3f%%, preacc5: %.3f%%' % (preacc1, preacc5))
    pass


def main(MLP_name, MLP_milestones, MLP_gamma, MLP_epochs, lr, optim_type, MLP_weight_decay, MLP_momentum,
         availabel_cudas, KD_temp):
    train_datasets, test_datasets, config, classes_per_task = get_dataset()
    pretask_traindata = train_datasets[0]
    pretask_valdata = test_datasets[0]
    currenttask_train_data = train_datasets[1]
    currenttask_val_data = test_datasets[1]
    pre_FE_model = torch.load("pre_FE_cls_examplar_0.pth")
    FE_model = torch.load("FE_cls_examplar_0.pth")
    pre_active_classes = list(range(10))
    acc1, acc5, _ = current_task_validate(pre_FE_model, pretask_valdata, pre_active_classes)
    print(f"acc1: {acc1}, acc5: {acc5}")
    active_classes = list(range(10 * 2))
    acc1, acc5, _ = current_task_validate(FE_model, pretask_valdata, pre_active_classes)
    print(f"acc1: {acc1}, acc5: {acc5}")
    acc1, acc5, _ = current_task_validate(FE_model, currenttask_val_data, active_classes)
    print(f"acc1: {acc1}, acc5: {acc5}")
    # print(pre_FE_model)
    feature_mapper_train(MLP_name, pre_FE_cls=pre_FE_model, FE_cls=FE_model, training_dataset=currenttask_train_data,
                         test_dataset=currenttask_val_data, active_classes=active_classes,
                         MLP_milestones=MLP_milestones,
                         MLP_gamma=MLP_gamma, MLP_epochs=MLP_epochs, FM_lr=lr, optim_type=optim_type,
                         MLP_weight_decay=MLP_weight_decay, MLP_momentum=MLP_momentum, availabel_cudas=availabel_cudas,
                         KD_temp=KD_temp, pre_task_testdata=pretask_valdata)


if __name__ == "__main__":
    MLP_name = "MLP_3_layers"
    MLP_milestones = [20, 30]
    MLP_gamma = 0.2
    MLP_epochs = 40
    lr = 0.001
    optim_type = "adam"
    MLP_weight_decay = 5e-4
    MLP_momentum = 0.9
    availabel_cudas = "0"
    KD_temp = 1
    main(MLP_name, MLP_milestones, MLP_gamma, MLP_epochs, lr, optim_type, MLP_weight_decay, MLP_momentum,
         availabel_cudas, KD_temp)
