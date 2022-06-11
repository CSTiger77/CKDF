import argparse
import os
import random
import sys
import time
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

sys.path.append("/share/home/kcli/CL_research/iCaRL_ILtFA")
from CIFAR.alg_model import resnetforcifar
from public.util_models import FE_cls, FE_2fc_cls, FE_3fc_cls

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from CIFAR.alg_model.config.config import Config

from public.data import get_dataset
from public.utils import AverageMeter, accuracy, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model_name', type=str, default=Config.model_name, help=' CNN model_name')
    parser.add_argument('--dataset_root',
                        type=str,
                        default=Config.dataset_root,
                        help='cifar dataset path')
    parser.add_argument('--lr',
                        type=float,
                        default=Config.lr,
                        help='learning rate')
    parser.add_argument('--momentum',
                        type=float,
                        default=Config.momentum,
                        help='momentum')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=Config.weight_decay,
                        help='weight decay')
    parser.add_argument('--gamma',
                        type=float,
                        default=Config.gamma,
                        help='gamma')
    parser.add_argument('--milestones',
                        type=list,
                        default=Config.milestones,
                        help='optimizer milestones')
    parser.add_argument('--epochs',
                        type=int,
                        default=Config.epochs,
                        help='num of training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=Config.batch_size,
                        help='batch size')
    parser.add_argument('--num_classes',
                        type=int,
                        default=Config.num_classes,
                        help='model classification num')
    parser.add_argument('--feature_dim',
                        type=int,
                        default=Config.feature_dim,
                        help='feature_dim')
    parser.add_argument('--num_workers',
                        type=int,
                        default=Config.num_workers,
                        help='number of worker to load data')
    parser.add_argument('--checkpoints',
                        type=str,
                        default=Config.checkpoint_path,
                        help='path for saving trained models')
    parser.add_argument('--log',
                        type=str,
                        default=Config.log,
                        help='path to save log')
    parser.add_argument('--evaluate',
                        type=str,
                        default=Config.evaluate,
                        help='path for evaluate model')
    parser.add_argument('--rate', type=float, default=Config.rate, help='model size')
    parser.add_argument('--seed', type=int, default=Config.seed, help='seed')
    parser.add_argument('--print_interval',
                        type=bool,
                        default=Config.print_interval,
                        help='print interval')
    parser.add_argument('--availabel_cudas',
                        type=str,
                        default=Config.availabel_cudas,
                        help='print interval')

    return parser.parse_args()


class cifar_train:
    def __init__(self, model_name, dataset_root, epochs, logger, lr, momentum, weight_decay, batch_size, num_workers,
                 milestones,
                 gamma, checkpoints, availabel_cudas, num_classes, feature_dim,
                 seed=0, rate=1):
        self.model_name = model_name
        self.root = dataset_root
        self.epochs = epochs
        self.logger = logger
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.milestones = milestones
        self.gamma = gamma
        self.rate = rate
        self.checkpoints = checkpoints
        self.train_dataset = get_dataset("CIFAR100", 'train', dir=self.root)
        self.val_dataset = get_dataset("CIFAR100", 'test', dir=self.root)
        self.availabel_cudas = availabel_cudas
        self.net = self.construct_model(rate)
        # self.net = self.load_model("/share/home/kcli/CL_research/iCaRL_ILtFA/log/FE_cls_cifar10_preTrain_log/"
        #                               "resnet34/rate_/"
        #                               "cifar10_resnet34_rate_1_FE_cls.pth")
        # self.net = self.load_model("/share/home/kcli/CL_research/iCaRL_ILtFA/checkpoints/cifar10_resnet34_rate_0.125_FEcls.pth")
        # self.net = self.load_FE_2fc_cls_model("/share/home/kcli/CL_research/iCaRL_ILtFA/checkpoints/cifar10_resnet34_rate_1_FE_2fc_cls_64.pth")

    def construct_model(self, rate):
        # model = torch.nn.DataParallel(
        #     FE_3fc_cls(resnetforcifar.__dict__[self.model_name](rate=rate, get_feature=True), int(512 * rate),
        #                self.feature_dim, self.num_classes))

        # model = torch.nn.DataParallel(
        #     FE_2fc_cls(resnetforcifar.__dict__[self.model_name](rate=rate, get_feature=True), int(512 * rate),
        #                self.feature_dim, self.num_classes))

        model = torch.nn.DataParallel(FE_cls(resnetforcifar.__dict__[self.model_name](rate=rate, get_feature=True),
                                             int(512 * rate), self.num_classes))
        model = model.cuda()
        return model

    def load_model(self, model_path):
        if self.availabel_cudas:
            # os.environ['CUDA_VISIBLE_DEVICES'] = self.availabel_cudas
            # device_ids = [i for i in range(len(self.availabel_cudas.strip().split(',')))]
            # model = torch.nn.DataParallel(FE_cls(resnetforimagenet.__dict__[self.model_name](rate=rate, get_feature=True),
            #                                      self.hidden_size, self.num_classes),
            #                               device_ids=device_ids).cuda()
            model = FE_cls(resnetforcifar.__dict__[self.model_name](rate=self.rate, get_feature=True),
                           int(512 * self.rate),
                           self.num_classes)
            model.load_state_dict(
                torch.load(model_path))
            model = torch.nn.DataParallel(model).cuda()
            # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
            # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            model = FE_cls(resnetforcifar.__dict__[self.model_name](rate=self.rate, get_feature=True),
                           int(512 * self.rate),
                           self.num_classes)
            model.load_state_dict(
                torch.load(model_path))
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
                               self.feature_dim, self.num_classes)
            model.load_state_dict(
                torch.load(model_path))
            model = torch.nn.DataParallel(model).cuda()
            # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
            # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            model = FE_2fc_cls(resnetforcifar.__dict__[self.model_name](rate=self.rate, get_feature=True),
                               int(512 * self.rate),
                               self.feature_dim, self.num_classes)
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
                               self.feature_dim, self.num_classes)
            model.load_state_dict(
                torch.load(model_path))
            model = torch.nn.DataParallel(model).cuda()
            # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
            # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            model = FE_3fc_cls(resnetforcifar.__dict__[self.model_name](rate=self.rate, get_feature=True),
                               int(512 * self.rate),
                               self.feature_dim, self.num_classes)
            model.load_state_dict(
                torch.load(model_path))
        print(type(model))
        print(model)
        return model
        pass

    def get_dataset(self, dataset_name, dataset_type):
        return get_dataset(dataset_name, type=dataset_type, download=True, capacity=None, dir=self.root)

    def validate_main(self, args):
        print("seed:", self.seed)
        print("model:", self.net)
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            cudnn.deterministic = True

        gpus = torch.cuda.device_count()
        self.logger.info(f'use {gpus} gpus')
        self.logger.info(f"args: {args}")

        cudnn.benchmark = True
        cudnn.enabled = True
        start_time = time.time()
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=True)
        # for name, param in self.net.named_parameters():
        #     self.logger.info(f"{name},{param.requires_grad}")
        print("len(train_loader):", len(train_loader))
        acc1, acc5, throughput = self.validate(val_loader)
        rlt = f"val: top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
        self.logger.info(
            rlt
        )
        print(rlt)

    def train_main(self, args):
        print("seed:", self.seed)
        print("model:", self.net)
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            cudnn.deterministic = True

        gpus = torch.cuda.device_count()
        self.logger.info(f'use {gpus} gpus')
        self.logger.info(f"args: {args}")

        cudnn.benchmark = True
        cudnn.enabled = True
        start_time = time.time()
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=True)
        print("len(train_loader):", len(train_loader))
        for name, param in self.net.named_parameters():
            self.logger.info(f"{name},{param.requires_grad}")
        criterion = torch.nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(self.net.parameters(),
                                    self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.gamma)
        best_acc = 0.0
        start_epoch = 1
        if not os.path.exists(self.checkpoints):
            os.makedirs(self.checkpoints)

        self.logger.info('start training')
        for epoch in range(start_epoch, args.epochs + 1):
            acc1, acc5, losses = self.train(train_loader, criterion, optimizer,
                                            scheduler, epoch, args)
            self.logger.info(
                f"train: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%, losses: {losses:.2f}"
            )

            acc1, acc5, throughput = self.validate(val_loader)
            rlt = f"val: epoch {epoch:0>3d}, top1 acc: {acc1:.2f}%, top5 acc: {acc5:.2f}%, throughput: {throughput:.2f}sample/s"
            self.logger.info(
                rlt
            )
            print(rlt)
            if acc1 > best_acc:
                torch.save(self.net.module.state_dict(),
                           os.path.join(args.checkpoints, "best.pth"))
                best_acc = acc1
            # remember best prec@1 and save checkpoint
            # torch.save(
            #     {
            #         'epoch': epoch,
            #         'best_acc': best_acc,
            #         'acc1': acc1,
            #         'loss': losses,
            #         'lr': scheduler.get_lr()[0],
            #         'model_state_dict': self.net.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'scheduler_state_dict': scheduler.state_dict(),
            #     }, os.path.join(args.checkpoints, 'latest.pth'))

        self.logger.info(f"finish training, best acc: {best_acc:.2f}%")
        training_time = (time.time() - start_time) / 3600
        self.logger.info(
            f"finish training, total training time: {training_time:.2f} hours")
        # torch.save(self.net, "reset34_rate1_8_cifar10_pretrain.pth")
        # checkpoint_file = os.path.join(self.checkpoints,
        #                                "cifar10_resnet34_rate_{}_FE_3fc_cls_{}.pth".format(self.rate, self.feature_dim))
        checkpoint_file = os.path.join(self.checkpoints,
                                       "cifar100_resnet34_rate_{}_FE_cls.pth".format(self.rate))
        torch.save(self.net.module.state_dict(), checkpoint_file)

    def train(self, train_loader, criterion, optimizer, scheduler, epoch, args):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        # switch to train mode
        self.net.train()

        iters = len(train_loader.dataset) // args.batch_size
        iter_index = 1

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()

            _, outputs = self.net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))
            if iter_index % args.print_interval == 0:
                self.logger.info(
                    f"train: epoch {epoch:0>3d}, iter [{iter_index:0>4d}, {iters:0>4d}], lr: "
                    f"{scheduler.get_last_lr()[0]:.6f}, top1 acc: {acc1.item():.2f}%, "
                    f"top5 acc: {acc5.item():.2f}%, loss_total: {loss.item():.2f}"
                )

            iter_index += 1
        scheduler.step()
        return top1.avg, top5.avg, losses.avg

    def validate(self, val_loader):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.net.eval()

        with torch.no_grad():
            end = time.time()
            total = 0
            correct = 0
            for inputs, labels in val_loader:
                data_time.update(time.time() - end)
                inputs, labels = inputs.cuda(), labels.cuda()
                _, outputs = self.net(inputs)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                top1.update(acc1.item(), inputs.size(0))
                top5.update(acc5.item(), inputs.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_temp = (predicted == labels).sum()
                correct += correct_temp
        throughput = 1.0 / (batch_time.avg / inputs.size(0))
        prect = 100. * correct / total
        return top1.avg, top5.avg, throughput


def main():
    args = parse_args()
    logger = get_logger(args.log)
    cifar_model = cifar_train(model_name=args.model_name, dataset_root=args.dataset_root, epochs=args.epochs,
                              logger=logger, lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, batch_size=args.batch_size,
                              num_workers=args.num_workers, milestones=args.milestones, gamma=args.gamma,
                              checkpoints=args.checkpoints, availabel_cudas=args.availabel_cudas,
                              num_classes=args.num_classes,
                              feature_dim=args.feature_dim,
                              seed=args.seed, rate=args.rate)
    cifar_model.train_main(args)
    # cifar_model.validate_main(args)


if __name__ == "__main__":
    # model = resnetforcifar.__dict__["resnet18"](rate=1/4, get_feature=True)
    # print(model.get_feature)
    # # print(model)
    # model2 = resnetforcifar.__dict__["resnet18"](rate=1 / 4)
    # print(model2.get_feature)
    main()
