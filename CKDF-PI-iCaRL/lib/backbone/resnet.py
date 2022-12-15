"""
https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
"""
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'resnet18',
    'resnet32',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'podnet_resnet18',
]


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, last_relu=True):
        super(BasicBlock, self).__init__()
        self.last_relu = last_relu
        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,
                      out_channels * BasicBlock.expansion,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BasicBlock.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion))

    def forward(self, x):
        if self.last_relu:
            return F.relu(self.residual_branch(x) + self.shortcut(x))
        else:
            return self.residual_branch(x) + self.shortcut(x)


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels,
                      out_channels,
                      stride=stride,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels,
                      out_channels * BottleNeck.expansion,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BottleNeck.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=False)(self.residual_branch(x) +
                                     self.shortcut(x))


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


class ResNet(nn.Module):
    def __init__(self, block, layers, rate=1, inter_layer=False):
        super(ResNet, self).__init__()
        self.inter_layer = inter_layer
        self.in_channels = int(64 * rate)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(64 * rate), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(64 * rate)), nn.ReLU(inplace=False))

        self.stage2 = self._make_layer(block, int(64 * rate), layers[0], 1)
        self.stage3 = self._make_layer(block, int(128 * rate), layers[1], 2)
        self.stage4 = self._make_layer(block, int(256 * rate), layers[2], 2)
        self.stage5 = self._make_layer(block, int(512 * rate), layers[3], 2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict["state_dict"] if "state_dict" in pretrain_dict else pretrain_dict
        from collections import OrderedDict

        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if "last_linear" not in k and "classifier" not in k and "linear" not in k and "fd" not in k:
                k = k.replace("backbone.", "")
                k = k.replace("fr", "layer3.4")
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x


class PODNet_ResNet(nn.Module):
    def __init__(self, block, layers, rate=1, last_relu=True):
        super(PODNet_ResNet, self).__init__()
        self.last_relu = last_relu
        self.in_channels = int(64 * rate)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(64 * rate), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(64 * rate)), nn.ReLU())

        self.stage2 = self._make_layer(block, int(64 * rate), layers[0], 1)
        self.stage3 = self._make_layer(block, int(128 * rate), layers[1], 2)
        self.stage4 = self._make_layer(block, int(256 * rate), layers[2], 2)
        self.stage5 = self._make_layer(block, int(512 * rate), layers[3], 2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        block_num = len(strides)
        for block_index in range(block_num):
            if block_index == block_num - 1:
                layers.append(block(self.in_channels, out_channels, strides[block_index], last_relu=False))
            else:
                layers.append(block(self.in_channels, out_channels, strides[block_index], last_relu=self.last_relu))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict["state_dict"] if "state_dict" in pretrain_dict else pretrain_dict
        from collections import OrderedDict

        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if "last_linear" not in k and "classifier" not in k and "linear" not in k and "fd" not in k:
                k = k.replace("backbone.", "")
                k = k.replace("fr", "layer3.4")
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def forward(self, x, **kwargs):
        out = self.conv1(x)

        x_1 = self.stage2(out)
        x_2 = self.stage3(self.end_relu(x_1))
        x_3 = self.stage4(self.end_relu(x_2))
        x_4 = self.stage5(self.end_relu(x_3))
        return x_1, x_2, x_3, x_4

    def end_relu(self, x):
        if self.last_relu:
            return F.relu(x)
        return x


def _resnet(block, layers, rate, **kwargs):
    model = ResNet(block, layers, rate, **kwargs)
    return model


def _resnet_for_podnet(block, layers, rate, last_relu, **kwargs):
    model = PODNet_ResNet(block, layers, rate, last_relu)
    return model


def resnet18(rate=1, **kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], rate,
                   **kwargs)


def podnet_resnet18(rate=1, last_relu=True, **kwargs):
    return _resnet_for_podnet(BasicBlock, [2, 2, 2, 2], rate, last_relu=last_relu,
                              **kwargs)


def resnet32(rate=1, **kwargs):
    return _resnet(BasicBlock, [3, 4, 5, 3], rate, **kwargs)


def resnet34(rate=1, **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], rate, **kwargs)


def resnet50(rate=1, **kwargs):
    return _resnet(BottleNeck, [3, 4, 6, 3], rate, **kwargs)


def resnet101(**kwargs):
    return _resnet(BottleNeck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return _resnet(BottleNeck, [3, 8, 36, 3], **kwargs)


def conv1x1_bn(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channel), nn.ReLU(inplace=False))


class ChannelDistillResNet50152(nn.Module):
    def __init__(self, num_classes=100):
        super(ChannelDistillResNet50152, self).__init__()
        self.student = resnet50(num_classes=num_classes, inter_layer=True)
        self.teacher = resnet152(pretrained=True,
                                 num_classes=num_classes,
                                 inter_layer=True)

        self.s_t_pair = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
        self.connector = nn.ModuleList(
            [conv1x1_bn(s, t) for s, t in self.s_t_pair])
        # freeze teacher
        for m in self.teacher.parameters():
            m.requires_grad = False

    def forward(self, x):
        ss = self.student(x)
        ts = self.teacher(x)
        for i in range(len(self.s_t_pair)):
            ss[i] = self.connector[i](ss[i])

        return ss, ts


if __name__ == "__main__":
    model = resnet32(rate=1, get_feature=True)
    print(model)
    # b = [[1,2,3], [2,3,4]]
    # a = {"dsa": b}
    # with open("list.json", 'w') as fw:
    #     json.dump(a, fw, indent=4)
