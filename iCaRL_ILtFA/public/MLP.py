import torch as pt
import torchvision as ptv
import numpy as np

__all__ = [
    'MLP_3_layers',
    'MLP_2_layers',
    'MLP_4_layers',
    'discriminate_2_layers',
    'discriminate_3_layers',
]

from torch import nn


class MLP_3_layers(pt.nn.Module):
    def __init__(self, input_dim, out_dim, rate=16):
        super(MLP_3_layers, self).__init__()
        self.dim = input_dim
        self.fc1 = pt.nn.Linear(input_dim, rate * input_dim)
        self.fc2 = pt.nn.Linear(rate * input_dim, rate * input_dim)
        self.fc3 = pt.nn.Linear(rate * input_dim, out_dim)
        # self.bn = pt.nn.LayerNorm(input.size()[1:])

    def forward(self, din):
        # self.bn = pt.nn.LayerNorm(din.size()[1:])
        x = self.fc1(din)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        dout = nn.functional.relu(x)
        return self.fc3(dout)
        # return pt.nn.functional.softmax(self.fc3(dout))


class MLP_2_layers(pt.nn.Module):
    def __init__(self, input_dim, out_dim, rate=16):
        super(MLP_2_layers, self).__init__()
        self.dim = input_dim
        self.fc1 = pt.nn.Linear(input_dim, rate * input_dim)
        self.fc2 = pt.nn.Linear(rate * input_dim, out_dim)

    def forward(self, din):
        x = self.fc1(din)
        dout = nn.functional.relu(x)
        return self.fc2(dout)
        # return pt.nn.functional.softmax(self.fc2(dout))


class MLP_4_layers(pt.nn.Module):
    def __init__(self, input_dim, out_dim, rate=16):
        super(MLP_4_layers, self).__init__()
        self.dim = input_dim
        rate = 16
        self.fc1 = pt.nn.Linear(input_dim, rate * input_dim)
        self.fc2 = pt.nn.Linear(rate * input_dim, rate * input_dim)
        self.fc3 = pt.nn.Linear(rate * input_dim, rate * input_dim)
        self.fc4 = pt.nn.Linear(rate * input_dim, out_dim)
        # self.bn = pt.nn.LayerNorm(input.size()[1:])

    def forward(self, din):
        # self.bn = pt.nn.LayerNorm(din.size()[1:])
        x = self.fc1(din)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        dout = nn.functional.relu(x)
        return self.fc4(dout)


class discriminate_2_layers(pt.nn.Module):
    def __init__(self, input_dim, out_dim, rate=16):
        super(discriminate_2_layers, self).__init__()
        self.dim = input_dim
        self.fc1 = pt.nn.Linear(input_dim, rate * input_dim)
        self.fc2 = pt.nn.Linear(rate * input_dim, out_dim)
        self.sigmoid = pt.nn.Sigmoid()  # 也是一个激活函数，二分类问题中，

    def forward(self, din):
        x = self.fc1(din)
        dout = nn.functional.relu(x)
        return self.sigmoid(self.fc2(dout))
        # return pt.nn.functional.softmax(self.fc2(dout))


class discriminate_3_layers(pt.nn.Module):
    def __init__(self, input_dim, out_dim, rate=16):
        super(discriminate_3_layers, self).__init__()
        self.dim = input_dim
        self.fc1 = pt.nn.Linear(input_dim, rate * input_dim)
        self.fc2 = pt.nn.Linear(rate * input_dim, rate * input_dim)
        self.fc3 = pt.nn.Linear(rate * input_dim, out_dim)
        self.sigmoid = pt.nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
        # sigmoid可以班实数映射到【0,1】，作为概率值，
        # 多分类用softmax函数
        # self.bn = pt.nn.LayerNorm(input.size()[1:])

    def forward(self, din):
        # self.bn = pt.nn.LayerNorm(din.size()[1:])
        x = self.fc1(din)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        dout = nn.functional.relu(x)
        return self.sigmoid(self.fc3(dout))
        # return pt.nn.functional.softmax(self.fc3(dout))
