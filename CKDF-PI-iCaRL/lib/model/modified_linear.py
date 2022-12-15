import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module


class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma

    def forward(self, input=None, **kwargs):
        if "is_nograd" in kwargs:
            with torch.no_grad():
                if input is not None:
                    if self.sigma is None:
                        out = F.linear(F.normalize(input, p=2, dim=1),
                                       self.weight)
                    else:
                        out = F.linear(F.normalize(input, p=2, dim=1),
                                       F.normalize(self.weight, p=2, dim=1))
                        out = self.sigma * out
                    return out
                else:
                    if self.sigma is None:
                        out = F.linear(self.weight,
                                       self.weight)
                    else:
                        out = F.linear(F.normalize(self.weight, p=2, dim=1),
                                       F.normalize(self.weight, p=2, dim=1))
                        out = self.sigma * out
                    return out
        else:
            if input is not None:
                if self.sigma is None:
                    out = F.linear(F.normalize(input, p=2, dim=1),
                                   self.weight)
                else:
                    out = F.linear(F.normalize(input, p=2, dim=1),
                                   F.normalize(self.weight, p=2, dim=1))
                    out = self.sigma * out
                return out
            else:
                if self.sigma is None:
                    out = F.linear(self.weight,
                                   self.weight)
                else:
                    out = F.linear(F.normalize(self.weight, p=2, dim=1),
                                   F.normalize(self.weight, p=2, dim=1))
                    out = self.sigma * out
                return out


class SplitCosineLinear(Module):
    # consists of two fc layers and concatenate their outputs
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1)  # concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out
        return out
