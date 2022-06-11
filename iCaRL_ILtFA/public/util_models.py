"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
import torch
import torch.nn as nn


# -------------------------------------------------------------------------------------------------#

# --------------------#
# ----- ResNet -----#
# --------------------#

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, rate=1):
        super().__init__()

        self.in_channels = int(64 * rate)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(64 * rate), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(64 * rate)),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, int(64 * rate), num_block[0], 1)
        self.conv3_x = self._make_layer(block, int(128 * rate), num_block[1], 2)
        self.conv4_x = self._make_layer(block, int(256 * rate), num_block[2], 2)
        self.conv5_x = self._make_layer(block, int(512 * rate), num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * rate) * block.expansion, num_classes)

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

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        feature = output.view(output.size(0), -1)
        output = self.fc(feature)

        return output


def resnet18(numclass, rate):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], numclass, rate)


def resnet34(numclass, rate):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], numclass, rate)


def resnet50(numclass, rate):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], numclass, rate)


def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])


# 中间层特征提取
class FeatureExtractor(torch.nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule  # 输出features && targets
        self.extracted_layers = extracted_layers

    # 自己修改forward函数
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name == "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs[-2].view(outputs[-2].size(0), -1), outputs[-1]


# 中间层特征提取
class FE_cls(torch.nn.Module):
    def __init__(self, FE, input_dim, class_num):
        super(FE_cls, self).__init__()
        self.FE = FE  # 只输出features
        self.cls = nn.Linear(input_dim, class_num)

    # 自己修改forward函数
    def forward(self, x):
        features = self.FE(x)
        targets = self.cls(features)
        return features, targets

    def get_cls_results(self, features):
        mode = self.training
        self.eval()
        with torch.no_grad():
            target = self.cls(features)
        self.train(mode=mode)
        return target

    def cls_forward(self, features):
        target = self.cls(features)
        return target


class FE_2fc_cls(torch.nn.Module):
    def __init__(self, FE, input_dim, feature_dim, class_num):
        super(FE_2fc_cls, self).__init__()
        self.FE = FE  # 只输出features
        self.fc = nn.Linear(input_dim, feature_dim)
        self.cls = nn.Linear(feature_dim, class_num)

    # 自己修改forward函数
    def forward(self, x):
        features = self.FE(x)
        features = self.fc(features)
        targets = self.cls(features)
        return features, targets

    def get_cls_results(self, features):
        mode = self.training
        self.eval()
        with torch.no_grad():
            target = self.cls(features)
        self.train(mode=mode)
        return target


class FE_3fc_cls(torch.nn.Module):
    def __init__(self, FE, input_dim, feature_dim, class_num):
        super(FE_3fc_cls, self).__init__()
        self.FE = FE  # 只输出features
        self.fc1 = nn.Linear(input_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, class_num)
        self.cls = nn.Linear(class_num, class_num)

    # 自己修改forward函数
    def forward(self, x):
        features = self.FE(x)
        features = self.fc1(features)
        x = self.fc2(features)
        x = nn.functional.relu(x)
        targets = self.cls(x)
        return features, targets

    def train_cls_with_features(self, features):
        x = self.fc2(features)
        x = nn.functional.relu(x)
        targets = self.cls(x)
        return targets

    def get_cls_results(self, features):
        mode = self.training
        self.eval()
        with torch.no_grad():
            x = self.fc2(features)
            x = nn.functional.relu(x)
            target = self.cls(x)
        self.train(mode=mode)
        return target


# 中间层特征提取
class MLP_for_FM(torch.nn.Module):
    def __init__(self, MLP):
        super(MLP_for_FM, self).__init__()
        self.FM = MLP

    # 自己修改forward函数
    def forward(self, x):
        features = self.FM(x)
        return features

    def get_mapping_features(self, prefeatures):
        mode = self.training
        self.eval()
        with torch.no_grad():
            features = self(prefeatures)
        self.train(mode=mode)
        return features


# 中间层特征提取
class MLP_for_FM_cls(torch.nn.Module):
    def __init__(self, MLP, feature_size, num_classes):
        super(MLP_for_FM_cls, self).__init__()
        self.FM = MLP
        self.fc = nn.Linear(feature_size, num_classes)

    # 自己修改forward函数
    def forward(self, x):
        features = self.FM(x)
        target_hat = self.fc(features)
        return features, target_hat

    def get_mapping_features(self, prefeatures):
        mode = self.training
        self.eval()
        with torch.no_grad():
            features, _ = self(prefeatures)
        self.train(mode=mode)
        return features


# 中间层特征提取
class MLP_cls_domain_dis(torch.nn.Module):
    def __init__(self, MLP, input_dim, class_num):
        super(MLP_cls_domain_dis, self).__init__()
        self.FE = MLP
        self.cls = nn.Linear(input_dim, class_num)
        self.domain_dis = nn.Linear(input_dim, 2)

    # 自己修改forward函数
    def forward(self, x):
        features = self.FE(x)
        targets = self.cls(features)
        domain_id = self.domain_dis(features)
        return features, targets, domain_id

    def get_mapping_features(self, prefeatures):
        self.eval()
        with torch.no_grad():
            return self(prefeatures)[0]


'''自定义loss'''


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


'''自定义层 bias_layer'''


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.params = nn.Parameter(torch.Tensor([1, 0]))

    def forward(self, x, current_classes_num):
        x[:, -current_classes_num:] *= self.params[0]
        x[:, -current_classes_num:] += self.params[1]
        return x


if __name__ == "__main__":
    # model = BiasLayer()
    # print(model)
    # x = torch.Tensor([[1,1,1], [2,2,2]])
    # current_classes_num = 2
    # result = model(x, current_classes_num)
    # print(model.parameters().data())
    # print(result)
    for i in range(1, 5):
        print(i)
