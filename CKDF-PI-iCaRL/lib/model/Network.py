import torch

import torch.nn as nn
import torch.nn.functional as F

from lib.backbone import resnet, resnet_cifar
from lib.modules import GAP


class fc_relu(nn.Module):
    def __init__(self, input_dim, out_dim, bias=False):
        super(fc_relu, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.fc = torch.nn.Linear(self.input_dim, self.out_dim, bias=bias)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input):
        assert input.size(1) == self.input_dim
        features = self.fc(input)
        nor_features = self.bn(features)
        output = self.relu(nor_features)
        return output

    def load_model(self, model_path, logger=None):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        if logger:
            logger.info("BarlowTwins Model has been loaded...")


class MLP_classifier(nn.Module):
    def __init__(self, input_feature_dim, layer_nums, output_feature_dim, hidden_layer_rate=1,
                 last_hidden_layer_use_relu=False, bias=False, all_classes=100):
        super(MLP_classifier, self).__init__()
        self.layer_nums = layer_nums
        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim
        self.last_hidden_layer_use_relu = last_hidden_layer_use_relu
        self.hidden_layer_rate = hidden_layer_rate
        self.hidden_fc_layers = self._make_fc(bias)
        self.cls_layer = torch.nn.Linear(self.output_feature_dim, all_classes, bias=True)

    def _make_fc(self, bias=False):
        hidden_fc_layers = []
        input_dim = self.input_feature_dim
        if self.layer_nums == 1:
            if self.last_hidden_layer_use_relu:
                hidden_fc_layers.append(
                    fc_relu(self.input_feature_dim, self.output_feature_dim, bias=bias)
                )
            else:
                hidden_fc_layers = torch.nn.Linear(self.input_feature_dim, self.output_feature_dim, bias=bias)

        else:
            for layer in range(self.layer_nums):
                if layer < self.layer_nums - 1:
                    hidden_fc_layers.append(
                        fc_relu(input_dim, int(self.hidden_layer_rate * self.input_feature_dim), bias=bias)
                    )
                    input_dim = int(self.hidden_layer_rate * self.input_feature_dim)
                else:
                    if self.last_hidden_layer_use_relu:
                        hidden_fc_layers.append(
                            fc_relu(input_dim, self.output_feature_dim, bias=bias)
                        )
                    else:
                        hidden_fc_layers.append(
                            torch.nn.Linear(input_dim, self.output_feature_dim, bias=bias)
                        )

                    input_dim = self.input_feature_dim
        return nn.Sequential(*hidden_fc_layers)

    def forward(self, din, **kwargs):
        assert din.size(1) == self.input_feature_dim
        calibrated_features = self.hidden_fc_layers(din)
        if "feature_flag" in kwargs:
            return calibrated_features
        else:
            return self.cls_layer(calibrated_features)


class Projector_head(nn.Module):
    def __init__(self, input_feature_dim, layer_nums, output_feature_dim,
                 hidden_layer_rate=1, last_hidden_layer_use_relu=False, bias=False):
        super(Projector_head, self).__init__()
        self.layer_nums = layer_nums
        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim
        self.last_hidden_layer_use_relu = last_hidden_layer_use_relu
        self.hidden_layer_rate = hidden_layer_rate
        self.hidden_fc_layers = self._make_fc(bias)

    def _make_fc(self, bias=False):
        hidden_fc_layers = []
        input_dim = self.input_feature_dim
        if self.layer_nums == 1:
            if self.last_hidden_layer_use_relu:
                hidden_fc_layers.append(
                    fc_relu(self.input_feature_dim, self.output_feature_dim, bias=bias)
                )
            else:
                hidden_fc_layers = torch.nn.Linear(self.input_feature_dim, self.output_feature_dim, bias=bias)

        else:
            for layer in range(self.layer_nums):
                if layer < self.layer_nums - 1:
                    hidden_fc_layers.append(
                        fc_relu(input_dim, int(self.hidden_layer_rate * self.input_feature_dim), bias=bias)
                    )
                    input_dim = int(self.hidden_layer_rate * self.input_feature_dim)
                else:
                    if self.last_hidden_layer_use_relu:
                        hidden_fc_layers.append(
                            fc_relu(input_dim, self.output_feature_dim, bias=bias)
                        )
                    else:
                        hidden_fc_layers.append(
                            torch.nn.Linear(input_dim, self.output_feature_dim, bias=bias)
                        )

                    input_dim = self.input_feature_dim
        return nn.Sequential(*hidden_fc_layers)

    def forward(self, din):
        assert din.size(1) == self.input_feature_dim
        calibrated_features = self.hidden_fc_layers(din)
        return calibrated_features


class resnet_model(nn.Module):
    def __init__(self, cfg, cnn_type=None, rate=1., output_feature_dim=None):
        super().__init__()
        self.cfg = cfg
        if cnn_type is not None:
            assert output_feature_dim is not None
            if "18" in cnn_type or "34" in cnn_type:
                self.extractor = nn.Sequential(*[resnet.__dict__[cnn_type](
                    rate=rate), GAP()])
            else:
                self.extractor = nn.Sequential(*[resnet_cifar.__dict__[cnn_type](
                    rate=rate), GAP()])
            self.linear_classifier = torch.nn.Linear(output_feature_dim,
                                                     self.cfg.DATASET.all_classes, bias=True)
        else:
            if "18" in self.cfg.extractor.TYPE or "34" in self.cfg.extractor.TYPE:
                self.extractor = nn.Sequential(*[resnet.__dict__[self.cfg.extractor.TYPE](
                    rate=self.cfg.extractor.rate), GAP()])
            else:
                self.extractor = nn.Sequential(*[resnet_cifar.__dict__[self.cfg.extractor.TYPE](
                    rate=self.cfg.extractor.rate), GAP()])
            self.linear_classifier = torch.nn.Linear(self.cfg.extractor.output_feature_dim,
                                                     self.cfg.DATASET.all_classes, bias=True)

    def forward(self, x, **kwargs):
        if "no_grad" in kwargs or "is_nograd" in kwargs:
            if "train_classifier" in kwargs:
                features = self.forward_func(x)
                return self.linear_classifier(features), features
            elif "get_classifier" in kwargs:
                with torch.no_grad():
                    features = self.forward_func(x)
                    return self.linear_classifier(features)
            elif "feature_flag" in kwargs:
                return self.forward_func(x)
            elif "herding_feature" in kwargs:
                return self.forward_func(x)
            elif "get_out_use_features" in kwargs:
                with torch.no_grad():
                    return self.linear_classifier(x)
            else:
                with torch.no_grad():
                    features = self.forward_func(x)
                    return self.linear_classifier(features), features
        else:
            if "train_classifier" in kwargs:
                features = self.forward_func(x)
                return self.linear_classifier(features)
            elif "train_extractor" in kwargs:
                features = self.extractor(x)
                features = features.view(features.shape[0], -1)
                return features
            elif "train_cls_use_features" in kwargs:
                return self.linear_classifier(x)
            else:
                features = self.extractor(x)
                features = features.view(features.shape[0], -1)
                return self.linear_classifier(features), features

    def forward_func(self, x):
        mode = self.extractor.training
        self.extractor.eval()
        with torch.no_grad():
            features = self.extractor(x)
        self.extractor.train(mode)
        features = features.view(features.shape[0], -1)
        return features

    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)


class FCN_model(nn.Module):
    def __init__(self, cfg):
        super(FCN_model, self).__init__()
        self.FCN = None
        self.FCN = Projector_head(input_feature_dim=cfg.FCTM.FCN.in_feature_dim, layer_nums=cfg.FCTM.FCN.layer_nums,
                                  output_feature_dim=cfg.FCTM.FCN.out_feature_dim,
                                  hidden_layer_rate=cfg.FCTM.FCN.hidden_layer_rate,
                                  last_hidden_layer_use_relu=cfg.FCTM.FCN.last_hidden_layer_use_relu)
        self.global_fc = nn.Linear(cfg.FCTM.FCN.in_feature_dim,
                                   cfg.DATASET.all_classes, bias=True)

    def forward(self, pre_model_feature, **kwargs):
        if "no_grad" in kwargs or "is_nograd" in kwargs:
            features = pre_model_feature
            with torch.no_grad():
                calibrated_features = self.FCN(features)
            if "feature_flag" in kwargs:
                return calibrated_features
            else:
                with torch.no_grad():
                    outputs = self.global_fc(calibrated_features)
                return outputs
        else:
            features = pre_model_feature
            calibrated_features = self.FCN(features)
            outputs = self.global_fc(calibrated_features)
            return {
                "all_logits": outputs,
            }
