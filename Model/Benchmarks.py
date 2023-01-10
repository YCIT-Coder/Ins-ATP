#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Benchmarks.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2021/8/2 16:19
# @Desc  :

import torch
import torch.nn as nn
from torchvision.models import vgg16, resnet18, inception_v3, GoogLeNet, resnet34


class vgg_net(nn.Module):
    """"""

    def __init__(self, ):
        """Constructor for vgg_net"""
        super(vgg_net, self).__init__()
        self.feature_extractor = vgg16(False, progress=False).features
        self.refine = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.refine(x)
        return x.view(x.size(0), x.size(1))


class inception_net(nn.Module):
    """"""

    def __init__(self, ):
        """Constructor for inception_net"""
        super(inception_net, self).__init__()
        self.feature_extractor = inception_v3(False, progress=False, aux_logits=False)
        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class googlenet_net(nn.Module):
    """"""

    def __init__(self, ):
        """Constructor for inception_net"""
        super(googlenet_net, self).__init__()
        self.feature_extractor = GoogLeNet(aux_logits=False)
        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class resnet_net(nn.Module):
    """"""

    def __init__(self, ):
        """Constructor for vgg_net"""
        super(resnet_net, self).__init__()
        layers = list(resnet34(False, progress=False).children())[0:-1]
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), x.size(1))


class BenchMarkNet(nn.Module):
    """"""

    def __init__(self, backbone='vgg'):
        """Constructor for BenchMarkNet"""
        super(BenchMarkNet, self).__init__()
        self.feature_extractor = None
        if backbone == 'vgg':
            self.feature_extractor = vgg_net()
        elif backbone == 'resnet':
            self.feature_extractor = resnet_net()
        elif backbone == 'inception':
            self.feature_extractor = inception_net()
        elif backbone == 'googlenet':
            self.feature_extractor = googlenet_net()
        else:
            pass
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.regressor(x)
        x = x.squeeze()
        return x


class BinaryBenchMarkNet(nn.Module):
    """"""

    def __init__(self, backbone='vgg', category=1):
        """Constructor for BenchMarkNet"""
        super(BinaryBenchMarkNet, self).__init__()
        self.feature_extractor = None
        if backbone == 'vgg':
            self.feature_extractor = vgg_net()
        elif backbone == 'resnet':
            self.feature_extractor = resnet_net()
        elif backbone == 'inception':
            self.feature_extractor = inception_net()
        elif backbone == 'googlenet':
            self.feature_extractor = googlenet_net()
        else:
            pass
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, category),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.regressor(x)
        x = x.squeeze()
        return x


if __name__ == '__main__':
    import torchsummary

    # net = BenchMarkNet('resnet')
    net = BenchMarkNet('inception')
    # net = BenchMarkNet('googlenet')
    # net = BenchMarkNet('vgg')
    # net = BenchMarkNet('resnet')
    # x = torch.randn(2, 3, 512, 512)
    # y = net(x)
    # print(y.size())
    torchsummary.summary(net, (3, 512, 512))
