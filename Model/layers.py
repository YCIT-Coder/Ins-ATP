#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : layers.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2021/6/16 23:08
# @Desc  :


import torch
import torch.nn as nn


class Conv2D(nn.Module):
    """"""

    def __init__(self, in_channel, out_channel, kernel=3, stride=1, padding=1, bias=False, ac=False, dropout=0.0):
        """Constructor for Conv2D"""
        super(Conv2D, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=bias))
        layers.append(nn.BatchNorm2d(out_channel))
        if ac:
            layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    """"""

    def __init__(self, in_channel, out_channel, kernel=3, stride=1, padding=1, bias=False, dropout=0.0):
        """Constructor for ResBlock"""
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=bias)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.ac = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel, stride, padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.bn1(self.conv1(x))
        out2 = self.bn2(self.conv2(self.ac(out2)))
        return self.ac(out1 + out2)


class Attention(nn.Module):
    """"""

    def __init__(self, feature_dim, hidden, patches):
        """Constructor for Attention"""
        super(Attention, self).__init__()
        self.v = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.Tanh()
        )
        self.w = nn.Sequential(
            nn.Linear(hidden, 1),
        )
        self.patches = patches

    def forward(self, x):
        out = self.v(x)
        out = self.w(out)
        # out = torch.tanh(out)
        out = out.view(-1, self.patches)
        # print(out.size())
        out = torch.softmax(out, 1)
        # print(out.size())
        return out


if __name__ == '__main__':
    x = torch.randn(20, 128)
    net = Attention(128, 256, 10)
    att = net(x)
    print(att)
    print(torch.sum(att, 1))
