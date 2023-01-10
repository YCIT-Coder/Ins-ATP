#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : regression.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2021/8/2 17:10
# @Desc  :

import torch
import random
from torch.nn import MSELoss
import torch.nn as nn
import numpy as np


class WeightedMSELoss(nn.Module):
    """
    根据对比损失加权
    """

    def __init__(self, weight=False):
        """Constructor for WeightedMSELoss"""
        super(WeightedMSELoss, self).__init__()
        self.weight = weight
        if self.weight:
            self.mse_loss = MSELoss(reduction='none')
        else:
            self.mse_loss = MSELoss()

    def forward(self, pred, target):
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)
        if self.weight:
            target_dist = ((target - target.T) >= 0).float()
            pred_dist = ((pred - pred.T) >= 0).float()
            mask = 1 - (target_dist == pred_dist).float()  # 位置出错的样本
            mask = torch.triu(mask, diagonal=1)
            margin = torch.abs(pred - target)
            margin = ((margin - margin.T) > 0).float()
            margin = torch.triu(margin, diagonal=1)
            weight = torch.sum(margin * mask, 1)
            weight = weight + 1e-8
            weight = weight / torch.sum(weight)
            weight = weight.view(pred.size())
            loss = self.mse_loss(pred, target) * (1 + weight)
            return torch.mean(loss)
        else:
            return self.mse_loss(pred, target)


class CascadeLoss(nn.Module):
    """"""

    def __init__(self, weight_decay=0.9, bit=11, margin=0.3):
        """Constructor for CascadeLoss"""
        super(CascadeLoss, self).__init__()
        self.weight_decay = -np.log(1.0 / weight_decay - 1)  # weight_decay ~ 1 用sigmoid控制
        self.bce_loss = nn.BCELoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bit = bit
        self.margin = margin

    def forward(self, pred, target, epoch=0):
        # print(pred.size(),target.size())
        weight = torch.arange(0, self.bit).to(pred.device)
        weight_decay = 1 / (1 + np.exp(-(self.weight_decay + epoch / 30.)))
        weight = torch.pow(weight_decay, weight)
        # weight = torch.softmax(weight, 0)

        # weight = weight.view(1, weight.size(0))
        # weight = weight.expand(pred.size(0), weight.size(1))
        # print(pred.size(), target.size())
        # mask1 = 1 - ((pred[:, :-1] >= self.th) == (target[:, :-1] >= self.th)).float()
        # mask2 = 1 - ((pred[:, :-1] <= 1 - self.th) == (target[:, :-1] <= 1 - self.th)).float()
        mask = (torch.abs(pred[:, :-1] - target[:, :-1]) - self.margin > 0).float()
        loss_bce = self.bce_loss(pred[:, :-1], target[:, :-1]) * mask
        # print(torch.sum(mask))
        loss_bce = torch.mean(loss_bce, 0) * weight[:-1]
        loss_mse = self.mse_loss(pred[:, -1], target[:, -1])
        loss_mse = torch.mean(loss_mse) * weight[-1]
        loss_bce = torch.mean(loss_bce)
        loss = loss_bce + loss_mse
        # print(epoch, weight)
        # print(loss_bce, loss_mse)
        return loss, loss_bce, loss_mse


class BinaryLoss(nn.Module):
    """"""

    def __init__(self, weight_decay=0.9, bit=11, margin=0.3):
        """Constructor for CascadeLoss"""
        super(BinaryLoss, self).__init__()
        self.weight_decay = -np.log(1.0 / weight_decay - 1)  # weight_decay ~ 1 用sigmoid控制
        self.bce_loss = nn.BCELoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bit = bit
        self.margin = margin

    def forward(self, pred, target, epoch=0):
        # print(pred.size(),target.size())
        weight = torch.arange(0, self.bit).to(pred.device)
        weight_decay = 1 / (1 + np.exp(-(self.weight_decay + epoch / 30.)))
        weight = torch.pow(weight_decay, weight)
        # weight = torch.softmax(weight, 0)

        # weight = weight.view(1, weight.size(0))
        # weight = weight.expand(pred.size(0), weight.size(1))
        # print(pred.size(), target.size())
        # mask1 = 1 - ((pred[:, :-1] >= self.th) == (target[:, :-1] >= self.th)).float()
        # mask2 = 1 - ((pred[:, :-1] <= 1 - self.th) == (target[:, :-1] <= 1 - self.th)).float()
        mask = (torch.abs(pred[:, :-1] - target[:, :-1]) - self.margin > 0).float()
        loss_bce = self.bce_loss(pred[:, :-1], target[:, :-1]) * mask
        # print(torch.sum(mask))
        loss_bce = torch.mean(loss_bce, 0) * weight[:-1]
        loss_mse = self.mse_loss(pred[:, -1], target[:, -1])
        loss_mse = torch.mean(loss_mse) * weight[-1]
        loss_bce = torch.mean(loss_bce)
        loss = loss_bce + loss_mse * 0
        # print(epoch, weight)
        # print(loss_bce, loss_mse)
        return loss, loss_bce, loss_mse


def set_seed(seed):
    torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)


set_seed(1000)

if __name__ == '__main__':
    from Data.Organoid import ATPMaper

    set_seed(1000)
    mod = 200000
    bin = 20000
    bit = int(np.ceil(np.log2(mod // bin) + 1e-8) + 1)
    loss_f = CascadeLoss(weight_decay=0.9, bit=bit)

    y = torch.rand(3, bit)
    y[:, 0:bit - 1] = torch.randint(0, 2, (3, bit - 1))
    # x[:, 0:bit - 1] = y[:, 0:bit - 1]
    for i in range(500):
        x = torch.rand(3, bit)
        # print(x[:, - 1])
        # print(y[:, - 1])
        # print(torch.abs(x[:, 0:bit - 1] - y[:, 0:bit - 1]))
        loss_f(x, y, i)

    # loss_f = WeightedMSELoss(weight=True)
    # x = torch.rand(5)
    # y = torch.rand(5)
    # print(x)
    # print(y)
    # print(ATPMaper(y))
    # print(loss_f(x, y))
