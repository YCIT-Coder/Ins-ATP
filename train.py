#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train_binary_MIL_benchmark.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2021/12/6 15:33
# @Desc  :

import os
import torch
import argparse
from Model.MIL_Benchmarks import MILCovid19, AD2D_MIL, DACMIL
from Losses.regression import MSELoss, CascadeLoss
from Data.Organoid import OrganoidBinary, ATPMaper
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import random

parser = argparse.ArgumentParser('params')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--batch', type=int, default=10)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--rs', type=int, default=0)
parser.add_argument('--bb', type=str, default='2')
parser.add_argument('--patch', type=int, default=16)
parser.add_argument('--d', type=int, default=1)
parser.add_argument('--mod', type=int, default=200000)
parser.add_argument('--bin', type=int, default=20000)
args = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)


set_seed(1000)

# define device
gpu_flag = False
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if torch.cuda.is_available():
    gpu_flag = True

# define data
bit = int(np.ceil(np.log2(args.mod // args.bin) + 1e-8) + 1)
train_data = OrganoidBinary('./H5_file/ATP_train_crop_{}.h5'.format(args.d), args.size, resample=(args.rs > 0),
                            mod=args.mod, bin_wid=args.bin, bit=bit)
# train_data = OrganoidData('./H5_file/ATP_data_crop_1.h5'.format(args.d), args.size)
train_loader = DataLoader(train_data, args.batch, shuffle=True, num_workers=1)
val_data = OrganoidBinary('./H5_file/ATP_val_crop_{}.h5'.format(args.d), args.size, mod=args.mod, bin_wid=args.bin,
                          bit=bit)
# val_data = OrganoidData('./H5_file/ATP_data_crop_2.h5'.format(args.d), args.size)
val_loader = DataLoader(val_data, args.batch, shuffle=True, num_workers=1)

# define Model

patch_size = args.size // np.int32(np.sqrt(args.patch))

if args.bb == '0':
    net = MILCovid19(in_channel=3, out_channel=32, category=bit, patches=args.patch, num_layers=5, num_stack=6,
                     image_size=args.size // int(np.sqrt(args.patch)))
elif args.bb == "1":
    net = AD2D_MIL(in_channel=3, hidden=512, category=bit, num_layer=5, image_size=args.size, patches=args.patch)
elif args.bb == "2":
    net = DACMIL(category=bit, patches=args.patch, image_size=args.size // int(np.sqrt(args.patch)))

# define loss
loss_f = CascadeLoss(weight_decay=0.5, bit=train_data.bit, margin=0.3)

# define trainer
trainer = Adam(net.parameters(), args.lr, betas=(0.5, 0.999))

# def lr schedule
lr_schedule = StepLR(trainer, 10, 0.9)

if __name__ == '__main__':

    log_path = './binary_mil_log/gpu_{}/backbone_{}/epoch_{}/image_size_{}/resample_{}/batch_size_{}/lr_{}_dataset_{}/mod_{}/bin_{}/patch_{}/'.format(
        args.gpu, args.bb, args.epoch, args.size, args.rs, args.batch, args.lr, args.d, args.mod, args.bin, args.patch)
    summary_writer = SummaryWriter(log_path)
    for epoch in range(args.epoch):
        losses = []
        losses_bce = []
        losses_mse = []
        net.train()
        for i, batch in enumerate(train_loader):
            x = batch['image'].float()
            y = batch['binary'].float()

            if gpu_flag:
                x = x.cuda()
                y = y.cuda()
                net = net.cuda()

            pre = net(x)
            loss, loss_bce, loss_mse = loss_f(pre, y, epoch)
            trainer.zero_grad()
            loss.backward()
            trainer.step()
            print('epoch:{},Iter:{},Loss:{}'.format(epoch, i, float(loss)))
            print(ATPMaper(pre, train_data.bin_wid))
            print(ATPMaper(y, train_data.bin_wid))
            losses.append(float(loss))
            losses_bce.append(float(loss_bce))
            losses_mse.append(float(loss_mse))
        summary_writer.add_scalars('loss',
                                   {'all': np.mean(losses), 'bce': np.mean(losses_bce), 'mse': np.mean(losses_mse)},
                                   epoch)
        del losses
        del losses_bce
        del losses_mse
        torch.cuda.empty_cache()

        net.eval()
        Y = []
        Pre = []
        Pre_Binary = []
        Y_Binary = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                x = batch['image'].float()
                y = batch['binary'].float()

                if gpu_flag:
                    x = x.cuda()
                    y = y.cuda()
                    net = net.cuda()

                pre = net(x)
                Y.append(ATPMaper(y, train_data.bin_wid).cpu().numpy())
                Pre.append(ATPMaper(pre, train_data.bin_wid).detach().cpu().numpy())
                Pre_Binary.append(pre[:, :-1].detach().cpu().numpy())
                Y_Binary.append(y[:, :-1].detach().cpu().numpy())

        Y = np.concatenate(Y, 0)
        Pre = np.concatenate(Pre, 0)
        Pre_Binary = np.concatenate(Pre_Binary, 0)
        Y_Binary = np.concatenate(Y_Binary, 0)
        ordered_index = sorted(range(len(Y)), key=lambda k: Y[k])

        fig = plt.figure()
        plt.plot(range(len(Y)), Y[ordered_index], color='C1')
        plt.scatter(range(len(Pre)), Pre[ordered_index], color='C2', marker='2')
        summary_writer.add_figure('comp', fig, epoch)
        plt.close()

        fig = plt.figure()
        err = np.abs(Y - Pre)
        plt.boxplot(err)
        summary_writer.add_figure('err', fig, epoch)
        plt.close()

        summary_writer.add_scalars('eval', {'L1': np.mean(np.abs(Pre - Y))}, epoch)
        summary_writer.add_scalar('lr', trainer.param_groups[0]["lr"], epoch)
        summary_writer.add_histogram('pred', Pre_Binary, epoch)
        summary_writer.add_histogram('target', Y_Binary, epoch)

        del Y
        del Pre
        torch.cuda.empty_cache()
        lr_schedule.step()
        torch.save(net.state_dict(), log_path + 'epoch_{}.pkl'.format(epoch))
