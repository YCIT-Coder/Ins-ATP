#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : predict.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2021/10/11 14:42
# @Desc  :

from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, \
    Normalize
from Data.crop_image import fine_mask
import torch
import numpy as np
from Data.Organoid import OrganoidData, ATPMaper
from torch.utils.data import DataLoader
from Model.Benchmarks import BenchMarkNet
from Model.MIL_Benchmarks import MILCovid19, AD2D_MIL

size = 512
mod = 200000
patch = 16

patch_size = size // np.int32(np.sqrt(patch))
model_name = 'our1'

dict_model = {
    'resnet': {
        'name': 'resnet',
        'param': './params/resnet.pkl',
    },
    'vgg': {
        'name': 'vgg',
        'param': './params/vgg.pkl',
    },
    'inception': {
        'name': 'inception',
        'param': './params/inception.pkl',
    },
    'googlenet': {
        'name': 'inception',
        'param': './params/google_inception.pkl',
    },
    'our0': {
        'param': './params/our00.pkl',
        'batch': 5,
        'size': 512,
        'mod': 200000,
        'patch': 16,
        'category': 1
    },
    'our1': {
        'param': './params/our1.pkl',
        'batch': 5,
        'size': 512,
        'mod': 200000,
        'patch': 256,
        'category': 1
    },
    'binary': {
        'param': './params/epoch_228.pkl',
        'batch': 5,
        'size': 512,
        'mod': 200000,
        'patch': 256,
        'bin': 20000,
        'category': 1
    }
}

try:
    bit = int(np.ceil(np.log2(dict_model[model_name]['mod'] // dict_model[model_name]['bin']) + 1e-8) + 1)
except Exception as e:
    bit = 1

if model_name in list(dict_model.keys())[:4]:
    net = BenchMarkNet(dict_model[model_name]['name'])
elif model_name == 'our0':
    net = MILCovid19(in_channel=3, out_channel=32, category=bit, patches=dict_model[model_name]['patch'], num_layers=5,
                     num_stack=6,
                     image_size=dict_model[model_name]['size'] // int(np.sqrt(dict_model[model_name]['patch'])))
else:
    net = AD2D_MIL(in_channel=3, hidden=512, category=bit, num_layer=5, image_size=dict_model[model_name]['size'],
                   patches=dict_model[model_name]['patch'])

net.load_state_dict(torch.load(dict_model[model_name]['param'], lambda storage, loc: storage))
net.eval()

transform = Compose([
    ToPILImage(),
    Resize((512, 512)),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def predict_ATP(image: np.array):
    global net
    net.eval()
    in_tensor = transform(image).float()
    if torch.cuda.is_available():
        in_tensor = in_tensor.to('cuda:0')
    in_tensor = in_tensor.unsqueeze(0)
    # print(in_tensor.size())
    pre_atp = net(in_tensor).detach().cpu().numpy().flatten()
    return np.int32(pre_atp[-1] * 200000)


if __name__ == '__main__':
    path = 'C:/Users/wow00/Desktop/dataval/1189-2/'
    import os
    import cv2

    csv_data = []
    for root, _, names in os.walk(path):
        for name in names:
            if not name.endswith('.tif'):
                continue
            image_path = os.path.join(root, name)
            img = cv2.imread(image_path)
            mask = cv2.imread(image_path, 0)
            x, y, w, h = fine_mask(mask)
            new_img = img[y:y + h, x:x + w]
            # cv2.imshow('1', new_img)
            # cv2.waitKey()
            atp = predict_ATP(new_img)
            line = name + ',' + str(atp) + '\n'
            csv_data.append(line)
            new_img = cv2.resize(new_img, (512, 512))
            print(name, atp)

    with open(path + '{}_atp.csv'.format(model_name), 'w') as f:
        f.writelines(csv_data)
