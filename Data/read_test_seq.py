#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : read_test_seq.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2021/10/11 9:47
# @Desc  :

import os
import cv2
import h5py
import numpy as np

if __name__ == '__main__':
    Poses = []
    path = 'D:/Organoid_Data/ATP_estimate'
    for root, _, names in os.walk(path):
        for name in names:
            if not name.endswith('.tif'):
                continue
            pos, _, T = name.replace('Aligned[ZProj[Bright Field]]_', '')[: -4].split('_')
            Poses.append(pos)
    Poses = set(Poses)

    with h5py.File('../H5_file/seq.h5', 'w') as f:
        for pos in Poses:
            gallery = []
            for i in range(1, 8):
                image_path = 'Aligned[ZProj[Bright Field]]_{}_1_00{}.tif'.format(pos, i)
                image_path = os.path.join(path, image_path)
                img = cv2.imread(image_path)
                print(img.shape, img.dtype)
                gallery.append(img)
            gallery = np.stack(gallery)
            f[pos] = gallery
