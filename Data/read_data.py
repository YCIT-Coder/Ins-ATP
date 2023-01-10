#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : read_data.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2021/8/2 8:44
# @Desc  :

import os
import cv2
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from crop_image import fine_mask

if __name__ == '__main__':
    # data = pd.read_excel('D:/Organoid_Data/ATP_estimate/0747-2/ATP.xlsx', index_col='id')
    annos = []
    images = []
    path = 'D:/Organoid_Data/ATP_estimate_3'
    for root, _, names in os.walk(path):
        if root == path:
            continue
        dir_name = os.path.basename(root).split('-')[0]
        anno_path = os.path.join(root, 'ATP_{}.xlsx'.format(dir_name))
        anno = pd.read_excel(anno_path, index_col='id')
        for name in names:
            if not name.endswith('.tif'):
                continue
            image_path = os.path.join(root, name)
            pos = name.split('_')[1]
            annos.append(anno.loc[pos[0], int(pos[1:])])
            # print(annos[-1])

            if annos[-1] > 0:
                img = cv2.imread(image_path)
                mask = cv2.imread(image_path, 0)
                x, y, w, h = fine_mask(mask)
                img = img[y:y + h, x:x + w]
                img = cv2.resize(img, (1992, 1992))
                images.append(img)
                print(img.shape)
                # img = cv2.resize(img, (800, 800))
                # img = cv2.putText(img, str(annos[-1]), (50, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
                # img = cv2.putText(img, image_path, (50, 70), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
                # cv2.imshow('img', img)
                # cv2.waitKey()

    # annos.sort()
    # plt.scatter(range(len(annos)), annos, marker='1')
    # plt.show()

    images = np.stack(images)
    print(images.dtype, images.shape)

    with h5py.File('../H5_file/ATP_data_crop_3.h5', 'w') as f:
        f['image'] = images
        f['atp'] = annos
        pass
