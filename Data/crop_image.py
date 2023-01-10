#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : crop_image.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2021/12/1 8:59
# @Desc  :

import os
import cv2
import numpy as np
import pandas as pd


# path = 'D:/Organoid_Data/ATP_estimate_1'


def fine_big(contours):
    ans = [0, 0, 0, 0]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > ans[-1] * ans[-2]:
            ans = [x, y, w, h]
    return ans


def fine_mask(image):
    image = cv2.medianBlur(image, 15)
    image = cv2.medianBlur(image, 15).astype(np.float32)
    image[image > 250] = 0
    # print(image.mean())
    mask = 1 - (image < image.mean()).astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = fine_big(contours)
    return x, y, w, h


if __name__ == '__main__':
    path = 'D:/Organoid_Data/ATP_estimate_3'
    # path = 'D:/Organoid_Data/test'
    annos = []
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
            print(pos, annos[-1])

            if annos[-1] > 0:
                img = cv2.imread(image_path)
                mask = cv2.imread(image_path, 0)
                x, y, w, h = fine_mask(mask)
                new_img = img[y:y + h, x:x + w]
                img = cv2.resize(img, (512, 512))
                new_img = cv2.resize(new_img, (512, 512))
                cv2.imshow('raw', img)
                cv2.imshow('img', new_img)
                cv2.waitKey()
