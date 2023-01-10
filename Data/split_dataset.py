#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : split_dataset.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2021/8/3 10:31
# @Desc  :

import h5py

start_index = 4

if __name__ == '__main__':
    with h5py.File("../H5_file/ATP_data_crop.h5", 'r') as f:
        image = f['image'][...]
        atp = f['atp'][...]
        ordered_index = sorted(range(len(atp)), key=lambda k: atp[k])  # 获取排序索引

        val_index = ordered_index[start_index::5]
        train_index = list(set(ordered_index) - set(val_index))
        print(len(ordered_index), ordered_index)
        print(len(train_index), train_index)
        print(len(val_index), val_index)

        with h5py.File('../H5_file/ATP_train_crop_{}.h5'.format(start_index), 'w') as f:
            f['image'] = image[train_index]
            f['atp'] = atp[train_index]

        with h5py.File('../H5_file/ATP_val_crop_{}.h5'.format(start_index), 'w') as f:
            f['image'] = image[val_index]
            f['atp'] = atp[val_index]
