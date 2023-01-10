#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : merge_h5.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2021/8/30 15:09
# @Desc  :

import h5py
import numpy as np

if __name__ == '__main__':
    # with h5py.File("../H5_file/ATP_data_crop.h5", 'w') as f:
    #     h1 = h5py.File("../H5_file/ATP_data_crop_1.h5", 'r')
    #     h2 = h5py.File("../H5_file/ATP_data_crop_2.h5", 'r')
    #     h3 = h5py.File("../H5_file/ATP_data_crop_3.h5", 'r')
    #
    #     print(len(h1['atp']))
    #     print(len(h2['atp']))
    #     print(len(h3['atp']))
    #
    #     image = np.concatenate([h1['image'][...], h2['image'][...], h3['image'][...]], 0)
    #     atp = np.concatenate([h1['atp'][...], h2['atp'][...], h3['atp'][...]], 0)
    #
    #     print(atp.shape)
    #     f['image'] = image
    #     f['atp'] = atp
    with h5py.File("../H5_file/ATP_data_crop.h5", 'r') as f:
        atp = f['atp'][...]
        with open('atp.csv', 'w') as csv:
            atp = atp.tolist()
            atp = [str(x) for x in atp]
            csv.writelines('\n'.join(atp))
