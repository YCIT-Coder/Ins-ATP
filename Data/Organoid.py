#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Organoid.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2021/8/2 15:57
# @Desc  :

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize, ToTensor, ToPILImage, \
    Normalize


def ATPMaper(pre, bin_wid=20000, th=0.5):
    base = torch.logspace(0, pre.size(-1) - 2, pre.size(-1) - 1, 2).__reversed__().to(pre.device)
    base = base.view(1, -1).expand(pre[:, :-1].size())
    value = (torch.sum((pre[:, :-1] > th).float() * base, -1) + pre[:, -1]) * bin_wid
    return value


class OrganoidData(Dataset):
    """"""

    def __init__(self, url='../H5_file/ATP_data_1.h5', size=224, resample=False, bin_wid=20000, mod=200000):
        """Constructor for OrganoidData"""
        super(OrganoidData, self).__init__()
        self.file = h5py.File(url, 'r')
        self.transform = Compose([
            ToPILImage(),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            # RandomEqualize(),
            # RandomAutocontrast(),
            Resize(size),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mod = mod
        self.index = np.array(range(len(self.file['atp'])))
        atps = self.file['atp'][...].astype(np.int32)
        atps = np.clip(atps, 0, self.mod)
        if resample:
            indices = []
            for i in range(0, atps.max() + 1, bin_wid):
                mask = ((atps >= i) * (atps < (i + bin_wid))) > 0
                indices.append(self.index[mask].tolist())
            max_indices_len = np.max([len(_) for _ in indices])

            new_indices = []
            for inner_indices in indices:
                if len(inner_indices) > 0:
                    mod = max_indices_len // len(inner_indices)
                    rem = max_indices_len % len(inner_indices)
                    if mod > 1:
                        inner_indices.extend(inner_indices * (mod - 1))
                    if rem > 0:
                        inner_indices.extend(inner_indices[:rem])
                    new_indices.append(inner_indices)
            new_indices = np.stack(new_indices)
            self.index = new_indices.flatten()
        self.atps = atps

        # with open('atp1.csv', 'w') as csv:
        #     atps = atps[self.index]
        #     atp = [str(x) for x in atps]
        #     csv.writelines('\n'.join(atp))

    def __getitem__(self, i):
        item = self.index[i]
        atp = self.atps[item] / self.mod
        image = self.file['image'][item]

        data = {
            'image': self.transform(image),
            'atp': float(atp)
        }
        return data

    def __len__(self):
        assert len(self.file['image']) == len(self.file['atp'])
        return len(self.index)


class Pre_data(Dataset):
    """"""

    def __init__(self, url='../H5_file/seq.h5', size=224):
        """Constructor for OrganoidData"""
        super(Pre_data, self).__init__()
        self.file = h5py.File(url, 'r')
        self.transform = Compose([
            ToPILImage(),
            # RandomHorizontalFlip(),
            # RandomVerticalFlip(),
            Resize(size),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.keys = list(self.file.keys())

    def __getitem__(self, item):
        key = self.keys[item]
        Images = []
        for image in self.file[key]:
            image = self.transform(image)
            Images.append(image)

        Images = torch.stack(Images, 0)

        return {'key': key, 'images': Images}

    def __len__(self):
        return len(self.keys)


class OrganoidBinary(Dataset):
    """"""

    def __init__(self, url='../H5_file/ATP_data_1.h5', size=224, resample=False, bin_wid=20000, mod=200000, bit=10):
        """Constructor for OrganoidBinary"""
        super(OrganoidBinary, self).__init__()
        self.file = h5py.File(url, 'r')
        self.transform = Compose([
            ToPILImage(),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            # RandomEqualize(),
            # RandomAutocontrast(),
            Resize((size, size,)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mod = mod
        self.index = np.array(range(len(self.file['atp'])))
        atps = self.file['atp'][...].astype(np.int32)
        atps = np.clip(atps, 0, self.mod)
        if resample:
            indices = []
            for i in range(0, atps.max() + 1, bin_wid):
                mask = ((atps >= i) * (atps < (i + bin_wid))) > 0
                indices.append(self.index[mask].tolist())
            max_indices_len = np.max([len(_) for _ in indices])

            new_indices = []
            for inner_indices in indices:
                if len(inner_indices) > 0:
                    mod = max_indices_len // len(inner_indices)
                    rem = max_indices_len % len(inner_indices)
                    if mod > 1:
                        inner_indices.extend(inner_indices * (mod - 1))
                    if rem > 0:
                        inner_indices.extend(inner_indices[:rem])
                    new_indices.append(inner_indices)
            new_indices = np.stack(new_indices)
            self.index = new_indices.flatten()
        self.atps = atps
        self.bin_wid = bin_wid
        self.bit = bit

    def __getitem__(self, i):
        item = self.index[i]
        atp = self.atps[item] / self.mod
        image = self.file['image'][item]
        binary = bin(self.atps[item] // self.bin_wid).replace('0b', '')
        binary = [int(element) for element in binary]
        decinal = self.atps[item] % self.bin_wid * 1.0 / self.bin_wid
        padding = self.bit - 1 - len(binary)
        if padding > 0:
            binary = np.array([0] * (self.bit - 1 - len(binary)) + binary + [decinal])
        else:
            binary = np.array(binary + [decinal])
        data = {
            'image': self.transform(image),
            'atp': float(atp),
            'binary': binary,
            # 'decimal': decinal
        }
        return data

    def __len__(self):
        assert len(self.file['image']) == len(self.file['atp'])
        return len(self.index)


class OrganoidBinaryResample(Dataset):
    """"""

    def __init__(self, url='../H5_file/ATP_data_1.h5', size=224, resample=False, bin_wid=20000, mod=200000, bit=10):
        """Constructor for OrganoidBinary"""
        super(OrganoidBinaryResample, self).__init__()
        self.file = h5py.File(url, 'r')
        self.transform = Compose([
            ToPILImage(),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            # RandomEqualize(),
            # RandomAutocontrast(),
            Resize((size, size,)),
            Resize((512, 512)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mod = mod
        self.index = np.array(range(len(self.file['atp'])))
        atps = self.file['atp'][...].astype(np.int32)
        atps = np.clip(atps, 0, self.mod)
        if resample:
            indices = []
            for i in range(0, atps.max() + 1, bin_wid):
                mask = ((atps >= i) * (atps < (i + bin_wid))) > 0
                indices.append(self.index[mask].tolist())
            max_indices_len = np.max([len(_) for _ in indices])

            new_indices = []
            for inner_indices in indices:
                if len(inner_indices) > 0:
                    mod = max_indices_len // len(inner_indices)
                    rem = max_indices_len % len(inner_indices)
                    if mod > 1:
                        inner_indices.extend(inner_indices * (mod - 1))
                    if rem > 0:
                        inner_indices.extend(inner_indices[:rem])
                    new_indices.append(inner_indices)
            new_indices = np.stack(new_indices)
            self.index = new_indices.flatten()
        self.atps = atps
        self.bin_wid = bin_wid
        self.bit = bit

    def __getitem__(self, i):
        item = self.index[i]
        atp = self.atps[item] / self.mod
        image = self.file['image'][item]
        binary = bin(self.atps[item] // self.bin_wid).replace('0b', '')
        binary = [int(element) for element in binary]
        decinal = self.atps[item] % self.bin_wid * 1.0 / self.bin_wid
        padding = self.bit - 1 - len(binary)
        if padding > 0:
            binary = np.array([0] * (self.bit - 1 - len(binary)) + binary + [decinal])
        else:
            binary = np.array(binary + [decinal])
        data = {
            'image': self.transform(image),
            'atp': float(atp),
            'binary': binary,
            # 'decimal': decinal
        }
        return data

    def __len__(self):
        assert len(self.file['image']) == len(self.file['atp'])
        return len(self.index)


if __name__ == '__main__':
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import cv2

    mod = 200000
    bin_wid = 2000
    bit = int(np.ceil(np.log2(mod // bin_wid) + 1e-8) + 1)
    print(bit)
    data = OrganoidBinary('../H5_file/ATP_train_crop_1.h5', size=512, resample=True, bin_wid=bin_wid, mod=mod, bit=bit)
    print(data.__len__())
    loader = DataLoader(data, 15, shuffle=True)
    Y = []
    Pre = []
    for batch in loader:
        atp = batch['atp'].numpy()
        if np.max(atp) == 1:
            print(atp)
            print(atp * data.mod)
            print(batch['binary'].numpy())
        else:
            # print(atp, atp * data.mod, batch['binary'].numpy())
            # print(np.max(atp))
            continue
        # image = batch["image"].squeeze().numpy().transpose([1, 2, 0])
        # cv2.imshow('1', (image + 1) / 2)
        # cv2.waitKey()
        # base = torch.logspace(0, 9, 10, 2).__reversed__()
        # base = base.view(1, -1).expand(batch['binary'][:, :-1].size())
        # value = (torch.sum(batch['binary'][:, :-1] * base, -1) + batch['binary'][:, -1]) * data.bin_wid
        # print(base).
        # print(atp, atp * data.mod, batch['binary'].numpy())
        value = ATPMaper(batch['binary'], data.bin_wid)
        print(value)
        break

    # pre = torch.rand(atp.shape[0])
    # Y.append(atp)
    # Pre.append(pre.detach().cpu().numpy())

    # Y = np.concatenate(Y, 0)
    # Pre = np.concatenate(Pre, 0)
    # ordered_index = sorted(range(len(Y)), key=lambda k: Y[k])
    # fig = plt.figure()
    # plt.plot(range(len(Y)), Y[ordered_index], color='C1')
    # plt.scatter(range(len(Pre)), Pre[ordered_index], color='C2', marker='2')
    # plt.show()
    # plt.close()
