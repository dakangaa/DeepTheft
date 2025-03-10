import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from copy import deepcopy
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import h5py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

labels_name = {'conv2d': 0, 'batch_norm': 1, 'relu_': 2,
               'max_pool2d': 3, 'adaptive_avg_pool2d': 4,
               'linear': 5, 'add_': 6, '_': 7}

models = {'vgg':0, 'vgg_bn':1, 'resnet_basicblock':2, 'resnet_bottleneck':3, 'custom_net':4, 'custom_net_bn':5}
samples = [256, 256, 2592, 2592, 2728, 2728]
test_samples = [26, 26, 268, 268, 282, 282]



class Rapl(torch.utils.data.Dataset):
    """
    包含x,y,z的数据集
    """ 
    def __init__(self, data, transform=None, target_transform=None):
        self.x, self.y = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        feature, target, label = x[:, 1:3], x[:, 3:4], y # 第一列数据是另外一个通道的数据, 在训练MetaModel_Hyper时会用
        feature = self.transform(feature) if self.transform is not None else feature
        target = self.target_transform(target) if self.target_transform is not None else target
        return feature, target, label

    def __len__(self):
        return len(self.x)


def collate_fn_batch(data):
    """
    定义如何生成一个batch
    左右填充xy为16的倍数, x用0填充, y用-1填充
    z没有改变: 在计算loss时会把padding去掉再计算
    """
    max_length = max([_x.shape[0] for (_x, _y, _z) in data])
    max_length += 16 - max_length % 16
    x, y, z = [], [], []
    for i, (_x, _y, _z) in enumerate(data):
        _x, _y = torch.as_tensor(_x).transpose(1, 0), torch.as_tensor(_y).transpose(1, 0)
        l = int((max_length - _x.shape[-1]) / 2) # 左填充
        r = max_length - _x.shape[-1] - l # 右填充

        _x = F.pad(_x, (l, r), "constant", 0) 
        _y = F.pad(_y, (l, r), "constant", -1)

        x = _x.unsqueeze(0) if i == 0 else torch.concat([x, _x.unsqueeze(0)])
        y = _y if i == 0 else torch.concat([y, _y])
        z.append(_z)

    return x, y, z


class RaplLoader(object):
    def __init__(self, batch_size, num_workers=0, test_index=None, mode="all", is_test=False):
        if test_index == None:
            self.test_index = []
            np.random.seed(0)
            for i in range(len(samples)):
                index = np.random.choice(samples[i], test_samples[i], replace=False)
                self.test_index.append(index)
        else:
            self.test_index = test_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = labels_name['_']
        self.is_test = is_test
        if is_test:
            self.test = self.preprocess(mode, is_test)
        else:
            self.train, self.val = self.preprocess(mode, is_test)
            # data ready

    def preprocess(self, mode, is_test):
        # 生成训练集和验证集
        train_x, train_y = [], []
        val_x, val_y = [], []
        if mode == "all":
            input_size = ["331", "299", "224", "192", "160"]
        else:
            input_size = [mode]
        # 数据路径
        data = h5py.File(r'../autodl-tmp/dataset/data.h5', 'r')
        # 加载数据 
        # !注意内存是否足够
        for k in data['data'].keys():
            if k.split(")")[1] in input_size:
                test_indexes = self.test_index[models[k.split(')')[0]]]
                i = int(k.split(')')[-1])
                if i in test_indexes:
                    val_x.append(data['data'][k][:])
                    val_y.append(data['position'][k][:])
                else:
                    train_x.append(data['data'][k][:])
                    train_y.append(data['position'][k][:])
        if is_test:
            return train_x+val_x ,train_y+val_y
        else:
            return (train_x, train_y), (val_x, val_y)

    def loader(self, data, shuffle=False, transform=None, target_transform=None):
        dataset = Rapl(data, transform=transform, target_transform=target_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, collate_fn=collate_fn_batch)
        #collate_fn:定义样本合并方式函数
        return dataloader

    def get_loader(self):
        if self.is_test:
            testloader = self.loader(self.test)
            return testloader
        else:
            trainloader = self.loader(self.train, shuffle=True)
            testloader = self.loader(self.val)
            return trainloader, testloader
