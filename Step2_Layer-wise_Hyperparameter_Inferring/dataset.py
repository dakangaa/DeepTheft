import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import h5py
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

models = {'vgg':0, 'vgg_bn':1, 'resnet_basicblock':2, 'resnet_bottleneck':3, 'custom_net':4, 'custom_net_bn':5}
samples = [256, 256, 2592, 2592, 2728, 2728]
# train_samples = [12, 12, 116, 116, 122, 122]
test_samples = [26, 26, 268, 268, 282, 282]

# 似乎没用:
# train_index = [] 
# np.random.seed(1)
# for i in range(len(samples)):
#     index = np.random.choice(samples[i], train_samples[i], replace=False)
#     train_index.append(index)


class Normalization(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        _range = np.max(input, axis=0) - np.min(input, axis=0) + 1e-7
        input = (input - np.min(input, axis=0)) / _range
        return input


class Resize(torch.nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, inputs):
        out = [inputs[int(i * inputs.shape[0] / self.length)] for i in range(self.length)] # 通过子采样resize样本到指定大小
        out = np.array(out).transpose([1, 0])
        return out


class ToTargets(torch.nn.Module):
    def __init__(self, mode, label):
        super().__init__()
        self.mode = mode
        self.label = label

    def forward(self, targets):
        # 其他超参数呢
        if self.mode == 'kernel_size':
            # {1, 3, 5, 7} --> {0, 1, 2, 2?}
            targets = targets[self.label]
            targets = (targets - 1) / 2
            targets = targets - 1 if targets == 3 else targets #? :让离散目标值变紧凑? 为什么要targets - 1 if targets == 3
        if self.mode == 'stride':
            # {1,2} --> {0,1}
            targets = targets[self.label]
            targets = targets - 1
        if self.mode == 'out_channels':
            # {2^4, 2^5, 2^6,...} --> {-2, -1, 0, ...}
            targets = targets[self.label]
            targets = np.log2(targets) - 6
        return targets


class Rapl(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        (self.feature, self.label) = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        feat, lab = self.feature[index][:, 1:3], self.label[index]
        feat = self.transform(feat) if self.transform is not None else feat
        lab = self.target_transform(lab) if self.target_transform is not None else lab
        return feat, lab

    def __len__(self):
        return len(self.feature)


class RaplLoader(object):
    def __init__(self, batch_size, mode, num_workers=0):
        test_index = []
np.random.seed(0)
for i in range(len(samples)):
    index = np.random.choice(samples[i], test_samples[i], replace=False)
    test_index.append(index)
        self.label = {'in_channels': 0, 'out_channels': 1, 'kernel_size': 2,
                      'stride': 3, 'padding': 4, 'dilation': 5,
                      'groups': 6, 'input_size': 7, 'output_size': 8}[mode] #mode直接对字典取值
        # 与层类别无关?
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = {'out_channels': 6, 'kernel_size': 3, 'stride': 2}[mode]
        self.train, self.val = self.preprocess()
        # train: 224input_size 样本的 conv层 三个RAPL通道
        # val: 对应的一个超参数kernel_size(可调)

        self.transform = transforms.Compose([
            Normalization(), # 归一化
            Resize(1024), # 子采样缩放到1024长度
        ]) # 对x处理的模块
        self.target_transform = transforms.Compose([
            ToTargets(mode, self.label),#对目标值进行缩放(K, S, C_o)
        ]) # 对y处理的模块

    def preprocess(self):
        train_x, train_y = [], []
        val_x, val_y = [], []
        x = h5py.File(r'../autodl-tmp/dataset/data.h5', 'r')
        y = h5py.File(r'../autodl-tmp/dataset/hp.h5', 'r')

        for k in x['data'].keys():
            if k.split(')')[1] == '224': # A：只对输入大小为224的样本训练
                d = x['data'][k][:]
                pos = x['position'][k][:]
                hp = y[k][:]
                hp = hp[hp[:, 0] != -1]
                hp = hp[hp[:, -1] != -1]
                #去除与输入通道数无关或与输出形状无关的层(?去掉MP和Linear)
                
                test_indexes = test_index[models[k.split(')')[0]]] # 训练样本index数组
                n = int(k.split(')')[-1]) # 当前样本index
                hp_index = 0
                for (i, j) in pos:
                    if d[:, -1][i] == 0: # ? 0:只取卷积层
                        if n in test_indexes:
                            val_x.append(d[i:j + 1, :-1])# x是三个RAPL通道
                            val_y.append(hp[hp_index])  # 只有卷积层的超参数
                        else:
                            train_x.append(d[i:j + 1, :-1])
                            train_y.append(hp[hp_index])
                        hp_index += 1
                assert hp_index == len(hp)

        return (train_x, train_y), (val_x, val_y)

    def loader(self, data, shuffle=False, transform=None, target_transform=None):
        dataset = Rapl(data, transform=transform, target_transform=target_transform) # 自定义一个DataSet类
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        return dataloader

    def get_loader(self):
        trainloader = self.loader(self.train, shuffle=True, transform=self.transform, target_transform=self.target_transform)
        valloader = self.loader(self.val, transform=self.transform, target_transform=self.target_transform)
        return trainloader, valloader
