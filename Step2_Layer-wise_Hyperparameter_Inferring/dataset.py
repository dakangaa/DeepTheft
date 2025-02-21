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
test_samples = [26, 26, 268, 268, 282, 282]

labels_name = {'conv2d': 0, 'batch_norm': 1, 'relu_': 2,
               'max_pool2d': 3, 'adaptive_avg_pool2d': 4,
               'linear': 5, 'add_': 6, '_': 7}

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
        out = np.array(out).transpose([1, 0]) # 对x转置
        return out


class ToTargets(torch.nn.Module):
    def __init__(self, mode, label, layer_type, regression=False):
        super().__init__()
        self.mode = mode
        self.label = label
        self.layer_type = layer_type
        self.is_regression = regression
        
    def forward(self, targets):
        # 每次只对一个target(层)调用
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
            if self.is_regression:
                if self.layer_type == "conv2d":    
                    O_c = targets[0] * targets[2]**2 * targets[1] * targets[8]**2
                    targets = np.concatenate([targets[0:3], [targets[8]], [np.log2(O_c)]], dtype=np.float32)
                if self.layer_type == "linear":
                    O_l = targets[0] * targets[1]
                    targets = np.concatenate([targets[0:2], [np.log2(O_l)]], dtype=np.float32)
            else :
                # {2^4, 2^5, 2^6,...} --> {-2, -1, 0, ...}
                targets = targets[self.label]
                targets = np.log2(targets) - 6
        return targets


class Rapl(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, target_transform=None, use_domain=False):
        self.use_domain = use_domain
        if self.use_domain:
            (self.feature, self.label, self.domain) = data
        else:
            (self.feature, self.label) = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.use_domain:
            feat, lab, dom = self.feature[index][:, 1:3], self.label[index], self.domain[index] 
        else:
            feat, lab = self.feature[index][:, 1:3], self.label[index]
        feat = self.transform(feat) if self.transform is not None else feat
        lab = self.target_transform(lab) if self.target_transform is not None else lab
        if self.use_domain:
            return feat, lab, dom
        else:
            return feat, lab

    def __len__(self):
        return len(self.feature)


class RaplLoader(object):
    def __init__(self, args, test_index=None, is_test=False, input_size=["224"], indirect_regression=False):
        self.use_domain = args.use_domain
        if test_index == None:
            self.test_index = []
            np.random.seed(0)
            for i in range(len(samples)):
                index = np.random.choice(samples[i], test_samples[i], replace=False)
                self.test_index.append(index)
        else:
            self.test_index = test_index
        self.label = {'in_channels': 0, 'out_channels': 1, 'kernel_size': 2,
                      'stride': 3, 'padding': 4, 'dilation': 5,
                      'groups': 6, 'input_size': 7, 'output_size': 8}[args.HyperParameter] #mode直接对字典取值
        self.layer_type = args.layer_type
        self.batch_size = args.batch_size
        self.num_workers = args.workers
        if indirect_regression:
            self.num_classes = {'out_channels': 1, 'kernel_size': 3, 'stride': 2}[args.HyperParameter]
        else:
            self.num_classes = {'out_channels': 6, 'kernel_size': 3, 'stride': 2}[args.HyperParameter]
            
        self.is_test = is_test
        self.input_size = input_size # 样本的input_size
        if not self.is_test:
            self.train, self.val = self.preprocess()
            # train: 224input_size 样本的 conv层 (只取两个channel)
            # val: 对应的一个超参数kernel_size(可调)
        else:
            self.test = self.preprocess()
                
        self.transform = transforms.Compose([
            Normalization(), # 归一化
            Resize(1024), # 子采样缩放到1024长度
        ]) # 对x处理的模块
        self.target_transform = transforms.Compose([
            ToTargets(args.HyperParameter, self.label, self.layer_type, indirect_regression),#对目标值进行缩放(K, S, C_o)
        ]) # 对y处理的模块

        # 确定运行环境
        self.device = args.device
        
        
    def preprocess(self):
        domain_index_dict = {"160":0, "192":1, "224":2, "299":3, "331":4}
        data_domain = [domain_index_dict[k] for k in self.input_size]
        train_x, train_y = [], []
        val_x, val_y = [], []
        if self.use_domain:
            train_domain = []
            val_domain = []
        if self.device == "laptop":
            x = h5py.File(r'dataset/data.h5', 'r')
            y = h5py.File(r'dataset/hp.h5', 'r')
        elif self.device == "autodl":
            x = h5py.File(r'../autodl-tmp/dataset/data.h5', 'r')
            y = h5py.File(r'../autodl-tmp/dataset/hp.h5', 'r')
        for k in x['data'].keys():
            domain = k.split(")")[1]
            domain = domain_index_dict[domain]
            if domain in data_domain : # A：只对输入大小为224的样本训练
                d = x['data'][k][:]
                pos = x['position'][k][:]
                hp = y[k][:]
                if self.layer_type == "linear":
                    # 大部分linear层的采样点数都为0
                    hp = hp[hp[:, -2] == -1]
                    hp = hp[hp[:, -1] == -1]
                elif self.layer_type == "conv2d":
                    hp = hp[hp[:, 0] != -1]
                    hp = hp[hp[:, -1] != -1]
                elif self.layer_type == "max_pool2d":
                    hp = hp[hp[:, 0] == -1]
                    
                test_indexes = self.test_index[models[k.split(')')[0]]] # 训练样本index数组
                n = int(k.split(')')[-1]) # 当前样本index
                hp_index = 0
                for (i, j) in pos:
                    if d[:, -1][j] == labels_name[self.layer_type]: 
                        if n in test_indexes:
                            val_x.append(d[i:j + 1, :-1])
                            val_y.append(hp[hp_index])  # 只有卷积层的超参数
                            if self.use_domain:
                                val_domain.append(domain)
                        else:
                            train_x.append(d[i:j + 1, :-1])
                            train_y.append(hp[hp_index])
                            if self.use_domain:
                                train_domain.append(domain)
                        hp_index += 1
                assert hp_index == len(hp)

        if self.is_test:
            if self.use_domain:
                return train_x + val_x, train_y + val_y, train_domain + val_domain
            else:
                return train_x + val_x, train_y + val_y
        else:
            if self.use_domain:
                return (train_x, train_y, train_domain), (val_x, val_y, val_domain)
            else:
                return (train_x, train_y), (val_x, val_y)

    def loader(self, data, shuffle=False, transform=None, target_transform=None):
        dataset = Rapl(data, transform=transform, target_transform=target_transform, use_domain=self.use_domain) # 自定义一个DataSet类
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=True)
        return dataloader

    def get_loader(self):
        if not self.is_test:
            # 训练数据
            trainloader = self.loader(self.train, shuffle=True, transform=self.transform, target_transform=self.target_transform)
            valloader = self.loader(self.val, transform=self.transform, target_transform=self.target_transform)
            return trainloader, valloader
        else:
            testloader = self.loader(self.test, shuffle=True, transform=self.transform, target_transform=self.target_transform)
            return testloader
        