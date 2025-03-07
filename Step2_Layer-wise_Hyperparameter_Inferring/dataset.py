import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import h5py
import os
import time
from utils import Timer
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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

class Crop(torch.nn.Module):
    # 裁剪/填充处理
    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, inputs):
        out = [inputs[i % inputs.shape[0]] for i in range(0, self.length)] 
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

# DEBUG
rapl_timer = Timer()
class Rapl(torch.utils.data.Dataset):
    """
    加载指定index的数据
    """
    def __init__(self, file_path, index_dict, transform, target_transform):
        super().__init__()
        self.index_dict = index_dict 
        self.bunch_size = 600 * 128
        self.begin = -1 # 当前bunch的位置
        self.end = -1
        self.length = len(index_dict)
        self.bunch_data = {"trace":[], "hp":[]}
        self.file_path = file_path
        self._load_bunch(0)

        self.transform = transform
        self.target_transform = target_transform

    
    def _load_bunch(self, index):
        rapl_timer.start()
        self.begin = index // self.bunch_size * self.bunch_size
        self.end = min(self.begin + self.bunch_size, self.length)
        self.bunch_data["trace"].clear()
        self.bunch_data["hp"].clear()
        with h5py.File(self.file_path, "r") as f:
            dataset_trace = f["trace"]
            dataset_hp = f["hp"]
            self.bunch_data["trace"] = [dataset_trace[self.index_dict[i]][:, 1:3] for i in range(self.begin, self.end)]
            self.bunch_data["hp"] = [dataset_hp[self.index_dict[i]][:] for i in range(self.begin, self.end)]
        rapl_timer.stop()
        
    def __getitem__(self, index):
        if index < self.begin or index >= self.end:
            self._load_bunch(index)
        trace = self.transform(self.bunch_data["trace"][index % self.bunch_size])
        hp = self.target_transform(self.bunch_data["hp"][index % self.bunch_size])
        return trace, hp
        

    def __len__(self):
        return self.length
    
    def shuffle(self):
        torch.manual_seed(int(time.time()))
        new_order = torch.randperm(len(self.index_dict))
        self.index_dict = [self.index_dict[new_order[i]] for i in range(len(self.index_dict))]

class RaplLoader(object):
    def __init__(self, args, no_val=False, input_size=["224"], indirect_regression=False):
        self.device = args.device
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
        self.is_test = no_val
        self.input_size = input_size # 样本的input_size
        self.no_val = no_val
        self.path = r"dataset/conv2d.h5" 
        self.seed = 0
        # 数据预处理
        use_crop = ["kernel_size", "stride"] #TODO: 检验kernelsize
        if args.HyperParameter in use_crop:
            self.transform = transforms.Compose([
                Normalization(), # 归一化
                Crop(1024), # 子采样缩放到1024长度
            ])
        else:
            self.transform = transforms.Compose([
                Normalization(), # 归一化
                Resize(1024), # 子采样缩放到1024长度
            ])
        # 对x处理的模块
        self.target_transform = transforms.Compose([
            ToTargets(args.HyperParameter, self.label, self.layer_type, indirect_regression),#对目标值进行缩放(K, S, C_o)
        ]) # 对y处理的模块

    def get_index_dict(self, input_size="224", no_val=False):
        offset = [442344, 884688, 1327032, 1769376, 2211720] 
        i = {"160":0, "192":1, "224":2, "299":3, "331":4}[input_size]
        begin = offset[i-1] if i-1 >= 0 else 0
        end = offset[i]
        length = end - begin
        val_rate = 0.10
        if no_val:
            return [str(v) for v in range(begin, end)]
        else:
            torch.manual_seed(self.seed)
            index_dict_val = (torch.randperm(end - begin) + begin).tolist()
            index_dict = [str(i) for i in index_dict_val[begin + int(length * val_rate) : end]]
            index_dict_val = [str(i) for i in index_dict_val[begin : begin + int(length * val_rate)]]
            return index_dict, index_dict_val


    def get_loader(self):
        # index_dict
        if self.no_val:
            index_dict = []
            for size in self.input_size:
                index_dict.extend(self.get_index_dict(input_size=size, no_val=self.no_val))
            self.dataset = Rapl(self.path, index_dict,self.transform, self.target_transform)
            dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
                # 是否打乱由index_dict决定
            return dataloader
        else:
            index_dict = []
            index_dict_val = []
            
            for size in self.input_size:
                _1, _2 = self.get_index_dict(input_size=size, no_val=self.no_val)
                index_dict.extend(_1)
                index_dict_val.extend(_2)
            self.dataset = Rapl(self.path, index_dict,self.transform, self.target_transform)
            self.dataset_val = Rapl(self.path, index_dict_val,self.transform, self.target_transform)
            dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
            dataloader_val = torch.utils.data.DataLoader(
                self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
            return dataloader, dataloader_val
    
    def shuffle_dataset(self):
        # 打乱训练集数据
        self.dataset.shuffle()    
    
                

        