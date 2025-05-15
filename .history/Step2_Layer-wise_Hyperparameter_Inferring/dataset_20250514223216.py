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

ALL_LOAD = True # 是否加载所有数据
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

class CopyPad(torch.nn.Module):
    # 裁剪/填充处理
    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, inputs):
        out = [inputs[i % inputs.shape[0]] for i in range(0, self.length)]
        out = np.array(out).transpose([1, 0]) # 对x转置
        return out

class ToTargets(torch.nn.Module):
    def __init__(self, hyperparameter, label, layer_type, regression=False):
        super().__init__()
        self.hyperparameter = hyperparameter
        self.label = label #目标超参数对应的列号
        self.layer_type = layer_type
        self.is_regression = regression

    def forward(self, targets):
        # targets：一层的所有超参数
        if self.hyperparameter == 'kernel_size' and self.layer_type == "conv2d":
            # {1, 3, 7} --> {0, 1, 2}
            targets = targets[self.label]
            targets = (targets - 1) / 2
            targets = targets - 1 if targets == 3 else targets
        elif self.hyperparameter == 'kernel_size' and self.layer_type == "max_pool2d":
            # {2,3} --> {0,1}
            targets = targets[self.label]
            targets = targets - 2
        if self.hyperparameter == 'stride':
            # {1,2} --> {0,1}
            targets = targets[self.label]
            targets = targets - 1
        if self.hyperparameter == 'out_channels':
            if self.is_regression:
                if self.layer_type == "conv2d":
                    # O_c = targets[0] * targets[2]**2 * targets[1] * targets[8]**2
                    # targets = np.concatenate([targets[0:3], [targets[8]], [np.log2(O_c)]], dtype=np.float32)
                    # 缩放到 [-1 ~ 1]
                    targets = targets[self.label]
                    lower = 6
                    upper = 11
                    targets = np.log2(targets) - lower # {0,1,2,3,4,5}
                if self.layer_type == "linear":
                    # O_l = targets[0] * targets[1]
                    # targets = np.concatenate([targets[0:2], [np.log2(O_l)]], dtype=np.float32)
                    targets = targets[self.label]
                    lower = 1000
                    upper = 4096
                    targets = (targets - lower) / (upper - lower) #{0,1}
            else :
                # {2^6, 2^7, 2^8...} --> {0, 1 ,2 ...}
                targets = targets[self.label]
                targets = np.log2(targets) - 6
        if self.hyperparameter == "padding":
            targets = targets[self.label]
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
        self.bunch_size = 2*600 * 128
        if ALL_LOAD:
            self.bunch_size = 2211720 # 最大可能的样本数
        self.begin = -1 # 当前bunch的位置
        self.end = -1
        self.length = len(index_dict)
        self.bunch_data = {"trace":[], "hp":[]}
        self.file_path = file_path
        self._load_bunch(0)

        self.transform = transform
        self.target_transform = target_transform
        rapl_timer.reset()

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
        print("已重新打乱数据")

class RaplLoader(object):
    def __init__(self, args, no_val=False, input_size=["224"], data_ratio=1.0):
        self.device = args.device
        self.label = {'in_channels': 0, 'out_channels': 1, 'kernel_size': 2,
                      'stride': 3, 'padding': 4, 'dilation': 5,
                      'groups': 6, 'input_size': 7, 'output_size': 8}[args.HyperParameter] #mode直接对字典取值
        self.layer_type = args.layer_type
        self.batch_size = args.batch_size
        self.num_workers = args.workers
        if args.layer_type == "conv2d":
            self.num_classes = {'out_channels': 6, 'kernel_size': 3, 'stride': 2}[args.HyperParameter]
        elif args.layer_type == "max_pool2d":
            self.num_classes = {'kernel_size': 2, "padding":2}[args.HyperParameter]
        elif args.layer_type == "linear":
            self.num_classes = 2 #1000, 4096
        self.input_size = input_size # 样本的input_size
        self.no_val = no_val
        self.path = f"dataset/{self.layer_type}.h5"
        self.seed = 0
        self.data_ratio = data_ratio
        # 数据预处理
        use_copypad = ["kernel_size", "stride", "out_channels"]
        if self.layer_type == "conv2d":
            if args.HyperParameter in use_copypad:
                self.transform = transforms.Compose([
                    Normalization(), # 归一化
                    CopyPad(1024), # 子采样缩放到1024长度
                ])
            else:
                self.transform = transforms.Compose([
                    Normalization(), # 归一化
                    Resize(1024), # 子采样缩放到1024长度
                ])
        else:
            self.transform = transforms.Compose([
                Resize(1024), # 子采样缩放到1024长度
            ])
        self.target_transform = transforms.Compose([
            ToTargets(args.HyperParameter, self.label, self.layer_type, args.regression),#对目标值进行缩放(K, S, C_o)
        ])

    def _sample_indices(self, indices: list) -> list:
        """
        在固定 seed 下，对 indices 先打乱，再取前 data_ratio 部分
        """
        total = len(indices)
        sample_count = max(1, int(total * self.data_ratio))
        # 先打乱
        torch.manual_seed(self.seed)
        perm = torch.randperm(total).tolist()
        sampled = [indices[i] for i in perm[:sample_count]]
        return sampled

    def get_index_dict(self, input_size="224", no_val=False):
        layer_type_index = {"conv2d":0, "max_pool2d":1, "linear":2}
        offset = [[442344, 884688, 1327032, 1769376, 2211720],  #conv2d
                  [20488, 40976, 61464, 81952, 102440],         #max_pool2d
                  [12128, 24261, 36402, 48553, 60706]][layer_type_index[self.layer_type]]          #linear
        i = {"160":0, "192":1, "224":2, "299":3, "331":4}[input_size]
        begin = offset[i-1] if i-1 >= 0 else 0
        end = offset[i]
        length = end - begin
        val_rate = 0.10
        if no_val:
            raw = [str(v) for v in range(begin, end)]
            return self._sample_indices(raw)
        else:
            torch.manual_seed(self.seed)
            index_dict_val = (torch.randperm(length) + begin).tolist()
            raw_train = [str(i) for i in index_dict_val[int(length * val_rate) : ]]
            raw_val = [str(i) for i in index_dict_val[0 : int(length * val_rate)]]
            index_dict = self._sample_indices(raw_train)
            index_dict_val = self._sample_indices(raw_val)
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



