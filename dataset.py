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
LABEL_DICT = {'in_channels': 0, 'out_channels': 1, 'kernel_size': 2,
                      'stride': 3, 'padding': 4, 'dilation': 5,
                      'groups': 6, 'input_size': 7, 'output_size': 8}

class ToTargets(torch.nn.Module):
    def __init__(self, hyperparameter, label, layer_type, regression=False):
        super().__init__()
        self.hyperparameter = hyperparameter
        self.label = label #目标超参数对应的列号
        self.layer_type = layer_type
        self.is_regression = regression

    def forward(self, targets):
        # targets：一层的所有超参数
        assert isinstance(targets, np.ndarray), f"ToTargets.forward expects numpy.ndarray, got {type(targets)}"
        targets = np.transpose(targets)
        if self.hyperparameter == 'kernel_size' and self.layer_type == "conv2d":
            # {1, 3, 7} --> {0, 1, 2}
            targets = targets[self.label]
            targets = (targets - 1) / 2
            targets = np.where(targets==3, targets - 1, targets)
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

rapl_timer = Timer()
class Rapl(torch.utils.data.Dataset):
    """
    读取、加载指定index的数据
    """
    def __init__(self, file_path, domains, target_transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        self.target_transform = target_transform

        self.data = dict() #{trace:,hp:}
        self._load(domains)
        self.length = self.data["hp"].shape[0]

    def _load(self, domains):
        """
        将指定domains的数据加载到data中
        """
        assert len(domains) > 0
        with h5py.File(self.file_path, "r") as f:
            list_t = []
            list_h = []
            for idx in range(len(domains)):
                list_t.append(f["trace"][domains[idx]][:, 1:3, :])
                list_h.append(f["hp"][domains[idx]][:])
            self.data["trace"] = np.concatenate(list_t, axis=0)
            self.data["hp"] = np.concatenate(list_h, axis=0)


    def __getitem__(self, index):
        trace = self.data["trace"][index,:,:]
        hp = self.target_transform(self.data["hp"][index,:])
        return trace, hp

    def __getitems__(self, indices):
        traces = self.data["trace"][indices,:,:]
        hps = self.target_transform(self.data["hp"][indices,:])
        return list(zip(traces,hps))
    
    def __len__(self):
        return self.length

class RaplLoader(object):
    # 生成dataloader
    def __init__(self, args, no_val=False, input_size=["224"]):
        self.label = LABEL_DICT[args.HyperParameter] #mode直接对字典取值
        self.layer_type = args.layer_type
        self.batch_size = args.batch_size
        self.num_workers = args.workers
        if args.layer_type == "conv2d":
            self.num_classes = {'out_channels': 6, 'kernel_size': 3, 'stride': 2}[args.HyperParameter]
        elif args.layer_type == "max_pool2d":
            self.num_classes = {'kernel_size': 2, "padding":2}[args.HyperParameter]
        elif args.layer_type == "linear":
            self.num_classes = 2 #1000, 4096
        self.domains = input_size # 数据域
        self.prefetch_factor = args.prefetch_factor
        self.no_val = no_val #是否需要验证集
        self.path = os.path.join(args.data_path, f"{self.layer_type}.h5")
        # 数据预处理
        self.target_transform = transforms.Compose([
            ToTargets(args.HyperParameter, self.label, self.layer_type, args.regression),#对目标值进行缩放(K, S, C_o)
        ])
        
        self.seed = 42
        self.val_rate = 0.1

    def get_loader(self):
        dataset = Rapl(self.path, self.domains, self.target_transform)
        if self.no_val:
            # 用于验证
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, prefetch_factor=self.prefetch_factor)
                # 是否打乱由index_dict决定
            return dataloader
        else:
            val_size = int(len(dataset) * self.val_rate)
            train_size = len(dataset) - val_size
            generator = torch.Generator().manual_seed(self.seed)
            dataset_train, dataset_val = torch.utils.data.random_split(dataset,[train_size,val_size], generator=generator)
            dataloader_train = torch.utils.data.DataLoader(
                dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, prefetch_factor=self.prefetch_factor)
            dataloader_val = torch.utils.data.DataLoader(
                dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, prefetch_factor=self.prefetch_factor)
            return dataloader_train, dataloader_val





