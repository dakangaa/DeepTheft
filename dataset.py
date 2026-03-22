import torch
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
import h5py
import os
from utils import Timer

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
LABEL_DICT = {'in_channels': 0, 'out_channels': 1, 'kernel_size': 2,
                      'stride': 3, 'padding': 4, 'dilation': 5,
                      'groups': 6, 'input_size': 7, 'output_size': 8}

class ToTargets(torch.nn.Module):
    def __init__(self, hyperparameter, label, layer_type, regression=False):
        super().__init__()
        self.hyperparameter = hyperparameter
        self.label = label
        self.layer_type = layer_type
        self.is_regression = regression

    def forward(self, targets):
        assert isinstance(targets, np.ndarray), f"ToTargets.forward expects numpy.ndarray, got {type(targets)}"
        targets = np.transpose(targets)
        if self.hyperparameter == 'kernel_size':
            if self.layer_type == "conv2d":
                targets = targets[self.label]
                targets = (targets - 1) / 2
                targets = np.where(targets==3, targets - 1, targets)
            elif self.layer_type == "max_pool2d":
                targets = targets[self.label]
                targets = targets - 2
        if self.hyperparameter == 'stride':
            targets = targets[self.label]
            targets = targets - 1
        if self.hyperparameter == 'out_channels':
            assert self.is_regression == True
            if self.layer_type == "conv2d":
                targets = targets[self.label]
                lower = 6
                upper = 11
                targets = np.log2(targets) - lower
            if self.layer_type == "linear":
                targets = targets[self.label]
                lower = 1000
                upper = 4096
                targets = (targets - lower) / (upper - lower)
        if self.hyperparameter == "padding":
            targets = targets[self.label]
        return targets

rapl_timer = Timer()
class Rapl(torch.utils.data.Dataset):
    def __init__(self, file_path, domains, target_transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        self.target_transform = target_transform

        self.data = dict()
        self._load(domains)
        self.length = self.data["hp"].shape[0]

    def _load(self, domains):
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
    def __init__(self, args, no_val=False, input_size=["224"]):
        self.label = LABEL_DICT[args.HyperParameter]
        self.layer_type = args.layer_type
        self.batch_size = args.batch_size
        self.num_workers = args.workers
        if args.layer_type == "conv2d":
            self.num_classes = {'out_channels': 6, 'kernel_size': 3, 'stride': 2}[args.HyperParameter]
        elif args.layer_type == "max_pool2d":
            self.num_classes = {'kernel_size': 2, "padding":2}[args.HyperParameter]
        elif args.layer_type == "linear":
            self.num_classes = 2
        self.domains = input_size
        self.prefetch_factor = args.prefetch_factor
        self.no_val = no_val
        self.path = os.path.join(args.data_path, f"{self.layer_type}.h5")
        self.target_transform = transforms.Compose([
            ToTargets(args.HyperParameter, self.label, self.layer_type, args.regression),
        ])

        self.seed = 42
        self.val_rate = 0.1

    def get_loader(self):
        dataset = Rapl(self.path, self.domains, self.target_transform)
        if self.no_val:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, prefetch_factor=self.prefetch_factor)
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





