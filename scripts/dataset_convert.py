"""

This script shreds and preprocesses the data to the same length to speed up data loading, ultimately generating h5 files.
"""
import h5py
import numpy as np
import argparse
import torchvision.transforms as transforms
import torch
model_index_dict = {'vgg':0, 'vgg_bn':1, 'resnet_basicblock':2, 'resnet_bottleneck':3, 'custom_net':4, 'custom_net_bn':5}
layer_index_dict = {'conv2d': 0, 'batch_norm': 1, 'relu_': 2,
               'max_pool2d': 3, 'adaptive_avg_pool2d': 4,
               'linear': 5, 'add_': 6, '_': 7}
domain_index_dict = {"160":0, "192":1, "224":2, "299":3, "331":4}
index_domain_dict = {0: "160", 1: "192", 2: "224", 3: "299", 4: "331"}

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
        indices = np.linspace(0, inputs.shape[0] - 1, self.length, dtype=int)
        out = inputs[indices].transpose([1, 0])
        return out

class CopyPad(torch.nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, inputs):
        indices = np.arange(self.length) % inputs.shape[0]
        out = inputs[indices].transpose([1, 0])
        return out

def signal_cut(layer_type, domain):
    data_x, data_y = [], []
    trace_num = []
    datah5 = h5py.File(r'./dataset/data.h5', 'r')
    hph5 = h5py.File(r'./dataset/hp.h5', 'r')
    for k in datah5['data'].keys():
        if k.split(")")[1] == domain:
            traces = datah5['data'][k][:, :-1]
            label = datah5["data"][k][:, -1]
            pos = datah5['position'][k][:]
            hp = hph5[k][:]
            if layer_type == "linear":
                hp = hp[hp[:, -2] == -1]
                hp = hp[hp[:, -1] == -1]
            elif layer_type == "conv2d":
                hp = hp[hp[:, 0] != -1]
                hp = hp[hp[:, -1] != -1]
            elif layer_type == "max_pool2d":
                hp = hp[hp[:, 0] == -1]

            hp_index = 0
            temp_x = []
            temp_y = []
            for (i, j) in pos:
                if label[j] == layer_index_dict[layer_type]:
                    temp_x.append(traces[i:j + 1, :])
                    temp_y.append(hp[hp_index])
                    hp_index += 1
            if hp_index == len(hp):
                data_x.extend(temp_x)
                data_y.extend(temp_y)
    datah5.close()
    hph5.close()
    return data_x, data_y

def signal_transform(signal_segments, layer_type):
    if layer_type == "conv2d":
        transform = transforms.Compose([
            Normalization(),
            CopyPad(1024),
        ])
    else:
        transform = transforms.Compose([
            Resize(1024),
        ])
    return [transform(segment) for segment in signal_segments]

def preprocess(layer_type):
    signal_domains = {}
    hp_domains = {}
    signal_nums = []
    for dom in range(5):
        signals, hps = signal_cut(layer_type, index_domain_dict[dom])
        signal_nums.append(len(signals))
        signal_trans = signal_transform(signals, layer_type)
        signal_stack = np.stack(signal_trans).astype(np.float32)
        hp_stack = np.stack(hps).astype(np.float32)
        domain_str = index_domain_dict[dom]
        print(domain_str, 'signal_stack', signal_stack.shape, signal_stack.dtype, 'bytes', signal_stack.size * signal_stack.itemsize)
        orig_bytes = 0
        with h5py.File('./dataset/data.h5', 'r') as _d:
            for kk in _d['data'].keys():
                if kk.split(')')[1] == domain_str:
                    ds = _d['data'][kk]
                    orig_bytes += int(np.prod(ds.shape)) * np.dtype(ds.dtype).itemsize
        print(domain_str, 'orig_data bytes (sum of datasets matching domain):', orig_bytes)

        signal_domains[index_domain_dict[dom]] = signal_stack
        hp_domains[index_domain_dict[dom]] = hp_stack
    print(signal_nums)
    return signal_domains, hp_domains

if __name__ == "__main__":

    layer_types = ["conv2d", "linear", "max_pool2d"]
    for layer_type in layer_types:
        trace_domains, hp_domains = preprocess(layer_type)
        output_file_path = rf'./dataset/new_dataset/{layer_type}.h5'
        with h5py.File(output_file_path, 'w') as f:
            trace_group = f.create_group('trace')
            for i in range(5):
                domain_str = index_domain_dict[i]
                trace_group.create_dataset(
                    domain_str,
                    data=trace_domains[domain_str].astype(np.float32),
                )
            hp_group = f.create_group('hp')
            for i in range(5):
                domain_str = index_domain_dict[i]
                hp_group.create_dataset(
                    domain_str,
                    data=hp_domains[domain_str].astype(np.float32),
                )

        print(layer_type, "OK")