import h5py
import numpy as np
import argparse

model_index_dict = {'vgg':0, 'vgg_bn':1, 'resnet_basicblock':2, 'resnet_bottleneck':3, 'custom_net':4, 'custom_net_bn':5}
layer_index_dict = {'conv2d': 0, 'batch_norm': 1, 'relu_': 2,
               'max_pool2d': 3, 'adaptive_avg_pool2d': 4,
               'linear': 5, 'add_': 6, '_': 7}
# 生成conv层的数据
def preprocess(layer_type):
    domain_index_dict = {"160":0, "192":1, "224":2, "299":3, "331":4}
    domains = range(5)
    data_x, data_y = [], []
    offset = []
    datah5 = h5py.File(r'../autodl-tmp/dataset/data.h5', 'r')
    hph5 = h5py.File(r'../autodl-tmp/dataset/hp.h5', 'r')
    for dom in domains:
        for k in datah5['data'].keys():
            if domain_index_dict[k.split(")")[1]] == dom:
                traces = datah5['data'][k][:, :-1]
                label = datah5["data"][k][:, -1]
                pos = datah5['position'][k][:]
                hp = hph5[k][:]
                # 筛选hp
                if layer_type == "linear":
                    # 大部分linear层的采样点数都为0
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
                # 如果不能正确匹配，那么该轮就舍弃
                if hp_index == len(hp):
                    data_x.extend(temp_x)
                    data_y.extend(temp_y)
        offset.append(len(data_x))

    return data_x, data_y, offset

parser = argparse.ArgumentParser(description='create dataset')
parser.add_argument("--layer_type", type=str)
args = parser.parse_args()

# 调用 preprocess 函数生成 data_x 和 data_y
layer_type = args.layer_type
data_x, data_y, offset = preprocess(layer_type)
print(f"offset: {offset}")
# 将 data_x 和 data_y 写入新的 HDF5 文件
output_file_path = rf'../autodl-tmp/dataset/{layer_type}.h5'
with h5py.File(output_file_path, 'w') as f:
    # 创建 trace 数据集
    trace_group = f.create_group('trace')
    for i, arr in enumerate(data_x):
        trace_group.create_dataset(str(i), data=arr)  # 存储每个 numpy 数组

    # 创建 hp 数据集
    hp_group = f.create_group('hp')
    for i, arr in enumerate(data_y):
        hp_group.create_dataset(str(i), data=arr)  # 存储每个 numpy 数组

print("OK")