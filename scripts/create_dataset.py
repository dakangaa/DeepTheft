import h5py
import numpy as np
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
            if domain_index_dict(k.split(")")[1]) == dom: 
                d = datah5['data'][k][:]
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
                for (i, j) in pos:
                    if d[:, -1][j] == layer_index_dict[layer_type]: 
                        data_x.append(d[i:j + 1, :-1]) # 不要最后一列（pos）
                        data_y.append(hp[hp_index])  
                        hp_index += 1
                assert hp_index == len(hp)
        offset.append(len(data_x))

    return data_x, data_y, offset

# 调用 preprocess 函数生成 data_x 和 data_y
layer_type = "conv2d"  # 示例：处理 conv2d 层
data_x, data_y, offset = preprocess(layer_type)

# 将 data_x 和 data_y 写入新的 HDF5 文件
output_file_path = rf'../autodl-tmp/dataset/{layer_type}.h5'
with h5py.File(output_file_path, 'w') as f:
    # 创建 trace 数据集
    trace_dtype = h5py.vlen_dtype(np.float32)  # 使用变长数据类型存储不同长度的数组
    trace_dataset = f.create_dataset('trace', (len(data_x),), dtype=trace_dtype)
    for i, arr in enumerate(data_x):
        trace_dataset[i] = arr  # 存储每个 numpy 数组

    # 创建 hp 数据集
    hp_dtype = h5py.vlen_dtype(np.float32)  # 使用变长数据类型存储不同长度的数组
    hp_dataset = f.create_dataset('hp', (len(data_y),), dtype=hp_dtype)
    for i, arr in enumerate(data_y):
        hp_dataset[i] = arr  # 存储每个 numpy 数组

print("OK")