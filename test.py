# 查看h5文件结构
import h5py
file_data = h5py.File(r'dataset/data.h5', 'r')
file_hp = h5py.File(r'dataset/hp.h5', 'r')

conv2d_index = {'conv2d': 0, 'batch_norm': 1, 'relu_': 2,
               'max_pool2d': 3, 'adaptive_avg_pool2d': 4,
               'linear': 5, 'add_': 6, '_': 7}["conv2d"]
kernel_size_index = {'in_channels': 0, 'out_channels': 1, 'kernel_size': 2,
                'stride': 3, 'padding': 4, 'dilation': 5,
                'groups': 6, 'input_size': 7, 'output_size': 8}["kernel_size"]
layerhp_indexes = {0, 3, 5}
distribution = dict()

for k in file_data["data"].keys(): # 遍历每个架构
     if k.split(")")[1] == "192": # 限制输入大小为192×192
        layer_size = file_data["position"][k].shape[0]
        layer_index = 0 # 架构中的每一层
        layerhp_index = 0 # 有超参数的层
        while layer_index < layer_size:
            begin = file_data["position"][k][layer_index, 0] 
            layer = file_data["data"][k][begin, 3]
            if layer in layerhp_indexes:
                if layer == conv2d_index: # 限制卷积层
                    # 统计频率
                    kernel_size = file_hp[k][layerhp_index, kernel_size_index]
                    if kernel_size in distribution.keys():
                        distribution[kernel_size] += 1
                    else:
                        distribution[kernel_size] = 1
                layerhp_index += 1
            layer_index += 1

print(distribution)                
