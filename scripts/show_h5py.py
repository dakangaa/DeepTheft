import h5py

def print_hdf5_structure(file, indent=0):
    """
    递归打印 HDF5 文件的结构（组和数据集）。
    
    :param file: HDF5 文件或组对象
    :param indent: 缩进级别（用于格式化输出）
    """
    for key in file.keys():
        item = file[key]
        # 打印当前对象的名称和类型
        if isinstance(item, h5py.Group):
            print(' ' * indent + f"Group: {key}")
            # 递归遍历子组
            print_hdf5_structure(item, indent + 4)
        elif isinstance(item, h5py.Dataset):
            print(' ' * indent + f"Dataset: {key} (shape={item.shape}, dtype={item.dtype})")
        else:
            print(' ' * indent + f"Unknown: {key}")

# 打开 HDF5 文件
file_path = '../autodl-tmp/dataset/conv2d.h5'
with h5py.File(file_path, 'r') as f:
    print(f"文件结构: {file_path}")
    print_hdf5_structure(f)