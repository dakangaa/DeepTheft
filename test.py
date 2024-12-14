# 查看h5文件结构
import h5py
data = h5py.File(r'/root/autodl-tmp/dataset/data.h5', 'r')
def print_structure(obj):
    if isinstance(obj, h5py.Group):
        for k in obj.keys():
            # print(obj[k].name, ":\t", obj[k])
            print_structure(obj[k])
            if isinstance(obj[k], h5py.Dataset):
                name_split = k.split(")")
                name2 = name_split[0]+")"+name_split[1]
                if name2 not in _:
                    _[name2] = 1
                else:
                    _[name2] += 1      

_ = dict()
print_structure(data)
print(_)