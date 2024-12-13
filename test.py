import h5py
data = h5py.File(r'/root/autodl-tmp/dataset/data.h5', 'r')
def print_structure(name, obj, deepth):
    print(f"{name}: {type(obj)}")
    if isinstance(obj, h5py.Group) and deepth < 4:
        if deepth == 2 :
            set_nettype = set()
            set_num1 = set()
            set_num2 = set()
            for key in obj:
                strList = key.split(")")
                set_nettype.add(strList[0])
                num1 = int(strList[1])
                num2 = int(strList[2])
                set_num1.add(num1)
                set_num2.add(num2)
            print(set_nettype)
            print(set_num1)
            print("size_num2", len(set_num2))
        else:
            for key in obj:
                print_structure(f"{name}/{key}", obj[key], deepth+1)

print_structure('/', data, 1)