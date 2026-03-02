'''
    用于改变pth文件的结构:将list改成scalar value,添加optimizer和scheduler
'''
import torch
import argparse
import re
import MateModel_Hyper
from pathlib import Path



def show_pth(path):
    checkpoint = torch.load(path, weights_only=False)
    print(checkpoint.keys())
    print(checkpoint["epoch"])
    print(checkpoint["acc"])
    print(checkpoint["f1"])
    print(checkpoint["loss_value"])
    # print(checkpoint[""])

def convert_pth(path):
    checkpoint = torch.load(path, weights_only=False)
    if isinstance(checkpoint["acc"], list):
        checkpoint["acc"] = checkpoint["acc"][0]
        checkpoint["f1"] = checkpoint["f1"][0]
        checkpoint["loss_value"] = checkpoint["loss_value"][0]

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.feat_dim = 128
    args.head = "mlp"
    layer_type = re.split('[_/]', path)[8]
    if layer_type == "conv2d":
        hyperParameter = re.split('[_/]', path)[9]
        args.num_classes = {'out': 6, 'kernel': 3, 'stride': 2}[hyperParameter]
    elif layer_type == "max":
        hyperParameter = re.split('[_/]', path)[10]
        args.num_classes = {'kernel': 2, "padding":2}[hyperParameter]
    elif layer_type == "linear":
        args.num_classes = 2 #1000, 4096
    else:
        raise BaseException
    device = torch.device('cuda')
    net = MateModel_Hyper.Model(args=args).to(device)
    net.load_state_dict(checkpoint["net"])
    
    start_epoch = checkpoint["epoch"]

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20,
    )
    for _ in range(start_epoch):
        scheduler.step()

    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["scheduler"] = scheduler.state_dict()

    torch.save(checkpoint, path)


folder = Path("/home/dacom/Code/DeepTheft/results/MateModel_Hyper")

for file in folder.rglob("*"):
    if file.is_file():
        path = str(file)
        show_pth(path)
        convert_pth(path)
        show_pth(path)