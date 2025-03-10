import MateModel_Hyper
import torch
import dataset
import train
import torch.nn as nn
import argparse

# 对未知input_size测试
def eval(epoch, args, loader, prototypes):
    net.eval()
    timer = train.Timer()
    timer.start()
    with torch.no_grad():
        accuracy, F1 = 0, 0
        f1.reset()
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device).float(), targets.to(device).long()
            features = net(inputs)
            feat_dot_prototypes = torch.matmul(features, prototypes.T)
            pred = feat_dot_prototypes.max(1)[1]
            accuracy, p, r, F1 = f1(pred, targets)
            if (batch_idx+1)%100 == 0:
                timer.stop()
                print(f"[{batch_idx+1}/{len(loader)}] : {batch_idx*args.batch_size/timer.sum():.3f}samples/sec")
                timer.start()
            
    logs = '{} - TrainEpoch:[{}]\t Acc:{:.3f}\t P:{:.3f}\t R:{:.3f}\t F1:{:.3f}\t'
    print(logs.format(args.mode, epoch, accuracy, p, r, F1))
    return F1

parser = argparse.ArgumentParser(description='Test on unknown input_size')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--batch_size', default=1280, type=int, help='mini-batch size')
parser.add_argument('--input_size', default="192", type=str, help='test input_size') # TEST domain
parser.add_argument("--layer_type", default="conv2d", type=str, help="layer_type which hyperParameter is belong to")
parser.add_argument("--HyperParameter", "-H", default="kernel_size", type=str, help="测试的超参数")   # option: kernel_size, stride, out_channels
parser.add_argument("--origin_domain_num", "-o", default=1, type=int, help="训练的源域数量")
parser.add_argument("--use_domain", action="store_true", help="是否使用源域信息") # Deprecated
parser.add_argument('--head', default='mlp', type=str, help='mlp or linear head')
parser.add_argument('--feat_dim', default = 128, type=int, help='feature dim')
parser.add_argument("--device", type=str, default="laptop", help="laptop or autodl")

args = parser.parse_args()
args.pretrain = False
args.mode = "TEST"
device = torch.device("cuda")

print("Loading data...")
data = dataset.RaplLoader(args, input_size = [args.input_size], is_test=True)
test_loader = data.get_loader()
args.num_classes = data.num_classes

print("Loading Model...")
path = 'results/MateModel_Hyper' + '/' + args.HyperParameter + "_" + str(args.origin_domain_num) + "_train" + '_ckpt.pth'
print(f"load file path:{path}")
check_point = torch.load(path) 
prototypes = check_point["loss"]["disLoss.prototypes"]
net = MateModel_Hyper.Model(args, input_channels=2)
net.load_state_dict(check_point["net"])
net.to(device)
last_acc = check_point["acc"]
train_epoch = check_point["epoch"]
f1 = train.F1_score(num_classes=data.num_classes)
eval(train_epoch, args, test_loader, prototypes)
