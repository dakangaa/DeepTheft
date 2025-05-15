import MateModel_Hyper
import torch
import dataset
import argparse
from utils import F1_score, Timer
import numpy as np

# 对未知input_size测试
def eval(epoch, args, loader, prototypes):
    net.eval()
    timer = Timer()
    timer.start()
    with torch.no_grad():
        # accuracy, p, r, F1 = 0, 0, 0, 0
        metrics_sum = np.zeros(4)
        f1.reset()
        if args.regression:
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device).float(), targets.to(device).long()
                features = net(inputs)
                if args.layer_type == "conv2d":
                    lower = 0
                    upper = 5
                    feat_dot_prototype = torch.matmul(features, prototypes) # -1 ~ 1
                    pred = torch.round((feat_dot_prototype + 1) / 2 * (upper - lower)).long()
                elif args.layer_type == "linear":
                    lower = 0 #1000
                    upper = 1 #4096
                    feat_dot_prototype = torch.matmul(features, prototypes) # -1 ~ 1
                    pred = (feat_dot_prototype > 0).long()
                if (batch_idx+1)%100 == 0:
                    timer.stop()
                    print(f"[{batch_idx+1}/{len(loader)}] : {batch_idx*args.batch_size/timer.sum():.3f}samples/sec")
                    timer.start()
                metrics_sum = np.array(f1(pred, targets))

        else:
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device).float(), targets.to(device).long()
                features = net(inputs)
                feat_dot_prototypes = torch.matmul(features, prototypes.T)
                pred = feat_dot_prototypes.max(1)[1]
                if (batch_idx+1)%100 == 0:
                    timer.stop()
                    print(f"[{batch_idx+1}/{len(loader)}] : {batch_idx*args.batch_size/timer.sum():.3f}samples/sec")
                    timer.start()
                metrics_sum = np.array(f1(pred, targets))

    logs = '{} - TrainEpoch:[{}]\t Acc:{:.3f}\t P:{:.3f}\t R:{:.3f}\t F1:{:.3f}\t'
    print(logs.format(args.mode, epoch, metrics_sum[0], metrics_sum[1], metrics_sum[2], metrics_sum[3]))
    return metrics_sum[3]

parser = argparse.ArgumentParser(description='Test on unknown input_size')
parser.add_argument('--path', default='results/de_Lvar', type=str, help='load_path')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--batch_size', default=1280, type=int, help='mini-batch size')
parser.add_argument("--layer_type", type=str, help="layer_type which hyperParameter is belong to")
parser.add_argument("--HyperParameter", "-H", default="kernel_size", type=str, help="测试的超参数")   # option: kernel_size, stride, out_channels
parser.add_argument("--origin_domain_num", "-o", default=1, type=int, help="训练的源域数量")
parser.add_argument('--head', default='mlp', type=str, help='mlp or linear head')
parser.add_argument('--feat_dim', default = 128, type=int, help='feature dim')
parser.add_argument("--device", type=str, default="laptop", help="laptop or autodl")
parser.add_argument("--test_domain", default="331", type=str, help="目标域")
parser.add_argument("--regression", action="store_true", help="是否为回归任务")
args = parser.parse_args()
args.mode = "TEST"

if args.HyperParameter != "out_channels":
    assert args.regression == False # 除了out_channels都不需要回归任务
device = torch.device("cuda")

print("Loading data...")
data = dataset.RaplLoader(args, input_size = [args.test_domain], no_val=True)
test_loader = data.get_loader()
args.num_classes = data.num_classes

print("Loading Model...")
if args.regression:
    path = args.path + '/' + args.layer_type + "_" + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + args.test_domain + "_" + "regression" + '_ckpt.pth'
else:
    path = args.path + '/' + args.layer_type + "_" + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + args.test_domain + "_" + "train" + '_ckpt.pth'
print(f"load path : {path}")
check_point = torch.load(path, weights_only=False)
if args.regression:
    prototypes = torch.zeros(args.feat_dim).cuda()
    prototypes[0] = 1
else:
    prototypes = check_point["loss"]["disLoss.prototypes"]
net = MateModel_Hyper.Model(args, input_channels=2)
net.load_state_dict(check_point["net"])
net.to(device)
last_acc = check_point["acc"]
train_epoch = check_point["epoch"]
f1 = F1_score(num_classes=data.num_classes)
eval(train_epoch, args, test_loader, prototypes)
