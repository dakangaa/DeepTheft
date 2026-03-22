import model
import torch
import dataset
import argparse
from utils import F1_score, Timer
import numpy as np
import os
import pandas as pd

# 对未知input_size测试
def eval(epoch, args, loader, prototypes, net, device, f1):
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
    return metrics_sum[0], metrics_sum[3] #acc, f1



def test(args):
    args.mode = "TEST"

    if args.HyperParameter != "out_channels":
        args.regression = False # 除了out_channels都不需要回归任务
    else:
        args.regression = True
    device = torch.device("cuda")

    print("Loading data...")
    data = dataset.RaplLoader(args, input_size = [args.test_domain], no_val=True)
    test_loader = data.get_loader()
    args.num_classes = data.num_classes

    print("Loading Model...")
    if args.origin_domain_num == 4:
        test_domain_str = args.test_domain
    else:
        test_domain_str = "331"
    if args.regression:
        path = args.path + '/' + args.layer_type + "_" + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + test_domain_str + "_" + "regression" + '_ckpt.pth'
    else:
        path = args.path + '/' + args.layer_type + "_" + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + test_domain_str + "_" + "classification" + '_ckpt.pth'
    print(f"load path : {path}")
    check_point = torch.load(path, weights_only=False)
    if args.regression:
        prototypes = torch.zeros(args.feat_dim).cuda()
        prototypes[0] = 1
    else:
        prototypes = check_point["loss"]["LossSep.prototypes"]
    net = model.Model(args, input_channels=2)
    net.load_state_dict(check_point["net"])
    net.to(device)
    last_acc = check_point["acc"]
    train_epoch = check_point["epoch"]
    f1 = F1_score(num_classes=data.num_classes)
    return eval(train_epoch, args, test_loader, prototypes, net, device, f1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test on unknown input_size')
    # test
    parser.add_argument("--layer_type", type=str, default="conv2d", help="layer_type which hyperParameter is belong to, should be one of conv2d, max_pool2d, linear")
    parser.add_argument("--HyperParameter", "-H", default="out_channels", type=str, help="hyperparameter to predict")   # option: kernel_size, stride, out_channels
    parser.add_argument("--test_domain", default="331", type=str, help="target domain for testing, should be one of 160, 192, 224, 299, 331")
    parser.add_argument("--origin_domain_num", "-o", default=4, type=int, help="number of origin domains") # 源域在除了测试域的剩余域中顺序取

    # data
    parser.add_argument('--path', default='results/MateModel_Hyper', type=str, help='save path for checkpoint')
    parser.add_argument('--data_path', default='dataset/new_dataset', type=str, help='path for dataset')
    parser.add_argument('--workers', default=3, type=int, help='number of data loading workers')
    parser.add_argument('--prefetch_factor', default=2, type=int, help='prefetch number of one loader worker')
    parser.add_argument('--batch_size', default=1280, type=int, help='mini-batch size')

    # model
    parser.add_argument('--feat_dim', default = 128, type=int, help='feature dim')


    args = parser.parse_args()

    acc, f1 = test(args)
