import MateModel_Hyper
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
        path = args.path + '/' + args.layer_type + "_" + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + test_domain_str + "_" + "train" + '_ckpt.pth'
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
    return eval(train_epoch, args, test_loader, prototypes, net, device, f1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test on unknown input_size')
    # test
    parser.add_argument("--layer_type", type=str, default="conv2d", help="layer_type which hyperParameter is belong to")
    parser.add_argument("--HyperParameter", "-H", default="stride", type=str, help="测试的超参数")   # option: kernel_size, stride, out_channels
    parser.add_argument("--origin_domain_num", "-o", default=4, type=int, help="训练的源域数量")
    parser.add_argument("--test_domain", default="331", type=str, help="目标域")

    # data
    parser.add_argument('--path', default='results/MateModel_Hyper', type=str, help='load_path')
    parser.add_argument('--data_path', default='dataset/new_dataset', type=str)
    parser.add_argument('--prefetch_factor', default=2, type=int, help='prefetch number of one loader worker')
    parser.add_argument('--workers', default=3, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=1280, type=int, help='mini-batch size')

    # model
    parser.add_argument('--head', default='mlp', type=str, help='mlp or linear head')
    parser.add_argument('--feat_dim', default = 128, type=int, help='feature dim')

    args = parser.parse_args()
    ws = [0.1, 0.5, 1.0, 2.0, 5.0]
    proto_ms = [0.5, 0.8, 0.9, 0.95, 0.99]
    ts = [0.01, 0.05, 0.1, 0.2, 0.5]

    test_domains = ["160", "192", "224", "299", "331"]
    table = {"test_domain": test_domains}
    mean_row = {
        'test_domain': 'mean'
    }
    model_dir = "results/ts"
    for t in ts:
        args.path = os.path.join(model_dir, f"{t:.2f}")
        # accs = list()
        f1s = list()
        for args.test_domain in test_domains:
            _1, _2 = test(args)
            # accs.append(_1)
            f1s.append(_2)
        table[f"t={t:.2f}"] = f1s
        mean_row[f"t={t:.2f}"] = np.mean(f1s)




    # 保存到 Excel
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(table)
    df.loc[len(df)] = mean_row
    excel_path = 'results/parameter_sensitivity.xlsx'
    try:
        if os.path.exists(excel_path):
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, index=False, sheet_name="t")
        else:
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
                df.to_excel(writer, index=False, sheet_name="t")
        print(f"Saved results to {excel_path}")
    except Exception as e:
        print(f"Failed to write Excel file: {e}")
