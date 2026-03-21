import torch
import torch.backends.cudnn as cudnn
import os
import argparse
import MateModel_Hyper
from dataset import RaplLoader
import loss
import utils
import torch.nn as nn
import numpy as np
import sys


train_timer = utils.Timer()

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def train_step(epoch, net, trainloader, criterion, optimizer, f1, device):
    net.train()

    train_timer.start()
    metrics = np.zeros(5) # train_loss, accuracy, p, r, F1
    f1.reset()
    for batch_idx, data in enumerate(trainloader):
        # if args.use_domain:
        #     assert len(data) == 3
        #     inputs, targets, domain = data[0].to(device).float(), data[1].to(device).long(), data[2].to(device).long()
        # else:
        assert len(data) == 2
        inputs, targets = data[0].to(device).float(), data[1].to(device).long()
        domain = None
        optimizer.zero_grad()
        if args.regression:
            features = net(inputs)
            loss, pred = criterion(features, targets)
        else:
            loss, pred, loss_dis, loss_comp = criterion(net, inputs, targets, domain)
        loss.backward()
        optimizer.step()

        metrics[0] = loss.item()
        metrics[1:] = f1(pred, targets)#accuracy, p, r, F1

        if (batch_idx+1) % 1000 == 0:
            elapsed = train_timer.stop()
            logs = '{} - Epoch:[{}][{}/{}]\tLoss:{:.3f}\tAcc:{:.3f}\tP:{:.3f}\tR:{:.3f}\tF1:{:.3f}\t{:.3f}samples/sec'
            print(logs.format('TRAIN', epoch, (batch_idx+1), len(trainloader), metrics[0],
                              metrics[1], metrics[2], metrics[3], metrics[4],
                                1000 * args.batch_size / elapsed))
            print("\n")
            train_timer.start()
            f1.reset()
    train_timer.stop()
    return metrics[0], metrics[4]


@torch.no_grad()
def eval_step(epoch, arg, loader, net, criterion, f1, device):
    net.eval()

    metrics = np.zeros(5) # train_loss, accuracy, p, r, F1
    f1.reset()
    for batch_idx, data in enumerate(loader):
        # if args.use_domain:
        #     assert len(data) == 3
        #     inputs, targets, domain = data[0].to(device).float(), data[1].to(device).long(), data[2].to(device).long()
        # else:
        assert len(data) == 2
        inputs, targets = data[0].to(device).float(), data[1].to(device).long()
        domain = None
        if args.regression:
            features = net(inputs)
            loss, pred = criterion(features, targets)
        else:
            loss, pred, _, _ = criterion(net, inputs, targets, domain)

        metrics[0] = loss.item()
        metrics[1:] = f1(pred, targets)#accuracy, p, r, F1

    eval_loss, accuracy, p, r, F1 = metrics[:]
    logs = '{} - Epoch: [{}]\t Loss: {:.3f}\t Acc: {:.3f}\t P: {:.3f}\t R: {:.3f}\t F1: {:.3f}\t'
    print(logs.format(arg, epoch, eval_loss, accuracy, p, r, F1))
    return eval_loss, accuracy, F1


def save_step(epoch, acc, f1, loss, net, criterion, optimizer, scheduler):
    global best_f1, best_loss
    if f1 > best_f1:
        print('saving...')
        state = {
            'net': net.state_dict(),
            'epoch': epoch+1,
            "acc": acc,
            "f1": f1,
            "loss": criterion.state_dict(),
            "loss_value": loss,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if args.regression:
            path = args.path + '/' + args.layer_type + "_" + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + args.test_domain + "_" + "regression" + '_ckpt.pth'
        else:
            path = args.path + '/' + args.layer_type + "_" + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + args.test_domain + "_" + "train" + '_ckpt.pth'
        print("save path:" + path)

        if not os.path.exists(args.path):
            os.makedirs(args.path)
        torch.save(state, path)

        best_f1 = f1
        best_loss = loss
    else:
        if args.regression:
            path = args.path + '/' + args.layer_type + "_" + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + args.test_domain + "_" + "regression" + '_ckpt.pth'
        else:
            path = args.path + '/' + args.layer_type + "_" + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + args.test_domain + "_" + "train" + '_ckpt.pth'

        if not os.path.exists(args.path):
            os.makedirs(args.path)
        last_checkpoint = torch.load(path, weights_only=False)
        last_checkpoint["epoch"] = epoch + 1
        torch.save(last_checkpoint, path)
        print("此次epoch, 模型性能没有提高")

def train(args, net, trainloader, valloader, criterion, optimizer, scheduler, f1, device):
    start_epoch = scheduler.last_epoch + 1      # 已经跑过的 epoch
    max_epoch   = scheduler.T_max
    for epoch in range(start_epoch, min(start_epoch + args.epochs, max_epoch)):
        print(f">>>>>>>>>>>>>>>>>> EPOCH {epoch} <<<<<<<<<<<<<<<<<<")
        print(f"lr:{scheduler.get_last_lr()}")
        train_loss, train_acc = train_step(epoch, net, trainloader, criterion, optimizer, f1, device)
        val_loss, val_acc, val_f1 = eval_step(epoch, "VAL", valloader, net, criterion, f1, device)
        save_step(epoch, val_acc, val_f1, val_loss, net, criterion, optimizer, scheduler)
        scheduler.step()
        print("\n")

def experiment(args):
    if args.log:
        log_file_path = f"{args.log_path}/{args.layer_type}_{args.HyperParameter}_{args.origin_domain_num}_{args.test_domain}_w{args.w:.1f}_p{args.proto_m:.2f}_t{args.temperature:.2f}.log"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        original_stdout = sys.stdout
        log_file = open(log_file_path, 'a')
        sys.stdout = Tee(sys.stdout, log_file)

    print("--------------------- EXP START ---------------------")
    if args.HyperParameter != "out_channels":
        args.regression = False # 除了out_channels都不需要回归任务
    else:
        args.regression = True
    learning_rate = {"kernel_size":0.001, "stride":0.001, "out_channels":0.001, "padding":0.001}
    args.lr = learning_rate[args.HyperParameter]
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    # 设定源域
    input_size = ["160", "192", "224", "299", "331"]
    input_size = [i for i in input_size if i != args.test_domain][0 : args.origin_domain_num]

    data = RaplLoader(args, no_val=False, input_size=input_size)
    args.num_classes = data.num_classes

    trainloader, valloader = data.get_loader()
    net = MateModel_Hyper.Model(args=args).to(device)
    if args.regression:
        criterion = loss.RegressionLoss(args).to(device)
    else:
        criterion = loss.ClassificationLoss(args, net, valloader).to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    # 模型重载
    if args.resume:
        if args.regression:
            # 重载预训练
            path = args.path + '/' + args.layer_type + "_" + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + args.test_domain + "_" + "regression" + '_ckpt.pth'
        else:
            path = args.path + '/' + args.layer_type + "_" + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + args.test_domain + "_" + "train" + '_ckpt.pth'
        print("load path:" + path)

        global best_f1, best_loss
        checkpoint = torch.load(path, weights_only=False)
        print('Loading...')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['acc']
        best_f1 = checkpoint["f1"]
        criterion.load_state_dict(checkpoint["loss"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']) # 从上一次的最佳checkpoint开始
        if scheduler.T_max != args.max_epochs:
            print(f"WARNING: loaded scheduler's max_epoch {scheduler.T_max} is different from args.max_epochs {args.max_epochs}, using args.max_epochs")
            scheduler.T_max = args.max_epochs
        best_loss = checkpoint["loss_value"]
        print(f"best_acc:{best_acc:.2f} best_f1:{best_f1:.2f}")
    else:
        start_epoch = -1
        best_acc = 0
        best_f1 = 0
        best_loss = [float("inf")]

    f1 = utils.F1_score(num_classes=data.num_classes) # y_pred y_true

    train(args, net, trainloader, valloader, criterion, optimizer, scheduler, f1, device)

    if args.log:
        sys.stdout = original_stdout
        log_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepTheft Training')
    # training
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--epochs', default=30, type=int, help='number of epochs to run')
    parser.add_argument("--max_epochs", default=30, type=int, help="total num of epochs")

    # data
    parser.add_argument('--path', default='results/MateModel_Hyper', type=str, help='save_path')
    parser.add_argument('--data_path', default='dataset/new_dataset', type=str)
    parser.add_argument('--workers', default=3, type=int, help='number of data loading workers')
    parser.add_argument('--prefetch_factor', default=2, type=int, help='prefetch number of one loader worker')

    # log
    parser.add_argument('--log_path', default='results/log', type=str)
    parser.add_argument('--log', action='store_true', help='save log')


    # experiment
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument("--layer_type", type=str, default="linear", help="layer_type which hyperParameter is belong to")
    parser.add_argument("--HyperParameter", "-H", default="out_channels", type=str, help="训练的超参数")   # option: kernel_size, stride, out_channels
    parser.add_argument("--test_domain", default="331", type=str, help="目标域")
    parser.add_argument("--origin_domain_num", "-o", default=1, type=int, help="源域数量") # 源域在除了测试域的剩余域中顺序取

    # model
    parser.add_argument('--head', default='mlp', type=str, help='mlp or linear head')
    parser.add_argument('--feat_dim', default = 128, type=int, help='feature dim')
    parser.add_argument("-w", default=1.0, type=float, help="compLoss的权重")
    parser.add_argument("--temperature", default=0.1, type=float, help="温度系数tao")
    parser.add_argument('--proto_m', default= 0.95, type=float, help='momentum of prototype update') # 论文中的alpha

    args = parser.parse_args()
    args.resume = False

    args.log = True

    args.layer_type = "conv2d"
    args.HyperParameter = "stride"

    ws = [0.1, 0.5, 1.0, 2.0, 5.0]
    proto_ms = [0.5, 0.8, 0.9, 0.95, 0.99]
    ts = [0.01, 0.05, 0.1, 0.2, 0.5]
    test_domains = ["160", "192", "224", "299", "331"]

    args.origin_domain_num = 4

    # args.w = 1.0
    # args.temperature = 0.1
    # args.proto_m = 0.95
    # # for args.w in ws:
    # #     for args.test_domain in test_domains:
    # #         print("w:" + str(args.w))
    # #         args.path = "results/ws/" + f"{args.w:.1f}"
    # #         experiment(args)

    #p0.7不稳定，训练p0.8
    args.w = 1.0
    args.temperature = 0.1
    args.proto_m = 0.8
    for args.test_domain in test_domains:
        print("proto_m:" + str(args.proto_m))
        args.path = "results/proto_ms/" + f"{args.proto_m:.2f}"
        experiment(args)

    # t0.01loss=nan，修改后重新训练
    args.w = 1.0
    args.temperature = 0.01
    args.proto_m = 0.95
    for args.test_domain in test_domains:
        print("temperature:" + str(args.temperature))
        args.path = "results/ts/" + f"{args.temperature:.2f}"
        experiment(args)

    # p0.99收敛慢，增加训练epoch到40
    args.resume = True
    args.w = 1.0
    args.temperature = 0.1
    args.proto_m = 0.99
    args.epochs = 40
    args.max_epochs = 40
    for args.test_domain in test_domains:
        print("[resume] proto_m:" + str(args.proto_m))
        args.path = "results/proto_ms/" + f"{args.proto_m:.2f}"
        experiment(args)
