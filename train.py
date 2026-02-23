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

train_timer = utils.Timer()

def train_step(epoch):
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
def eval_step(epoch, arg, loader):
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


def save_step(epoch, acc, f1, loss):
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

def train():
    for epoch in range(start_epoch, start_epoch+args.epochs):
        print(f">>>>>>>>>>>>>>>>>> EPOCH {epoch} <<<<<<<<<<<<<<<<<<")
        print(f"lr:{scheduler.get_last_lr()}")
        train_loss, train_acc = train_step(epoch)
        val_loss, val_acc, val_f1 = eval_step(epoch, "VAL", valloader)
        save_step(epoch, val_acc, val_f1, val_loss)
        scheduler.step()
        print("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepTheft Training')
    # training
    parser.add_argument("--device", type=str, help="运行的机器")
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs to run')
    parser.add_argument("--max_epochs", default=20, type=int, help="total num of epochs")
    
    # data
    parser.add_argument('--path', default='results/MateModel_Hyper', type=str, help='save_path')
    parser.add_argument('--data_path', default='dataset/new_dataset', type=str)
    parser.add_argument('--workers', default=3, type=int, help='number of data loading workers')
    parser.add_argument('--prefetch_factor', default=2, type=int, help='prefetch number of one loader worker')
    
    # experiment
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument("--layer_type", type=str, help="layer_type which hyperParameter is belong to")
    parser.add_argument("--HyperParameter", "-H", default="kernel_size", type=str, help="训练的超参数")   # option: kernel_size, stride, out_channels
    parser.add_argument("--test_domain", default="331", type=str, help="目标域")
    parser.add_argument("--origin_domain_num", "-o", default=4, type=int, help="源域数量") # 源域在除了测试域的剩余域中顺序取

    # model
    parser.add_argument('--head', default='mlp', type=str, help='mlp or linear head')
    parser.add_argument('--feat_dim', default = 128, type=int, help='feature dim')
    parser.add_argument('--regression', action="store_true", help="是否为回归任务")
    parser.add_argument("-w", default=1, type=float, help="compLoss的权重")
    parser.add_argument("--temperature", default=0.1, type=float, help="温度系数tao")
    parser.add_argument('--proto_m', default= 0.95, type=float, help='momentum of prototype update')

    args = parser.parse_args()
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
        criterion = loss.Loss(args, net, valloader).to(device)

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

        checkpoint = torch.load(path, weights_only=False)
        print('Loading...')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['acc']
        best_f1 = checkpoint["f1"]
        criterion.load_state_dict(checkpoint["loss"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_loss = checkpoint["loss_value"]
        print(f"best_acc:{best_acc:.2f} best_f1:{best_f1:.2f}")
    else:
        start_epoch = -1
        best_acc = 0
        best_f1 = 0
        best_loss = [float("inf")]

    f1 = utils.F1_score(num_classes=data.num_classes) # y_pred y_true

    train()
