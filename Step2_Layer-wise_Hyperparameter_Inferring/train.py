import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import argparse
import MateModel_Hyper
from dataset import RaplLoader
import time
import numpy as np
import loss

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class F1_score(nn.Module):
    def __init__(self, num_classes, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

    def reset(self):
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 1, "y为正确类别数据"
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.num_classes)
        y_pred = F.one_hot(y_pred, self.num_classes)

        self.tp += (y_true * y_pred).sum(0)
        self.tn += ((1 - y_true) * (1 - y_pred)).sum(0)
        self.fp += ((1 - y_true) * y_pred).sum(0)
        self.fn += (y_true * (1 - y_pred)).sum(0)

        precision = self.tp / (self.tp + self.fp + self.epsilon) # 精确率：预测为正的样本中预测正确的比例
        recall = self.tp / (self.tp + self.fn + self.epsilon)  # 召回率：实际为正的样本中预测正确的比例

        accuracy = self.tp.sum() / (self.tp.sum() + self.tn.sum() + self.fp.sum() + self.fn.sum())
        accuracy = accuracy.item() * self.num_classes # 抵消分母的n倍样本量

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.mean().item() 
        return accuracy*100., precision.mean().item()*100., recall.mean().item()*100., f1*100.


def train_step(epoch):
    net.train()

    timer = Timer()
    timer.start()
    train_loss, accuracy, F1, loss1, loss2 = 0, 0, 0, 0, 0
    f1.reset()
    for batch_idx, data in enumerate(trainloader):
        if args.use_domain:
            assert len(data) == 3
            inputs, targets, domain = data[0].to(device).float(), data[1].to(device).long(), data[2].to(device).long()
        else:
            assert len(data) == 2
            inputs, targets = data[0].to(device).float(), data[1].to(device).long()
            domain = None 
        optimizer.zero_grad()
        if args.pretrain:
            pred = net(inputs)
            loss = criterion(pred, targets)
            pred = torch.argmax(pred, dim=1)
        else:
            loss, pred, loss_dis, loss_comp = criterion(net, inputs, targets, domain)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loss1 += loss_dis.item()
        loss2 += loss_comp.item()
        accuracy, p, r, F1 = f1(pred, targets)

        timer.stop()
        if (batch_idx+1) % 100 == 0:
            logs = '{} - Epoch:[{}][{}/{}]\tLoss:{:.3f}\tLoss_Dis:{:.3f}\tLoss_Comp:{:.3f}\tAcc:{:.3f}\tP:{:.3f}\tR:{:.3f}\tF1:{:.3f}\t{:.3f}samples/sec'
            print(logs.format('TRAIN', epoch, (batch_idx+1), len(trainloader), train_loss / (batch_idx + 1), loss1 / (batch_idx + 1), loss2 / (batch_idx + 1),
                              accuracy, p, r, F1, (batch_idx+1) * args.batch_size / timer.sum()))
            timer.start()
    return train_loss / len(trainloader), F1


@torch.no_grad()
def eval_step(epoch, arg, loader):
    net.eval()

    eval_loss, accuracy, F1 = 0, 0, 0
    f1.reset()
    for batch_idx, data in enumerate(loader):
        if args.use_domain:
            assert len(data) == 3
            inputs, targets, domain = data[0].to(device).float(), data[1].to(device).long(), data[2].to(device).long()
        else:
            assert len(data) == 2
            inputs, targets = data[0].to(device).float(), data[1].to(device).long()
            domain = None 
        if args.pretrain:
            pred = net(inputs)
            loss = criterion(pred, targets)
            pred = torch.argmax(pred, dim=1)
        else:
            loss, pred = criterion(net, inputs, targets, domain)

        eval_loss += loss.item()
        accuracy, p, r, F1 = f1(pred, targets)

    logs = '{} - Epoch: [{}]\t Loss: {:.3f}\t Acc: {:.3f}\t P: {:.3f}\t R: {:.3f}\t F1: {:.3f}\t'
    print(logs.format(arg, epoch, eval_loss / len(loader), accuracy, p, r, F1))
    return eval_loss / len(loader), accuracy, F1


def save_step(epoch, acc, test_index, f1, loss):
    global best_f1, best_loss
    if args.pretrain:
        isSave = sum(f1) > sum(best_f1)
    else:
        isSave = sum(loss) < sum(best_loss)
    if isSave:
        print('saving...', end='\n\n')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            "acc": acc, 
            "test_index": test_index,
            "f1": f1,
            "loss": criterion.state_dict(),
            "loss_value": loss
        }    
        if args.pretrain:
            path = args.path + '/' + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + "pretrain" + '_ckpt.pth'
        else:
            if args.use_domain:
                path = args.path + '/' + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + "train_usedomain" + '_ckpt.pth'
            else:
                path = args.path + '/' + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + "train" + '_ckpt.pth'

        if not os.path.exists(args.path):
            os.makedirs(args.path)
        torch.save(state, path)
        print("save_path:" + path)

        best_f1 = f1
        best_loss = loss
    else:
        print("此次epoch, 模型性能没有提高")


def train():
    for epoch in range(start_epoch, start_epoch+args.epochs):
        # epoch += 1
        train_loss, train_acc = train_step(epoch)
        val_loss, val_acc, val_f1 = eval_step(epoch, "VAL", valloader)
        save_step(epoch, [val_acc], data.test_index, [val_f1], [val_loss])
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepTheft Training')
    parser.add_argument("--device", type=str, help="运行的机器")
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--path', default='results/MateModel_Hyper', type=str, help='save_path')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument("--layer_type", default="conv2d", type=str, help="layer_type which hyperParameter is belong to")
    parser.add_argument("--HyperParameter", "-H", default="kernel_size", type=str, help="训练的超参数")   # option: kernel_size, stride, out_channels
    
    parser.add_argument("--pretrain", action="store_true", help="是否为预训练")
    parser.add_argument('--head', default='mlp', type=str, help='mlp or linear head')
    parser.add_argument('--feat_dim', default = 128, type=int, help='feature dim')
    parser.add_argument("--origin_domain_num", "-o", default=4, type=int, help="源域数量")
    parser.add_argument("--use_domain", action="store_true", help="是否使用源域信息") # Deprecated
    
    parser.add_argument("--w", default=1, type=float, help="compLoss的权重")
    parser.add_argument("--temperature", default=0.1, type=float, help="温度系数tao")
    parser.add_argument('--proto_m', default= 0.95, type=float, help='weight of prototype update')
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    # 数据重载
    input_size = ["160", "192", "224", "299", "331"][0 : args.origin_domain_num]

    if args.resume:
        first_train = False #判断是否第一次正式训练
        if args.pretrain:
            # 重载预训练
            path = args.path + '/' + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + "pretrain" + '_ckpt.pth'
            checkpoint = torch.load(path)
            test_index = checkpoint["test_index"] 
            data = RaplLoader(args, input_size=input_size, test_index = test_index)
        else:
            # 重载正式训练
            if args.use_domain:
                path = args.path + '/' + args.HyperParameter + "_" + str(args.origin_domain_num) + "_train_usedomain" + '_ckpt.pth' 
            else:
                path = args.path + '/' + args.HyperParameter + "_" + str(args.origin_domain_num) + "_train" + '_ckpt.pth' 
            if not os.path.exists(path):
                # 正式训练未进行，使用预训练参数
                first_train = True # 第一次正式训练
                path = args.path + '/' + args.HyperParameter + "_" + str(args.origin_domain_num) + "_pretrain" + '_ckpt.pth'
            checkpoint = torch.load(path)
            test_index = checkpoint["test_index"] 
            data = RaplLoader(args, input_size=input_size, test_index = test_index)
    else:
        # 初始预训练
        assert args.pretrain == True, "正式训练需要加载预训练数据"
        data = RaplLoader(args, input_size=input_size) 
    args.num_classes = data.num_classes
    print("load_path:" + path)
        
    trainloader, valloader = data.get_loader()
    if args.pretrain:
        net = MateModel_Hyper.Model(args=args).to(device) 
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        net = MateModel_Hyper.Model(args=args).to(device) 
        criterion = loss.Loss(args, net, valloader).to(device)

    # 模型重载
    if args.resume:
        print('Loading...')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint["f1"]
        if first_train:
            # 第一次正式训练
            best_acc = [0]
            best_f1 = [0]
            best_loss = [float("inf")] # 正式训练的总loss
        else:
            # 正式训练
            criterion.load_state_dict(checkpoint["loss"])
            best_loss = checkpoint["loss_value"]
            best_acc = checkpoint["acc"]
            best_f1 = checkpoint["f1"]
    else:
        # 预训练
        best_acc = [0]
        start_epoch = 0
        best_f1 = [0]

    f1 = F1_score(num_classes=data.num_classes) # y_pred y_true

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train()
