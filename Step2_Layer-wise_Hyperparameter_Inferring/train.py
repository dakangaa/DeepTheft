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
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.num_classes)
        y_pred = F.one_hot(torch.argmax(y_pred, dim=1), self.num_classes)

        self.tp += (y_true * y_pred).sum(0)
        self.tn += ((1 - y_true) * (1 - y_pred)).sum(0)
        self.fp += ((1 - y_true) * y_pred).sum(0)
        self.fn += (y_true * (1 - y_pred)).sum(0)

        precision = self.tp / (self.tp + self.fp + self.epsilon) # 精确率：预测为正的样本中预测正确的比例
        recall = self.tp / (self.tp + self.fn + self.epsilon)  # 召回率：实际为正的样本中预测正确的比例

        accuracy = self.tp.sum() / (self.tp.sum() + self.tn.sum() + self.fp.sum() + self.fn.sum())
        accuracy = accuracy.item() * self.num_classes # 乘以类别？

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.mean().item() #类别与样本无关，每一种类别出现的概率是一样的
        return accuracy*100., precision.mean().item()*100., recall.mean().item()*100., f1*100.


def train_step(epoch):
    net.train()

    timer = Timer()
    timer.start()
    train_loss, accuracy, F1 = 0, 0, 0
    f1.reset()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device).float(), targets.to(device).long()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        accuracy, p, r, F1 = f1(outputs, targets)

        timer.stop()
        if (batch_idx+1) % 100 == 0:
            logs = '{} - Epoch: [{}][{}/{}]\t Loss: {:.3f}\t Acc: {:.3f}\t P: {:.3f}\t R: {:.3f}\t F1: {:.3f}\t {:.3f}samples/sec'
            print(logs.format('TRAIN', epoch, (batch_idx+1), len(trainloader), train_loss / (batch_idx + 1), accuracy,
                              p, r, F1, (batch_idx+1) * args.batch_size / timer.sum()))
            timer.start()
    return train_loss / len(trainloader), F1


@torch.no_grad()
def eval_step(epoch, arg, loader):
    net.eval()

    eval_loss, accuracy, F1 = 0, 0, 0
    f1.reset()
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device).float(), targets.to(device).long()

        outputs = net(inputs)
        loss = CEloss(outputs, targets)

        eval_loss += loss.item()
        accuracy, p, r, F1 = f1(outputs, targets)

    logs = '{} - Epoch: [{}]\t Loss: {:.3f}\t Acc: {:.3f}\t P: {:.3f}\t R: {:.3f}\t F1: {:.3f}\t'
    print(logs.format(arg, epoch, eval_loss / len(loader), accuracy, p, r, F1))
    return eval_loss / len(loader), F1


def save_step(epoch, acc, test_index):
    global best_acc
    if sum(acc) > sum(best_acc):
        print('saving...', end='\n\n')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            "acc": acc, 
            "test_index": test_index
        }
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        torch.save(state, args.path + '/' + args.HyperParameter + '_ckpt.pth')
        best_acc = acc
    else:
        print("此次epoch, 精确度没有提高")


def train():
    for epoch in range(start_epoch, start_epoch+args.epochs):
        # epoch += 1
        train_loss, train_acc = train_step(epoch)
        val_loss, val_acc = eval_step(epoch, "VAL", valloader)
        save_step(epoch, [val_acc], data.test_index)
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepTheft Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--path', default='results/MateModel_Hyper', type=str, help='save_path')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument("--HyperParameter", "-H", default="kernel_size", type=str, help="训练的超参数")   # option: kernel_size, stride, out_channels
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if args.resume:
        checkpoint = torch.load(args.path + '/' + args.HyperParameter + '_ckpt.pth', weights_only=True)
        test_index = checkpoint["test_index"] 
        data = RaplLoader(batch_size=args.batch_size, num_workers=args.workers, mode=args.HyperParameter, test_index = test_index) # ?mode:一次训练一个超参数模型?
    else:
        data = RaplLoader(batch_size=args.batch_size, num_workers=args.workers, mode=args.HyperParameter) # ?mode:一次训练一个超参数模型?
        
    trainloader, valloader = data.get_loader()

    net = MateModel_Hyper.Model(num_classes=data.num_classes).to(device)
    if args.resume:
        print('Loading...')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        best_acc = [0]
        start_epoch = 0

    CEloss = nn.CrossEntropyLoss() 
    f1 = F1_score(num_classes=data.num_classes)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train()
