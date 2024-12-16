import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
import MateModel_Stru
from dataset import RaplLoader
import Levenshtein
import os
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

def levenshtein_accuracy(predict, targets, labels):
    def decode(result_not_allign):
        _, indices = result_not_allign.max(-1)
        model_out = indices.tolist()

        decoded = [model_out[0]]
        prev_element = model_out[0]
        for idx in range(1, len(model_out)):
            if prev_element != model_out[idx]:
                decoded.append(model_out[idx])
            prev_element = model_out[idx]

        return decoded

    str_predict, str_labels = [], []
    for (p, t, l) in zip(predict.permute(1, 0, 2), targets, labels):
        p = p[t >= 0, :]
        decoded = decode(p)
        str_predict.append(''.join(str(i) for i in decoded))
        str_labels.append(''.join(str(i) for i in l.tolist()))

    errs = [Levenshtein.distance(str_predict[i], str_labels[i]) for i in range(len(str_predict))]
    tokens = [max(len(str_predict[i]), len(str_labels[i])) for i in range(len(str_predict))]
    LDA = [1 - errs[i] / tokens[i] for i in range(len(str_predict))]
    return LDA


def LDA(out, targets, position):
    out = out.permute(2, 0, 1).log_softmax(-1)

    labels = []
    for (t, pos) in zip(targets, position):
        t = t[t >= 0]
        labels.append(t[pos[:, 0]])

    LDA = levenshtein_accuracy(out, targets, labels)
    return LDA


def UPloss(predicts, targets, position):
    predicts = predicts.softmax(1)
    losses = []
    for (p, t, pos) in zip(predicts, targets, position):
        p = p[:, t >= 0]
        t = t[t >= 0]
        loss = p.new_zeros((1,))
        n = 0
        while n < len(pos):
            i, j = pos[n, 0], pos[n, 1]
            loss[0] += p[t[i], i:j+1].mean().log()
            n += 1

        losses.append(-loss[0][None] / n)

    output = torch.cat(losses, 0)
    output = output.mean()
    return output

def train_step(epoch, loader):
    net.train()
    
    timer = Timer()
    timer.start()
    train_loss = 0 # lossSum
    acc, levenshtein_acc = 0, 0
    total, correct = 0, 0
    for batch_idx, (inputs, targets, position) in enumerate(loader):
        inputs, targets = inputs.to(device).float(), targets.to(device).long()

        optimizer.zero_grad()
        out = net(inputs) # out:(batch, type/channel, sample_len)
        loss = CEloss(out, targets) + UPloss(out, targets, position)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = out.max(1) # 独热编码转类别编码
        labeled = (targets >= 0) #mask:填充部分为0
        total += labeled.sum()
        correct += ((predicted == targets) * labeled).sum()
        acc = 100. * correct / total # 对样本点的准确率(SA)

        if (batch_idx+1) % 10 == 0:
            logs = '{} - Epoch: [{}][{}/{}]\t Loss: {:.3f}\t ACC(SA): {:.3f}\t {:.3f}samples/sec'
            print(logs.format('TRAIN', epoch, batch_idx+1, len(loader), 
                              train_loss / (batch_idx + 1), acc, 
                              10 * args.batch_size / timer.stop()))
            timer.start()
    return train_loss / len(loader), acc


@torch.no_grad()
def eval_step(loader):
    net.eval()

    eval_ce_loss, eval_up_loss = 0, 0
    seg_acc, seg_n = 0, 0
    levenshtein_acc, levenshtein_n = 0, 0
    for batch_idx, (inputs, targets, position) in enumerate(loader):
        inputs, targets = inputs.to(device).float(), targets.to(device).long()

        out = net(inputs)
        loss1 = CEloss(out, targets)
        loss2 = UPloss(out, targets, position)
        eval_ce_loss += loss1.item()
        eval_up_loss += loss2.item()

        _, predicted = out.max(1)
        labeled = (targets >= 0)
        total = labeled.sum(1)
        correct = ((predicted == targets) * labeled).sum(1)
        acc = 100. * correct / total
        seg_acc += sum(acc)
        seg_n += len(acc)

        lda = LDA(out, targets, position)
        levenshtein_acc += sum(lda)
        levenshtein_n += len(lda)

    seg_acc /= seg_n
    levenshtein_acc /= levenshtein_n / 100.
    logs = 'VAL - CELoss: {:.3f}\t UPLoss: {:.3f}\t SA: {:.3f}%\t LDA: {:.3f}%\t'
    print(logs.format(eval_ce_loss / len(loader), eval_up_loss / len(loader), seg_acc, levenshtein_acc))
    return (eval_ce_loss+eval_up_loss) / len(loader), (levenshtein_acc + seg_acc) / 2


def save_step(epoch, acc):
    global best_acc
    if sum(acc) > sum(best_acc):
        print('Saving...', end='\n\n')
        state = {
            'net': net.state_dict(),
            'acc': best_acc,
            'epoch': epoch,
        }
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        torch.save(state, args.path + '/ckpt.pth')
        best_acc = acc
    else:
        print("(LA+SA)/2 精确度未上升")


def train():
    
    for epoch in range(start_epoch, start_epoch+args.epochs):
        # M：epoch += 1
        train_loss, train_acc = train_step(epoch, trainloader) # acc:SA
        val_loss, val_acc = eval_step(valloader) # acc;LA
        save_step(epoch, [val_acc])
        scheduler.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepTheft Training')#L:设置运行参数
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='mini-batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--path', default='results/MateModel_Stru', type=str, help='save_path') # 中途保存的文件和继续读取的文件
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True  #L：对固定大小的输入进行优化
    else:
        device = torch.device('cpu')

    print('Loading data...') # 加载数据到内存中
    timer_load_data = Timer()
    timer_load_data.start()
    data = RaplLoader(batch_size=args.batch_size, num_workers=args.workers) 
    trainloader, valloader = data.get_loader()
    timer_load_data.stop()
    print(f"Finish loading: {timer_load_data.sum():.3f} sec")
    # 创建新网络
    net = MateModel_Stru.Model(num_classes=data.num_classes).to(device) # ? :LSTM的输出和输出
    # 恢复参数
    if args.resume:
        checkpoint = torch.load(args.path + '/ckpt.pth', weights_only=False) # ckpt: checkpoint
        net.load_state_dict(checkpoint['net']) #L：state_dict
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        best_acc = [0]
        start_epoch = 0

    CEloss = nn.CrossEntropyLoss(ignore_index=-1) # -1为填充，不考虑y中为-1的样本点

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train()
