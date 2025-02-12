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
        
def process_out_channels(outputs, targets):
    """
    将回归转换成分类, 将outputs和targets转换成独热编码
    """
    if args.layer_type == "conv2d":
        targets_C_o = targets[:, 1]
        targets_log = torch.round(torch.log2(targets_C_o)).to(torch.int32).flatten(-1) 
        assert targets_log.ndim == 1

        outputs_C_o = 2**(outputs[:,-1]) / (targets[:,0] * targets[:, 2]**2 * targets[:,3]**2)
        outputs_log = torch.round(torch.log2(outputs_C_o)).to(torch.int32).flatten(-1)
        assert outputs_log.ndim == 1
    elif args.layer_type == "linear":
        targets_F_o = targets[:, 1]
        targets_log = torch.round(torch.log2(targets_F_o)).to(torch.int32).flatten(-1)
        assert targets_log.ndim == 1

        outputs_F_o = 2**(outputs[:,-1]) / targets[:,1]
        outputs_log = torch.round(torch.log2(outputs_F_o)).to(torch.int32).flatten(-1)
        assert outputs_log.ndim == 1
        
    all_classes = torch.unique(torch.cat((targets_log, outputs_log)))
    num_classes = all_classes.shape[0]
    class_mapping = { class_name.item() : idx for idx, class_name in enumerate(all_classes)}
    targets_log = torch.tensor([class_mapping[class_name.item()] for class_name in targets_log ], dtype=torch.long, device=torch.device("cuda"))
    outputs_log = torch.tensor([class_mapping[class_name.item()] for class_name in outputs_log ], dtype=torch.long, device=torch.device("cuda"))
    
    targets = F.one_hot(targets_log, num_classes=num_classes)
    outputs = F.one_hot(outputs_log, num_classes=num_classes)
    return outputs, targets, num_classes
    
class F1_score_out_channels(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.samples = 0
        self.true = 0
        
        # self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

    # def reset(self):
    #     self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

    def forward(self, outputs, targets):
        y_pred, y_true, num_classes= process_out_channels(outputs, targets)
        
        tp = (y_true * y_pred).sum(0)
        tn = ((1 - y_true) * (1 - y_pred)).sum(0)
        fp = ((1 - y_true) * y_pred).sum(0)
        fn = (y_true * (1 - y_pred)).sum(0)

        precision = tp / (tp + fp + self.epsilon) # 精确率：预测为正的样本中预测正确的比例
        recall = tp / (tp + fn + self.epsilon)  # 召回率：实际为正的样本中预测正确的比例
        
        self.samples += y_true.shape[0]
        self.true += (y_true * y_pred).sum()
        accuracy =self.true / self.samples

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.mean().item() #类别与样本无关，每一种类别出现的概率是一样的
        return accuracy * 100, precision.mean().item()*100., recall.mean().item()*100., f1*100.
    
def train_step(epoch):
    net.train()

    timer = Timer()
    timer.start()
    train_loss, accuracy, F1 = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device).float(), targets.to(device).float()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = MSEloss(outputs.reshape(-1), targets[:,-1])
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
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device).float(), targets.to(device).float()

        outputs = net(inputs)
        loss = MSEloss(outputs.reshape(-1), targets[:,-1])

        eval_loss += loss.item()
        accuracy, p, r, F1 = f1(outputs, targets)

    logs = '{} - Epoch: [{}]\t Loss: {:.3f}\t Acc: {:.3f}\t P: {:.3f}\t R: {:.3f}\t F1: {:.3f}\t'
    print(logs.format(arg, epoch, eval_loss / len(loader), accuracy, p, r, F1))
    return eval_loss / len(loader), accuracy, F1


def save_step(epoch, acc, test_index, f1):
    global best_acc
    if sum(acc) > sum(best_acc):
        print('saving...', end='\n\n')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            "acc": acc, 
            "test_index": test_index,
            "f1": f1
        }
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        torch.save(state, args.path + '/out_channels_ckpt.pth')
        best_acc = acc
    else:
        print("此次epoch, 精确度没有提高")


def train():
    for epoch in range(start_epoch, start_epoch+args.epochs):
        # epoch += 1
        train_loss, train_acc = train_step(epoch)
        val_loss, val_acc, f1 = eval_step(epoch, "VAL", valloader)

        save_step(epoch, [val_acc], data.test_index, [f1])
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepTheft Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--path', default='results/MateModel_Hyper', type=str, help='save_path')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument("--layer_type", default="conv2d", type=str, help="layer_type which hp belong to")
    parser.add_argument("--input_size", default="224", type=str, help="input_size of target model")
    args = parser.parse_args()
    args.HyperParameter = "out_channels"
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if args.resume:
        checkpoint = torch.load(args.path + '/out_channels_ckpt.pth')
        test_index = checkpoint["test_index"] 
        data = RaplLoader(args, test_index = test_index, indirect_regression=True) 
    else:
        data = RaplLoader(args, indirect_regression=True)
        
    trainloader, valloader = data.get_loader()

    net = MateModel_Hyper.Model(output_size=data.num_classes).to(device)
    if args.resume:
        print('Loading...')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint["f1"]
    else:
        best_acc = [0]
        start_epoch = 0
        best_f1 = [0]

    MSEloss = nn.MSELoss()
    f1 = F1_score_out_channels()

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train()
