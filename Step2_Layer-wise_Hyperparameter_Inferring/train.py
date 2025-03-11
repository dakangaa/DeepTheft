import torch
import torch.backends.cudnn as cudnn
import os
import argparse
import MateModel_Hyper
from dataset import RaplLoader, rapl_timer
import utils
import torch.nn as nn

def train_step(epoch):
    net.train()

    timer = utils.Timer()
    timer.start()
    train_loss, accuracy, F1, loss1, loss2 = 0, 0, 0, 0, 0
    f1.reset()
    for batch_idx, data in enumerate(trainloader):
        assert len(data) == 2
        inputs, targets = data[0].to(device).float(), data[1].to(device).long()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        accuracy, p, r, F1 = f1(outputs, targets)

        timer.stop()
        if (batch_idx+1) % 100 == 0:
            logs = '{} - Epoch:[{}][{}/{}]\tLoss:{:.3f}\tAcc:{:.3f}\tP:{:.3f}\tR:{:.3f}\tF1:{:.3f}\t{:.3f}samples/sec'
            print(logs.format('TRAIN', epoch, (batch_idx+1), len(trainloader), train_loss / (batch_idx + 1),
                              accuracy, p, r, F1, (batch_idx+1) * args.batch_size / timer.sum()))
            print(f"loading bunch use time {rapl_timer.sum() / timer.sum() * 100:.2f}%")
            timer.start()
    return train_loss / len(trainloader), F1


@torch.no_grad()
def eval_step(epoch, arg, loader):
    net.eval()

    eval_loss, accuracy, F1 = 0, 0, 0
    f1.reset()
    for batch_idx, data in enumerate(loader):
        assert len(data) == 2
        inputs, targets = data[0].to(device).float(), data[1].to(device).long()

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        eval_loss += loss.item()
        accuracy, p, r, F1 = f1(outputs, targets)

    logs = '{} - Epoch: [{}]\t Loss: {:.3f}\t Acc: {:.3f}\t P: {:.3f}\t R: {:.3f}\t F1: {:.3f}\t'
    print(logs.format(arg, epoch, eval_loss / len(loader), accuracy, p, r, F1))
    return eval_loss / len(loader), accuracy, F1


def save_step(epoch, acc, f1, loss):
    global best_f1, best_loss
    path = args.path + '/' + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + args.test_domain + "_" + "train" + '_ckpt.pth'
    print("save path:" + path)

    if sum(acc) > sum(best_acc):
        print('saving...')
        state = {
            'net': net.state_dict(),
            'epoch': epoch+1,
            "acc": acc,
            "f1": f1,
            "loss": criterion.state_dict(),
            "loss_value": loss
        }

        if not os.path.exists(args.path):
            os.makedirs(args.path)
        torch.save(state, path)

        best_f1 = f1
        best_loss = loss
    else:
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        last_checkpoint = torch.load(path)
        last_checkpoint["epoch"] = epoch + 1
        torch.save(last_checkpoint, path)
        print("此次epoch, 模型性能没有提高")


def train():
    for epoch in range(start_epoch, start_epoch+args.epochs):
        print(f">>>>>>>>>>>>>>>>>> EPOCH {epoch} <<<<<<<<<<<<<<<<<<")
        print(f"lr:{scheduler.get_last_lr()}")
        train_loss, train_acc = train_step(epoch)
        val_loss, val_acc, val_f1 = eval_step(epoch, "VAL", valloader)
        save_step(epoch, [val_acc], [val_f1], [val_loss])
        scheduler.step()
        data.shuffle_dataset()
        print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepTheft Training')
    parser.add_argument("--device", type=str, help="运行的机器")
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs to run')
    parser.add_argument("--max_epochs", default=20, type=int, help="total num of epochs")
    parser.add_argument('--path', default='results/MateModel_Hyper', type=str, help='save_path')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument("--layer_type", default="conv2d", type=str, help="layer_type which hyperParameter is belong to")
    parser.add_argument("--HyperParameter", "-H", default="kernel_size", type=str, help="训练的超参数")   # option: kernel_size, stride, out_channels
    parser.add_argument("--test_domain", default="331", type=str, help="目标域")

    parser.add_argument('--head', default='mlp', type=str, help='mlp or linear head')
    parser.add_argument('--feat_dim', default = 128, type=int, help='feature dim')
    parser.add_argument("--origin_domain_num", "-o", default=4, type=int, help="源域数量")
    parser.add_argument("--use_domain", action="store_true", help="是否使用源域信息") # Deprecated

    parser.add_argument("-w", default=1, type=float, help="compLoss的权重")
    parser.add_argument("--temperature", default=0.1, type=float, help="温度系数tao")
    parser.add_argument('--proto_m', default= 0.95, type=float, help='weight of prototype update')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')

    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    # 设定源域
    input_size = ["160", "192", "224", "299", "331"]
    input_size = [i for i in input_size if i != args.test_domain][0 : args.origin_domain_num]

    if args.resume:
        path = args.path + '/' + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + args.test_domain + "_" + "train" + '_ckpt.pth'
        checkpoint = torch.load(path)
        data = RaplLoader(args, no_val=False, input_size=input_size)
        print("load path:" + path)
    else:
        data = RaplLoader(args, no_val=False, input_size=input_size)
    args.num_classes = data.num_classes

    trainloader, valloader = data.get_loader()
    net = MateModel_Hyper.Model(num_classes=data.num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # 模型重载
    if args.resume:
        print('Loading...')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint["f1"]
        criterion.load_state_dict(checkpoint["loss"])
        best_loss = checkpoint["loss_value"]
        best_acc = checkpoint["acc"]
        best_f1 = checkpoint["f1"]
    else:
        best_acc = [0]
        start_epoch = -1
        best_f1 = [0]
        best_acc = [0]
        best_f1 = [0]
        best_loss = [float("inf")] # 正式训练的总loss
    print(f"best_acc:{best_acc[0]:.2f} best_f1:{best_f1[0]:.2f}")

    f1 = utils.F1_score(num_classes=data.num_classes) # y_pred y_true

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # 恢复训练需要显式设置initial_lr参数
    if start_epoch >= 0:
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = args.lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, last_epoch=start_epoch)
    train()
