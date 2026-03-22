import torch
import torch.backends.cudnn as cudnn
import os
import argparse
import model
from dataset import RaplLoader
import loss
import utils
import numpy as np


train_timer = utils.Timer()

def train_step(epoch, net, trainloader, criterion, optimizer, f1, device):
    net.train()

    train_timer.start()
    metrics = np.zeros(5) # train_loss, accuracy, p, r, F1
    f1.reset()
    for batch_idx, data in enumerate(trainloader):
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
        metrics[1:] = f1(pred, targets)

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

    metrics = np.zeros(5)
    f1.reset()
    for batch_idx, data in enumerate(loader):
        assert len(data) == 2
        inputs, targets = data[0].to(device).float(), data[1].to(device).long()
        domain = None
        if args.regression:
            features = net(inputs)
            loss, pred = criterion(features, targets)
        else:
            loss, pred, _, _ = criterion(net, inputs, targets, domain)

        metrics[0] = loss.item()
        metrics[1:] = f1(pred, targets)

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
            path = args.path + '/' + args.layer_type + "_" + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + args.test_domain + "_" + "classification" + '_ckpt.pth'
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
            path = args.path + '/' + args.layer_type + "_" + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + args.test_domain + "_" + "classification" + '_ckpt.pth'

        if not os.path.exists(args.path):
            os.makedirs(args.path)
        last_checkpoint = torch.load(path, weights_only=False)
        last_checkpoint["epoch"] = epoch + 1
        torch.save(last_checkpoint, path)
        print("此次epoch, 模型性能没有提高")

def train(args, net, trainloader, valloader, criterion, optimizer, scheduler, f1, device):
    start_epoch = scheduler.last_epoch + 1
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
    print("--------------------- EXP START ---------------------")
    if args.HyperParameter != "out_channels":
        args.regression = False
    else:
        args.regression = True
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    input_size = ["160", "192", "224", "299", "331"]
    input_size = [i for i in input_size if i != args.test_domain][0 : args.origin_domain_num]

    data = RaplLoader(args, no_val=False, input_size=input_size)
    args.num_classes = data.num_classes

    trainloader, valloader = data.get_loader()
    net = model.Model(args=args).to(device)
    if args.regression:
        criterion = loss.LossReg(args).to(device)
    else:
        criterion = loss.LossCla(args, net, valloader).to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    if args.resume:
        if args.regression:
            path = args.path + '/' + args.layer_type + "_" + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + args.test_domain + "_" + "regression" + '_ckpt.pth'
        else:
            path = args.path + '/' + args.layer_type + "_" + args.HyperParameter + "_" + str(args.origin_domain_num) + "_" + args.test_domain + "_" + "classification" + '_ckpt.pth'
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
        scheduler.load_state_dict(checkpoint['scheduler'])
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

    f1 = utils.F1_score(num_classes=data.num_classes)

    train(args, net, trainloader, valloader, criterion, optimizer, scheduler, f1, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepTheft Training')
    # training
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs to run')
    parser.add_argument("--max_epochs", default=200, type=int, help="T_max for CosineAnnealingLR")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")

    # data
    parser.add_argument('--path', default='results/MateModel_Hyper', type=str, help='save path for checkpoint')
    parser.add_argument('--data_path', default='dataset/new_dataset', type=str, help='path for dataset')
    parser.add_argument('--workers', default=3, type=int, help='number of data loading workers')
    parser.add_argument('--prefetch_factor', default=2, type=int, help='prefetch number of one loader worker')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')

    # experiment
    parser.add_argument('--resume', '-r', action='store_true', help='whether to resume from checkpoint')
    parser.add_argument("--layer_type", type=str, default="conv2d", help="layer_type which hyperParameter is belong to, should be one of conv2d, max_pool2d, linear")
    parser.add_argument("--HyperParameter", "-H", default="out_channels", type=str, help="hyperparameter to predict")
    parser.add_argument("--test_domain", default="331", type=str, help="target domain for testing, should be one of 160, 192, 224, 299, 331")
    parser.add_argument("--origin_domain_num", "-o", default=4, type=int, help="number of origin domains")

    # model
    parser.add_argument('--feat_dim', default = 128, type=int, help='feature dim')
    parser.add_argument("-w", default=1.0, type=float, help="weight of loss_cla")
    parser.add_argument("--temperature", default=0.1, type=float, help="temperature tau")
    parser.add_argument('--alpha', default= 0.95, type=float, help='momentum of prototype update')

    args = parser.parse_args()
    args.resume = False

    experiment(args)
