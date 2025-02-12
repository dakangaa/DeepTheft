import MateModel_Hyper
import torch
import dataset
import train
import torch.nn as nn
import argparse

def eval_step(epoch, arg, loader):
    net.eval()
    timer = train.Timer()
    timer.start()
    
    eval_loss, accuracy, F1 = 0, 0, 0
    f1.reset()
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device).float(), targets.to(device).long()

        outputs = net(inputs)
        loss = CEloss(outputs, targets)

        eval_loss += loss.item()
        accuracy, p, r, F1 = f1(outputs, targets)
        if (batch_idx+1)%100 == 0:
            timer.stop()
            print(f"[{batch_idx+1}/{len(loader)}] : {batch_idx*batch_size/timer.sum():.3f}samples/sec")
            timer.start()
            
    logs = '{} - TrainEpoch: [{}]\t Loss: {:.3f}\t Acc: {:.3f}\t P: {:.3f}\t R: {:.3f}\t F1: {:.3f}\t'
    print(logs.format(arg, epoch, eval_loss / len(loader), accuracy, p, r, F1))
    return eval_loss / len(loader), F1

parser = argparse.ArgumentParser(description='Test in different samples of input_size')
parser.add_argument('--batch_size', default=1280, type=int, help='mini-batch size')
parser.add_argument('--input_size', default="192", type=str, help='test input_size')
parser.add_argument("--HyperParameter", "-H", default="kernel_size", type=str, help="测试的超参数")   # option: kernel_size, stride, out_channels
args = parser.parse_args()
hp = args.HyperParameter
batch_size = args.batch_size
input_size = args.input_size
device = torch.device("cuda")

print("Loading data...")
data = dataset.RaplLoader(batch_size=batch_size, mode=hp, is_test = True, input_size = input_size)
test_loader = data.get_loader()
print("Loading Model...")
check_point = torch.load('results/MateModel_Hyper' + '/' + hp + '_ckpt.pth') 
net = MateModel_Hyper.Model(output_size=data.num_classes, input_channels=2)
net.load_state_dict(check_point["net"])
net.to(device)
last_acc = check_point["acc"]
train_epoch = check_point["epoch"]
CEloss = nn.CrossEntropyLoss() 
f1 = train.F1_score(num_classes=data.num_classes)
eval_step(train_epoch, "TEST in " + hp, test_loader)
