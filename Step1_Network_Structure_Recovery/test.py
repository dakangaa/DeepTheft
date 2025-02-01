import MateModel_Stru
import torch
import dataset
import train
import torch.nn as nn
import argparse

@torch.no_grad()
def eval_step(info,loader):
    net.eval()

    eval_ce_loss, eval_up_loss = 0, 0
    seg_acc, seg_n = 0, 0
    levenshtein_acc, levenshtein_n = 0, 0
    for batch_idx, (inputs, targets, position) in enumerate(loader):
        inputs, targets = inputs.to(device).float(), targets.to(device).long()

        out = net(inputs)
        loss1 = CEloss(out, targets)
        loss2 = train.UPloss(out, targets, position)
        eval_ce_loss += loss1.item()
        eval_up_loss += loss2.item()

        _, predicted = out.max(1)
        labeled = (targets >= 0)
        total = labeled.sum(1)
        correct = ((predicted == targets) * labeled).sum(1)
        acc = 100. * correct / total
        seg_acc += sum(acc)
        seg_n += len(acc)

        lda = train.LDA(out, targets, position)
        levenshtein_acc += sum(lda)
        levenshtein_n += len(lda)

    seg_acc /= seg_n
    levenshtein_acc /= levenshtein_n / 100.
    logs = '{} - CELoss: {:.3f}\t UPLoss: {:.3f}\t SA: {:.3f}%\t LDA: {:.3f}%\t'
    print(logs.format(info, eval_ce_loss / len(loader), eval_up_loss / len(loader), seg_acc, levenshtein_acc))
    return (eval_ce_loss+eval_up_loss) / len(loader), (levenshtein_acc + seg_acc) / 2

parser = argparse.ArgumentParser(description='Step1 : Test in different samples of input_size')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
parser.add_argument('--input_size', default="331", type=str, help='test input_size')
args = parser.parse_args()
device = torch.device("cuda")
target_input_size = "224"

print("Loading data...")
data = dataset.RaplLoader(batch_size=args.batch_size, mode=args.input_size, is_test=True)
test_loader = data.get_loader()

print("Loading Model...")
check_point = torch.load('results/MateModel_Stru' + '/' + target_input_size + '_ckpt.pth') 
net = MateModel_Stru.Model(num_classes=data.num_classes, input_channels=2)
net.load_state_dict(check_point["net"])
net.to(device)
last_acc = check_point["acc"]
train_epoch = check_point["epoch"]
CEloss = nn.CrossEntropyLoss(ignore_index=-1) # -1为填充，不考虑y中为-1的样本点

eval_step("target: " + target_input_size + "\nTEST in" + args.input_size, test_loader)
