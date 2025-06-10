import torch

from MateModel_Hyper import Model
from dataset import RaplLoader, rapl_timer
from utils import Timer

class Args():
    def __init__(self):
        self.mode = "regression"
        self.HyperParameter = "out_channels"
        self.test_domain = "331"
        self.origin_domain_num = 4

        self.num_classes = {'out_channels': 6, 'kernel_size': 3, 'stride': 2}[self.HyperParameter]
        self.feat_dim = 128
        self.head = "mlp"
        self.device = torch.device("cuda")
        self.layer_type = "conv2d"
        self.batch_size = 128
        self.workers = 3
        if self.mode == "regression":
            self.regression = True

args = Args()
ckpt_path = "_".join(["results/MateModel_Hyper/conv2d", args.HyperParameter, str(args.origin_domain_num), args.test_domain, args.mode, "ckpt.pth"])
checkpoint = torch.load(ckpt_path, weights_only=False)
net = Model(args).to(args.device)
net.load_state_dict(checkpoint["net"])
input_size = ["160", "192", "224", "299", "331"]
input_size = [i for i in input_size if i != args.test_domain][0 : args.origin_domain_num]
data = RaplLoader(args, no_val=True, input_size=input_size)
test_loader = data.get_loader()

import numpy as np
all_features = [[],[],[],[]]
all_labels = []
net.eval()
timer = Timer()
timer.start()
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(args.device).float(), targets.numpy()
        down_xn = net.layer_wise_get_feat(inputs)
        for i in range(0, 4):
            down_xi = down_xn[i].cpu().numpy()
            down_xi = down_xi.reshape(down_xn[i].shape[0], -1)
            all_features[i].append(down_xi)
        all_labels.append(targets)
        if (batch_idx+1)%100 == 0:
            timer.stop()
            print(f"[{batch_idx+1}/{len(test_loader)}] : {batch_idx*args.batch_size/timer.sum():.3f}samples/sec")
            timer.start()
all_features = [np.concatenate(feats, axis=0) for feats in all_features]
all_labels = np.concatenate(all_labels, axis=0)

for i, feats in enumerate(all_features, start=1):
    np.save(f"feature_map/{args.HyperParameter}_{args.origin_domain_num}_{args.test_domain}_encoder_{i}.npy", feats)
np.save(f"feature_map/{args.HyperParameter}_{args.origin_domain_num}_{args.test_domain}_labels.npy", all_labels)