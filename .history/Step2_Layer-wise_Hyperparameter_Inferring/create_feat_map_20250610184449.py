import torch

from MateModel_Hyper import Model
from dataset import RaplLoader, rapl_timer

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