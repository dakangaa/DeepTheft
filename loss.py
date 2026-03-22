import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class LossSim(nn.Module):
    '''
    Compactness Loss with class-conditional prototypes
    类内变异？
    '''
    def __init__(self, args, temperature=0.1, base_temperature=0.1):
        super(LossSim, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, prototypes, labels, domains):

        prototypes = F.normalize(prototypes, dim=1)
        proxy_labels = torch.arange(0, self.args.num_classes).cuda()
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, proxy_labels).float().cuda()
        feat_dot_prototype = torch.div(
            torch.matmul(features, prototypes.T),
            self.temperature)
        logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits = feat_dot_prototype - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss

    def init_class_prototypes(self):
        self.model.eval()
        start = time.time()
        prototype_counts = [0]*self.args.num_classes
        with torch.no_grad():
            prototypes = torch.zeros(self.args.num_classes,self.args.feat_dim).cuda()
            for i, (input, target, domain) in enumerate(self.loader):
                input, target = input.cuda().float(), target.cuda().long()
                features = self.model(input)
                for j, feature in enumerate(features):
                    prototypes[target[j].item()] += feature
                    prototype_counts[target[j].item()] += 1
            for cls in range(self.args.num_classes):
                prototypes[cls] /=  prototype_counts[cls]
            duration = time.time() - start
            print(f'Time to initialize prototypes: {duration:.3f}')
            prototypes = F.normalize(prototypes, dim=1)
            self.prototypes = torch.nn.Parameter(prototypes)

class LossSep(nn.Module):
    '''
    Dispersion Loss with EMA prototypes
    类间分离
    '''
    def __init__(self, args, model, loader, temperature=0.1, base_temperature=0.1):
        super(LossSep, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.register_buffer("prototypes", torch.zeros(self.args.num_classes,self.args.feat_dim))
        self.model = model
        self.loader = loader
        self.init_class_prototypes()

    def forward(self, features, labels):
        prototypes = self.prototypes
        num_cls = self.args.num_classes
        for j in range(len(features)):
            prototypes[labels[j].item()] = F.normalize(prototypes[labels[j].item()] *self.args.alpha
                                                       + features[j]*(1-self.args.alpha), dim=0)
        self.prototypes = prototypes.detach()
        labels = torch.arange(0, num_cls).cuda()
        labels = labels.contiguous().view(-1, 1)

        mask = (1- torch.eq(labels, labels.T).float()).cuda()

        logits = torch.div(
            torch.matmul(prototypes, prototypes.T),
            self.temperature)
        masked_logits = logits.masked_fill(mask == 0, -1e10)
        log_sum_exp = torch.logsumexp(masked_logits, dim=1)
        mean_prob_neg = log_sum_exp - torch.log(mask.sum(1) + 1e-8)
        if torch.isnan(mean_prob_neg).sum().item() > 0:
            print(f"nan:{torch.isnan(mean_prob_neg).sum().item()}")

        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
        loss = self.temperature / self.base_temperature * mean_prob_neg.mean()
        return loss

    def init_class_prototypes(self):
        self.model.eval()
        start = time.time()
        prototype_counts = [0]*self.args.num_classes
        with torch.no_grad():
            prototypes = torch.zeros(self.args.num_classes,self.args.feat_dim).cuda()
            for i, values in enumerate(self.loader):
                assert len(values) == 2
                input, target = values
                input, target = input.cuda().float(), target.cuda().long()
                features = self.model(input)
                for j, feature in enumerate(features):
                    prototypes[target[j].item()] += feature
                    prototype_counts[target[j].item()] += 1
            for cls in range(self.args.num_classes):
                assert prototype_counts[cls] > 0
                prototypes[cls] /=  prototype_counts[cls]
            duration = time.time() - start
            print(f'Time to initialize prototypes: {duration:.3f}')
            prototypes = F.normalize(prototypes, dim=1)
            self.prototypes = prototypes

class LossReg(nn.Module):
    def __init__(self, args):
        super(LossReg, self).__init__()
        self.P_upper = torch.zeros(args.feat_dim).cuda()
        self.P_upper[0] = 1
        self.layer_type = args.layer_type

    def forward(self, features, targets):
        if self.layer_type == "conv2d":
            lower = 0
            upper = 6
            feat_dot_prototype = torch.matmul(features, self.P_upper)
            pred = torch.floor((feat_dot_prototype + 1) / 2 * (upper - lower)).long()
            r = (targets + 0.5) / (upper - lower) * 2 - 1
            loss = ((feat_dot_prototype - r) ** 2).mean()
        elif self.layer_type == "linear":
            lower = 0
            upper = 1
            feat_dot_prototype = torch.matmul(features, self.P_upper)
            r = (targets - lower)/ (upper - lower) * 2 - 1
            loss = ((feat_dot_prototype - r) ** 2).mean()
            pred = (feat_dot_prototype > 0).long()
        return loss, pred
class LossCla(nn.Module):
    def __init__(self, args, net, loader):
        super(LossCla, self).__init__()
        self.disLoss = LossSep(args, net, loader, temperature=args.temperature)
        self.comLoss = LossSim(args, temperature=args.temperature)
        self.w = args.w
        self.temperature = args.temperature

    def forward(self, net, input, target, domain=None):
        features = net(input)
        feat_dot_prototype = torch.div(torch.matmul(features, self.disLoss.prototypes.T), self.temperature)
        logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits = feat_dot_prototype - logits_max.detach()

        pred = logits.data.max(1)[1]

        loss_dis = self.disLoss(features, target)
        loss_comp = self.comLoss(features, self.disLoss.prototypes, target, domain)
        return loss_dis + self.w*loss_comp, pred, loss_dis, loss_comp
