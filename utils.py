import numpy as np
import time
import torch.nn.functional as F
import torch.nn as nn
import torch
class Timer:
    def __init__(self):
        self.times = []

    def reset(self):
        self.times = []

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

class F1_score(nn.Module):
    def __init__(self, num_classes, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0
        self.count = torch.zeros(num_classes).cuda()

    def reset(self):
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 1, "y为正确类别label"
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.num_classes)
        y_pred = F.one_hot(y_pred, self.num_classes)
        self.count += y_true.sum(0)

        self.tp += (y_true * y_pred).sum(0)
        self.tn += ((1 - y_true) * (1 - y_pred)).sum(0)
        self.fp += ((1 - y_true) * y_pred).sum(0)
        self.fn += (y_true * (1 - y_pred)).sum(0)

        precision = self.tp / (self.tp + self.fp + self.epsilon)
        recall = self.tp / (self.tp + self.fn + self.epsilon)

        accuracy = self.tp.sum() / (self.tp.sum() + self.tn.sum() + self.fp.sum() + self.fn.sum())
        accuracy = accuracy.item() * self.num_classes

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = ((f1*self.count).sum()/self.count.sum()).item()
        precision = ((precision*self.count).sum()/self.count.sum()).item()
        recall = ((recall*self.count).sum()/self.count.sum()).item()
        return accuracy*100., precision*100., recall*100., f1*100.
