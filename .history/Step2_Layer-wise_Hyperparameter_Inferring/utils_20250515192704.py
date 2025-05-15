import numpy as np
import time
import torch.nn.functional as F
import torch.nn as nn
import torch
class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []

    def reset(self):
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

class F1_score(nn.Module):
    def __init__(self, num_classes, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0
        self.type_count = torch.zeros(num_classes).cuda()
        self.true_count = 0

    def reset(self):
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0
        self.type_count = torch.zeros_like(self.type_count).cuda()
        self.true_count = 0

    def forward(self, y_pred, y_true):
        # 多类别分类的F1计算：计算每个类的F1分数，再取平均
        # 返回整体测试集的指标
        assert y_pred.ndim == 1, "y为正确类别label"
        assert y_true.ndim == 1
        onehot_true = F.one_hot(y_true, self.num_classes)
        onehot_pred = F.one_hot(y_pred, self.num_classes)
        self.type_count += onehot_true.sum(0)
        self.true_count += (y_true == y_pred).sum().item()

        self.tp += (onehot_true * onehot_pred).sum(0)
        self.tn += ((1 - onehot_true) * (1 - onehot_pred)).sum(0)
        self.fp += ((1 - onehot_true) * onehot_pred).sum(0)
        self.fn += (onehot_true * (1 - onehot_pred)).sum(0)

        precision = self.tp / (self.tp + self.fp + self.epsilon) # 精确率：预测为正的样本中预测正确的比例
        recall = self.tp / (self.tp + self.fn + self.epsilon)  # 召回率：实际为正的样本中预测正确的比例

        accuracy = self.true_count / self.type_count.sum().item()

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = ((f1*self.type_count).sum()/self.type_count.sum()).item()
        precision = ((precision*self.type_count).sum()/self.type_count.sum()).item()
        recall = ((recall*self.type_count).sum()/self.type_count.sum()).item()
        return accuracy*100., precision*100., recall*100., f1*100.
