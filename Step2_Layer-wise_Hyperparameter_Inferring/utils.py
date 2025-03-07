import numpy as np
import time
import torch.nn.functional as F
import torch.nn as nn

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

class F1_score(nn.Module):
    def __init__(self, num_classes, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

    def reset(self):
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 1, "y为正确类别数据"
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.num_classes)
        y_pred = F.one_hot(y_pred, self.num_classes)

        self.tp += (y_true * y_pred).sum(0)
        self.tn += ((1 - y_true) * (1 - y_pred)).sum(0)
        self.fp += ((1 - y_true) * y_pred).sum(0)
        self.fn += (y_true * (1 - y_pred)).sum(0)

        precision = self.tp / (self.tp + self.fp + self.epsilon) # 精确率：预测为正的样本中预测正确的比例
        recall = self.tp / (self.tp + self.fn + self.epsilon)  # 召回率：实际为正的样本中预测正确的比例

        accuracy = self.tp.sum() / (self.tp.sum() + self.tn.sum() + self.fp.sum() + self.fn.sum())
        accuracy = accuracy.item() * self.num_classes # 抵消分母的n倍样本量

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.mean().item() 
        return accuracy*100., precision.mean().item()*100., recall.mean().item()*100., f1*100.
