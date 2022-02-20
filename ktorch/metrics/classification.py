import torch

__all__ = ['TopK', 'ConfusionMatrix']


class TopK(object):
    def __init__(self, topk):
        self.topk = topk
        self.maxk = max(self.topk)
        self.correct_k = None
        self.num_examples = 0
        
    def update(self, output, target):
        if self.correct_k is None:
            self.correct_k = torch.zeros((len(self.topk),), dtype=torch.int64, device=output.device)
        with torch.no_grad():
            maxk = min(self.maxk, output.shape[1])
            _, pred = torch.topk(output, maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target[None])
            for ind, k in enumerate(self.topk):
                self.correct_k[ind] += torch.sum(correct[:k], dtype=torch.int64)
        self.num_examples += len(output)
        
    def reset(self):
        self.correct_k.zero_()
        self.num_examples = 0

    def compute(self):
        return self.correct_k.float() / self.num_examples
        

class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, actual, predicted):
        if self.mat is None:
            self.mat = torch.zeros((self.num_classes, self.num_classes), 
                                    dtype=torch.int64, device=actual.device)
        with torch.no_grad():
            mask = (actual >= 0) & (actual < self.num_classes)
            inds = self.num_classes * actual[mask].to(torch.int64) + predicted[mask]
            self.mat += torch.bincount(inds, minlength=self.num_classes**2).reshape(
                self.num_classes, self.num_classes)
        
    def reset(self):
        self.mat.zero_()

    def compute(self):
        return self.mat

