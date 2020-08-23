# classification.py

import torch
import numpy as np

import pdb

__all__ = ['Classification', 'Top1Classification', 'BiOrdinalClassify', 'Agergs_classification']


class Classification:
    def __init__(self, topk=(1,)):
        self.topk = topk

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        maxk = max(self.topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Top1Classification:
    def __init__(self):
        pass

    def __call__(self, output, target):
        target = target.long()
        batch_size = target.size(0)

        pred = output.data.max(1)[1]
        res = float(pred.eq(target.data).cpu().sum()) / float(batch_size)

        return res

class BiOrdinalClassify:
    def __init__(self):
        pass
    def __call__(self, output, target):
        batch_size = target.size(0)

        output = torch.gt(output, 0.5)
        pred = torch.sum(output, dim=1).long()
        target = torch.sum(target, dim=1).long()
        res = float(pred.eq(target.data).cpu().sum()) / float(batch_size)

        return res

class Agergs_classification:
    def __init__(self):
        self.agebins = [0,20,25,30,35,40,45,50,60,120]

    def __call__(self, output, target):
        batch_size = target.size(0)

        preds = []
        for age in output:
            idx = np.digitize(int(age), self.agebins)
            preds.append(int(idx))
        preds = torch.LongTensor(preds)
        res = float(preds.eq(target.data.cpu()).sum()) / float(batch_size)
        
        return res