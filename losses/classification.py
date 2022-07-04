# classification.py

from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from torch.autograd import Variable


__all__ = ['Classification', 'BinaryClassify', 'AM_Softmax', 'CrossEntropy', 'Softmax']


# REVIEW: does this have to inherit nn.Module?
class Classification(nn.Module):
    def __init__(self, if_cuda=False):
        super(Classification, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, inputs, targets):
        targets = targets.long()
        loss = self.loss(inputs, targets)
        return loss

class BinaryClassify(nn.Module):
    def __init__(self, weight_file=None, if_cuda=False):
        super(BinaryClassify, self).__init__()
        loss_weight = torch.Tensor(np.load(weight_file))
        self.loss = nn.BCELoss(weight=loss_weight)

    def __call__(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss

# REVIEW: does this have to inherit nn.Module?
class Softmax(nn.Module):
    def __init__(self, nfeatures, nclasses, if_cuda=False):
        super(Softmax, self).__init__()
        self.fc = nn.Linear(nfeatures, nclasses)
        torch.nn.init.xavier_normal_(self.fc.weight)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        # targets = targets.long()
        targets = targets.float()
        y = self.fc(inputs)
        loss = self.loss(y, targets)
        return [y,loss]

class CrossEntropy(nn.Module):
    def __init__(self, if_cuda=False):
        super(CrossEntropy, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        y = self.logsoftmax(inputs)
        loss = torch.mean(torch.sum(-1*torch.mul(targets, y), dim=1))
        return loss


class AM_Softmax_old(nn.Module):
    def __init__(self, nfeatures, nclasses, m=1.0, if_cuda=False):
        super(AM_Softmax_old, self).__init__()
        self.nclasses = nclasses
        self.nfeatures = nfeatures
        self.m = m
        self.if_cuda = if_cuda

        self.weights = nn.parameter.Parameter(torch.Tensor(nclasses, nfeatures))
        torch.nn.init.xavier_normal_(self.weights.data)

        self.scale = nn.parameter.Parameter(torch.Tensor(1))
        torch.nn.init.constant_(self.scale.data, 1.00)

        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.nfeatures)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, labels):
        batch_size = labels.size(0)
        inputs = F.normalize(inputs,p=2,dim=1)
        weights_ = F.normalize(self.weights, p=2,dim=1)
        dist_mat = torch.mm(inputs, weights_.transpose(1,0))

        # create one_hot class label
        label_one_hot = torch.FloatTensor(batch_size, self.nclasses).zero_()
        if self.if_cuda:
            label_one_hot = label_one_hot.cuda()
        labels = labels.long()
        if len(labels.size()) == 1:
            labels = labels[:,None]
        label_one_hot.scatter_(1,labels,1) # Tensor.scatter_(dim,index,src)
        label_one_hot = Variable(label_one_hot)


        logits_pos = dist_mat[label_one_hot==1]
        logits_neg = dist_mat[label_one_hot==0]
        if self.if_cuda:
            scale_ = torch.log(torch.exp(Variable(torch.FloatTensor([1.0]).cuda()))
                + torch.exp(self.scale))
        else:
            scale_ = torch.log(torch.exp(Variable(torch.FloatTensor([1.0])))
                + torch.exp(self.scale))

        logits_pos = logits_pos.view(batch_size, -1)
        logits_neg = logits_neg.view(batch_size, -1)
        logits_pos = torch.mul(logits_pos, scale_)
        logits_neg = torch.mul(logits_neg, scale_)
        logits_neg = torch.log(torch.sum(torch.exp(logits_neg), dim=1))[:,None]

        loss = torch.mean(F.softplus(torch.add(logits_neg - logits_pos, self.m)))+1e-2*scale_*scale_
        
        return loss, scale_

class AM_Softmax(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, nfeatures, nclasses, s=64.0, m=0.35):
        super(AM_Softmax, self).__init__()
        self.s = s
        self.m = m
        self.weights = nn.parameter.Parameter(torch.Tensor(nclasses, nfeatures))
        nn.init.xavier_uniform_(self.weights)
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

        self.scale = nn.parameter.Parameter(torch.Tensor(1))

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, label):
        label = label.long()
        input = F.normalize(input,p=2,dim=1)
        weights = F.normalize(self.weights,p=2,dim=1)
        cosine = torch.mm(input, weights.t())
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)

        loss = self.loss(output, label)

        return [output, loss]
