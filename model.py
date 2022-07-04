# model.py

import math

import evaluate

import losses
import models
from torch import nn

import pdb

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1)
        m.bias.data.zero_()


class Model:
    def __init__(self, args):
        self.ngpu = args.ngpu
        self.cuda = args.cuda
        self.model_type = args.model_type
        self.model_options = args.model_options
        self.loss_type = args.loss_type
        self.loss_options = args.loss_options
        self.evaluation_type = args.evaluation_type
        self.evaluation_options = args.evaluation_options

    def setup(self, checkpoints):
        nets = {}
        nets['feat'] = getattr(models, self.model_type[0])(**self.model_options[0])
        nets['discrim'] = getattr(models, self.model_type[1])(**self.model_options[1])

        evaluation = getattr(evaluate, self.evaluation_type)(
            **self.evaluation_options)
        criterion = {}
        criterion['id'] = getattr(losses, self.loss_type[0])(**self.loss_options[0])
        criterion['gender'] = getattr(losses, self.loss_type[1])(**self.loss_options[1])
        criterion['age'] = getattr(losses, self.loss_type[2])(**self.loss_options[2])
        criterion['race'] = getattr(losses, self.loss_type[3])(**self.loss_options[3])
        criterion['conf'] = getattr(losses, self.loss_type[4])(**self.loss_options[4])
        criterion['mi'] = getattr(losses, self.loss_type[5])(**self.loss_options[5])

        if self.cuda:
            keys = list(nets)
            for key in keys:                
                nets[key] = nn.DataParallel(nets[key], device_ids=list(range(self.ngpu)))
                nets[key] = nets[key].cuda()
            keys = list(criterion)
            for key in keys:
                criterion[key] = criterion[key].cuda()

        model_dict = {}
        model_dict['model'] = nets
        model_dict['loss'] = criterion

        if checkpoints.latest('resume') is None:
            pass
            # model.apply(weights_init)
        else:
            try:
                model_dict = checkpoints.load(model_dict, checkpoints.latest('resume'))
            except:
                pass 

        return nets, model_dict, evaluation
