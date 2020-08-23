# checkpoints.py

import os
import torch
from torch import nn
import torch.optim as optim

import pickle

import pdb

class Checkpoints:
    def __init__(self, args):
        self.args = args
        self.dir_save = args.save_dir
        self.model_filename = args.resume
        self.save_results = args.save_results
        self.cuda = args.cuda

        if self.save_results and not os.path.isdir(self.dir_save):
            os.makedirs(self.dir_save)

    def latest(self, name):
        if name == 'resume':
            return self.model_filename

    def save(self, acc, models, epoch, step, best):
        keys = list(models)
        assert(len(keys) == 2)
        if best is True:
            nets = {}
            keys = list(models['model'])
            for key in keys:
                nets[key] = models['model'][key].state_dict()
            filename_model = '%s/model_epoch_%d_%s_%.2f.pkl' % (self.dir_save, epoch, str(step), acc)
            with open(filename_model, 'wb') as f:
                pickle.dump(nets, f)
            loss = {}
            keys = list(models['loss'])
            for key in keys:
                loss[key] = models['loss'][key].state_dict()
            filename_loss = '%s/loss_epoch_%d_%s_%.2f.pkl' % (self.dir_save, epoch, str(step), acc)
            with open(filename_loss, 'wb') as f:
                pickle.dump(loss, f)

    def load(self, models, filename):
        if os.path.isfile(filename):
            model = models['model']
            print("=> loading checkpoint '{}'".format(filename))
            if self.cuda:
                state_dict = torch.load(filename)
            else:
                state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
            model_dict = model.state_dict()
            update_dict = {}
            valid_keys = list(model_dict)
            state_keys = list(state_dict)
            state_ind = 0
            for key in valid_keys:
                # if key.endswith('num_batches_tracked'):
                #     continue
                update_dict[key] = state_dict[state_keys[state_ind]]
                state_ind += 1 
            model.load_state_dict(update_dict)
            models['model'] = model
            return models

        elif os.path.isdir(filename):
            filename_model = os.path.join(filename, 'model_epoch_31_final_0.997500.pkl')
            filename_loss = os.path.join(filename, 'loss_epoch_31_final_0.997500.pkl')
            with open(filename_model, 'rb') as f:
                nets = pickle.load(f)
            keys = list(models['model'])
            for key in keys:
                models['model'][key].load_state_dict(nets[key], strict=False)
                
            with open(filename_loss, 'rb') as f:
                loss = pickle.load(f)
            keys = list(models['loss'])
            for key in keys:
                models['loss'][key].load_state_dict(loss[key], strict=False)

            if self.cuda:
                keys = list(models['model'])
                for key in keys:                
                    models['model'][key] = nn.DataParallel(models['model'][key], device_ids=list(range(self.args.ngpu)))
                    models['model'][key] = models['model'][key].cuda()
                keys = list(models['loss'])
                for key in keys:
                    models['loss'][key] = models['loss'][key].cuda()
            
            return models
        raise (Exception("=> no checkpoint found at '{}'".format(filename)))
