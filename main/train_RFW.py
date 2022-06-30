# train.py

import time
import plugins
import itertools

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import random

import pdb

class Trainer:
    def __init__(self, args, model, criterion, evaluation):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.evaluation = evaluation

        self.nepochs = args.nepochs

        self.lr = args.learning_rate
        self.optim_method = args.optim_method
        self.optim_options = args.optim_options
        self.scheduler_method = args.scheduler_method

        self.optimizer_cnn = getattr(optim, self.optim_method)(
            model['feat'].parameters(), lr=self.lr, **self.optim_options)
        
        module_list = nn.ModuleList([criterion['id'],
            criterion['race'], model['discrim']])
        self.optimizer_cls = getattr(optim, self.optim_method)(
            module_list.parameters(), lr=self.lr, **self.optim_options)
        
        if self.scheduler_method is not None:
            if self.scheduler_method != 'Customer':
                self.scheduler = getattr(optim.lr_scheduler, self.scheduler_method)(
                    self.optimizer_cnn, **args.scheduler_options)

        # for classification
        self.labels = torch.zeros(args.batch_size).long()
        self.inputs = torch.zeros(
            args.batch_size,
            args.resolution_high,
            args.resolution_wide
        )

        if args.cuda:
            self.labels = self.labels.cuda()
            self.inputs = self.inputs.cuda()

        self.inputs = Variable(self.inputs)
        self.labels = Variable(self.labels)

        # logging training
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TrainLogger.txt',
            args.save_results
        )
        params_loss = ['LearningRate','Loss_cls_demog', 'Loss_cls_id',\
            'Loss_conf_demog', 'Loss_conf_id', 'Loss_cls_mi', 'Loss_conf_mi']
        self.log_loss.register(params_loss)

        # monitor training
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'LearningRate': {'dtype': 'running_mean'},
            'Loss_cls_demog': {'dtype': 'running_mean'},
            'Loss_cls_id': {'dtype': 'running_mean'},
            'Loss_cls_mi': {'dtype': 'running_mean'},
            'Loss_conf_demog': {'dtype': 'running_mean'},
            'Loss_conf_id': {'dtype': 'running_mean'},
            'Loss_conf_mi': {'dtype': 'running_mean'},
        }
        self.monitor.register(self.params_monitor)

        # visualize training
        self.visualizer = plugins.Visualizer(args.port, args.env, 'Train')
        params_visualizer = {
            'LearningRate': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'learning_rate',
                    'layout': {'windows': ['train', 'test'], 'id': 0}},
            'Loss_cls_demog': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'loss_cls',
                    'layout': {'windows': ['train', 'test'], 'id': 0}},
            'Loss_cls_id': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'loss_cls',
                    'layout': {'windows': ['train', 'test'], 'id': 0}},
            'Loss_cls_mi': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'loss_cls',
                    'layout': {'windows': ['train', 'test'], 'id': 0}},
            'Loss_conf_demog': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'loss_conf',
                    'layout': {'windows': ['train', 'test'], 'id': 0}},
            'Loss_conf_id': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'loss_conf',
                    'layout': {'windows': ['train', 'test'], 'id': 0}},
            'Loss_conf_mi': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'loss_conf',
                    'layout': {'windows': ['train', 'test'], 'id': 0}},
            'Train_Image': {'dtype': 'image', 'vtype': 'image',
                            'win': 'train_image'},
            'Train_Images': {'dtype': 'images', 'vtype': 'images',
                             'win': 'train_images'},
        }
        self.visualizer.register(params_visualizer)

        # display training progress
        self.print_formatter = 'Train [%d/%d][%d/%d] '
        for item in params_loss:
            self.print_formatter += item + " %.4f "
        # self.print_formatter += "Scale %.4f "

        self.losses = {}
        self.binage = torch.Tensor([10,22.5,27.5,32.5,37.5,42.5,47.5,55,75])

    def model_train(self):
        self.model['feat'].train()
        self.model['discrim'].train()

    def train(self, epoch, dataloader, checkpoints, acc_best):
        dataloader = dataloader['train']
        self.monitor.reset()

        torch.cuda.empty_cache()

        # switch to train mode
        self.model_train()

        end = time.time()

        stored_models = {}

        for i, (inputs, idlabels, racelabels, fmetas) in enumerate(dataloader):
            # keeps track of data loading time
            data_time = time.time() - end

            ############################
            # Update network
            ############################

            batch_size = inputs.size(0)
            batch_idx = list(range(batch_size))

            self.inputs.data.resize_(inputs.size()).copy_(inputs)
            self.labels.data.resize_(idlabels.size()).copy_(idlabels)
            self.labels = self.labels.view(-1)

            racelabels = racelabels[1]
            if self.args.cuda:
                racelabels = racelabels.cuda()
            racelabels = Variable(racelabels)

            outputs = self.model['feat'](self.inputs)

            ########### adversarial training for discriminator to reduce MI #################
            permute_idx = {}
            keys = list(self.criterion)
            for key in keys:
                permute_idx[key] = random.sample(batch_idx, batch_size)
            permute_outputs = torch.cat((outputs[permute_idx['race'],0:512],
                outputs[permute_idx['id'],1*512:2*512]), dim=1)
            inputs_discrim = torch.cat((outputs, permute_outputs), dim=0)
            outputs_discrim = self.model['discrim'](inputs_discrim)

            labels_discrim = torch.cat((torch.ones(batch_size),torch.zeros(batch_size)), dim=0)
            if self.args.cuda:
                labels_discrim = labels_discrim.cuda()
            labels_discrim = Variable(labels_discrim)
            ########### adversarial training for discriminator to reduce MI #################

            loss_cls_demog = self.criterion['race'](outputs[:,0:512], racelabels)[1]
            loss_cls_id =  self.criterion['id'](outputs[:,1*512:2*512], self.labels)[1]
            loss_cls_mi = self.criterion['mi'](outputs_discrim, labels_discrim)
            loss1 = loss_cls_demog + loss_cls_id + loss_cls_mi

            conflabels_id = 1.0/float(self.args.nclasses)*torch.ones(batch_size, int(self.args.nclasses))
            conflabels_race = 0.25*torch.ones(batch_size, 4)
            conflabels_mi = 0.5*torch.ones(batch_size, 2)
            if self.args.cuda:
                conflabels_id = conflabels_id.cuda()
                conflabels_race = conflabels_race.cuda()
                conflabels_mi = conflabels_mi.cuda()
            conflabels_id = Variable(conflabels_id)
            conflabels_race = Variable(conflabels_race)
            conflabels_mi = Variable(conflabels_mi)

            loss_conf_demog = \
                self.criterion['conf'](self.criterion['id'](outputs[:,0*512:1*512],self.labels)[0],  # race confusion
                conflabels_id)

            loss_conf_id = \
                self.criterion['conf'](self.criterion['race'](outputs[:,1*512:2*512],racelabels)[0], 
                conflabels_race)
            loss_conf_mi = self.criterion['conf'](outputs_discrim, torch.cat((conflabels_mi,
                conflabels_mi), dim=0))
            loss2 = loss_conf_demog + loss_conf_id + loss_conf_mi

            self.optimizer_cls.zero_grad()
            self.optimizer_cnn.zero_grad()
            
            loss1.backward(retain_graph=True)
            self.optimizer_cls.step()

            loss2.backward()
            self.optimizer_cnn.step()

            self.losses['Loss_cls_demog'] = loss_cls_demog.item()
            self.losses['Loss_cls_id'] = loss_cls_id.item()
            self.losses['Loss_conf_demog'] = loss_conf_demog.item()
            self.losses['Loss_conf_id'] = loss_conf_id.item()
            self.losses['Loss_cls_mi'] = loss_cls_mi.item()
            self.losses['Loss_conf_mi'] = loss_conf_mi.item()
            for param_group in self.optimizer_cnn.param_groups:
                self.cur_lr = param_group['lr']
            self.losses['LearningRate'] = self.cur_lr
            self.monitor.update(self.losses, batch_size)

            # print batch progress
            print(self.print_formatter % tuple(
                [epoch + 1, self.nepochs, i+1, len(dataloader)] +
                [self.losses[key] for key in self.params_monitor]))

            if i%10000 == 0:
                stored_models['model'] = self.model
                stored_models['loss'] = self.criterion
                checkpoints.save(acc_best, stored_models, epoch, i, True)
        
        # update the log file
        loss = self.monitor.getvalues()
        self.log_loss.update(loss)

        # print epoch progress
        # print(self.print_formatter % tuple(
        #     [epoch + 1, self.nepochs, i+1, len(dataloader)] +
        #     [loss[key] for key in self.params_monitor]))

        # update the visualization
        loss['Train_Image'] = inputs[0]
        loss['Train_Images'] = inputs
        self.visualizer.update(loss)

        # update the learning rate
        if self.scheduler_method is not None:
            if self.scheduler_method == 'ReduceLROnPlateau':
                self.scheduler.step(loss['Loss_cls_id'])
            elif self.scheduler_method == 'Customer':
                if epoch+1 in self.args.lr_schedule: 
                    self.lr *= 0.1
                    self.optimizer_cnn = getattr(optim, self.optim_method)(
                        self.model['feat'].parameters(), lr=self.lr, **self.optim_options)
                    
                    module_list = nn.ModuleList([self.criterion['id'], self.criterion['race'],
                        self.model['discrim']])
                    self.optimizer_cls = getattr(optim, self.optim_method)(
                        module_list.parameters(), lr=self.lr, **self.optim_options)

            else:
                self.scheduler.step()

        total_loss = self.monitor.getvalues('Loss_cls_demog') + \
            self.monitor.getvalues('Loss_cls_id') + \
            self.monitor.getvalues('Loss_conf_demog') + \
            self.monitor.getvalues('Loss_conf_id')
        return total_loss
