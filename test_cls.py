# test.py

import h5py
import sys
import math
import io

import time
import plugins
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import pdb

class Tester:
    def __init__(self, args, model, criterion, evaluation):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.evaluation = evaluation

        self.nepochs = args.nepochs

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

        # logging testing
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TestLogger.txt',
            args.save_results
        )
        params_loss = ['ACC']
        self.log_loss.register(params_loss)

        # monitor testing
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'ACC': {'dtype': 'running_mean'},
        }
        self.monitor.register(self.params_monitor)

        # visualize testing
        self.visualizer = plugins.Visualizer(args.port, args.env, 'Test')
        params_visualizer = {
            'ACC': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'acc',
                    'layout': {'windows': ['train', 'test'], 'id': 1}},
            # 'Test_Image': {'dtype': 'image', 'vtype': 'image',
            #                'win': 'test_image'},
            # 'Test_Images': {'dtype': 'images', 'vtype': 'images',
            #                 'win': 'test_images'},
        }
        self.visualizer.register(params_visualizer)

        # display training progress
        self.print_formatter = 'Test [%d/%d]] '
        for item in ['ACC']:
            self.print_formatter += item + " %.4f "

        self.losses = {}
        # self.binage = torch.Tensor([10,22.5,27.5,32.5,37.5,42.5,47.5,55,75])
        # self.binage = torch.Tensor([10,25,35,45,55,75])
        self.binage = torch.Tensor([19,37.5,52.5,77])

    def model_eval(self):
        self.model['feat'].eval()

    def test(self, epoch, dataloader):
        dataloader = dataloader['test']
        batch = dataloader.batch_size
        self.monitor.reset()
        torch.cuda.empty_cache()

        # switch to eval mode
        self.model_eval()

        end = time.time()

        features = []
        labels = []

        # extract query features
        for i, (inputs,input_labels,attrs,fmetas) in enumerate(dataloader):
            # keeps track of data loading time
            data_time = time.time() - end
            end = time.time()

            ############################
            # Evaluate Network
            ############################
            with torch.no_grad():
                self.inputs.resize_(inputs.size()).copy_(inputs)
                self.labels.resize_(input_labels.size()).copy_(input_labels)

            embeddings = self.model['feat'](self.inputs)

            feat_time = time.time() - end
            
            features.append(embeddings[:,1*512:2*512].data.cpu().numpy())
            labels.append(input_labels.data.numpy())

            torch.sum(embeddings).backward()

        # for i, (inputs,input_labels,attrs,fmetas) in enumerate(dataloader):
        #     if i == 1:
        #         break
        #     # keeps track of data loading time
        #     inputs = inputs[epoch:batch]
        #     input_labels = input_labels[epoch:batch]
        #     data_time = time.time() - end
        #     end = time.time()

        #     ############################
        #     # Evaluate Network
        #     ############################
        #     with torch.no_grad():
        #         self.inputs.resize_(inputs.size()).copy_(inputs)
        #         self.labels.resize_(input_labels.size()).copy_(input_labels)

        #     embeddings = self.model['feat'](self.inputs)

        #     feat_time = time.time() - end
            
        #     features.append(embeddings[:,1*512:2*512].data.cpu().numpy())
        #     labels.append(input_labels.data.numpy())

        #     torch.sum(embeddings).backward()

        labels = np.concatenate(labels, axis=0)
        features = np.concatenate(features, axis=0)
        results,_,_ = self.evaluation(features, labels)
        self.losses['ACC'] = results
        batch_size = 1
        self.monitor.update(self.losses, batch_size)

        # print batch progress
        # print(self.print_formatter % tuple(
        #     [epoch + 1, self.nepochs] +
        #     [results]))
        # print(f"Epoch: {epoch+1}\nResults:{results}")
        print(f"Epoch: {epoch+1}\n")
            
        # update the log file
        loss = self.monitor.getvalues()
        self.log_loss.update(loss)

        # update the visualization
        self.visualizer.update(loss)

        # np.savez('/research/prip-gongsixu/codes/biasface/results/model_analysis/result_agel20.npz',
        #     preds=preds.cpu().numpy(), labels=labels.cpu().numpy())
        
        return results

    def extract_features(self, dataloader):
        dataloader = dataloader['test']
        self.model_eval()
        torch.cuda.empty_cache()

        # extract features
        filenames = []
        features = []
        labels = []
        for i, (inputs, input_labels, attrs, fmetas) in enumerate(dataloader):

            # print(inputs, input_labels, attrs, fmetas)
            with torch.no_grad():
                self.inputs.resize_(inputs.size()).copy_(inputs)
                # self.inputs.data.resize_(inputs.size()).copy_(inputs)

            self.model['feat'].zero_grad()
            embeddings = self.model['feat'](self.inputs)

            ###### feature concatenation ########
            # temp = torch.cat((embeddings[:,0:512], embeddings[:,2*512:3*512], 
            #     embeddings[:,3*512:4*512]), 1)
            
            features.append(embeddings.data.cpu().numpy())
            labels.append(input_labels.data.numpy())

            torch.sum(embeddings).backward()

        labels = np.concatenate(labels, axis=0)
        features = np.concatenate(features, axis=0)
        print(features.shape)

        # save the features
        subdir = os.path.dirname(self.args.feat_savepath)
        if os.path.isdir(subdir) is False:
            os.makedirs(subdir)
        np.savez(self.args.feat_savepath, feat=features, label=labels)
        # with open(os.path.splitext(self.args.feat_savepath)[0]+'_list.txt','w') as f:
        #     for filename in filenames:
        #         f.write(filename+'\n')

    def extract_features_h5py(self, dataloader, num_images):
        dataloader = dataloader['test']
        self.model_eval()
        torch.cuda.empty_cache()

        filename_save = '../results/features/msceleb_debface_demog.h5py'
        f_h5py = h5py.File(filename_save, 'w')
        dt_feat = h5py.special_dtype(vlen=np.dtype('float64'))
        dt_path = h5py.special_dtype(vlen=str)
        
        f_h5py.create_dataset('paths', shape=(num_images,), dtype=dt_path)

        f_h5py.create_dataset('features', shape=(num_images,), dtype=dt_feat)

        # extract features
        filenames = []
        features = []
        labels = []
        for i, (inputs, input_labels, attrs, fmetas) in enumerate(dataloader):

            batch_size = inputs.size(0)

            self.inputs.data.resize_(inputs.size()).copy_(inputs)

            self.model['feat'].zero_grad()
            embeddings = self.model['feat'](self.inputs)

            ###### feature concatenation ########
            temp = torch.cat((embeddings[:,0:512], embeddings[:,2*512:3*512], 
                embeddings[:,3*512:4*512]), 1)
            f_h5py['paths'][i*batch_size:(i+1)*batch_size] = fmetas
            f_h5py['features'][i*batch_size:(i+1)*batch_size] = temp.data.cpu().numpy()

            torch.sum(embeddings).backward()

    def test_demog(self, demog_type, dataloader):
        dataloader = dataloader['test']
        self.monitor.reset()
        torch.cuda.empty_cache()

        # switch to eval mode
        self.model_eval()

        demog_target = {'gender':1, 'age':2, 'race':3}
        target_ind = demog_target[demog_type]

        end = time.time()

        predictions = []
        labels = []

        # extract query features
        nimg_total = 0
        for i, (inputs,input_labels,attrs,fmetas) in enumerate(dataloader):
            # keeps track of data loading time
            data_time = time.time() - end
            end = time.time()

            batch_size = inputs.size(0)
            nimg_total += batch_size

            ############################
            # Evaluate Network
            ############################

            self.inputs.data.resize_(inputs.size()).copy_(inputs)
            self.labels.data.resize_(input_labels.size()).copy_(input_labels)

            embeddings = self.model['feat'](self.inputs)
            features = embeddings[:,(target_ind-1)*512:target_ind*512]

            feat_time = time.time() - end

            predict = self.criterion[demog_type](features, self.labels)[0]

            predictions.append(predict.data.cpu().numpy())
            labels.append(input_labels.data.numpy())

            torch.sum(predict).backward()
        
        labels = np.concatenate(labels, axis=0)
        predictions = np.concatenate(predictions, axis=0)

        pred = np.argmax(predictions, axis=1)
        results = float(float(np.sum((pred==labels).astype(int))) / float(nimg_total))

        # np.savez('/research/prip-gongsixu/codes/biasface/results/model_analysis/result_agel20.npz',
        #     preds=preds.cpu().numpy(), labels=labels.cpu().numpy())

        return results
