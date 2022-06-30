# dataloader.py

import os

import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from datasets import *

import pdb

__all__ = ['Dataloader']

class Dataloader:
    def __init__(self, args):
        self.args = args

        self.resolution = (args.resolution_wide, args.resolution_high)
        self.input_size = (args.input_wide, args.input_high)

    def setup(self, dataloader_type, dataset_options):
        if 'transform' in dataset_options.keys():
            temp = dataset_options['transform']
            transform = transforms.Compose(
                self.preprocess(dataset_options['transform']))
            dataset_options['transform'] = transform

        if dataloader_type == 'FileListLoader':
            dataset = FileListLoader(**dataset_options)
        elif dataloader_type == 'CSVListLoader':
            dataset = CSVListLoader(**dataset_options)
        elif dataloader_type == 'ClassSamplesDataLoader':
            dataset = ClassSamplesDataLoader(**dataset_options)
        elif dataloader_type == 'GenderCSVListLoader':
            dataset = GenderCSVListLoader(**dataset_options)
        elif dataloader_type == 'H5pyLoader':
            dataset = H5pyLoader(**dataset_options)
        elif dataloader_type == 'H5pyCSVLoader':
            dataset = H5pyCSVLoader(**dataset_options)
        elif dataloader_type == 'DemogCSVListLoader':
            dataset = DemogCSVListLoader(**dataset_options)
        elif dataloader_type is None:
            print("No data assigned!")
        else:
            raise(Exception("Unknown Training Dataset"))

        dataset_options['transform'] = temp

        return dataset

    def preprocess(self, preprocess):
        process_list = []
        keys = [ \
            'Resize', \
            'CenterCrop', \
            'RandomCrop', \
            'RandomHorizontalFlip', \
            'RandomVerticalFlip', \
            'RnadomRotation', \
            'ToTensor', \
            'Normalize', \
        ]
        for key in keys:
            if key in preprocess.keys():
                if key == keys[0]:
                    process_list.append(getattr(transforms, key)(self.input_size))
                elif key == keys[1] or key == keys[2]:
                    process_list.append(getattr(transforms, key)(self.resolution))
                elif key == keys[3] or key == keys[4]:
                    process_list.append(getattr(transforms, key)())
                elif key == keys[5]:
                    process_list.append(getattr(transforms, key)(preprocess[key]))
                elif key == keys[6]:
                    process_list.append(getattr(transforms, key)())
                else:
                    process_list.append(transforms.Normalize(preprocess[key][0],
                        preprocess[key][1]))
        return process_list

    def create(self, dataset=None, flag=None):
        dataloader = {}
        if dataset is not None:
            dataloader['test'] = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.test_batch_size,
                num_workers=int(self.args.nthreads),
                shuffle=False,pin_memory=True
            )
            return dataloader
            
        else:
            if flag == "Train":
                self.dataset_train = self.setup(self.args.dataset_train, self.args.dataset_options_train)
                dataloader['train'] = torch.utils.data.DataLoader(
                    self.dataset_train,
                    batch_size=self.args.batch_size,
                    num_workers=int(self.args.nthreads),
                    shuffle=True,pin_memory=True
                )
                return dataloader

            elif flag == "Test":
                self.dataset_test = self.setup(self.args.dataset_test, self.args.dataset_options_test)
                dataloader['test'] = torch.utils.data.DataLoader(
                    self.dataset_test,
                    batch_size=self.args.test_batch_size,
                    num_workers=int(self.args.nthreads),
                    shuffle=False,pin_memory=True
                )
                return dataloader

            elif flag is None:
                self.dataset_train = self.setup(self.args.dataset_train, self.args.dataset_options_train)
                self.dataset_test = self.setup(self.args.dataset_test, self.args.dataset_options_test)
                dataloader['train'] = torch.utils.data.DataLoader(
                    self.dataset_train,
                    batch_size=self.args.batch_size,
                    num_workers=int(self.args.nthreads),
                    shuffle=True,pin_memory=True
                )
                dataloader['test'] = torch.utils.data.DataLoader(
                    self.dataset_test,
                    batch_size=self.args.test_batch_size,
                    num_workers=int(self.args.nthreads),
                    shuffle=False,pin_memory=True
                )
                return dataloader

    def __str__(self) -> str:
        return str(self.args) + '\n' + str(self.resolution) + '\n' + str(self.input_size)