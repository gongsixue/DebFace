# filelist.py

import os
import io
import math
import utils as utils
import torch.utils.data as data
import datasets.loaders as loaders
from PIL import Image
import h5py

import torch
import random

import pdb

class H5pyLoader(data.Dataset):
    def __init__(self, ifile, root=None, split=1.0,
        transform=None, loader='loader_image'):

        self.root = root
        self.transform = transform
        if loader is not None:
            self.loader = getattr(loaders, loader)

        self.f_h5py = h5py.File(ifile[0], 'r')
        
        if ifile[1].endswith('txt'):
            lines = utils.readtextfile(ifile[1])
            imagelist = []
            for x in lines:
                x = x.rstrip('\n')
                filename = os.path.splitext(os.path.basename(x))[0]
                labelname = os.path.basename(os.path.dirname(x))
                temp = [os.path.join(labelname, filename + '.jpg')]
                temp.append(labelname)
                imagelist.append(temp)

        labellist = [x[1] for x in imagelist]

        self.images = imagelist
        self.classname = labellist
        self.classname = list(set(self.classname))
        self.classname.sort()


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if len(self.images) > 0:
            if self.root is not None:
                path = os.path.join(self.root,self.images[index][0])
            else:
                path = self.images[index][0]

            label = self.classname.index(self.images[index][1])
            fmeta = path

            feature = self.f_h5py['features'][index]

        else:
            image = []
            label = None
            fmeta = None        

        return feature, label, label, fmeta
