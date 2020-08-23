# filelist.py

import os
import io
import math
import utils as utils
import torch.utils.data as data
import datasets.loaders as loaders
from PIL import Image
import h5py
import csv

import torch
import random

import pdb

class H5pyCSVLoader(data.Dataset):
    def __init__(self, ifile, root=None, split=1.0,
        transform=None, loader='loader_image'):

        self.root = root
        self.transform = transform
        if loader is not None:
            self.loader = getattr(loaders, loader)

        self.f_h5py = h5py.File(ifile[0], 'r')

        datalist = []
        idlist = []
        genderlist = []
        agelist = []
        racelist = []

        with open(ifile[1], 'r') as csvfile:
            rows = csv.reader(csvfile, delimiter='\t')
            for i,row in enumerate(rows):
                attrs = []
                attrs.append(row[0])
                attrs.append(os.path.dirname(row[0]))
                attrs.append(int(row[1]))
                attrs.append(int(row[2]))
                attrs.append(int(row[3]))

                datalist.append(attrs)
                idlist.append(attrs[1])
                genderlist.append(attrs[2])
                agelist.append(attrs[3])
                racelist.append(attrs[4])

        self.data = datalist
        self.idclass = list(set(idlist))
        self.genderclass = list(set(genderlist))
        self.ageclass = list(set(agelist))
        self.raceclass = list(set(racelist))
        
        self.idclass.sort()
        self.genderclass.sort()
        self.ageclass.sort()
        self.raceclass.sort()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if len(self.data) > 0:
            if self.root is not None:
                path = os.path.join(self.root,self.data[index][0])
            else:
                path = self.data[index][0]

            idlabel = self.idclass.index(self.data[index][1])
            genderlabel = self.genderclass.index(self.data[index][2])
            agelabel = self.ageclass.index(self.data[index][3])
            racelabel = self.raceclass.index(self.data[index][4])
            fmeta = path

            im_bytes = self.f_h5py['images'][index]
            image = Image.open(io.BytesIO(im_bytes))

            if self.transform is not None:
                image = self.transform(image)

        else:
            image = []
            idlabel = None
            genderlabel = None
            agelabel = None
            racelabel = None
            fmeta = None        

        return image, idlabel, genderlabel, agelabel, racelabel, fmeta
