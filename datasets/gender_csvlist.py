# filelist.py

import os
import csv
import math
import torch
from random import shuffle
import torch.utils.data as data
import datasets.loaders as loaders
import numpy as np
from PIL import Image
import pdb

class GenderCSVListLoader(data.Dataset):
    def __init__(self, ifile, root=None,
                 transform=None, loader='loader_image',
                 ):

        self.root = root
        self.ifile = ifile
        self.transform = transform
        if loader is not None:
            self.loader = getattr(loaders, loader)

        self.nattributes = 0
        datalist = []
        classname = []
        if ifile is not None:
            with open(ifile, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter='\t')
                for i,row in enumerate(reader):
                    if i == 0:
                        continue
                    if self.nattributes <= len(row):
                        self.nattributes = len(row)
                    file, dir, age, gender = row[0].split(',')
                    # if 'NaN' in row:
                    #     idx = [x[0] for x in enumerate(row) if x[1] == 'NaN']
                    #     for j in idx:
                    #         row[j] = -1
                    # gender = int(row[1]) # gender!
                    if gender == "male":
                        gender = 0
                    elif gender == 'female':
                        gender = 1
                    if gender != -1:
                        path = dir
                        datalist.append(dir)
                        ############## Gender!!!! ##############
                        classname.append(gender) # gender!
                        ############## Gender!!!! ##############

        self.data = datalist

        self.classname = classname

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if len(self.data) > 0:
            # if self.root is not None:
            #     path = os.path.join(self.root, self.data[index][0])
            # else:
            #     path = self.data[0]
            #     print(path)
            path = "datasets/AFAD/AFAD-Full/" + self.data[0]
            age, gender, id = self.data[0].split('/')
            image = self.loader(path)

            # attributes = self.data[index]
            # fmetas = attributes[0]
            fmetas = len(self.data[0].split('/'))
            attributes = [int(age), int(self.classname[0])]
            # attributes = [int(x) for x in attributes]
            ############## Gender!!!! ##############
            gender = self.classname
            label = gender[0]
            ############## Gender!!!! ##############
            # if len(attributes) < self.nattributes:
            #     length = self.nattributes - len(attributes)
            #     for i in range(length):
            #         attributes.append(-1)

            if self.transform is not None:
                image = self.transform(image)

            attributes = torch.Tensor(attributes)
            return image, label, attributes, fmetas
