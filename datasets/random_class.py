# triplet.py

import os
import utils as utils
import csv
import numpy as np

import torch
import torch.utils.data as data
import datasets.loaders as loaders

import pdb


def make_dataset_classfolders(ifile):
    tmpdata = utils.readcsvfile(ifile, "\t")
    classes = []
    for i in range(len(tmpdata)):
        classes.append(tmpdata[i][1])
    classes = list(set(classes))
    classes.sort()

    datalist = {}
    for i in range(len(classes)):
        datalist[i] = []

    for i in range(len(tmpdata)):
        row = tmpdata[i]
        datalist[classes.index(row[1])].append(row)

    return datalist, classes

def make_dataset_age_csvlist(ifile):
    nattributes = 0
    datalist = []
    classname = []

    if ifile is not None:
        with open(ifile, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for i,row in enumerate(reader):
                if nattributes <= len(row):
                    nattributes = len(row)
                if 'NaN' in row:
                    idx = [x[0] for x in enumerate(row) if x[1] == 'NaN']
                    for j in idx:
                        row[j] = -1
                path = row[0]
                datalist.append(row)
                ############## Age!!!! ##############
                classname.append(int(row[2])) # age!
                ############## Age!!!! ##############

    label_dict = {}
    # agebins = [0,20,25,30,35,40,45,50,60,120]
    # agebins = [0,20,30,40,50,60,120]
    agebins = [0,30,45,60,200]
    classname = list(set(classname))
    for item in classname:
        idx = np.digitize(item, agebins)-1
        label_dict[item] = idx

    data_dict = {}
    nclasses = len(agebins) - 1
    for i in range(len(datalist)):
        age = int(datalist[i][2])
        idx = label_dict[age]
        if idx not in data_dict:
            data_dict[idx] = [datalist[i]]
        else:
            data_dict[idx].append(datalist[i])

    return nattributes, label_dict, data_dict

def make_dataset_age_bicsvlist(ifile):
    nattributes = 0
    datalist = []
    classname = []

    if ifile is not None:
        with open(ifile, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for i,row in enumerate(reader):
                if nattributes <= len(row):
                    nattributes = len(row)
                if 'NaN' in row:
                    idx = [x[0] for x in enumerate(row) if x[1] == 'NaN']
                    for j in idx:
                        row[j] = -1
                path = row[0]
                datalist.append(row)
                ############## Age!!!! ##############
                classname.append(int(row[2])) # age!
                ############## Age!!!! ##############

    label_dict = {}
    # agebins = [0,20,25,30,35,40,45,50,60,120]
    agebins = [0,20,30,40,50,60,120]
    classname = list(set(classname))
    label2int = {}
    for item in classname:
        idx = int(np.digitize(item, agebins))
        label_dict[item] = [1]*idx
        label_dict[item].extend([0]*(len(agebins)-1-idx))
        label_dict[item] = label_dict[item][1:]
        if idx-1 not in label2int:
            label2int[idx-1] = label_dict[item]

    data_dict = {}
    nclasses = len(list(label2int))
    for i in range(len(datalist)):
        age = int(datalist[i][2])
        label = label_dict[age]
        idx = np.sum(label)
        if idx not in data_dict:
            data_dict[idx] = [datalist[i]]
        else:
            data_dict[idx].append(datalist[i])

    return nattributes, label_dict, data_dict


def make_dataset_gender_csvlist(ifile):
    nattributes = 0
    datalist = []
    classname = []

    if ifile is not None:
        with open(ifile, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for i,row in enumerate(reader):
                if nattributes <= len(row):
                    nattributes = len(row)
                if 'NaN' in row:
                    idx = [x[0] for x in enumerate(row) if x[1] == 'NaN']
                    for j in idx:
                        row[j] = -1
                gender = int(row[1]) # gender!
                if gender != -1:
                    path = row[0]
                    datalist.append(row)
                    ############## Gender!!!! ##############
                    classname.append(gender) # gender!
                    ############## Gender!!!! ##############

    label_dict = {}
    classname = list(set(classname))
    for i,item in enumerate(classname):
        label_dict[item] = i

    data_dict = {}
    nclasses = len(classname)
    for i in range(len(datalist)):
        gender = int(datalist[i][1])
        idx = label_dict[gender]
        if idx not in data_dict:
            data_dict[idx] = [datalist[i]]
        else:
            data_dict[idx].append(datalist[i])

    return nattributes, label_dict, data_dict

class Iterator(object):

    def __init__(self, imagelist):
        self.length = len(imagelist)
        self.temp = torch.randperm(self.length)
        self.current = 0

    def __iter__(self):
        return self

    def next(self):
        value = self.temp[self.current]
        self.current += 1
        if self.current == self.length:
            self.current = 0
            self.temp = torch.randperm(self.length)
        return value


class ClassSamplesDataLoader(data.Dataset):
    def __init__(
        self, root, ifile, num_images, transform=None,
        loader='loader_image', train_type='age_cls'
    ):

        self.root = root
        self.num_images = num_images
        if train_type=='age_bcls':
            nattributes, label_dict, datalist = make_dataset_age_bicsvlist(ifile)
        elif train_type=='age_cls':
            nattributes, label_dict, datalist = make_dataset_age_csvlist(ifile)
        elif train_type=='gender_cls':
            nattributes, label_dict, datalist = make_dataset_gender_csvlist(ifile)

        self.train_type = train_type

        if len(datalist) == 0:
            raise(RuntimeError("No images found"))

        if loader is not None:
            self.loader_input = getattr(loaders, loader)

        self.transform = transform
        if len(datalist) > 0:
            self.classes = label_dict
            self.datalist = datalist
            self.nattributes = nattributes

        self.num_classes = len(list(self.datalist))
        self.class_iter = {}
        for i in range(self.num_classes):
            self.class_iter[i] = Iterator(self.datalist[i])

    def __len__(self):
        return self.num_classes

    def __getitem__(self, index):
        images = []
        fmetas = []
        labels = []
        attributes = []
        for i in range(self.num_images):
            ind = self.class_iter[index].next()
            name = self.datalist[index][ind][0]
            name = os.path.join(self.root, name)
            image = self.loader_input(name)
            images.append(self.transform(image))
            fmetas.append(self.datalist[index][ind][0])
            row = [int(x) for x in self.datalist[index][ind][1:]]
            # row.insert(0, self.classes.index(self.datalist[index][ind][1]))
            attributes.append(torch.Tensor(row))
            if self.train_type == 'age_cls' or self.train_type == 'age_bcls':
                label = self.classes[int(self.datalist[index][ind][2])]
            elif self.train_type == 'gender_cls':
                label = self.classes[int(self.datalist[index][ind][1])]
            labels.append(torch.Tensor([label]))
        
        images = torch.stack(images)
        labels = torch.stack(labels)
        attributes = torch.stack(attributes)
        return images, labels, attributes, fmetas
