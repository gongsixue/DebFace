import os
from sympy import E
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import pdb

__all__ = ['normalize', 'get_FARs', 'ROC_plot',\
    'get_index_dict', 'read_pairfile', 'get_pair_and_label',\
    'get_labels_from_txt', 'get_pairwise_score_label',\
    'get_genpairs_imppairs', 'curve_plot',\
    'get_pair_and_label_rfw', 'get_index_dict_rfw']

def normalize(x, ord=None, axis=None, epsilon=10e-12):
    ''' Devide the vectors in x by their norms.'''
    if axis is None:
        axis = len(x.shape) - 1
    norm = np.linalg.norm(x, ord=None, axis=axis, keepdims=True)
    x = x / (norm + epsilon)
    return x

def get_FARs(tool='numpy',cuda=False):
    tmp1 = [10**x for x in range(-8,0,1)]
    tmp2 = list(range(1,10,1))
    tmp = np.kron(tmp1,tmp2)
    tmp = np.insert(tmp,0,0.0)
    tmp = np.insert(tmp,tmp.size,1.0)
    FARs = tmp
    if tool == 'torch':
        FARs = torch.Tensor(tmp)
        if cuda:
            FARs = FARs.cuda()
    return FARs

def ROC_plot(TARs, FARs, legends, savedir, legend_loc='lower right'):
    # Both TARs and FARs are type 'list'
    assert(len(TARs) == len(FARs))
    for i in range(len(FARs)):
        FARs[i] = 0.1*FARs[i]

    font_prop = font_manager.FontProperties(size=13)
    plt.figure()
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    lw = 2
    for i in range(len(TARs)):
        plt.semilogx(100*FARs[i], 100*TARs[i], color=colors[i],
            lw=lw, label=legends[i])
    plt.xlim([10e-4, 1])
    plt.xticks(fontsize = 13)
    plt.ylim([0.0, 100])
    plt.yticks(fontsize = 13)
    plt.xlabel('False Accept Rate (%)', fontproperties=font_prop)
    plt.ylabel('Verification Rate (%)', fontproperties=font_prop)
    plt.legend(loc=legend_loc, prop=font_prop)
    # plt.legend(loc=legend_loc)
    plt.grid(True)
    plt.savefig(os.path.join(savedir, 'ROC_11lh.pdf'))
    plt.show()
    plt.clf()

def get_index_dict(imagepaths_filename):
    index_dict = {}
    with open(imagepaths_filename, 'r') as f:
        lines = list(f)
        imagepaths = [x.split('\n')[0] for x in lines]
    for i,path in enumerate(imagepaths):
        index_dict[os.path.splitext(os.path.basename(path))[0]] = i
    return index_dict

def get_index_dict_rfw(imagepaths_filename):
    index_dict = {}
    with open(imagepaths_filename, 'r') as f:
        lines = list(f)
        imagepaths = [x.split('\n')[0].split('\t')[0] for x in lines]
    for i,path in enumerate(imagepaths):
        sub_name = [d for d in path.split('/') if d != '']
        sub_name = '/'.join(sub_name[-3:])
        index_dict[os.path.splitext(sub_name)[0]] = i
    return index_dict

def read_pairfile(pairs_filename, dataset):
    pairs = []
    with open(pairs_filename, 'r') as f:
        if dataset == 'LFW':
            for line in f.readlines()[0:]:
                pair = line.strip().split()
                pairs.append(pair)
        elif dataset == 'RFW':
            for line in f.readlines()[0:]:
                pair = line.strip().split()
                pairs.append(pair)
    return pairs

def get_pair_and_label(pairfiles, index_dict):
    nrof_skipped_pairs = 0
    pair_indices = []
    issame_label = []
    for pair in pairfiles:
        if len(pair) == 3:
            path0 = pair[0] + '_' + '%04d' % int(pair[1])
            path1 = pair[0] + '_' + '%04d' % int(pair[2])
            issame = True
        elif len(pair) == 4:
            path0 = pair[0] + '_' + '%04d' % int(pair[1])
            path1 = pair[2] + '_' + '%04d' % int(pair[3])
            issame = False
        pair_indices.append((index_dict[path0],index_dict[path1]))
        issame_label.append(issame)
    pair_indices = np.array(pair_indices)
    issame_label = np.array(issame_label)

    return issame_label, pair_indices

def get_pair_and_label_rfw(pairfiles, index_dict, race):
    nrof_skipped_pairs = 0
    pair_indices = []
    issame_label = []
    for pair in pairfiles:
        if len(pair) == 3:
            path0 = os.path.join(race, pair[0], pair[0] + '_' + '%04d' % int(pair[1]))
            path1 = os.path.join(race, pair[0], pair[0] + '_' + '%04d' % int(pair[2]))
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(race, pair[0], pair[0] + '_' + '%04d' % int(pair[1]))
            path1 = os.path.join(race, pair[2], pair[2] + '_' + '%04d' % int(pair[3]))
            issame = False
        pair_indices.append((index_dict[path0],index_dict[path1]))
        issame_label.append(issame)
    pair_indices = np.array(pair_indices)
    issame_label = np.array(issame_label)

    return issame_label, pair_indices

def get_labels_from_txt(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
        # class_sorted = [x.split('\t')[0].split('/')[-2] for x in lines]
        class_sorted = [x.split('\t')[0].split('/')[-2] for x in lines if len(x) == 3]
        # class_sorted = [x[0].split(',') for x in lines if len(x) == 3]
        # class_sorted = [x.split('.') for x in lines]
        for i in class_sorted:
            if len(i) != 3: print(i)
        classname = []
        classname[:] = class_sorted[:]
        class_sorted.sort()
    labels = [int(class_sorted.index(x)) for x in classname]
    labels = np.array(labels)
    return labels

def get_pairwise_score_label(score_mat, label):
    n = label.size 
    # assert score_mat.shape[0]==score_mat.shape[1]==n
    triu_indices = np.triu_indices(n, 1)
    if len(label.shape)==1:
        label = label[:, None]
    label_mat = label==label.T
    score_vec = score_mat[triu_indices]
    label_vec = label_mat[triu_indices]
    return score_vec, label_vec

def get_genpairs_imppairs(label):
    # the returned genid and impid are tuple with lengh of 2
    # if convert to the numpy, shape = (2*N)
    # N is the number of pairs
    n = label.size
    triu_indices = np.triu_indices(n,1)
    if len(label.shape) == 1:
        label = label[:,None]

    label_mat = label==label.T
    temp = np.zeros(label_mat.shape, dtype=bool)
    temp[triu_indices] = True
    genlab = label_mat&temp
    genid = np.where(genlab==True)

    temp = np.ones(label_mat.shape, dtype=bool)
    temp[triu_indices] = False
    implab = label_mat|temp
    impid = np.where(implab==False)
    return genid,impid

def curve_plot(x_axis, y_axis, savepath, font_size=13):
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)
    font_prop = font_manager.FontProperties(size=font_size)
    plt.figure()
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    lw = 2
    plt.plot(x_axis, y_axis, color=colors[0],lw=lw)
    legends = ['112x112', '100x100', '90x90', '75x75', '56x56', '28x28', '14x14', '7x7']
    plt.xticks(x_axis, legends, rotation=45, fontsize = font_size)
    plt.ylim([50,100])
    plt.yticks(fontsize = font_size)
    plt.xlabel('Image Size', fontproperties=font_prop)
    plt.ylabel('Gender Accuracy', fontproperties=font_prop)
    
    plt.grid(True)
    plt.savefig(savepath, bbox_inches='tight')
    plt.show()
    plt.clf()