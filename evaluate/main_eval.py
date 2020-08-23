import numpy as np
import os

from verification import FaceVerification
from identification import identification
from eval_utils import *

import pdb

label_filename = '/scratch/gongsixue/face_resolution/cfp_ff_aligned_list.txt'
feat_filenamelist = ['/scratch/gongsixue/face_resolution/feats/feat_cfp_112x112.npz', \
    '/scratch/gongsixue/face_resolution/feats/feat_cfp_100x100.npz', \
    '/scratch/gongsixue/face_resolution/feats/feat_cfp_90x90.npz', \
    '/scratch/gongsixue/face_resolution/feats/feat_cfp_75x75.npz', \
    '/scratch/gongsixue/face_resolution/feats/feat_cfp_56x56.npz', \
    '/scratch/gongsixue/face_resolution/feats/feat_cfp_28x28.npz', \
    '/scratch/gongsixue/face_resolution/feats/feat_cfp_14x14.npz', \
    '/scratch/gongsixue/face_resolution/feats/feat_cfp_7x7.npz', \
    ]
imgsize_list = [112,100,90,75,56,28,14,7]
fig_savepath = '/research/prip-gongsixu/codes/face_resolution/gender.pdf'
savedir = '/research/prip-gongsixu/codes/face_resolution'

############# verification ################
# pair_index_filename = '/user/pripshare/Databases/FaceDatabasesPublic/cfp-dataset/cfp-dataset/Protocol/Split/FF'
# obj = FaceVerification(label_filename,
#     protocol='BLUFR', metric='cosine',
#     nthreads=8,multiprocess=True,
#     pair_index_filename=pair_index_filename,template_filename=None,
#     pairs_filename=None,nfolds=10,
#     nimgs=None, ndim=None)

# legends = ['112x112', '100x100', '90x90', '75x75', '56x56', '28x28', '14x14', '7x7']
# TARs = []
# FARs = []
# for feat_filename in feat_filenamelist:
#     data = np.load(feat_filename)
#     feat = data['feat']
#     TAR,FAR,thresholds = obj(feat)
#     TARs.append(TAR)
#     FARs.append(FAR)
# # curve_plot(imgsize_list, avgs, fig_savepath, font_size=13)
# ROC_plot(TARs, FARs, legends, savedir, legend_loc='lower right')
############# verification ################

############# identification ################
# idx_probe = np.load('/scratch/gongsixue/face_resolution/identification/idx_probe.npy')
# idx_gallery = np.load('/scratch/gongsixue/face_resolution/identification/idx_gallery.npy')

# data = np.load(feat_filenamelist[0])
# feat_ori = data['feat']

# DIR_list = []
# for feat_filename in feat_filenamelist:
#     data = np.load(feat_filename)
#     feat = data['feat']
#     DIRs = identification(label_filename, feat, feat_ori, idx_probe, idx_gallery, get_retrievals=False)
#     print(DIRs)
#     DIR_list.append(DIRs[0])
# curve_plot(imgsize_list, DIR_list, fig_savepath, font_size=13)
############# identification ################

############### plot ##############
acclist = [97.06,97.02,97.02,96.90,96.62,94.70,66.34,56.80]
# acclist = [61.27,61.22,60.58,60.65,45.76,34.97,12.88,4.09]
curve_plot(imgsize_list, acclist, fig_savepath, font_size=13)
############### plot ##############
