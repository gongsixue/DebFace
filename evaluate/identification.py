import os
import numpy as np
import scipy.io
import math
import pickle
from eval_utils import *

import pdb

# Compare between every row of x1 and every row of x2
def cosineSimilarity(x1,x2):
    assert x1.shape[1]==x2.shape[1]
    epsilon = 1e-10
    x2 = x2.transpose()
    x1_norm = np.sqrt(np.sum(np.square(x1), axis=1, keepdims=True))
    x2_norm = np.sqrt(np.sum(np.square(x2), axis=0, keepdims=True))
    x1 = x1 / (x1_norm+epsilon)
    x2 = x2 / (x2_norm+epsilon)
    dist = np.dot(x1, x2)
    return dist

def identification(label_filename, feat_probe, feat_gallery, 
    idx_probe, idx_gallery, get_retrievals=False):
    labels = get_labels_from_txt(label_filename)
    label_probe = labels[idx_probe].reshape(-1,1)
    label_gallery = labels[idx_gallery].reshape(-1,1)

    feature_probe = feat_probe[idx_probe,:]
    feature_gallery = feat_gallery[idx_gallery,:]
    
    # Close-set
    scores = cosineSimilarity(feature_probe, feature_gallery)
    DIRs, _, _ = DIR_FAR(scores, label_probe==label_gallery.T, ranks=[1, 5, 10])

    if get_retrievals:
        _, _, _,  mate_indices, success, sort_idx_mat_m, sorted_score_mat_m = DIR_FAR(scores, 
            label_probe==label_gallery.T, ranks=[1], get_retrievals=True)

        return feature_probe, feature_gallery, mate_indices, success, sort_idx_mat_m, sorted_score_mat_m

    else:

        return DIRs

def DIR_FAR(score_mat, label_mat, ranks=[1], FARs=[1.0], get_retrievals=False):
    ''' Closed/Open-set Identification. 
        A general case of Cummulative Match Characteristic (CMC) 
        where thresholding is allowed for open-set identification.
    args:
        score_mat:            a P x G matrix, P is number of probes, G is size of gallery
        label_mat:            a P x G matrix, bool
        ranks:                a list of integers
        FARs:                 false alarm rates, if 1.0, closed-set identification (CMC)
        get_retrievals:       not implemented yet
    return:
        DIRs:                 an F x R matrix, F is the number of FARs, R is the number of ranks, 
                              flatten into a vector if F=1 or R=1.
        FARs:                 an vector of length = F.
        thredholds:           an vector of length = F.
    '''
    assert score_mat.shape==label_mat.shape
    assert np.all(label_mat.astype(np.float32).sum(axis=1) <=1 )
    # Split the matrix for match probes and non-match probes
    # subfix _m: match, _nm: non-match
    # For closed set, we only use the match probes
    mate_indices = label_mat.astype(np.bool).any(axis=1)
    score_mat_m = score_mat[mate_indices,:]
    label_mat_m = label_mat[mate_indices,:]
    score_mat_nm = score_mat[np.logical_not(mate_indices),:]
    label_mat_nm = label_mat[np.logical_not(mate_indices),:]
    mate_indices = np.argwhere(mate_indices).flatten()

    print('mate probes: %d, non mate probes: %d' % (score_mat_m.shape[0], score_mat_nm.shape[0]))

    # Find the thresholds for different FARs
    max_score_nm = np.max(score_mat_nm, axis=1)
    label_temp = np.zeros(max_score_nm.shape, dtype=np.bool)
    if len(FARs) == 1 and FARs[0] >= 1.0:
        # If only testing closed-set identification, use the minimum score as thrnp.vstack((eshold
        # in case there is no non-mate probes
        thresholds = [np.min(score_mat) - 1e-10]
    else:
        # If there is open-set identification, find the thresholds by FARs.
        assert score_mat_nm.shape[0] > 0, "For open-set identification (FAR<1.0), there should be at least one non-mate probe!"
        thresholds = find_thresholds_by_FAR(max_score_nm, label_temp, FARs=FARs)

    # Sort the labels row by row according to scores
    sort_idx_mat_m = np.argsort(score_mat_m, axis=1)[:,::-1]
    sorted_label_mat_m = np.ndarray(label_mat_m.shape, dtype=np.bool)
    sorted_score_mat_m = score_mat_m.copy()
    for row in range(label_mat_m.shape[0]):
        sort_idx = (sort_idx_mat_m[row, :])
        sorted_label_mat_m[row,:] = label_mat_m[row, sort_idx]
        sorted_score_mat_m[row,:] = score_mat_m[row, sort_idx]
        
    # Calculate DIRs for different FARs and ranks
    gt_score_m = score_mat_m[label_mat_m]
    assert gt_score_m.size == score_mat_m.shape[0]

    DIRs = np.zeros([len(FARs), len(ranks)], dtype=np.float32)
    FARs = np.zeros([len(FARs)], dtype=np.float32)
    success = np.ndarray((len(FARs), len(ranks)), dtype=np.object)
    for i, threshold in enumerate(thresholds):
        for j, rank  in enumerate(ranks):
            score_rank = gt_score_m >= threshold
            retrieval_rank = sorted_label_mat_m[:,0:rank].any(axis=1)
            DIRs[i,j] = (score_rank & retrieval_rank).astype(np.float32).mean()
            if get_retrievals:
                success[i,j] = (score_rank & retrieval_rank)
        if score_mat_nm.shape[0] > 0:
            FARs[i] = (max_score_nm >= threshold).astype(np.float32).mean()

    if DIRs.shape[0] == 1 or DIRs.shape[1] == 1:
        DIRs = DIRs.flatten()
        success = success.flatten()

    if get_retrievals:
        return DIRs, FARs, thresholds, mate_indices, success, sort_idx_mat_m, sorted_score_mat_m

    return DIRs, FARs, thresholds