import numpy as np
import os
import math

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import pdb

def acc_per_class_1offset(filename_result):
    agebins = [0,20,30,40,50,60,120]

    results = np.load(filename_result)
    preds = results['preds']
    labels = results['labels']
    ages = results['ages']

    classname = list(set(labels))
    num_hits = 0
    for i,clas in enumerate(classname):
        idx = np.where(labels==clas)[0]
        pred_clas = preds[idx]
        age_clas = ages[idx]

        gt_off1 = np.array([np.digitize(item-5, agebins)-1 for item in age_clas])[:,None]
        gt_up1 = np.array([np.digitize(item+5, agebins)-1 for item in age_clas])[:,None]
        gt = np.tile(clas, (len(idx), 1))
        newgt = np.concatenate((gt_off1, gt_up1, gt), axis=1)

        hits_class = [1 for x,y in zip(pred_clas,newgt) if x in y]
        # rate = float(len(num_hits)) / float(len(idx))
        # print('Class {} -- {}'.format(i, rate))
        num_hits += len(hits_class)
    rate = float(num_hits) / float(len(preds))
    print('Average Acc -- {}'.format(rate))

def acc_distr_per_class(filename_result):
    results = np.load(filename_result)
    preds = results['preds']
    labels = results['labels']

    classname = list(set(labels))
    pdb.set_trace()
    for i,clas in enumerate(classname):
        idx = np.where(labels==clas)[0]
        pred_clas = preds[idx]
        rates = []
        for j in range(len(classname)):
            pred_idx = np.where(pred_clas==classname[j])[0]
            if j == 0:
                rate = float(len(pred_idx)) / float(len(idx))
            if j == 1:
                rate = float(len(pred_idx)) / float(len(idx))
            if j == 2:
                rate = float(len(pred_idx)) / float(len(idx))
            if j == 3:
                rate = float(len(pred_idx)) / float(len(idx))
            if j == 4:
                rate = float(len(pred_idx)) / float(len(idx)) 
            if j == 5:
                rate = float(len(pred_idx)) / float(len(idx)) 
            if j == i:
                print('Class {} vs {} -- {}'.format(i,j,rate))
            rates.append(rate)

        opacity = 1.0
        binWidth = 0.5
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        plt.bar(classname, rates, binWidth, 
            alpha=opacity, color=colors[0])

        plt.savefig('/research/prip-gongsixu/codes/biasface/results/model_analysis/age0_hist_{}.png'.format(i))
        plt.show()

def main():
    filename_result = '/research/prip-gongsixu/codes/biasface/results/model_analysis/result_agel20.npz'
    acc_per_class_1offset(filename_result)

if __name__ == '__main__':
    main()

####### Results! ##########

####### Results! ##########
