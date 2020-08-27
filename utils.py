# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:54:48 2020

@author: renluqin

@brief: This script includes some useful functions that would be used
        in other scripts.
"""

import os
from skimage import io
import numpy as np
from numpy import trapz


def LoadImagesFromFolder(folder):
    """
	@brief      Load all images from a given folder.
                
    @param      folder      The path of image folder.
	
	@return     images      Array of images of dimension (height, length, 3)
	"""
    images = []
    files = os.listdir(folder)
    files.sort()
    for filename in files:
        img = io.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def IoUQ(box1, box2):
    """
	@brief      Calculate the ratio of the intersection area of the two boxes 
                to the union area (IoU)
	
	@param      box1     Boxes of dimension 7: index_img,i,j,h,l,score, index
    
    @param      box2     Boxes of dimension 5: index_img,i,j,h,l
	
	@return     IoU      float64
	"""
    imNumber, ycorner1, xcorner1, height1, width1, score, index = box1
    imNumber, ycorner2, xcorner2, height2, width2 = box2

    w_intersection = min(xcorner1 + width1, xcorner2 + width2) - max(xcorner1, xcorner2)
    h_intersection = min(ycorner1 + height1, ycorner2 + height2) - max(ycorner1, ycorner2)

    if w_intersection <= 0 or h_intersection <= 0: # No overlap
        return 0

    I = w_intersection * h_intersection

    U = width1 * height1 + width2 * height2 - I # Union = Total Area - I

    return I / U


def PrecisionRecall(label_test, label_train):
    """
	@brief      According to the real face coordinates in label_test, 
                calculate the precision and rappel rate of labels in label_train 
                where the face coordinates are obtained by training.
                
    @param      label_test          Array of real face labels of dimension 5: (img_index, i, j, h, l)
    
    @param      label_train         Array of real face labels of dimension 6: (img_index, i, j, h, l, score)
	
	@return     precision           float : True positive / (True positive + False positive)
    
    @return     recall              float : True positive / (True positive + False negative)
    
    @return     f_score             float : 2 * precision * recall / (precision + recall)
    
    @return     recall_over_time    list of recalls for each label
    
    @return     precision_over_time list of precisions for each label
    
    @return     auc                 float : area under the PR curve 
	"""
    # add index column to label_train
    l = [i for i in range(len(label_train))]
    l = np.asarray(l).reshape(-1,1)
    label_train = np.hstack((label_train, l))
    # VP will contain all true positive labels
    # FP will contain all false positive labels
    VP=[]
    FP=[]
    VP_index=[]
    for img1 in label_test:
        img_train = np.asarray(list(filter(lambda x: x[0] == img1[0], label_train)))
        flag=0
        for img2 in img_train:
            a=IoUQ(img2,img1)
            if a>0.5:
                VP.append(img1)
                VP_index.append(img2[6])
                flag=1
                break
        if flag==0:
           FP.append(img1)              
    VP = np.asarray(VP)
    FP = np.asarray(FP)
    precision = len(VP)/len(label_train)
    recall = len(VP)/len(label_test)
    f_score = 2*precision*recall/(precision+recall)
    
    # Traverse each label in label_train, 
    # and calculate each recall and precision, and save it to draw PR curve
    recall_over_time=[]
    precision_over_time=[]
    match=0
    i=1
    # sort label_train by descending score 
    label_train_sorted = np.asarray(sorted(label_train, key = lambda x:x[5], reverse=True))
    for img in label_train_sorted:
        if img[6] in VP_index:
            match=match+1
        recall_over_time.append(match/len(label_test))
        precision_over_time.append(match/i)
        i=i+1      
    
    # Compute the area under the PR curve using the composite trapezoidal rule.
    auc = trapz(precision_over_time, x=recall_over_time)
 
    return precision, recall, f_score, recall_over_time, precision_over_time, auc



















