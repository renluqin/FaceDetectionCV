# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:36:07 2020

@author: renluqin

@brief: This script is to generate a classifier model from the training images.
"""
import numpy as np
from skimage import io, transform
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.svm import SVC
import joblib

# Initialize X individual-variable table
# Each row represents an image
# Each column represents a hog feature
X = np.zeros((6329, 1215))
# Loading training images : positive set
for i in range(1284):
    I = rgb2gray(io.imread('E:/FranceB/SY32/SY32_Pj/data/pos/pos%d.jpg'%(i+1)))
    X[i,:] = hog(transform.resize(I,[60,40]))
# Loading training images : negative set
for i in range(1009):
    for j in range(5):
        path = 'E:/FranceB/SY32/SY32_Pj/data/neg/' + '%d'%(i+1) + '_' + '%d'%(j+1) + '.jpg'
        I = rgb2gray(io.imread(path))
        X[5*i+j+1284,:] = hog(transform.resize(I,[60,40]))
# Generation of learning labels
y = np.concatenate((np.ones(1284), -np.ones(5045)))

# fit the svm classifier
clf_svm = SVC(C=1, gamma=0.1, kernel='rbf',probability=True)
clf_svm.fit(X, y)
# save the classifier model
joblib.dump(clf_svm, "hog_svm_model.m")









