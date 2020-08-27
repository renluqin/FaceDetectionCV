# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:12:51 2020

@author: renluqin

@brief: This script is to use the sliding window to scan 
        whether there is a face on each image (based on a pre-trained classifier), 
        and finally give the position and score of the face in each image.
"""

import numpy as np
from skimage.color import rgb2gray
import time
import joblib
from skimage.feature import hog
import os
from skimage import transform

def SlidingWindow(images, clf, minSize, scale):
    """
	@brief      Traverse all images, move the window on each image to detect 
                whether there is a face. 
                The size of the window is unchanged, 
                the input image is the original size, 
                its scale will continue to shrink in each iteration.
                
    @param      images      List of images
                
    @param      clf         Binary Classifier
	
	@param      minSize     Array: [height,length], Image minimum zoom size
    
    @param      scale       The reduction ratio of the image in each iteration
	
	@return     boxs        Array of boxes of dimension 6: index_img,i,j,h,l,score
	"""
    h=minSize[0]
    l=minSize[1]
    boxs = []
    for index in range(len(images)):
        I = rgb2gray(images[index])
        origin_shape = I.shape
        ratio = 1
        while I.shape[0] > h and I.shape[1] > l:
            for i in range(0, I.shape[0], I.shape[0]//30 + 1):
                for j in range(0, I.shape[1], I.shape[1]//30 + 1):
                    if (i+h) < I.shape[0] and (j+l) < I.shape[1]:
                        crop = hog(I[i:i+h, j:j+l])
                        if crop.shape[0] == 1215:
                            score = clf.decision_function(X=crop.reshape(1,-1))
                            if score > 0.4:
                                box = [index+1,int(ratio*i),int(ratio*j),int(ratio*h),int(ratio*l),score.item()]
                                boxs.append(box)
            height = int(I.shape[0] * scale)
            length = int(I.shape[1] * scale)
            I = transform.resize(I,[height,length])
            ratio = origin_shape[0] / height
        print("progress:{0}%".format(round((index + 1) * 100 / len(images))), end="\r")
        time.sleep(0.01)
    return np.array(boxs)


    






































