# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 22:02:10 2020

@author: renluqin

@brief: This script is to generate detections of faces from the given images.
"""
import numpy as np
from utils import LoadImagesFromFolder
from sliding_window import SlidingWindow
from training_set_generation import ExtractTargetAndSave
import joblib
from cleaning_results import Deduplication, BestCandidates

def detections(image_path):
    """
	@brief      This function is to generate detections of faces from the given images.
                
    @param      image_path  String of the folder path of the images to be detected.
	"""
    # load the trained classifier model
    clf = joblib.load("hog_svm_model.m")
    # load images of which faces needed to be detected
    images_test = LoadImagesFromFolder(image_path)
    """
        Use function SlidingWindow() to detect faces on each image_test.
        
        This function takes about 67 minutes.
        
        MinSize is set to [60,40], which means the size of the sliding window 
        is fixed at 60*40 px, and this is also the minimum detection window. 
        
        Faces smaller than this window will not be detected.
        
        Scale is set to 0.8, which means at each step, the image will be reduced by 0.8 times. 
    
    """
    boxs = SlidingWindow(images_test, clf, [60,40], 0.8)
    # If the IoU of 2 boxes is larger than 0.12, then remove the box with lower score.
    candidates = Deduplication(boxs,0.12)
    # According to the descending order of the box score of each image, 
    # the first 80% of the boxes are saved as the final detection box.
    results = BestCandidates(candidates,0.8)
    np.savetxt('detection.txt',results,fmt="%d %d %d %d %d %.6f")

# """
# TEST
# """
# detect faces for the given images
# images_test = LoadImagesFromFolder('E:/FranceB/SY32/SY32_Pj/project_test/test/')
# detections('E:/FranceB/SY32/SY32_Pj/project_test/test/')
# # load the detection.txt
# labels = np.loadtxt('detection.txt', dtype = int)
# # extract detected faces from to the folder
# ExtractTargetAndSave(labels, images_test, 'E:/FranceB/SY32/SY32_Pj/results/04152346/')

















