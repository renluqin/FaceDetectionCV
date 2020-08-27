# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:45:52 2020

@author: renluqin

@brief: This script is used to generate the training set, including
        a positive set with faces and a negative set without faces.
"""

import numpy as np
import random
from skimage import io
from utils import LoadImagesFromFolder


def ExtractTargetAndSave(labels, images, path):
    """
	@brief      Extract and save targets(faces) from given images 
                with the coordinates in labels.
                
    @param      labels      Array: [index_img, i, j, h, l]
                                -(i,j) : coordinates of the upper left corner of the box
                                -h : box's height
                                -l : box's length
    @param      images      Array of image
    
    @param      path        Where the extracted targets are saved.
	"""
    i = 1
    for x in labels:
        index = x[0]-1
        crop = images[index][x[1]:x[1]+x[3], x[2]:x[2]+x[4]]
        face = path + '%d'%(i)+ '.jpg'
        io.imsave(face, crop) 
        i = i+1
        
  
def IoUCOCO(box1, box2):
    """
	@brief      Calculate the ratio of the intersection area of the two boxes 
                to the union area (IoU)
	
	@param      box1,box2     Boxes of dimension 4: i,j,h,l
	
	@return     IoU           float64
	"""
    ycorner1, xcorner1, height1, width1 = box1
    ycorner2, xcorner2, height2, width2 = box2

    w_intersection = min(xcorner1 + width1, xcorner2 + width2) - max(xcorner1, xcorner2)
    h_intersection = min(ycorner1 + height1, ycorner2 + height2) - max(ycorner1, ycorner2)

    if w_intersection <= 0 or h_intersection <= 0: # No overlap
        return 0

    I = w_intersection * h_intersection

    U = width1 * height1 + width2 * height2 - I # Union = Total Area - I

    return I / U    


def RandomBoxsInImage(images, labels):  
    """
	@brief      For each image in images, generate random negative boxes which are not faces.
                The IoU of the negative box and a positive box (in labels) is less than 0.2
	
	@param      images      Array of image
    
    @param      labels      Array: [index_img, i, j, h, l]
	
	@return     neg_labels  Array: [i, j, h, l]
	"""
    neg_labels = []
    i = 0
    for img in images:
        i_labels = labels[np.where(labels[:,0]==i+1)][:,1:5]
        j = 0
        while j < 5:
            box = np.zeros((1, 4))
            box[0,0] = random.randint(0,int(img.shape[0]*0.9))
            box[0,1] = random.randint(0,int(img.shape[1]*0.9))
            box[0,2] = random.randint(0,img.shape[0]-box[0,0]-1)
            box[0,3] = random.randint(0,img.shape[1]-box[0,1]-1) 
            box_HL_ratio = box[0,2]/box[0,3]
            if box_HL_ratio > 1 and box_HL_ratio < 3.3 and box[0,2] > 21 and box[0,2] < 407 and box[0,3] > 13 and box[0,3] < 275:
                for face_box in i_labels:
                    if box_HL_ratio > 1 and box_HL_ratio < 3.3 and IoUCOCO(box[0],face_box) < 0.2:
                        neg_labels.append(box[0])
                        j = j + 1
                        break
        i = i + 1
    return np.array(neg_labels,dtype=int)
    

def SaveNegSet(images, neg_labels, path):  
    """
	@brief      For each image in images, generate random negative boxes which are not faces.
                The IoU of the negative box and a positive box (in labels) is less than 0.2
	
	@param      images      Array of image
    
    @param      neg_labels  Array: [i, j, h, l]
    
    @param      path        Where the extracted negative set is saved.
	"""
    j = 1
    for i in range(len(neg_labels)):
        index_img = int(i/5)
        x = neg_labels[i]
        crop = images[index_img][x[0]:x[0]+x[2], x[1]:x[1]+x[3]]
        noface = path + '%d'%(index_img+1) + '_' + '%d'%(j) + '.jpg'
        io.imsave(noface, crop) 
        if j >= 5:
            j = 1
        else:
            j = j + 1
            
# """
# TEST
# """   
# print('TEST')  

                
# #load label_train.txt
# x_label = np.loadtxt("project_train/label_train.txt", dtype=int)       
# #load 1000 training images
# images_train = LoadImagesFromFolder('E:/FranceB/SY32/SY32_Pj/project_train/train/')
# #Extract and save targets(faces) from 1000 images with the coordinates in label_train.txt.
# #ExtractTargetAndSave(x_label, images_train, 'E:/FranceB/SY32/SY32_Pj_Test/data/pos/')

# #Generate random negative boxes in each image
# neg_labels = RandomBoxsInImage(images_train, x_label) 
# #extract and save 5 negative images for each image in the training set
# #SaveNegSet(images_train, neg_labels, 'E:/FranceB/SY32/SY32_Pj_Test/data/neg/')



   
















    
    
    
    
    
    
    
    