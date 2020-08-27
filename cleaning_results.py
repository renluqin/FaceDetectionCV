# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:28:54 2020

@author: renluqin

@brief: This script is to clean up the results: 
        1. remove the boxes with lower scores if too many intersecting areas, 
        2. sort the boxes by scores
        3. filter out the boxes with higher scores in each image
        4. finally save them into a .txt format file.
"""
import numpy as np
from skimage import io

def IoUCOCO(box1, box2):
    """
	@brief      Calculate the ratio of the intersection area of the two boxes 
                to the union area (IoU)
	
	@param      box1,box2     Boxes of dimension 6: index_img,i,j,h,l,score
	
	@return     IoU           float64
	"""
    index1, ycorner1, xcorner1, height1, width1, score1 = box1
    index2, ycorner2, xcorner2, height2, width2, score2 = box2

    w_intersection = min(xcorner1 + width1, xcorner2 + width2) - max(xcorner1, xcorner2)
    h_intersection = min(ycorner1 + height1, ycorner2 + height2) - max(ycorner1, ycorner2)

    if w_intersection <= 0 or h_intersection <= 0: # No overlap
        return 0

    I = w_intersection * h_intersection

    U = width1 * height1 + width2 * height2 - I # Union = Total Area - I

    return I / U    


def Deduplication(boxs, threshold):
    """
	@brief      If the IoU of the two boxes is greater than threshold, 
                remove the box with the lower score
	
	@param      boxs            Array of boxes of dimension 6
    
    @param      threshold       float of the threshold of the size of the area where two boxes coincide
	
	@return     results         Array of cleaned boxes
	"""
    area=[]
    delete=[]
    for i in range(len(boxs)-1):
        j=i+1
        img1=boxs[i]
        img2=boxs[j]
        while img1[0] == img2[0] and j<len(boxs):
            a=IoUCOCO(img1,img2)
            if a > threshold:
                area.append(a)
                if img1[5]>img2[5]:
                    delete.append(j)
                else:
                    delete.append(i)
            j=j+1
            if j<len(boxs):
                img2=boxs[j]
    
    myDelete = list(set(delete))
    myDelete.sort(reverse=True)
    results = np.delete(boxs,myDelete,0)
    return results


def BestCandidates(boxs, proportion):
    """
	@brief      Filter out the boxes with higher scores in each image (TOP proportion*100%)
	
	@param      boxs            Array of boxes of dimension 6
    
    @param      proportion      float of the proportion of candidates in the same box
	
	@return     results         Array of best boxes
	"""
    results = np.zeros((0,6))
    for i in range(len(boxs)):
        candidates = boxs[np.where(boxs[:,0] == i+1)]
        sort_arg = np.argsort(candidates[:,5])
        sort_arg = sort_arg[::-1][0:int(len(sort_arg)*proportion)+1]
        candidates = candidates[sort_arg]
        results = np.vstack((results,candidates))
    return results


# """
# Save cleaned results in detection.txt
# """
# results = Deduplication(boxs)
# results = BestCandidates(results)
# np.savetxt('detection_2.txt',results,fmt="%d %d %d %d %d %.6f")


# """
# Test : Display image and face detection frame in the drawing window.
# """
# I = io.imread('project_test/test/0010.jpg')
# plt.imshow(I)
# for x in results[23:,:]:
#     currentAxis=plt.gca()
#     rect=patches.Rectangle((x[2], x[1]),x[4],x[3],linewidth=1,edgecolor='r',facecolor='none')
#     currentAxis.add_patch(rect)













