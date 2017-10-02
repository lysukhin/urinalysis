# -*- coding: utf-8 -*-
import numpy as np
import cv2



def get_weighted_manhattan_distance(img1,img2):
    """
    Current state:
    calculate mean values of channels of images img1, img2:
    and then calculate manhattan distance with weights over them.

    weights = [1,5,4] were chosen using grid search 
    (! both images should be in LAB color space !)
    
    Takes as input:
    
    - img1, img2 - numpy 3d arrays 
    
    Returns:
    
    - distance - float, distance between two input images
    """
    weights = [2.,5.,3.]
    point1 = np.mean(img1,axis=(0,1))
    point2 = np.mean(img2,axis=(0,1))
    d = 0.
    
    for i in xrange(point1.shape[0]):
        d += weights[i]*np.abs(point1[i] - point2[i])
        
    return d
    
def preprocessing(img):
    
    """
    Current state: 
    - do blur with kernel size = (5,5)
    - convert from RGB to LAB space
    
    Takes as input:
    
    - img - numpy 3d array (color image)
    
    Returns:
    
    - img - processed image, numpy 3d array 
    
    
    """
    img = cv2.blur(img,(5,5))
    return cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    

def one_nearest_neighbor(train_images,test_img,verbose=False,
                         preprocessing=preprocessing,
                         get_distance=get_weighted_manhattan_distance,
                         return_probabilities=False):
    """
    Takes as input: 
    
    - train_images = {"class : train_image"}, where train_image is numpy 3d array and class is anything
    
    - test_image  - numpy 3d array
    
    - function get_distance(img1,img2) that returns float
    
    - function preprocessing(img) that returns image (numpy 3d array)
    
    - return_probabilities - boolean
    
    - verbose - boolean
    
    ----------------------------
    Returns:
    
    - predicted_class of type class
    
    - or predicted_class, list_of_probabilities if return_probabilities == True
    
    
    ---------------------
        
    Accuracy on current dataset is 86% to recognize correct class
    and 100% to make mistake +/- 1 class. 
    
    !This is need to be checked on other datasets!
    
    probabilities are counted like softmax(distance): 
    
        probabilities = np.exp(-(d - np.max(d)))/np.sum(np.exp(-(d - np.max(d))))
    """

    if preprocessing is not None: #Preprocess test_image if preprocessing was defined
        test_img = preprocessing(test_img)

    predicted_class = None #Setting predicted class to None
    min_dist = None #Setting minimal distance to None

    distances = []
    for class_name,train_img in train_images.iteritems():
        
        if preprocessing is not None:  #Preprocess train_image if preprocessing was defined
            train_img = preprocessing(train_img)

        dist = get_distance(test_img,train_img)
        
        if min_dist is None:
            min_dist = dist
            
        if dist <= min_dist:
            min_dist = dist
            predicted_class = class_name


        
        distances.append(dist)
    
    
    if verbose:
        print "all distances:", distances
        print "predicted class:", predicted_class
        print
            
    if return_probabilities:
        d = np.array(distances)/10.
        probabilities = np.exp(-(d - np.max(d)))/np.sum(np.exp(-(d - np.max(d))))
        return predicted_class,probabilities
    
    return predicted_class
