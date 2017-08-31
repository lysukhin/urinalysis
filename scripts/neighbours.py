# -*- coding: utf-8 -*-
import numpy as np
import cv2



def get_manhattan_distance(img1,img2):
    
    point1 = np.mean(img1,axis=(0,1))
    point2 = np.mean(img2,axis=(0,1))
    d = 0.
    
    for i in xrange(point1.shape[0]):
        d += np.abs(point1[i] - point2[i])
        
    return d
    
def preprocessing(img):
    img = cv2.blur(img,(5,5))
    return cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    

def one_nearest_neighbor(train_images,test_img,verbose=False,
                         preprocessing=preprocessing,
                         get_distance=get_manhattan_distance,
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
    
    - or predicted_class, list_of_probabilities of return_probabilities == True
    
    
    ---------------------
    
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