# -*- coding: utf-8 -*-


import numpy as np
import seaborn as sns
import pandas as pd
import folder_handling as fh
import os
import glob
from collections import OrderedDict
import matplotlib.pyplot as plt
import cv2



def get_distance(img1,img2):
        
        point1 = np.mean(img1,axis=(0,1))
        point2 = np.mean(img2,axis=(0,1))
        d = 0.
        for i in xrange(point1.shape[0]):
            d += (point1[i] - point2[i])**2
        return np.sqrt(d)


def one_nearest_neighbor(train_images,test_img,verbose=False,
                         preprocessing=None,
                         get_distance=get_distance,
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


class TestDistancePreprocessing(object):
    
    def __init__(self):
        self.mistakes_strict = None
        self.mistakes_soft = None
        self.well_predicted_strict = None
        self.well_predicted_soft = None
        self.prob_mistakes = None
        self.data_train = None
    
    def test(self,data_train,data_test,
             get_distance=get_distance, 
             preprocessing=None):
        
        """
        takes as input:
        - data_train dictionary {"indicator":{"illumination":{"class":images}}}
        
        - data_test dictionary {"indicator":{"illumination":{"class":images}}}
        
        - get_distance function
        
        - preprocessing function

        """
    
        self.mistakes_strict = []
        self.mistakes_soft = []
        self.well_predicted_strict = []
        self.well_predicted_soft = []
        self.prob_mistakes = []
        self.data_train = data_test
        self.data_test = data_train
        
        for indicator_type in data_test:
            for illumination in data_test[indicator_type]:
                for true_class in data_test[indicator_type][illumination]:
                    test_img = data_test[indicator_type][illumination][true_class]
                    train_images = data_train[indicator_type][illumination]
                    pred_val,p = one_nearest_neighbor(train_images,
                                        test_img=test_img,verbose=False,
                                        preprocessing=preprocessing,
                                        get_distance=get_distance,
                                        return_probabilities=True)

                    self.well_predicted_strict.append(true_class == pred_val)

                    self.well_predicted_soft.append(np.abs(int(true_class) - int(pred_val)) <= 1)


                    if pred_val != true_class:
                        self.mistakes_strict.append((indicator_type,illumination,"True class :{}".format(true_class),
                                     "Predicted class :{}".format(pred_val)))
                        self.prob_mistakes.append(p)

                        if abs(int(pred_val) - int(true_class)) > 1:
                            self.mistakes_soft.append((indicator_type,illumination,"True class :{}".format(true_class),
                                     "Predicted class :{}".format(pred_val)))

                

    def get_acc_strict(self):
            if not self.well_predicted_strict is None:
                return np.sum(self.well_predicted_strict)/float(len(self.well_predicted_strict))
            return None
        
    def get_acc_soft(self):
            if not self.well_predicted_soft is None:
                return np.sum(self.well_predicted_soft)/float(len(self.well_predicted_soft))
            return None
        
    def get_mistakes_strict(self):
            return self.mistakes_strict
        
    def get_mistakes_soft(self):
            return self.mistakes_soft
        
    def print_report(self):
            strict_acc = self.get_acc_strict()
            soft_acc = self.get_acc_soft()
            print "Точность в предсказании верного класса:\n {} %".format(round(strict_acc*100),1)
            print "Точность в предсказании класса с отклонением не более: 1\n {} %".format(round(soft_acc*100,1))
            
    def plot_mistakes(self):
        if not self.mistakes_strict is None:
            for item in self.mistakes_strict:
                plt.figure(figsize=(10,3))
                ind_type, illum, true_class, predicted_class = [x.split(':')[-1] for x  in item]
                img_train_true = self.data_train[ind_type][illum][true_class]
                img_train_chosen = self.data_train[ind_type][illum][predicted_class]
                img_test = self.data_test[ind_type][illum][true_class]

                plt.subplot(1,3,1)
                plt.imshow(img_test)
                plt.title('test image')

                plt.subplot(1,3,2)
                plt.imshow(img_train_chosen)
                plt.title('predicted image, class {}'.format(predicted_class))

                plt.subplot(1,3,3)
                plt.imshow(img_train_true)
                plt.title('true test image, class {}'.format(true_class))

                plt.suptitle(ind_type)
                plt.tight_layout()