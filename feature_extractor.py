# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 10:32:00 2023

@author: azarf
"""


from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot  as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



def feature_extractor(wn,data,label,num_components,plot,description = 'data'):
    mymean = np.mean(data,axis = 1, keepdims = True)
    centered_spectrum = data - mymean
    # Principal component Analysis
    pca = PCA(n_components = num_components)
    pca.fit(centered_spectrum)
    feature = pca.transform(centered_spectrum)   
    percentage = np.sum(pca.explained_variance_ratio_)

      # Principal component Analysis
    clf = LinearDiscriminantAnalysis()
    clf.fit(feature, label)
    features = clf.transform(feature)
    
    # calculating the ration of the peak at 601 to the peak at 628 
    #as the final feature
    tmp = np.absolute(wn-601)
    inx_601 = tmp.argmin()

    tmp = np.absolute(wn-628)
    inx_628 = tmp.argmin()
    

    ratio = np.divide(data[:,inx_601],data[:,inx_628])
    ratio[ratio > 5] = 1

    F = np.append(features,np.reshape(ratio,(feature.shape[0],1)), axis = 1)
    if plot == 1:
        Red = np.zeros((feature.shape[0],1))
        Green = np.zeros((feature.shape[0],1))
        Blue = np.zeros((feature.shape[0],1))
        Red[label == 1] = 0
        Red[label == 2] = 0.8500
        Red[label == 3] = 0.9260
        Red[label == 4] = 0
        Green[label == 1] = 0.4470
        Green[label == 2] = 0.3250
        Green[label == 3] = 0.6940
        Green[label == 4] = 0
        Blue[label == 1] = 0.7410
        Blue[label == 2] = 0.0980
        Blue[label == 3] = 0.1250
        Blue[label == 4] = 0
        mycolors = np.append(Red,Green, axis = 1)
        mycolors = np.append(mycolors,Blue, axis = 1)
        plt.figure('LDA1')
        plt.scatter(F[:,0], F[:,1], c = mycolors)
        plt.xlabel('LD1')
        plt.ylabel('LD2')

        plt.show()
        plt.figure('LDA2')
        plt.scatter(F[:,1], F[:,2], c = mycolors)

        plt.xlabel('LD2')
        plt.ylabel('LD3')



    return np.append(data, F, axis = 1), percentage