# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 09:06:29 2022

@author: azarf
"""




from data_loader import data_loader
from Performance import performance
from feature_extractor import feature_extractor
from sklearn.model_selection import train_test_split
import pickle


path2model = 'C:\\SpectraPlasmonic\\model'
path2data = 'C:\\SpectraPlasmonic\\SP_Data\\'

wn, spectra, Labels = data_loader(path2data)


X_train, X_test, y_train, y_test = train_test_split(spectra, Labels, test_size= 0.33, shuffle=True)


model = pickle.load(open(path2model, 'rb'))
Feature_test, percentage_test = feature_extractor(wn,X_test,y_test,5,0,description = 'data')



conf_max = performance(model,Feature_test, y_test, ['A','B','C','D'])

# testing_data will be a [sample x feature] numpy array and should
# return a [sample x class] numpy array.