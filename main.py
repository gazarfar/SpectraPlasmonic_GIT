"""
Created on Tue Dec 20 13:45:53 2022

@author: azarf
"""

############################################
# Packages & functions & path2data
############################################

import pickle
import numpy as np
from feature_extractor import feature_extractor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from data_loader import data_loader
from Performance import performance

path2data = 'C:\\SpectraPlasmonic\\SP_Data\\'
############################################
#Loading the data
############################################
wn, spectra, Labels = data_loader(path2data)


############################################
# Creating training and test Sets for
# an initial training and test of the model
############################################

X_train, X_test, y_train, y_test = train_test_split(spectra, Labels, test_size= 0.33, shuffle=True)


############################################
# Feature extraction by PCA
############################################
print('-------------------------------------------------------------------')
print('Feature Extraction')
print('-------------------------------------------------------------------\n')





# find the index of minimum element from the array


num_components = 5
Feature, percentage = feature_extractor(wn,X_train,y_train,num_components,1,description = 'data')
Feature_test, percentage_test = feature_extractor(wn,X_test,y_test,num_components,0,description = 'data')

#F, percentage = feature_extractor(wn,X_test,y_test,num_components,1,description = 'data')

print('The first ' + str(num_components) +'PC components includes ' + str(percentage) + '% of the variations in the spectra')
# print('As shown in the PCA figure by projecting the spectra to PC coordinate the spectra are already grouped into three clusters')
# print('Since the clusters can be seperated by straight lines, a Logistic Regression model with cross validation will work perfectly for classification')
print('-------------------------------------------------------------------')

# # ############################################
# # # Logistic Regression and model training
# # ############################################ 
print('-------------------------------------------------------------------')
print('Training the model by 66% of the data and testing on the rest')
print('-------------------------------------------------------------------\n')

model = LogisticRegression(solver='sag',max_iter=200,warm_start=True, class_weight="balanced", random_state=1).fit(Feature, y_train);



print('-------------------------------------------------------------------')
print('Testing the perfomance of the model on the 66% training data')
print('-------------------------------------------------------------------\n')

conf_max = performance(model,Feature, y_train, ['A','B','C','D'])

print('-------------------------------------------------------------------')
print('Testing the perfomance of the model on the 33% test data')
print('-------------------------------------------------------------------\n')

conf_max = performance(model,Feature_test, y_test, ['A','B','C','D'])

print('-------------------------------------------------------------------')
print('The model is furthered trained using the 33% remaining test dataset to be prepared for an independent test')
print('-------------------------------------------------------------------\n')

#further training the model with rest of the data
model.fit(Feature_test, y_test)
conf_max = performance(model,Feature_test, y_test, ['A','B','C','D'])

print('-------------------------------------------------------------------')
print('Saving the model')
print('-------------------------------------------------------------------\n')

# # #saving the model
filename = 'model'
#pickle.dump(model, open(filename, 'wb'))

