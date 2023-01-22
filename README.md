

Module description
The following modules are provided:
Main.py: The main module used for understanding the data and training the model, initialize the parameter Path2data for running the code and see the step-by-step training process.

data_loader: loads the data, and returns the spectra, class labels from the data dictionary. For simplicity the labels are changed from ‘A’, ‘B’,’C’,’D’ to 0, 1, 2, and 3. This module also normalizes the spectra to the peak at 601 wavenumber to reduce the effect of concentration.

Feature_extractor.py: Applied principal component analysis (PCA) followed by linear discriminant analysis (LDA) for extracting features from the spectral data. PCA was performed using 5 components (explaining more than 90% of the dataset). Then an LDA is performed to further reduce the featured to only three (one less than the number of classes). Since the number of samples are small only 73, the extracted PCA-LDA features are added to the spectra itself to increase the accuracy of the model.
To increase the performance of the model I also added the ration of the peak at 601 wavenumbers to 628 as another feature. Ending up with total of 1605 features.

Performance.py: Evaluate the performance of the model using confusion matrixes and the ROC curves

Test.py: Initialize the path2data and path2model and tests the performance 
