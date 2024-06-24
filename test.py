#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Python script to calculate specific ML model performance ON HOLDOUT TEST DATA. 
This script uses the MIT license.
"""

###############################################################################

# Author information
__author__ = "Kevin McCoy"
__copyright__ = "Copyright 2024, Ahmad"
__credits__ = ["Kevin McCoy", "Christine Peterson", "Moiz Ahmad"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Kevin McCoy"
__email__ = ["kmccoy1@rice.edu", "cbpeterson@mdanderson.org", "MAhmad@mdanderson.org"]
__status__ = "released"
__date__ = "2024-03-18" # Last modified date

###############################################################################

# Import basic libraries
import pandas as pd
import numpy as np
import random
import math
from sklearn import metrics
from sklearn import preprocessing

# Import various models used
from sklearn.ensemble import RandomForestClassifier


###############################################################################

# Data cleaning function
def data_cleaning():
    
    # Read in data
    print("Loading Dataset...")
    df_og = pd.read_excel(FILENAME)

    # Extract dependent variables of interest
    labels_data = df_og[['label0','label1','label2']].to_numpy()
    y0 = labels_data[:,0] # 1 = no PV, 0 = PV

    # Clean Data
    df = df_og.drop(['label0','label1','label2'], axis=1) #Drop label columns from list of X columns
    metadata = df[['Unnamed: 0','Study ID','ParticleNumber in Study','Num_slices']]
    df = df.drop(['Unnamed: 0','Study ID','ParticleNumber in Study','Num_slices'],axis=1) #Drop study metadata

    # Drop features 
    df = df.drop(['Slice', 'Exp_Skew', 'Exp_Kurt'],axis=1) # Missing data in this feature column

    # Drop features with high correlation
    df = df.drop(['XM', 'BX', 'FeretX', 'YM', 'BY', 'FeretY'], axis=1)
    
    # Optionally, drop 'Mean' feature
    df = df.drop(['Mean'], axis=1)

    # Export to numpy array
    X = df.to_numpy()

    # Get study ID
    patient = metadata['Study ID'].to_numpy() # Patient number

    # Get test and train indices
    test_percentage = 0.2 # Edit as you like
    num_patients = max(patient)
    test_idx = random.sample(range(1, num_patients+1), math.floor(test_percentage*num_patients))
    test_rows = metadata['Study ID'].isin(test_idx).to_numpy()

    # Return tuple of data
    return (y0, X, test_rows, patient)

###############################################################################

# Function to predict on test set
def predict(pred_method, clf, X_val, val_patient):
    
    # Predict on test set normally
    if pred_method == 'standard':
        y_pred = clf.predict(X_val)

    # Extract at least 3 postive objects per person
    else:
        
        # Get probability of positive class
        y_pred = clf.predict_proba(X_val)

        # Initialize output array
        y_out = np.empty((len(y_pred)))
        
        # Loop through each patient
        for i, proba in enumerate(y_pred):
            
            # Get current patient
            curr_patient = val_patient[i]
            
            # Get all probabilities for current patient
            mask = (val_patient == curr_patient)
            probs = y_pred[mask.flatten(), :]
            
            # Sort probabilities
            sorted_index_array = np.argsort(probs[:,1])
            sorted_probs = probs[sorted_index_array]
            
            # If there are at least 3 positive objects, extract them
            if np.any(proba[1] == sorted_probs[-3:, 1]) or proba[1] > 0.50:
                y_out[i] = 1
            else:
                y_out[i] = 0
        
        # Set output array to be the predicted values
        y_pred = y_out
    
    # Return predicted values
    return y_pred

###############################################################################

def binary_clf():

    # Read in data
    (y0, X, test_rows, patient) = data_cleaning()

    # Split data into test and train
    X_train = X[~test_rows, :]
    X_test = X[test_rows, :]
    
    # For single class, invert so that 1 is postive reading
    y_train = (y0[~test_rows] * -1) + 1
    y_test = (y0[test_rows] * -1) + 1

    # normalize data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Fit model
    clf.fit(X_train, y_train)
    
    # Get patient numbers
    test_patient = patient[test_rows]

    # Predict on test set
    # pred_method = 'full' # Extract at least 3 postive objects per person
    pred_method = 'standard' # Standard prediction method
    y_pred = predict(pred_method, clf, X_test, test_patient)

    # Calculate performance metrics
    prec = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)

    # Print performance metrics
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"AUC: {auc:.3f}")

###############################################################################

if __name__ == "__main__":
    
    random.seed(8675309) # Jenny, I got your number
     
    FILENAME = r"./data/og_data.xlsx"

    # Model used after hyperparameter tuning
    clf = RandomForestClassifier(max_samples=0.66, class_weight={0: 1, 1: 1}, n_estimators=1000, n_jobs=-1, random_state=8675309)

    # Execute a simulation
    binary_clf()
    
