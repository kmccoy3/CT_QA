#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Python script to calculate specific ML model performance. 
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
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

###############################################################################

# Predefine chunking function
def chunk(xs, n):
    """ Split list [xs] into list of [n] sublists

    Args:
        xs (list): list to chunk
        n (int): number of splits to generate

    Returns:
        list: a list of [n] sublists 
    """
    
    ys = list(xs)
    random.shuffle(ys)
    ylen = len(ys)
    size = int(ylen / n)
    chunks = [ys[0+size*i : size*(i+1)] for i in range(n)]
    leftover = ylen - size*n
    edge = size*n
    for i in range(leftover):
            chunks[i%n].append(ys[edge+i])
    return chunks

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

    # Drop features with missing data
    df = df.drop(['Slice', 'Exp_Skew', 'Exp_Kurt'],axis=1) # Missing data in this feature column

    # Drop features with high correlation
    df = df.drop(['XM', 'BX', 'FeretX', 'YM', 'BY', 'FeretY'], axis=1)
    
    # Optionally, drop 'Mean' feature
    df = df.drop(['Mean'], axis=1)

    # Export to numpy array
    X = df.to_numpy()

    # Get study ID
    patient = metadata['Study ID'].to_numpy() # Patient numbers

    # Get test and train indices
    test_percentage = 0.2 # Edit as you see fit
    num_patients = max(patient)
    test_idx = random.sample(range(1, num_patients+1), math.floor(test_percentage*num_patients))
    train_val_idx = list(set(range(1, num_patients+1)) - set(test_idx))

    # Split non-test data into k folds
    k = 5
    partitions = chunk(train_val_idx, k)

    # Get rows for each fold
    fold1_rows = np.where(metadata['Study ID'].isin(partitions[0]).to_numpy())[0]
    fold2_rows = np.where(metadata['Study ID'].isin(partitions[1]).to_numpy())[0]
    fold3_rows = np.where(metadata['Study ID'].isin(partitions[2]).to_numpy())[0]
    fold4_rows = np.where(metadata['Study ID'].isin(partitions[3]).to_numpy())[0]
    fold5_rows = np.where(metadata['Study ID'].isin(partitions[4]).to_numpy())[0]

    # Create tuple of folds
    train_val_folds = (fold1_rows, fold2_rows, fold3_rows, fold4_rows, fold5_rows)

    # Return tuple of data
    return (y0, X, train_val_folds, patient)

###############################################################################

def predict(pred_method, clf, X_val, val_patient):
    
    # Predict on validation set normally
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
    (y0, X, train_val_folds, patient) = data_cleaning()

    # Initialize performance lists
    precision_list = []
    recall_list = []
    auc_list = []

    for j, val_rows in enumerate(train_val_folds):

        # Get training rows
        train_rows = np.concatenate([train_val_folds[i] for i in range(5) if i != j])

        # Split data into train and validation sets                   
        X_train = X[train_rows, :]
        X_val = X[val_rows, :]
        
        # For single class, invert so that 1 is postive reading
        y_train = (y0[train_rows] * -1) + 1
        y_val = (y0[val_rows] * -1) + 1

        # normalize data
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Train model
        print(f"Running Fold #{j+1}...")
        clf.fit(X_train, y_train)
        
        # Get patient numbers
        val_patient = patient[val_rows]

        # Predict on validation set
        pred_method = 'full' # Extract at least 3 postive objects per person
        # pred_method = 'standard' # Standard prediction method
        y_pred = predict(pred_method, clf, X_val, val_patient)

        # Calculate performance metrics
        prec = metrics.precision_score(y_val, y_pred)
        recall = metrics.recall_score(y_val, y_pred)
        auc = metrics.roc_auc_score(y_val, y_pred)

        # Save metrics to list
        precision_list.append(prec)
        recall_list.append(recall)
        auc_list.append(auc)
        

    # Print average of performance metrics
    print(f"Average Precision: {np.mean(precision_list):.3f}")
    print(f"Average Recall: {np.mean(recall_list):.3f}")
    print(f"Average AUC: {np.mean(auc_list):.3f}")

###############################################################################

if __name__ == "__main__":
    
    random.seed(8675309) # Jenny, I got your number
     
    FILENAME = "./data/og_data.xlsx"

    # Base models tested
    # clf = GaussianNB()
    # clf = LinearDiscriminantAnalysis()
    # clf = QDA()
    # clf = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=8675309)
    # clf = KNeighborsClassifier(n_jobs=-1)
    # clf = SVC(class_weight='balanced', random_state=8675309)
    # clf = RandomForestClassifier(class_weight = 'balanced', n_jobs=-1, random_state=8675309)
    # clf = XGBClassifier(n_jobs=-1, random_state=8675309)
    # clf = MLPClassifier(random_state=8675309)

    # Model used after hyperparameter tuning
    clf = RandomForestClassifier(max_samples=0.66, class_weight={0: 1, 1: 1}, n_estimators=1000, n_jobs=-1, random_state=8675309)

    # Execute a simulation
    binary_clf()
