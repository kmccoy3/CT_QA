#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Python script to conduct grid search cross validation on binary class data. 
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
from sklearn.model_selection import GridSearchCV

# Import various models used
from sklearn.ensemble import RandomForestClassifier

###############################################################################

# Predefine chunking function
def chunk(xs, n):
    """ Split list [xs] into list of [n] sublists

    Args:
        xs (list): list to chunk
        n (int): number of splits to generate

    Returns:
        list: a list of [n] lists 
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
    y0 = labels_data[:,0]

    # Clean Data
    df = df_og.drop(['label0','label1','label2'], axis=1) #Drop label columns from list of X columns
    metadata = df[['Unnamed: 0','Study ID','ParticleNumber in Study','Num_slices']]
    df = df.drop(['Unnamed: 0','Study ID','ParticleNumber in Study','Num_slices'],axis=1) #Drop study metadata
    # Drop missing features 
    df = df.drop(['Slice', 'Exp_Skew', 'Exp_Kurt'],axis=1) # Missing data in this feature column

    # Drop features with high correlation
    df = df.drop(['XM', 'BX', 'FeretX', 'YM', 'BY', 'FeretY'], axis=1)
    
    # Optionally, drop 'Mean' feature
    df = df.drop(['Mean'], axis=1)    

    # Export to numpy array
    X = df.to_numpy()

    # Get study ID
    study_ID = metadata['Study ID'].to_numpy() # Patient number

    # Get test and train indices
    test_percentage = 0.2 # Edit as you like
    num_patients = max(study_ID)
    test_idx = random.sample(range(1, num_patients+1), math.floor(test_percentage*num_patients))
    train_val_idx = list(set(range(1, num_patients+1)) - set(test_idx))

    # Split into k folds
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
    return (y0, X, train_val_folds)

###############################################################################

def main():
    
    # Read in data
    (y0, X, train_val_folds) = data_cleaning()
    
    # Invert y0
    y0 = 1 - y0
    
    # Initialize cv object
    cv = list()

    # Create cross validation folds
    for j, val_rows in enumerate(train_val_folds):
        train_rows = np.concatenate([train_val_folds[i] for i in range(5) if i != j])
        cv.append((train_rows, val_rows))

    # Start grid search
    print("Starting Grid Search...")
    model = GridSearchCV(clf, parameters, cv=cv, scoring='precision', n_jobs=-1, verbose=1)
    model.fit(X, y0)

    # Print results
    print(f"Best parameters {model.best_params_}")
    print(f"Best average score achieved: {model.best_score_}")

###############################################################################

if __name__ == "__main__":
    
    random.seed(8675309) # Jenny, I got your number
     
    FILENAME = r"./data/og_data.xlsx"

    # Random Forest
    clf = RandomForestClassifier(max_samples=0.66, random_state=8675309)
    parameters = {'n_estimators':[50, 100, 500, 1000],
                  'class_weight':({0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 10}, 'balanced'),
                  'max_features': ['sqrt', 'log2']
                  }
    
    # Execute a simulation
    main()
