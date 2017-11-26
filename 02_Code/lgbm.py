#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 16:12:41 2017

@author: lukaskemmer
"""

from load_data import *
from data_visualization import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import feature_selection, decomposition
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from functions import *
from feature_engineering import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

## ============================ 0. Set parameters ========================== ##

# Define parameters
load_new_data = True # Set True to reload and format data
subset_data = False # Set Treu to select a random subsample of X, Y for testing
subset_size = 100000 # Set the size of the random subsample
estimate_feature_importance = False # Use xgb to estimate feature importance
train_model = True # Set True if model should be trained
make_predictions = False # Set true if prediction for submission should be made
k = 5 # Number of folds for cross-validation
np.random.seed(0) # Random seed to produce comparable results

# Set model parameters for LGBM
lgb_params = {}
lgb_params['n_estimators'] = 1100
lgb_params['max_depth'] = 4
lgb_params['learning_rate'] = 0.02
lgb_params['feature_fraction'] = 0.9
lgb_params['bagging_freq'] = 1
lgb_params['random_state'] = 0

# Set mandatory features to be used
train_features = ["ps_car_13",  
                  "ps_reg_03", 
                  "ps_ind_03", 
                  "ps_ind_15", 
                  "ps_reg_02",  
                  "ps_car_14",  
                  "ps_car_12",  
                  "ps_reg_01", 
                  "ps_car_15",  
                  "ps_ind_01", 
                  "ps_car_11",  
                  #"ps_calc_09", # Maybe drop
                  #"ps_calc_05", # Maybe drop
                  "ps_ind_14",
                  "ps_ind_17_bin",  
                  "ps_ind_08_bin",
                  "ps_ind_09_bin",
                  "ps_ind_18_bin",
                  "ps_ind_12_bin",
                  "ps_ind_16_bin", 
                  "ps_ind_07_bin", 
                  "ps_ind_06_bin",  
                  "ps_ind_05_cat",
                  "ps_car_01_cat",
                  "ps_car_07_cat",
                  "ps_car_03_cat",
                  "ps_car_06_cat",
                  "ps_car_04_cat",
                  "ps_car_09_cat",
                  "ps_car_02_cat",
                  "ps_ind_02_cat",
                  "ps_car_05_cat",
                  "ps_car_08_cat",
                  "ps_ind_04_cat",
                  "ps_car_11_cat"]

## ======================= 1. Load and visualize data ====================== ##
print("\n1. Loading and visualizing data ...\n")
if load_new_data:
    # Load raw data
    X_train_raw, y_train_raw, X_test_raw, X_test_ids, column_names = read_data()
    
    # Format data
    X_train_formatted, X_test_formatted = format_data(X_train_raw.copy(),
                                                      X_test_raw.copy(),
                                                      column_names)
    
    # Describe data
    X_train_summary, X_test_summary, X_train_freq, X_test_freq, X_train_missing, X_test_missing = describe_data(X_train_formatted, X_test_formatted)

# Copy formatted data
X_train = X_train_formatted.copy()
y_train = y_train_raw.copy()
X_test = X_test_formatted.copy()

# Randomly subset the data for faster computation during train/test
if subset_data:
    X_train = X_train.sample(n=subset_size, replace=False, 
                             random_state=0).reset_index(drop=True)
    y_train = y_train.sample(n=subset_size, replace=False, 
                             random_state=0).reset_index(drop=True)

# Plot distributions of features for training and testing data
visualize_features(X_train, "Training data", bin_features=False, 
                   cat_features=0, cont_features=False)
visualize_features(X_test, "Testing data", bin_features=False, 
                   cat_features=0, cont_features=False)

## ========================= 2. Feature engineering ======================== ##
print("\n2. Adding and selecting features ...\n")
# Select features
X_train = X_train[train_features]
X_test = X_test[train_features]

# Don't Replace NAs for LGBM
print("\n   a) Don't replace NAs for LGBM ...\n")
#X_train, X_test = replace_nas(X_train, X_test)

# Feature engineering
print("\n   b) Add features ...\n")    
X_train, X_test = add_features(X_train, X_test, y_train)

col_names = ["ps_car_07_cat",
               "ps_car_03_cat",
               "ps_car_09_cat",
               "ps_car_02_cat",
               "ps_ind_02_cat",
               "ps_car_05_cat"]
               #"ps_car_08_cat"]
               #"ps_ind_04_cat"]
               #"ps_ind_05_cat"]
               #"ps_car_10_cat"]

# Get dummies for features
X_train['name'] = 'X_train'
X_test['name'] = 'X_test'

# Merge X_train and X_test
X = X_train.append(X_test)

for col in col_names:
    # Create dummies
    dummies = pd.get_dummies(X[col].astype(str))
    # Adjust dummie names
    dummies.columns = col + "-" + dummies.columns
    X = pd.concat([X, dummies], axis = 1)
    # Drop categorical column
    X.drop(col)

# Split X_train and X_test
X_train = X.loc[X['name']=='X_train', X.columns != 'name']
X_test = X.loc[X['name']=='X_test', X.columns != 'name']    

## ===================== 3. Model training and evaluation ================== ##    
print("\n3. Training and validating model\n")

if train_model:   
    # Initialize model
    lgbm = LGBMClassifier(**lgb_params)
    
    # Create cross validation iterator
    kf = KFold(n_splits=k, random_state=1, shuffle=True)
    
    # Initialize array for evaluation results
    normalized_gini = []
    
    # Make copies for X, Y to b e used within CV
    X = X_train.copy()
    y = y_train.copy()
    y_pred = np.zeros(X_test.shape[0])
    
    for train_index, validation_index in kf.split(X, y):
        print("Cross-validation, Fold %d" % (len(normalized_gini)+1))
        
        # Split data into training and testing set
        X_train = X.iloc[train_index,:].copy()
        X_validate = X.iloc[validation_index,:].copy()
        y_train = y[train_index]
        y_validate = y[validation_index]
        
        # Train the model
        lgbm = lgbm.fit(X_train, y_train)
        
        # Test the model
        normalized_gini.append(test_model(lgbm, X_validate, y_validate))

        # Make test set prediction
        probs = lgbm.predict_proba(X_test[X_train.columns])[:,1]
        y_pred += np.log(probs/(1-probs))
        del X_train, X_validate, y_train, y_validate

    # Evaluate results from CV
    print("Normalized gini coefficient %f +/- %f" % (np.mean(normalized_gini), 
                                                     2*np.std(normalized_gini)))
    
    # Calculate prediction
    y_pred /= k
    y_pred =  1  /  ( 1 + np.exp(-y_pred) )


## =========================== 4. Output results =========================== ##
if make_predictions:
    print("\n4. Saving results\n")
    
    # Create output dataframe
    results = pd.DataFrame({'id':X_test_ids, 'target':y_pred})
    
    # Output results
    results.to_csv('../03_Results/lgbm_prediction.csv', sep=',', index=False)