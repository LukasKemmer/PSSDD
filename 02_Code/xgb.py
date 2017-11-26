#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:10:44 2017

@author: lukaskemmer
"""
from load_data import *
from data_visualization import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import feature_selection, decomposition
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from xgboost import plot_tree
from functions import *
from feature_engineering import *
from sklearn.preprocessing import OneHotEncoder

## ============================ 0. Set parameters ========================== ##

# Define parameters
load_new_data = True # Set True to reload and format data
subset_data = True # Set Treu to select a random subsample of X, Y for testing
subset_size = 1000 # Set the size of the random subsample
train_model = True # Set True if model should be trained
make_predictions = False # Set true if prediction for submission should be made
k = 5 # Number of folds for cross-validation
np.random.seed(0) # Random seed to produce comparable results

# Set model parameters for XGB
xgb_params = {'max_depth': 4, # Maximum tree depth for base learners
              'eta': 0.07, # Boosting learning rate
              'n_estimators' : 400, # Number of boosted trees to fit
              'silent': 1, # Whether to print messages while running boosting
              'objective' : 'binary:logistic',
              'booster' : 'gbtree', # Which booster to use: gbtree, gblinear or dart
              'n_jobs' : 8, # Number of parallel threads
              'gamma' : 10, # Minimum loss reduction required to make a further partition on a leaf node of the tree
              'min_child_weight' : 6, # Minimum sum of instance weight(hessian) needed in a child
              'max_delta_step' : 0, # Maximum delta step we allow each tree’s weight estimation to be
              'subsample' : .8, # Subsample ratio of the training instance
              'colsample_bytree' : .8, # Subsample ratio of columns when constructing each tree
              'colsample_bylevel' : 1, # Subsample ratio of columns for each split, in each level
              'reg_alpha' : 8, # L1 regularization term on weights
              'reg_lambda' : 1.3, # L2 regularization term on weights
              'scale_pos_weight' : 1.6, # Balancing of positive and negative weights
              'base_score' : 0.5} # The initial prediction score of all instances, global bias

# Define features to be used
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
    X_train_raw, y_train_raw, X_test_raw, X_test_ids, X_train_ids, column_names = read_data()
    
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

# Add combinations of features
print("\n   a) Add feature combinations ...\n")
# Create list with features that should be combined  
combinations = [('ps_reg_01', 'ps_car_02_cat'),('ps_reg_01', 'ps_car_04_cat')]
# Add combinations
X_train, X_test = add_combination_features(X_train, X_test, combinations)

# Encode categorical data
print("\n   b) Add encoded categorical features ...\n")
for col in [col for col in X_train.columns if '_cat' in col]:
    X_train[col + "_avg"], X_test[col + "_avg"] = target_encode(
            trn_series=X_train[col],
            tst_series=X_test[col],
            target=y_train,
            min_samples_leaf=200,
            smoothing=10)    

# Create dummies for categorical features
print("\n   c) Add dummies for categorical features ...\n")
# Create list with columns for which dummies should be created    
col_names = ["ps_car_07_cat",
             "ps_car_03_cat",
             "ps_car_09_cat",
             "ps_car_02_cat",
             "ps_ind_02_cat",
             "ps_car_05_cat"]
# Add dummie features
X_train, X_test = create_dummies(X_train, X_test, col_names)

## ===================== 3. Model training and evaluation ================== ##    
print("\n3. Training and validating model\n")
# Split data into first-stage training set and second stage validation set
X_train["id"] = X_train_ids
X_train, X_validation_2, y_train, y_validation_2 = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0)

# Reset indexes for cross-validation
X_train = X_train.reset_index(drop=True).drop('id', axis=1)
y_train = y_train.reset_index(drop=True)

if train_model:   
    # Initialize model
    xgb = XGBClassifier(**xgb_params)
    
    # Create cross validation iterator
    kf = KFold(n_splits=k, random_state=1, shuffle=True)
    
    # Initialize array for evaluation results
    normalized_gini = []
    
    # Make copies for X, Y to b e used within CV
    X = X_train.copy()
    y = y_train.copy()
    y_pred = np.zeros(X_test.shape[0])
    y_pred_validation = np.zeros(X_validation_2.shape[0])
    
    for train_index, validation_index in kf.split(X, y):
        print("Cross-validation, Fold %d" % (len(normalized_gini)+1))
        
        # Split data into training and testing set
        X_train = X.iloc[train_index,:].copy()
        X_validate = X.iloc[validation_index,:].copy()
        y_train = y[train_index]
        y_validate = y[validation_index]
        
        # Train the model
        if False:
            eval_set=[(X_validate, y_validate)]
            xgb = xgb.fit(X_train, 
                          y_train, 
                          eval_set=eval_set,
                          eval_metric=gini_xgb,
                          early_stopping_rounds=30,
                          verbose=False)
        
        xgb = xgb.fit(X_train, y_train)
        
        # Test the model
        normalized_gini.append(test_model(xgb, X_validate, y_validate))

        # Make test set prediction
        y_pred += xgb.predict_proba(X_test[X_train.columns])[:,1]
        
        # Make predictions for the second stage training set
        y_pred_validation += xgb.predict_proba(X_validation_2[X_train.columns])[:,1]

        del X_train, X_validate, y_train, y_validate        

    # Evaluate results from CV
    print("Normalized gini coefficient %f +/- %f" % (np.mean(normalized_gini), 
                                                     2*np.std(normalized_gini)))
    
    # Calculate prediction as average of fold-prediction
    y_pred /= k
    y_pred_validation /= k

## =========================== 4. Output results =========================== ##
if True:
    # Create dataframes for second stage model
    second_stage_train_test = pd.DataFrame({'id':X_validation_2.id, 'xgb_pred':y_pred_validation, 'target':y_validation_2})
    
    # Output results
    second_stage_train_test.to_csv('../03_Results/xgb_2_stage_train_test.csv', sep=',', index=False)

if make_predictions:
    print("\n4. Saving results\n")
    
    # Create output dataframes
    submission = pd.DataFrame({'id':X_test_ids, 'target':y_pred})
    
    # Output results
    submission.to_csv('../03_Results/xgb_prediction.csv', sep=',', index=False)
