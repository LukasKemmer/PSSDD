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
from sklearn import model_selection, tree, linear_model, feature_selection, decomposition
from xgboost import XGBClassifier
from xgboost import plot_tree
from functions import *
from sklearn.naive_bayes import GaussianNB

## ============================ 0. Set parameters ========================== ##

load_new_data = True # Set True to reload and format data
subset_data = True # Set True to use only 10000 records for training
train_mode = True # Set True if model should be trained
make_predictions = False # Set true if prediction for submission should be made
k = 5 # Number of folds for cross-validation
na_threshold = 100 # All features with more %NA will be dropped
num_features = 20 # Number of top features that will be used
np.random.seed(123123) # Random seed to produce comparable results

## ======================== 1. Load and prepare data ======================= ##

if load_new_data:
    # Load raw data
    X_train_raw, X_sub_raw, column_names = read_data()
    
    # Format data
    X_train_formatted, X_sub_formatted = format_data(X_train_raw.copy(),
                                                     X_sub_raw.copy(),
                                                     column_names)
    
    # Describe data
    X_train_summary, X_sub_summary, X_train_freq, X_sub_freq, X_train_missing, X_sub_missing = describe_data(X_train_formatted, X_sub_formatted)

# Copy formatted data
X_train = X_train_formatted.copy()
X_sub = X_sub_formatted.copy()

# Randomly subset the data for faster computation during train/test
if subset_data:
    X_train = X_train.sample(n=1000, replace=False).reset_index(drop=True)

# Drop columns that have more than 'na_threshold' % NAs, based on X_train data!
irrelevant_columns = list(X_train_missing[X_train_missing>na_threshold].index)
X_train = X_train.drop(irrelevant_columns, axis=1)
X_sub = X_sub.drop(irrelevant_columns, axis=1)

# Replace remaining NAs with median
if True:
    X_train = X_train.fillna(X_train.median())
    X_sub = X_sub.fillna(X_sub.median())

# Drop all records containing NAs
if False:
    X_train = X_train.dropna(axis=0, how = 'any')
    X_sub = X_sub.dropna(axis=0, how = 'any')

## ============================ 2. Visualize data ========================== ##

visualize_features(X_train, bin_features=False, cat_features=False, 
                   cont_features=False)

pair_plots(X_train, bin_features=False, cat_features=False, 
           cont_features=False)

## ========================== 2. Feature selection ========================= ##

if False:
    # Compute chi squared for feature selection
    chisq = feature_selection.chi2(X_train.drop(['id', 'target'], axis=1), 
                                   X_train.target)

    # Select top x features
    top_features = [X_train.columns[i+2] for i in 
                    (-chisq[0]).argsort()[0:num_features]]
    top_features.extend(['id', 'target'])
    X_train = X_train[top_features]
    
    # Perform PCA to identify main components
    #pca = decomposition.PCA(n_components = 8)
    #helper = X_train[['target', 'id']]
    #X_train = pd.DataFrame(pca.fit_transform(X_train.drop(['target', 'id'], 
    #                       axis=1)))
    #X_train = pd.concat([X_train, helper], axis=1)

## ========================= 3. Feature engineering ======================== ##

# Create dummy features for categorical features
X_train = pd.concat([X_train, pd.get_dummies(X_train.select_dtypes(
        include=['category']))], axis=1)
X_sub = pd.concat([X_sub, pd.get_dummies(X_sub.select_dtypes(
        include=['category']))], axis=1)


# Drop categorical features when using XGBoost
X_train = X_train.drop([col for col in X_train.columns if '_cat' in col], 
                       axis=1)
X_sub = X_sub.drop([col for col in X_sub.columns if '_cat' in col], 
                     axis=1)
# Drop categorical features
#X_train = X_train.select_dtypes(exclude=['category'])
#X_sub = X_sub.select_dtypes(exclude=['category'])

## ===================== 4. Model training and evaluation ================== ##    

# Select a model
#model = tree.DecisionTreeClassifier() # Decision tree
#model = ensemble.RandomForestClassifier(n_estimators=100) # Random forest
#model = linear_model.LogisticRegression() # Logistic regression
#model = naive_bayes.GaussianNB(priors = [1-np.mean(X_train_raw.target), 
#                                        np.mean(X_train_raw.target)])


model = XGBClassifier(max_depth=4, # Maximum tree depth for base learners
                      learning_rate=0.07, # Boosting learning rate (xgb’s “eta”)
                      n_estimators = 400, # Number of boosted trees to fit
                      silent = True, # Whether to print messages while running boosting
                      objective='binary:logistic', # learning objective
                      #booster='gbtree', # Which booster to use: gbtree, gblinear or dart
                      n_jobs = 8, # Number of parallel threads
                      gamma=10, # Minimum loss reduction required to make a further partition on a leaf node of the tree
                      min_child_weight=6, # Minimum sum of instance weight(hessian) needed in a child
                      #max_delta_step=0, # Maximum delta step we allow each tree’s weight estimation to be
                      subsample=.8, # Subsample ratio of the training instance
                      colsample_bytree=.8, # Subsample ratio of columns when constructing each tree
                      #colsample_bylevel=1, # Subsample ratio of columns for each split, in each level
                      reg_alpha=8, # L1 regularization term on weights
                      reg_lambda=1.3, # L2 regularization term on weights
                      scale_pos_weight=1.6) # Balancing of positive and negative weights
                      #base_score=0.5) # The initial prediction score of all instances, global bias

if train_mode:    
    # Create cross validation iterator
    kf = model_selection.KFold(n_splits=k)
    
    # Split X_train into X_train and Y_train
    Y_train = X_train.target
    X_train = X_train.drop(['target', 'id'], axis=1)
    
    # Initialize array for evaluation results
    normalized_gini = []
    
    # Make copies for X, Y to b e used within CV
    X = X_train.copy()
    Y = Y_train.copy()
    Y_pred = np.zeros(X_sub.shape[0])
    
    for train_index, test_index in kf.split(X):
        print("Cross-validation, Fold %d" % (len(normalized_gini)+1))
        
        # Split data into training and testing set
        X_train = X.iloc[train_index,:]
        X_test = X.iloc[test_index,:]
        Y_train = Y[train_index]
        Y_test = Y[test_index]

        # Fit the model
        model = model.fit(X_train, Y_train,
                          eval_metric=gini_xgb,
                          #early_stopping_rounds=50,
                          verbose=False)

        # Test the model
        normalized_gini.append(test_model(model, X_test, Y_test))

        # Make test set prediction
        Y_pred += model.predict_proba(X_sub[X_train.columns])[:,1]

    # Evaluate results from CV
    print("Normalized gini coefficient %f +/- %f" % (np.mean(normalized_gini), 
                                                     2*np.std(normalized_gini)))
    
    # Plot single tree
    plot_tree(model)
    plt.show()

## =========================== 5. Output results =========================== ##

# Calculate prediction
#Y_pred /= k

# Create output dataframe
#results = pd.DataFrame({'id':X_sub['id'], 'target':Y_pred})

# Output results
#results.to_csv('../03_Results/prediction.csv', sep=',', index=False)

if make_predictions:

    # Split X_train into X_train and Y_train
    Y_train = X_train.target
    X_train = X_train.drop(['target', 'id'], axis=1)
    
    # Fit the model
    model = model.fit(X_train, Y_train,
                      eval_metric=gini_xgb,
                      #early_stopping_rounds=50,
                      verbose=False)    
    
    # Predict probabilities
    Y_probs = model.predict_proba(X_sub[X_train.columns])

    # Create output dataframe
    results = pd.DataFrame({'id':X_sub['id'], 'target':Y_probs[:,1]})

    # Output results
    results.to_csv('../03_Results/prediction.csv', sep=',', index=False)
