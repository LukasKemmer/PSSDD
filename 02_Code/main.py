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
from sklearn import model_selection, tree, linear_model, metrics, feature_selection, decomposition
from xgboost import XGBClassifier
from functions import *
from sklearn.naive_bayes import GaussianNB

## ============================ 0. Set parameters ========================== ##

load_new_data = True # Set True to reload and format data
subset_data = False # Set True to use only 10000 records for training
na_threshold = 40 # All features with more %NA will be dropped
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
    X_train = X_train.sample(n=10000, replace=False)

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



## ===================== 4. Model training and evaluation ================== ##    

# Undersample training data to get 1:1 distribution of TRUE / FALSE
if False:
    X_train_True = X_train[X_train.target==1]
    X_train_False = X_train[X_train.target==0].sample(n=X_train_True.shape[0], 
                           replace=False, axis=0)
    X_train = X_train_True.append(X_train_False)

if True:    
    # Split training data into training and validation dataset
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_train.drop(['target', 'id'], axis=1), 
                                                                        X_train.target, test_size=0.2)
    
    # Select a model
    #model = tree.DecisionTreeClassifier() # Decision tree
    #model = ensemble.RandomForestClassifier(n_estimators=100) # Random forest
    #model = linear_model.LogisticRegression() # Logistic regression
    #model = naive_bayes.GaussianNB(priors = [1-np.mean(X_train_raw.target), 
    #                                        np.mean(X_train_raw.target)])
    model = XGBClassifier(max_depth=4,
                          objective="binary:logistic",
                          learning_rate=0.07, 
                          subsample=.8,
                          min_child_weight=6,
                          colsample_bytree=.8,
                          scale_pos_weight=1.6,
                          gamma=10,
                          reg_alpha=8,
                          reg_lambda=1.3)
    
    # Drop categorical features when using XGBoost
    X_train = X_train.drop([col for col in X_train.columns if '_cat' in col], 
                           axis=1)
    X_test = X_test.drop([col for col in X_test.columns if '_cat' in col], 
                           axis=1)
    
    # Fit the model
    model = model.fit(X_train, Y_train,
                      #eval_metric=gini_xgb,
                      #early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      verbose=False)

    # Predict results
    Y_probs = model.predict_proba(X_test)
    Y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = model.score(X_test, Y_test)
    print('Accuracy: ', accuracy)
    
    # Calculate and plot confusion matrix
    cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['0', '1'], normalize = True,
                          title='Normalized confusion matrix')
    
    # Calculate normalized gini index
    normalized_gini = eval_gini(np.array(Y_test), np.array(Y_probs[:,1]))
    print('Normalized gini coefficient: ', normalized_gini)

## =========================== 5. Output results =========================== ##

if True:
    # Predict classes
    #Y_predicted = model.predict(X_sub.drop('id', axis=1))
    Y_probs = model.predict_proba(X_sub[X_train.columns])

    # Create output dataframe
    results = pd.DataFrame({'id':X_sub['id'], 'target':Y_probs[:,1]})

    # Output results
    results.to_csv('../03_Results/prediction.csv', sep=',', index=False)
