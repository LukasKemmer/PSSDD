#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:11:11 2017

@author: lukaskemmer
"""
import pandas as pd
from sklearn.model_selection import train_test_split

def read_data():
    # Use Pandas to read data into data frames. NA Values are marked by '-1'
    X_test = pd.read_csv('../01_Data/test.csv', na_values='-1')
    X_train = pd.read_csv('../01_Data/train.csv', na_values='-1')

    # Split X_train in X_train and y_train
    y_train = X_train.target
    X_train_ids = X_train.id
    X_train = X_train.drop(['target', 'id'], axis=1)
    
    # Save the IDs for the testing set
    X_test_ids = X_test.id
    X_test = X_test.drop(['id'], axis=1)
    
    return X_train, y_train, X_test, X_test_ids, X_train_ids, X_train.columns

def format_data(X_train, X_test, column_names):
    # Adjust data types for columns with binary data
    bin_cols = [c for c in column_names if '_bin' in c]
    X_train[bin_cols] = X_train[bin_cols].astype('bool')
    X_test[bin_cols] = X_test[bin_cols].astype('bool')
    
    # Adjust data types for columns with categorical data
    '''
    for c in [c for c in column_names if '_cat' in c]:        
        X_train[c] = X_train[c].astype('category')
        X_test[c] = X_test[c].astype('category')
        '''
    return X_train, X_test

def describe_data(X_train, X_sub):
    # Describe the dataframes
    X_train_summary = X_train.describe()
    X_sub_summary = X_sub.describe()
    
    # Generate a (relative) frequency table for the data
    X_train_freq = [X_train[column].value_counts(normalize=True) for column in X_train.columns]
    X_sub_freq = [X_sub[column].value_counts(normalize=True) for column in X_sub.columns]
    
    # Generate a list with % NA for each column
    X_train_missing = X_train.isnull().mean() * 100
    X_sub_missing = X_sub.isnull().mean() * 100

    return X_train_summary, X_sub_summary, X_train_freq, X_sub_freq, X_train_missing, X_sub_missing