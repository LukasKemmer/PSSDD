#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:11:11 2017

@author: lukaskemmer
"""
import pandas as pd

def read_data():
    # Use Pandas to read data into data frames. NA Values are marked by '-1'
    X_sub = pd.read_csv('../01_Data/test.csv', na_values='-1')
    X_train = pd.read_csv('../01_Data/train.csv', na_values='-1')
    
    return X_train, X_sub, X_train.columns

def format_data(X_train, X_sub, column_names):
    # Set missing values to none
    #X_train[X_train==-1] = None
    #X_sub[X_sub==-1] = None
    
    # Adjust data types for columns with binary data
    bin_cols = [c for c in column_names if '_bin' in c]
    X_train[bin_cols] = X_train[bin_cols].astype('bool')
    X_sub[bin_cols] = X_sub[bin_cols].astype('bool')
    
    # Adjust data types for columns with categorical data
    for c in column_names:
        if '_cat' in c:
            X_train[c] = X_train[c].astype('category')
            X_sub[c] = X_sub[c].astype('category')

    return X_train, X_sub

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
    
    