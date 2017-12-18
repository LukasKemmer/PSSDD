#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:11:11 2017

@author: lukaskemmer
"""
import pandas as pd

def read_data():
    """
    Loads and returns test and train data
    X_train (DataFrame) : Training data (without target column)
    y_train (Series) : target column from training data
    X_test (DataFrame) : Testing data
    X_test_ids (Series) : ID column from testing data
    X_train_ids (Series) : ID column from training data
    X_train.columns (Array) : Column names from training data
    """
    # Use Pandas to read data into data frames. NA Values are marked by '-1'
    X_test = pd.read_csv('../01_Data/test.csv', na_values='-1')
    X_train = pd.read_csv('../01_Data/train.csv', na_values='-1')

    # Split X_train in X_train and y_train
    y_train = X_train.target
    
    # Save the IDs for the testing and training set set
    X_test_ids = X_test.id
    X_train_ids = X_train.id
    X_train = X_train.drop(['target', 'id'], axis=1)

    # Drop id in the testing set
    X_test = X_test.drop(['id'], axis=1)
    
    return X_train, y_train, X_test, X_test_ids, X_train_ids, X_train.columns

def format_data(X_train, X_test, column_names):
    """
    Formats training and testing data
    X_train : Training data
    X_test : Testing data
    column_names : Array with column names of X_train
    X_train (DataFrame) : Formatted X_train
    X_test (DataFrame) : Formatted X_test
    """
    # Adjust data types for columns with binary data
    bin_cols = [c for c in column_names if '_bin' in c]
    X_train[bin_cols] = X_train[bin_cols].astype('bool')
    X_test[bin_cols] = X_test[bin_cols].astype('bool')
    
    return X_train, X_test

def describe_data(X_train, X_test):
    """
    Creates descriptions of training and testing data
    X_train : Training data
    X_test : Testing data
    X_train_summary (DataFrame) : Summary of training data
    X_test_summary (DataFrame) : Summary of testing data
    X_train_freq (list) : Frequency table for training data
    X_test_freq (list) : Frequency table for testing data
    X_train_missing (Series) : Series for % missing in train
    X_test_missing (Series) : Series for % missing in test
    """
    # Describe the dataframes
    X_train_summary = X_train.describe()
    X_test_summary = X_test.describe()
    
    # Generate a (relative) frequency table for the data
    X_train_freq = [X_train[column].value_counts(normalize=True) for column in X_train.columns]
    X_test_freq = [X_test[column].value_counts(normalize=True) for column in X_test.columns]
    
    # Generate a list with % NA for each column
    X_train_missing = X_train.isnull().mean() * 100
    X_test_missing = X_test.isnull().mean() * 100

    return X_train_summary, X_test_summary, X_train_freq, X_test_freq, X_train_missing, X_test_missing