#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 22:08:50 2017

@author: lukaskemmer
"""
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import itertools

def replace_nas(X_train, X_test):
    """
    Replaces NA in training and testing data by median
    X_train : Training data
    X_test : Testing data
    X_train (DataFrame) : X_train with replaced NAs
    X_test (DataFrame) : X_test with replaced NAs
    """
    # Calculate median of X_train and X_test combined
    median = X_train.append(X_test).median()
    
    # Replace remaining NAs with median
    X_train = X_train.fillna(median)
    X_test = X_test.fillna(median)

    return X_train, X_test

def interaction_features(X):
    """
    Creates interaction-features for input data
    X : Input data frame
    X (DataFrame) : Data frame with added interaction features
    """
    # Add interaction features "mult" and "ps_car"
    X['mult'] = X['ps_reg_01'] * X['ps_reg_03'] * X['ps_reg_02']
    X['ps_car'] = X['ps_car_13'] * X['ps_reg_03'] * X['ps_car_13']

    # Add interaction features for all features that are not _cat or _bin
    cols = [col for col in X.columns if not "_cat" in col and not "_bin" in col]
    for c in itertools.combinations(cols ,r=2):
        X[c[0]+'-'+c[1]] = X[c[0]].astype('float64') * X[c[1]].astype('float64')
        
    return X

def polynomial_features(X):
    """
    Creates polynomial features of power up until 6
    X : Input data frame
    X (DataFrame) : Data frame with added polynomial features
    """
    # Create polynomial features only for continuos features
    # No point in polynomial features for ordinal variables (int) because polynomes
    # Wont change order when of ints > 0 (which is the case)
    for col in X.select_dtypes(include=['float64']).columns:
        X[col+'_pow2'] = np.power(X[col], 2)
        X[col+'_pow3'] = np.power(X[col], 3)
        X[col+'_pow4'] = np.power(X[col], 4)
        X[col+'_pow5'] = np.power(X[col], 5)
        X[col+'_pow6'] = np.power(X[col], 6)
    return X


def create_dummies(X_train, X_test, col_names):
    """
    Creates dummie features for columns in col_names (categorical)
    X_train : Training data
    X_test : Testing data
    col_names : Array with column names for which dummies should be created
    X_train (DataFrame) : X_train with dummie features
    X_test (DataFrame) : X_test with dummie features
    """
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
        X.drop(col, axis=1, inplace=True) # bugfix
    
    # Split X_train and X_test
    X_train = X.loc[X['name']=='X_train', X.columns != 'name']
    X_test = X.loc[X['name']=='X_test', X.columns != 'name']   
    
    return X_train, X_test

def add_combination_features(X_train, X_test, combs):
    """
    Adds combination features for columns in combs
    X_train : Training data
    X_test : Testing data
    combs : Column names for which combination features should be created
    X_train (DataFrame) : Formatted X_train
    X_test (DataFrame) : Formatted X_test
    """
    for n_c, (f1, f2) in enumerate(combs):
        # Create feature combinations
        name1 = f1 + "_plus_" + f2
        X_train[name1] = X_train[f1].astype(str) + "_" + X_train[f2].astype(str)
        X_test[name1] = X_test[f1].astype(str) + "_" + X_test[f2].astype(str)
        
        # Encode labels
        lbl = LabelEncoder()
        lbl.fit(list(X_train[name1].values) + list(X_test[name1].values))
        X_train[name1] = lbl.transform(list(X_train[name1].values))
        X_test[name1] = lbl.transform(list(X_test[name1].values))
        
    return X_train, X_test

def get_feature_importance(X, y, model):
    """
    Calculates the feature importance of "model" (e.g. xgb or lgbm model)
    X : Training data
    y : Target of training data
    model : Model to be used to calculate feature_importance
    returns (DataFrame) : (sorted) Feature importances of the model
    """
    
    # Fit the model
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_

    # Return results in data frame sorted by feature importance
    return pd.DataFrame({'feature' : X.columns, 'importance' : importances}
                        ).sort_values('importance', ascending=False)

def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    
    # Apply average function to all target data
    prior = target.mean()
    
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index

    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    
    return ft_trn_series, ft_tst_series