#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 22:08:50 2017

@author: lukaskemmer
"""
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import itertools
from scipy.misc import comb
from xgboost import plot_importance

def replace_nas(X_train, X_test):
    # Calculate median of X_train and X_test combined
    median = X_train.append(X_test).median()
    
    # Replace remaining NAs with median
    X_train = X_train.fillna(median)
    X_test = X_test.fillna(median)

    return X_train, X_test

def interaction_features(X):
    X['mult'] = X['ps_reg_01'] * X['ps_reg_03'] * X['ps_reg_02']
    X['ps_car'] = X['ps_car_13'] * X['ps_reg_03'] * X['ps_car_13']
    for c in itertools.combinations(X.select_dtypes(include=['float64']).columns, r=2):
        X[c[0]+'-'+c[1]] = X[c[0]].astype('float64') * X[c[1]].astype('float64')
    return X

def polynomial_features(X):
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

def add_combination_features(X_train, X_test, combs):
    # add combinations
    combs = [('ps_reg_01', 'ps_car_02_cat'),
             ('ps_reg_01', 'ps_car_04_cat')]
    
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

def create_dummies(X_train, X_test, col_names):
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
    
    return X_train, X_test

def add_features(X_train, X_test, y_train):    
    # Add polynomial features
    print("\n     I) Add polynomial features ...\n")
    #X_train = polynomial_features(X_train)
    #X_test = polynomial_features(X_test)

    print("\n     II) Add interaction terms ...\n")    
    # Add interactions of features
    #X_train = interaction_features(X_train)
    #X_test = interaction_features(X_test)
    
    # add combinations
    combs = [('ps_reg_01', 'ps_car_02_cat'),
             ('ps_reg_01', 'ps_car_04_cat')]
    
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

    # Encode categorical data
    print("\n     III) Add encoded categorical features ...\n")
    for col in [col for col in X_train.columns if '_cat' in col]:
        X_train[col + "_avg"], X_test[col + "_avg"] = target_encode(
                trn_series=X_train[col],
                tst_series=X_test[col],
                target=y_train,
                min_samples_leaf=200,
                smoothing=10)
    
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

    
    return X_train, X_test

def get_feature_importance(X, y, model):
    
    # Fit the model
    model.fit(X, y)
    importances = model.feature_importances_

    return pd.DataFrame({'feature' : X.columns, 'importance' : importances}
                        ).sort_values('importance', ascending=False)

def target_encode(trn_series=None,    # Revised to encode validation series
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