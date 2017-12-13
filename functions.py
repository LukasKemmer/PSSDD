#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:57:26 2017

@author: lukaskemmer
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

def recursive_feature_elimination(X, y, model, num_features):    
    # Recursive feature elimination
    scores = []
    features = []
    while X.shape[1]>num_features:
        print(X.shape)
        X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=X.shape[1])

        # train model
        model = model.fit(X_train, y_train)

        # Find feature with least importance
        #column_to_delete = X_train.columns[np.argmin(model.feature_importances_)]
        print()
        column_to_delete = X_train.columns[np.argsort(model.feature_importances_)][0:1]
        print("Drop feature: " + column_to_delete)
        print("%d features remaining" % (X_train.shape[1]))
        scores.append(log_loss(y_validate, model.predict_proba(X_validate)))
        features.append(X.shape[1])
        # Drop least important feature
        X.drop(column_to_delete, axis=1, inplace=True)
        del X_train, X_validate, y_train, y_validate
    print(scores)
    print(features)
    plt.plot(scores)
    plt.show()
    return X.columns

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