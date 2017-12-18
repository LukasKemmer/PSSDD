#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:53:45 2017

@author: lukaskemmer
"""
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from sklearn.model_selection import train_test_split

# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
@nb.jit(nopython=True)
def eval_gini(y_true, y_prob):
    """
    Calculates gini coefficient
    y_true : Actual target
    y_prob : Predicted probabilities
    returns (float) : Gini coefficient
    """
    # Uses gini computation by (swaps_random-swaps_sorted)/swaps_random
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0 # resembles number of swaps needed to sort y_true
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    return 1 - 2 * gini / (ntrue * (n - ntrue))

def gini_normalized(a, p):
    """
    Returns the normalized gini coefficient
    a : Actual target values
    p : Predicted target values
    returns (float) : Normalized gini coefficient
    """
    return eval_gini(a, p) / eval_gini(a, a)

def gini_xgb(preds, dtrain):
    """
    Computes gini coefficient for XGB
    preds : Predictions
    dtrain : Actual target
    returns (list) : Gini coefficient
    """
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]

def test_model(model, X_validate, y_validate):
    """
    Tests model accuracy
    X_validate : Validation data
    y_validate : Target of validation data
    model : Trained model
    normalized gini (float) : normalized gini coefficient
    """
    # Predict results
    Y_probs = model.predict_proba(X_validate)
  
    # Calculate and print normalized gini index
    normalized_gini = eval_gini(np.array(y_validate), np.array(Y_probs[:,1]))
    print('Normalized gini coefficient: ', normalized_gini)
    return normalized_gini
    
def recursive_feature_elimination(X, y, model, num_features):   
    """
    Recursive feature elimnination wheare the least important feature gets dropped
    each iteration until unly num_features features remain
    X : Training data
    y : Target of training data
    model : Trained model
    num_features : Number of features that should remain after the recursive feature elimination
    returns (list) : list with the remaining features
    """
    # Recursive feature elimination
    scores = []
    features = []
    while X.shape[1]>num_features:
        print(X.shape)
        X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=X.shape[1])

        # train model
        model = model.fit(X_train, y_train)

        # Find feature with least importance
        column_to_delete = X_train.columns[np.argsort(model.feature_importances_)][0:1]
        print("Drop feature: " + column_to_delete)
        print("%d features remaining" % (X_train.shape[1]))
        scores.append(test_model(model, X_validate, y_validate))
        features.append(X.shape[1])

        # Drop least important feature
        X.drop(column_to_delete, axis=1, inplace=True)
        del X_train, X_validate, y_train, y_validate
    
    # Print / plot results
    print(scores)
    print(features)
    plt.plot(scores)
    plt.show()
    return X.columns