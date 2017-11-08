#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:53:45 2017

@author: lukaskemmer
"""
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.utils.extmath import cartesian
from data_visualization import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numba as nb

def gini(y_actual, y_pred):
  return 2*roc_auc_score(y_actual, y_pred)-1

# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
@nb.jit(nopython=True)
def eval_gini(y_true, y_prob):
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    return 1 - 2 * gini / (ntrue * (n - ntrue))

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]

def test_model(model, X_test, Y_test):
    # Predict results
    Y_probs = model.predict_proba(X_test)
    Y_pred = model.predict(X_test)
    '''
    # Calculate accuracy
    accuracy = model.score(X_test, Y_test)
    print('Accuracy: ', accuracy)
        
    # Calculate and plot confusion matrix
    cnf_matrix = confusion_matrix(Y_test, Y_pred)
    
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['0', '1'], normalize = True,
                          title='Normalized confusion matrix')
    '''    
    # Calculate normalized gini index
    normalized_gini = eval_gini(np.array(Y_test), np.array(Y_probs[:,1]))
    print('Normalized gini coefficient: ', normalized_gini)
    return normalized_gini
    