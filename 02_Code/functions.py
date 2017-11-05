#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:53:45 2017

@author: lukaskemmer
"""
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import cartesian
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