#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:03:18 2017

@author: lukaskemmer
"""
import pandas as pd
import numpy as np

# Set weights for submission files
weights = np.array([2, 1, 1])

# Load submission files
xgboost = pd.read_csv('../03_Results/xgb_prediction.csv')
ada_boost = pd.read_csv('../03_Results/ada_boost_prediction.csv')
logistic_regression = pd.read_csv('../03_Results/log_reg_prediction.csv')

# Merge submissions
submission = (weights[0]*xgboost + weights[1]*ada_boost +  \
              weights[2]*logistic_regression)/np.sum(weights)

# Adjust format
submission['id'] = submission['id'].astype('int32')

# Save submission
submission.to_csv('../03_Results/stacked_prediction.csv', sep=',', index=False)