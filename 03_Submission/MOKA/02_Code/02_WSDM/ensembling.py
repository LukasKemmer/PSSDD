#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:03:18 2017

@author: lukaskemmer
"""
import pandas as pd
import numpy as np

# Set weights for submission files
weights = np.array([1, 1, 1, 1, 1, 1])

# Load submission files
xgb = pd.read_csv('../03_Results/xgb.csv')
xgb_2 = pd.read_csv('../03_Results/xgb_2.csv')
xgb_3 = pd.read_csv('../03_Results/xgb_3.csv')
lgbm = pd.read_csv('../03_Results/lgbm.csv')
lgbm_2 = pd.read_csv('../03_Results/lgbm_2.csv')
lgbm_3 = pd.read_csv('../03_Results/lgbm_3.csv')

# Merge submissions
is_churn = np.exp((weights[0]*np.log(xgb.is_churn) +\
                 weights[1]*np.log(xgb_2.is_churn) + \
                 weights[2]*np.log(xgb_3.is_churn) + \
                 weights[3]*np.log(lgbm.is_churn) + \
                 weights[4]*np.log(lgbm_2.is_churn) + \
                 weights[5]*np.log(lgbm_3.is_churn)
                 ) / np.sum(weights))

# Create data frame for submission
submission = pd.DataFrame({"msno":xgb.msno, "is_churn":is_churn})

# Save submission
submission.to_csv('../03_Results/ensemble.csv', sep=',', index=False)