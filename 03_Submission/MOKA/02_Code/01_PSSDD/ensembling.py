#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:03:18 2017

@author: lukaskemmer
"""
import pandas as pd
import numpy as np

# Set weights for submission files
weights = np.array([2, 2, 1, 1, 1, 2, 2])

# Load submission files
xgboost = pd.read_csv('../03_Results/xgb_prediction.csv')
lgbm1 = pd.read_csv('../03_Results/lgbm_prediction_45_comb_1.csv')
lgbm2 = pd.read_csv('../03_Results/lgbm_prediction_45_comb_2.csv')
lgbm3 = pd.read_csv('../03_Results/lgbm_prediction_45_comb_3.csv')
lgbm4 = pd.read_csv('../03_Results/lgbm_prediction_change.csv')
lgbm_diego = pd.read_csv('../03_Results/lgbm_diego.csv')
nn_prediction = pd.read_csv('../03_Results/nn_prediction.csv')

#lgbm_prediction_change

# Merge submissions
target = np.exp((weights[0]*np.log(xgboost.target) +\
                 weights[1]*np.log(lgbm1.target) + \
                 weights[2]*np.log(lgbm2.target) + \
                 weights[3]*np.log(lgbm3.target) + \
                 weights[4]*np.log(lgbm4.target) + \
                 weights[5]*np.log(lgbm_diego.target) + \
                 weights[6]*np.log(nn_prediction.target)
                 ) / np.sum(weights))

# Adjust format
submission = pd.DataFrame({"id":xgboost.id.astype("int32"), "target":target})

# Save submission
submission.to_csv('../03_Results/stacked_submission.csv', sep=',', index=False)