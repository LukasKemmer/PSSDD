#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:08:48 2017

@author: lukaskemmer
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from load_data import read_data, load_last_user_logs, get_num_user_logs

## ============================ 0. Set parameters ========================== ##

# Set model parameters for LGBM
lgb_params = {
        "num_leaves" : 10, # Maximum tree leaves for base learners
        "min_data_in_leaf" : 200, # Minimum amount of data per leaf
        "max_depth" : 3, # Max. depth of the base learner
        "max_bin" : 10, # Max number of bins that feature values will be bucketed in
        "learning_rate" : 0.04, # Shrinkage rate
        "objective" : "binary", # Specifies the learning task
        "metric" : "binary_logloss", # Evaluation metric
        "sub_feature" : 0.5, # % of features to be selected randomly per round
        "num_iterations" : 1000, # Number of boosting iterations
        "random_state" : 0, # Random number seed
        "min_split_gain" : 0.05 # Minimum gain required for split
        } 

# Set number of folds for cross-validation
k = 5

## ========================= 1. Load and clean data ======================== ##
print("\n1. Load and data ...\n")
X_train, X_test, members, transactions = read_data()

## ========================= 2. Feature engineering ======================== ##
print("\n2. Adding and selecting features ...\n")

# Sort transactions by date
current_transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)

# Get features for current transaction
print("\n   a) Creating features from most recent transaction ...\n")

# get most recent transaction
current_transactions = current_transactions.drop_duplicates(subset=["msno"], keep="first")

# Calculate discount
current_transactions["discount"] = current_transactions["plan_list_price"] - current_transactions["actual_amount_paid"]

# Calculate amount per day
current_transactions["amt_per_day"] = current_transactions["actual_amount_paid"] / current_transactions["payment_plan_days"]
# set inf-values of amt_per_day to 0
current_transactions.loc[np.isinf(current_transactions.amt_per_day), "amt_per_day"] = 0

# Check if discount
current_transactions["is_discount"] = current_transactions.discount.apply(lambda x: 1 if x > 0 else 0)

# Calculate the number of membership days within the current transaction
current_transactions["membership_days"] = pd.to_datetime(current_transactions["membership_expire_date"], format="%Y%m%d").subtract(pd.to_datetime(current_transactions["transaction_date"], format="%Y%m%d")).dt.days.astype(int)
current_transactions.loc[current_transactions["membership_days"]<0, "membership_days"] = 0

# Add current transactions
X_train = pd.merge(X_train, current_transactions, how="left", on="msno")
X_test = pd.merge(X_test, current_transactions, how="left", on="msno")

# delete current_transactions
del current_transactions

# Add totals for transactions
print("\n   b) Create features for all transactions ...\n")
# aggregate transaction data by msno
total_transactions = transactions.groupby("msno").agg({
        "payment_plan_days" : np.sum,
        "actual_amount_paid" : np.sum,
        "transaction_date" : np.min,
        "plan_list_price" : [np.sum, np.mean]}).reset_index()

# update column names
total_transactions.columns = ["msno", 
                                "total_payment_plan_days", 
                                "total_actual_amount_paid", 
                                "first_transaction_date",
                                "total_plan_list_price", 
                                "mean_plan_list_price"]

total_transactions["total_discount"] = total_transactions.total_plan_list_price - \
                                         total_transactions.total_actual_amount_paid

# Delete transactions dataframe
del transactions

# Add total_transactions
X_train = pd.merge(X_train, total_transactions, how="left", on="msno")
X_test = pd.merge(X_test, total_transactions, how="left", on="msno")
del total_transactions

# Create new features
X_train["membership_duration"] = pd.to_datetime(
        X_train["membership_expire_date"], format="%Y%m%d").subtract(
                pd.to_datetime(X_train["first_transaction_date"], 
                               format="%Y%m%d")).dt.days.astype(int)
X_test["membership_duration"] = pd.to_datetime(
        X_test["membership_expire_date"], format="%Y%m%d").subtract(
                pd.to_datetime(X_test["first_transaction_date"], 
                               format="%Y%m%d")).dt.days.astype(int)

# Map genders to 1 and 2
print("\n   c) Map genders to 1 and 2 ...\n")
# add member data to train and test
X_train = pd.merge(X_train, members, how="left", on="msno")
X_test = pd.merge(X_test, members, how="left", on="msno")

gender = {'male':1, 'female':2}
X_test["gender"] = X_test["gender"].map(gender)
X_train["gender"] = X_train["gender"].map(gender)
del gender

# Create features from last logs
print("\n   d) Create features from last logs ...\n")
last_user_logs = load_last_user_logs()

X_train = pd.merge(X_train, last_user_logs, how='left', on='msno')
X_test = pd.merge(X_test, last_user_logs, how='left', on='msno')
del last_user_logs

# Create combination features
print("\n   e) Create and add combination features ...\n")
X_train['autorenew_&_not_cancel'] = ((X_train.is_auto_renew == 1) == (X_train.is_cancel == 0)).astype(np.int8)
X_test['autorenew_&_not_cancel'] = ((X_test.is_auto_renew == 1) == (X_test.is_cancel == 0)).astype(np.int8)

X_train['notAutorenew_&_cancel'] = ((X_train.is_auto_renew == 0) == (X_train.is_cancel == 1)).astype(np.int8)
X_test['notAutorenew_&_cancel'] = ((X_test.is_auto_renew == 0) == (X_test.is_cancel == 1)).astype(np.int8)

# Get number of logs per user
print("\n   f) Get number of logs per user ...\n")
num_user_logs = get_num_user_logs()

# Add num_user_logs
X_train = pd.merge(X_train, num_user_logs, how="left", on="msno")
X_test = pd.merge(X_test, num_user_logs, how="left", on="msno")
del num_user_logs

# Fill nans with 0
print("\n   h) Replace NAs ...\n")
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

## ===================== 3. Model training and evaluation ================== ##
print("\n3. Training and validating model\n")

# Initialize model
model = LGBMClassifier(**lgb_params)

# Create cross validation iterator
kf = StratifiedKFold(n_splits=k, random_state=0, shuffle=False)

# Initialize array for evaluation results
log_loss_val = []

# Initialize array for predictions
y_pred = []

# Make copies for X, Y to b e used within CV
X = X_train.drop(["msno", "is_churn"], axis=1).copy()
y = X_train["is_churn"].copy()

# (stratified) Cross validation
for train_index, validation_index in kf.split(X, y):
    print("Cross-validation, Fold %d" % (len(log_loss_val) + 1))

    # Split data into training and testing set
    X_train = X.iloc[train_index, :].copy()
    X_validate = X.iloc[validation_index, :].copy()
    y_train = y[train_index]
    y_validate = y[validation_index]
    
    # Train the model
    model = model.fit(X_train, y_train)
    
    # Test the model
    log_loss_val.append(log_loss(y_validate, model.predict_proba(X_validate)))
    print("Log loss: %f" % log_loss_val[-1])
    
    # Make predictions
    y_pred.append(np.log(model.predict_proba(X_test[X.columns])[:,1]))
    
    # delete temporal dataframes
    del X_train, X_validate, y_train, y_validate
    
# Evaluate results from CV
print("Log loss %f +/- %f" % (np.mean(log_loss_val), 2 * np.std(log_loss_val)))

## =========================== 4. Output results =========================== ##    
# Create output dataframe
submission = pd.DataFrame({'msno':X_test.msno, 'is_churn': np.exp(np.mean(y_pred, axis=0))})
    
# Output results
print("\n4. Saving results\n")
submission.to_csv('../03_Results/lgbm_2.csv', sep=',', index=False)

