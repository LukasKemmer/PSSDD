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



## ========================= 1. Load and clean data ======================== ##
train = pd.read_csv('../01_Data/train.csv')
train = pd.concat((train, pd.read_csv('../01_Data/train_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
test = pd.read_csv('../01_Data/sample_submission_v2.csv')
members = pd.read_csv('../01_Data/members_v3.csv')

transactions = pd.read_csv('../01_Data/transactions.csv')
transactions = pd.concat((transactions, pd.read_csv('../01_Data/transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)

## ========================= 2. Feature engineering ======================== ##


#transactions = pd.DataFrame(transactions['msno'].value_counts().reset_index())
current_transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
current_transactions = current_transactions.drop_duplicates(subset=['msno'], keep='first')

#discount
current_transactions['discount'] = current_transactions['plan_list_price'] - current_transactions['actual_amount_paid']
#amt_per_day
current_transactions['amt_per_day'] = current_transactions['actual_amount_paid'] / current_transactions['payment_plan_days']
#is_discount
current_transactions['is_discount'] = current_transactions.discount.apply(lambda x: 1 if x > 0 else 0)
#membership_duration
current_transactions['membership_days'] = pd.to_datetime(current_transactions['membership_expire_date']).subtract(pd.to_datetime(current_transactions['transaction_date'])).dt.days.astype(int)

train['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([train, test], axis = 0)

combined = pd.merge(combined, members, how='left', on='msno')
members = []; print('members merge...') 

gender = {'male':1, 'female':2}
combined['gender'] = combined['gender'].map(gender)

combined = pd.merge(combined, current_transactions, how='left', on='msno')
#transactions=[]; print('transaction merge...')

train = combined[combined['is_train'] == 1]
test = combined[combined['is_train'] == 0]

train.drop(['is_train'], axis = 1, inplace = True)
test.drop(['is_train'], axis = 1, inplace = True)

del combined

last_user_logs = []


df_iter = pd.read_csv('../01_Data/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)


i = 0 #~400 Million Records - starting at the end but remove locally if needed
for df in df_iter:
    if i>35:
        if len(df)>0:
            print(df.shape)
            p = Pool(cpu_count())
            df = p.map(transform_df, np.array_split(df, cpu_count()))   
            df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
            df = transform_df2(df)
            p.close(); p.join()
            last_user_logs.append(df)
            print('...', df.shape)
            df = []
    i+=1

def transform_df(df):
    df = pd.DataFrame(df)
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

def transform_df2(df):
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

last_user_logs.append(transform_df(pd.read_csv('../01_Data/user_logs_v2.csv')))
last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)
last_user_logs = transform_df2(last_user_logs)

train = pd.merge(train, last_user_logs, how='left', on='msno')
test = pd.merge(test, last_user_logs, how='left', on='msno')
last_user_logs=[]

## Add combination features
train['autorenew_&_not_cancel'] = ((train.is_auto_renew == 1) == (train.is_cancel == 0)).astype(np.int8)
test['autorenew_&_not_cancel'] = ((test.is_auto_renew == 1) == (test.is_cancel == 0)).astype(np.int8)

train['notAutorenew_&_cancel'] = ((train.is_auto_renew == 0) == (train.is_cancel == 1)).astype(np.int8)
test['notAutorenew_&_cancel'] = ((test.is_auto_renew == 0) == (test.is_cancel == 1)).astype(np.int8)

# Add totals for transactions
total_transactions = transactions.groupby("msno").agg({
        "payment_plan_days" : np.sum,
        "actual_amount_paid" : np.sum,
        "plan_list_price" : [np.sum, np.mean]}).reset_index()

# update name
total_transactions.columns = ["msno", 
                                "total_payment_plan_days", 
                                "total_actual_amount_paid", 
                                "total_plan_list_price", 
                                "mean_plan_list_price"]
    
total_transactions["total_discount"] = total_transactions.total_plan_list_price - \
                                         total_transactions.total_actual_amount_paid

# Merge
train = pd.merge(train, total_transactions, how="left", on="msno")
test = pd.merge(test, total_transactions, how="left", on="msno")

train = pd.merge(train, user_logs_raw, how="left", on="msno")
test = pd.merge(test, user_logs_raw, how="left", on="msno")

# Fill nans with 0
train = train.fillna(0)
test = test.fillna(0)

## ===================== 3. Model training and evaluation ================== ##
print("\n3. Training and validating model\n")

# Set model parameters for LGBM
lgb_params = {
        "num_leaves" : 10, # Maximum tree leaves for base learners
        "min_data_in_leaf" : 200, # Minimum amount of data per leaf
        "max_depth" : 4, # Max. depth of the base learner
        "max_bin" : 10, # Max number of bins that feature values will be bucketed in
        "learning_rate" : 0.04, # Shrinkage rate
        "objective" : "binary", # Specifies the learning task
        "metric" : "binary_logloss", # Evaluation metric
        "sub_feature" : 0.8, # % of features to be selected randomly per round
        "num_iterations" : 1000, # Number of boosting iterations
        "random_state" : 0, # Random number seed
        "min_split_gain" : 0.1}

k = 5

# Initialize model
model = LGBMClassifier(**lgb_params)

# Create cross validation iterator
kf = StratifiedKFold(n_splits=k, random_state=0, shuffle=False)

# Initialize array for evaluation results
log_loss_val = []

# Initialize array for predictions
y_pred = []

# Make copies for X, Y to b e used within CV
X = train.drop(["msno", "is_churn"], axis=1).copy()
y = train["is_churn"].copy()

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
    y_pred.append(model.predict_proba(test[X.columns])[:,1])
    
    # delete temporal dataframes
    del X_train, X_validate, y_train, y_validate
    
# Evaluate results from CV
print("Log loss %f +/- %f" % (np.mean(log_loss_val), 2 * np.std(log_loss_val)))

print("\n4. Saving results\n")
    
# Create output dataframes
submission = pd.DataFrame({'msno':test.msno, 'is_churn': np.mean(y_pred, axis=0)})
    
# Output results
submission.to_csv('../03_Results/lgbm_prediction.csv', sep=',', index=False)

