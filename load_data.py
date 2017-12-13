#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:28:17 2017

@author: lukaskemmer
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import gc; gc.enable()

def read_data():
    # Load and concat data
    train = pd.read_csv('../01_Data/train.csv')
    train = pd.concat((train, pd.read_csv('../01_Data/train_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
    test = pd.read_csv('../01_Data/sample_submission_v2.csv')
    members = pd.read_csv('../01_Data/members_v3.csv')
    transactions = pd.read_csv('../01_Data/transactions.csv')
    transactions = pd.concat((transactions, pd.read_csv('../01_Data/transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)

    return train, test, members, transactions

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

def load_last_user_logs():
        
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
        
    last_user_logs.append(transform_df(pd.read_csv('../01_Data/user_logs_v2.csv')))
    last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)
    last_user_logs = transform_df2(last_user_logs)
    
    return last_user_logs

def get_num_user_logs():
    user_logs = pd.read_csv("../01_Data/user_logs.csv", usecols=['msno'])
    user_logs = pd.DataFrame(user_logs['msno'].value_counts().reset_index())
    user_logs.columns = ['msno','logs_count']
    return user_logs

'''
def read_data():
    # Read the original files
    members = pd.read_csv("../01_Data/members_v3.csv")
    train_v1 = pd.read_csv("../01_Data/train.csv")
    train_v2 = pd.read_csv("../01_Data/data 2/churn_comp_refresh/train_v2.csv")
    transactions_v1 = pd.read_csv("../01_Data/transactions.csv")
    transactions_v2 = pd.read_csv("../01_Data/data 3/churn_comp_refresh/transactions_v2.csv")
    test = pd.read_csv("../01_Data/data/churn_comp_refresh/sample_submission_v2.csv")
    
    # Concatenate train and transaction 
    train = pd.concat([train_v1, train_v2], axis=0, ignore_index=True).reset_index(drop=True)
    transactions = pd.concat([transactions_v1, transactions_v2], axis=0, ignore_index=True).reset_index(drop=True)
    
    # Adjust data types
    transactions["payment_method_id"] = transactions["payment_method_id"].astype("category")
    transactions["payment_plan_days"] = transactions["payment_plan_days"].astype("int16")
    transactions["plan_list_price"] = transactions["plan_list_price"].astype("int16")
    transactions["actual_amount_paid"] = transactions["actual_amount_paid"].astype("int16")
    transactions["is_auto_renew"] = transactions["is_auto_renew"].astype("bool")
    transactions["is_cancel"] = transactions["is_cancel"].astype("bool")
    transactions["transaction_date"] = pd.to_datetime(transactions.transaction_date, format='%Y%m%d')
    transactions["membership_expire_date"] = pd.to_datetime(transactions.membership_expire_date, format='%Y%m%d')
    members["city"] = members["city"].astype("category")
    members["registered_via"] = members["registered_via"].astype("category")
    members["gender"] = np.where(members.gender=="female", True, np.where(members.gender=="male", False, np.nan))
    members["bd"] = members["bd"].astype("int16")
    train["is_churn"] = train["is_churn"].astype("bool")
    
    # Drop all transactions without msno in train
    #transactions = transactions[(transactions.msno.isin(train.msno))]
    
    # Merge members and train
    X_train = pd.merge(train, members, how="left", on="msno")
    X_test = pd.merge(test, members, how="left", on="msno")
    
    # Create y_train and X_train
    y_train = X_train.is_churn
    X_train.drop("is_churn", axis=1, inplace=True)
    X_test.drop("is_churn", axis=1, inplace=True)
    
    return X_train, X_test, y_train, transactions
'''