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
    """
    Loads and returns test and train data
    train (DataFrame) : training data
    test (DataFrame) : testing data
    members (DataFrame) : member information
    transactions (DataFrame) : transaction information
    """
    # Load and concat data
    train = pd.read_csv("../01_Data/train.csv")
    train = pd.concat((train, pd.read_csv("../01_Data/train_v2.csv")), axis=0, ignore_index=True).reset_index(drop=True)
    test = pd.read_csv("../01_Data/sample_submission_v2.csv")
    members = pd.read_csv("../01_Data/members_v3.csv")
    transactions = pd.read_csv("../01_Data/transactions.csv")
    transactions = pd.concat((transactions, pd.read_csv("../01_Data/transactions_v2.csv")), axis=0, ignore_index=True).reset_index(drop=True)

    return train, test, members, transactions

def transform_df(df):
    """
    Transforms input to DataFrame and returns the the last log data
    df: user-log data (Not DataFrame)
    df (DataFrame) : last user-log data per user (msno)
    """
    df = pd.DataFrame(df)
    df = df.sort_values(by=["date"], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=["msno"], keep="first")
    return df

def transform_df2(df):
    """
    Returns the the last log data. Input already assumed to be DataFrame
    df: user-log data (already DataFrame)
    df (DataFrame) : last user-log data per user (msno)
    """
    df = df.sort_values(by=["date"], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=["msno"], keep="first")
    return df

def load_last_user_logs():
    """
    Reads user_logs and returns last user logs per msno.
    last_user_logs (DataFrame) : last user-log data per user (msno)
    """
    last_user_logs = []
    
    df_iter = pd.read_csv("../01_Data/user_logs.csv", low_memory=False, iterator=True, chunksize=10000000)
    
    i = 0
    for df in df_iter:
        if i>35:
            if len(df)>0:
                p = Pool(cpu_count())
                df = p.map(transform_df, np.array_split(df, cpu_count()))   
                df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
                df = transform_df2(df)
                p.close(); p.join()
                last_user_logs.append(df)
                df = []
        i+=1
        
    last_user_logs.append(transform_df(pd.read_csv("../01_Data/user_logs_v2.csv")))
    last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)
    last_user_logs = transform_df2(last_user_logs)
    
    return last_user_logs

def get_num_user_logs():
    """
    Returns number of logs per user
    user_logs (DataFrame) : number of logs per user (msno)
    """
    # Read msno column of user_logs
    user_logs = pd.read_csv("../01_Data/user_logs.csv", usecols=["msno"])

    # Count number of log entries per msno
    user_logs = pd.DataFrame(user_logs["msno"].value_counts().reset_index())

    # Adjust column names
    user_logs.columns = ["msno","logs_count"]
    return user_logs
