import math
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Activation, Merge, Reshape, Dropout, LeakyReLU
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import custom_object_scope
from keras import callbacks

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE

import seaborn as sns
import theano
from load_data import read_data, load_last_user_logs, get_num_user_logs

## ========================= 1. Load and clean data ======================== ##

'''
train = pd.read_csv('../01_Data/train.csv')
train = pd.concat((train, pd.read_csv('../01_Data/train_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
test = pd.read_csv('../01_Data/sample_submission_v2.csv')
members = pd.read_csv('../01_Data/members_v3.csv')

transactions = pd.read_csv('../01_Data/transactions.csv')
transactions = pd.concat((transactions, pd.read_csv('../01_Data/transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
'''

print("\n1. Load and data ...\n")
train, test, members, transactions = read_data()
train['is_churn'] = 1

## ========================= 2. Feature engineering ======================== ##
print("\n2. Adding and selecting features ...\n")

# Prepare transactions
current_transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)

# Get features for current transaction
print("\n   a) Creating features from most recent transaction ...\n")
# get most recent transaction
current_transactions = current_transactions.drop_duplicates(subset=['msno'], keep='first')
# Calculate discount
current_transactions['discount'] = current_transactions['plan_list_price'] - current_transactions['actual_amount_paid']
# Calculate cost per day
current_transactions['payment_plan_days'] = current_transactions['payment_plan_days'].replace(0, 30)
current_transactions['amt_per_day'] = current_transactions['actual_amount_paid'] / current_transactions['payment_plan_days']
# Check if discount
current_transactions['is_discount'] = current_transactions.discount.apply(lambda x: 1 if x > 0 else 0)
# Calculate the number of membership days within the current transaction
current_transactions['membership_days'] = pd.to_datetime(current_transactions['membership_expire_date']).subtract(pd.to_datetime(current_transactions['transaction_date'])).dt.days.astype(int)

# Add current transactions
train = pd.merge(train, current_transactions, how="left", on="msno")
test = pd.merge(test, current_transactions, how="left", on="msno")

# Add totals for transactions
print("\n   b) Create features for all transactions ...\n")
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

del transactions

# Add total transactions
train = pd.merge(train, total_transactions, how="left", on="msno")
test = pd.merge(test, total_transactions, how="left", on="msno")

# Map genders to 1 and 2
print("\n   c) Map genders to 1 and 2 ...\n")
'''
train['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([train, test], axis = 0)

combined = pd.merge(combined, members, how='left', on='msno')
members = []; print('members merge...') 

gender = {'male':1, 'female':2}
combined['gender'] = combined['gender'].map(gender)

train = combined[combined['is_train'] == 1]
test = combined[combined['is_train'] == 0]

train.drop(['is_train'], axis = 1, inplace = True)
test.drop(['is_train'], axis = 1, inplace = True)

del combined
'''
train = pd.merge(train, members, how="left", on="msno")
test = pd.merge(test, members, how="left", on="msno")

gender = {'male':1, 'female':2}
test["gender"] = test["gender"].map(gender)
train["gender"] = train["gender"].map(gender)

'''
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
'''
# Create features from last logs
print("\n   d) Create features from last logs ...\n")
last_user_logs = load_last_user_logs()

train = pd.merge(train, last_user_logs, how='left', on='msno')
test = pd.merge(test, last_user_logs, how='left', on='msno')
del last_user_logs

# Create combination features
print("\n   e) Create and add combination features ...\n")
train['autorenew_&_not_cancel'] = ((train.is_auto_renew == 1) == (train.is_cancel == 0)).astype(np.int8)
test['autorenew_&_not_cancel'] = ((test.is_auto_renew == 1) == (test.is_cancel == 0)).astype(np.int8)

train['notAutorenew_&_cancel'] = ((train.is_auto_renew == 0) == (train.is_cancel == 1)).astype(np.int8)
test['notAutorenew_&_cancel'] = ((test.is_auto_renew == 0) == (test.is_cancel == 1)).astype(np.int8)

# Get number of logs per user
print("\n   f) Get number of logs per user ...\n")
num_user_logs = get_num_user_logs()

# Add num_user_logs
train = pd.merge(train, num_user_logs, how="left", on="msno")
test = pd.merge(test, num_user_logs, how="left", on="msno")

# Fill nans with 0
print("\n   h) Replace NA by 0 ...\n")
train = train.fillna(0)
test = test.fillna(0)

def preproc(X_train, X_val, X_test):

    input_list_train = []
    input_list_val = []
    input_list_test = []
    
    #the cols to be embedded: rescaling to range [0, # values)
    for c in categorical_features:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i       
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)
     
    #the rest of the columns
    other_cols = [c for c in X_train.columns if (not c in categorical_features)]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)
    
    return input_list_train, input_list_val, input_list_test 

def build_embedding_network(input_dim):
    models = []

    # categorical_features = ['payment_method_id', 'city', 'registered_via']
    # binary_features = ['is_auto_renew', 'is_cancel', 'gender']
    # numerical_features = ['bd', 'payment_plan_days', 'plan_list_price', 'actual_amount_paid', 'transaction_date_year', 'transaction_date_month', 'transaction_date_day', 'membership_expire_date_year', 'membership_expire_date_month', 'membership_expire_date_day', 'logs_count', 'registration_init_year', 'registration_init_month', 'registration_init_day']

    # col_vals_dict = {c: list(X_train[c].unique()) for c in categorical_features}

    # embed_cols = []
    # for c in col_vals_dict:
    #     if len(col_vals_dict[c])>2:
    #         embed_cols.append(c)
    #         print(c + ': %d values' % len(col_vals_dict[c])) #look at value counts to know the embedding dimensions
        
    # model_payment_method_id = Sequential()
    # model_payment_method_id.add(Embedding(37, 20, input_length=1))
    # model_payment_method_id.add(Reshape(target_shape=(20,)))
    # models.append(model_payment_method_id)

    # model_city = Sequential()
    # model_city.add(Embedding(21, 10, input_length=1))
    # model_city.add(Reshape(target_shape=(10,)))
    # models.append(model_city)

    # model_registered_via = Sequential()
    # model_registered_via.add(Embedding(5, 2, input_length=1))
    # model_registered_via.add(Reshape(target_shape=(2,)))
    # models.append(model_registered_via)

    # model_rest = Sequential()
    # model_rest.add(Dense(15, input_dim=17))
    # models.append(model_rest)

    model = Sequential()
    # model.add(Merge(models, mode='concat'))
    model.add(Dense(80, input_dim=input_dim))
    model.add(Activation(LeakyReLU(alpha=0.6)))
    model.add(Dropout(.35))
    model.add(Dense(20))
    model.add(Activation(LeakyReLU(alpha=0.6)))
    model.add(Dropout(.15))
    model.add(Dense(10))
    model.add(Activation(LeakyReLU(alpha=0.6)))
    model.add(Dropout(.15))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# print("4. Evaluate feature importance")

# svm = LinearSVC()
# # create the RFE model for the svm classifier 
# # and select attributes
# rfe = RFE(svm, X_train.shape[1]-5)
# rfe = rfe.fit(X_train, Y_train)

# # print summaries for the selection of attributes
# print(list(zip(X_train.columns, rfe.support_, rfe.ranking_)))

# # drop the 5 worst columns
# useless_column_indices = [j for j in range(X_train.shape[1]) if not rfe.support_[j]]
# useless_columns = X_train.columns[useless_column_indices]
# print("Dropping " + useless_columns)
# X_train = X_train.drop(useless_columns, axis=1)
# X_test = X_test.drop(useless_columns, axis=1)

## ===================== 3. Model training and evaluation ================== ##
print("\n3. Training and validating model\n")

# Initialize array for evaluation results
log_loss_val = []

# Initialize array for predictions
y_pred = []

# Make copies for X, Y to b e used within CV
X = train.drop(["msno", "is_churn"], axis=1).copy()
y = train["is_churn"].copy()

K = 8
runs_per_fold = 3
n_epochs = 15

cv_scores = []
full_val_preds = np.zeros(np.shape(X)[0])
y_preds = np.zeros((np.shape(test)[0],K))

kfold = StratifiedKFold(n_splits = K, 
                            random_state = 231, 
                            shuffle = True)

for i, (f_ind, outf_ind) in enumerate(kfold.split(X, y)):

    X_train_f, X_val_f = X.loc[f_ind].copy(), X.loc[outf_ind].copy()
    Y_train_f, y_val_f = y[f_ind], y[outf_ind]

    # intersection = pd.merge(X_val, X_train_f, how='inner', on=['id'])
    # print(intersection.shape)
    
    X_test_f = test.copy()
    
    # #upsampling adapted from kernel: 
    # #https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
    pos = (pd.Series(Y_train_f == 1))
    
    # # Add positive examples
    X_train_f = pd.concat([X_train_f, X_train_f.loc[pos]], axis=0)
    Y_train_f = pd.concat([Y_train_f, Y_train_f.loc[pos]], axis=0)
    
    # Shuffle data
    idx = np.arange(len(X_train_f))
    np.random.shuffle(idx)
    X_train_f = X_train_f.iloc[idx]
    Y_train_f = Y_train_f.iloc[idx]
    
    #preprocessing
    # proc_X_train_f, proc_X_val_f, proc_X_test_f = preproc(X_train_f, X_val_f, X_test_f)
    proc_X_train_f = X_train_f
    proc_X_val_f = X_val_f
    proc_X_test_f = X_test_f
    
    #track of prediction for cv scores
    val_preds = 0
    
    for j in range(runs_per_fold):    
        NN = build_embedding_network(X.shape[1])
        NN.fit(proc_X_train_f.values, Y_train_f.values, epochs=n_epochs, batch_size=4096, verbose=1)
   
        val_preds += NN.predict(proc_X_val_f.values)[:,0] / runs_per_fold
        y_preds[:,i] += NN.predict(proc_X_test_f[proc_X_train_f.columns].values)[:,0] / runs_per_fold
        
    full_val_preds[outf_ind] += val_preds
        
    cv_score = log_loss(y_val_f.values, val_preds)
    if math.isnan(cv_score):
        pretty_pred = pd.DataFrame({'true' : y_val_f.values, 
                       'pred' : val_preds},
                       columns = ['true','pred'])
        print(pretty_pred.head(1000))

    cv_scores.append(cv_score)
    print ('\nFold %i prediction cv score: %.5f\n' %(i,cv_score))
    
print('Mean out of fold log loss score: %.5f' % np.mean(cv_scores))
print('Full validation score: %.5f' % log_loss(Y_train.values, full_val_preds))

y_pred_final = np.mean(y_preds, axis=1)

submission = pd.DataFrame({'msno':test.msno, 'is_churn': y_pred_final})
submission.to_csv('../output/pred.csv', index=False)
