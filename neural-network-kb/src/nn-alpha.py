import math
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Activation, Merge, Reshape, Dropout
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import custom_object_scope
from keras import callbacks

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

import seaborn as sns
import theano

PATH_TO_DATA = '../input/'
PATH_TO_PREDICTIONS = '../output/'

MAX_TRAINING = 1000000
MAX_LOGS = 100000
MAX_TRANSACTIONS = 1000000
MAX_TEST = 9074700
MAX_MEMBERS = 1000000
NUMBER_OF_ESTIMATORS = 100

print("0. Reading...")

train = pd.read_csv(PATH_TO_DATA + 'train.csv', nrows=MAX_TRAINING)
train = pd.concat((train, pd.read_csv(PATH_TO_DATA+'train_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
test = pd.read_csv(PATH_TO_DATA + 'sample_submission_v2.csv')

'''
# Add the number of transactions
transactions = pd.read_csv(PATH_TO_DATA + 'transactions.csv', usecols=['msno'], nrows=MAX_TRANSACTIONS)
transactions = pd.DataFrame(transactions['msno'].reset_index())
transactions.columns = ['msno','trans_count']
train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')
transactions = []; print('transaction merge...')
print(train)
'''

# Adds all the transactions columns of the most recent (i think most recent) transaction.
# (we could see how to use all the data, if it is useful. We already have the number of transactions
transactions = pd.read_csv(PATH_TO_DATA + 'transactions.csv', nrows=MAX_TRANSACTIONS)
transactions = pd.concat((transactions, pd.read_csv(PATH_TO_DATA + 'transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
transactions = transactions.drop_duplicates(subset=['msno'], keep='first')
#transactions=[]

# Adds the number of user logs. For now only use this as the kernel showed that it was the only think useful.
# We could also see if 'number of recent logs' or thinks like that could help.
user_logs = pd.read_csv(PATH_TO_DATA + 'user_logs.csv', usecols=['msno'], nrows=MAX_LOGS)
user_logs = pd.concat((user_logs, pd.read_csv(PATH_TO_DATA + 'user_logs.csv', usecols=['msno'], nrows=MAX_LOGS)), axis=0, ignore_index=True).reset_index(drop=True)
user_logs = pd.DataFrame(user_logs['msno'].value_counts().reset_index())
user_logs.columns = ['msno','logs_count']
#user_logs = []; 

# adds the data of the members. Maps the genders to numbers.
members = pd.read_csv(PATH_TO_DATA + 'members_v3.csv', nrows=MAX_MEMBERS)

gender = {'male': 0, 'female': 1}
#members = []; 

# Memory reduction
print("1. Memory reduction and cleaning...")

#a) members.csv - consumption 273 MB -> 173 MB

#columns: msno | city | bd | gender | registered_via | registration_init_time

#first, we clean the bd (age) outliers
mask = members['bd'] <= 1
members.loc[mask, 'bd'] = members['bd'].median()

mask = members['bd'] > 100
members.loc[mask, 'bd'] = members['bd'].median()

#convert the int64 columns in smaller variables according to the need

members['city'] = members['city'].astype(np.int8) #max:22 min:1
members['bd'] = members['bd'].astype(np.int16) #max:117 min:1
members['registered_via'] = members['registered_via'].astype(np.int8) #max:13 min:3

#for the dates: separate, in new columns, the information in each of the initial columns in order to convert the new ones in smaller integer types and drop the old ones 
#but first, we clean the values that make no sense

members_dates=['registration_init_time'] # , 'expiration_date']
for c in members_dates:
    mask = members[c] > 100000000
    #members.loc[mask, c] = transactions[c].median()
    members.loc[mask, c] = 0

members['registration_init_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[:4]))
members['registration_init_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_init_day'] = members['registration_init_time'].apply(lambda x: int(str(x)[-2:]))
#members['expiration_date_year'] = members['expiration_date'].apply(lambda x: int(str(x)[:4]))
#members['expiration_date_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
#members['expiration_date_day'] = members['expiration_date'].apply(lambda x: int(str(x)[-2:]))

members['registration_init_year'] = members['registration_init_year'].astype(np.int16)
members['registration_init_month'] = members['registration_init_month'].astype(np.int8)
members['registration_init_day'] = members['registration_init_day'].astype(np.int8)
#members['expiration_date_year'] = members['expiration_date_year'].astype(np.int16)
#members['expiration_date_month'] = members['expiration_date_month'].astype(np.int8)
#members['expiration_date_day'] = members['expiration_date_day'].astype(np.int8)

members = members.drop('registration_init_time', 1)
#members = members.drop('expiration_date', 1)

#b) train.csv - consumption 15MB -> 8MB

#columns: msno | is_churn

train['is_churn'] = train['is_churn'].astype(np.int8)

#c) transactions.csv - consumption 1.4GB -> 513 MB

#columns: msno | payment_method_id | payment_plan_days | plan_list_price | 
#         actual_amount_paid | is_auto_renew | is_cancel | 
#        transaction_date_year | transaction_date_month | transaction_date_day

transactions['payment_method_id'] = transactions['payment_method_id'].astype(np.int8)
transactions['payment_plan_days'] = transactions['payment_plan_days'].astype(np.int16)
transactions['plan_list_price'] = transactions['plan_list_price'].astype(np.int16)
transactions['actual_amount_paid'] = transactions['actual_amount_paid'].astype(np.int16)
transactions['is_auto_renew'] = transactions['is_auto_renew'].astype(np.int8)
transactions['is_cancel'] = transactions['is_cancel'].astype(np.int8)

#for the dates: separate, in new columns, the information in each of the initial columns in order to convert the new ones in smaller integer types and drop the old ones 
#but first, we clean the values that make no sense

transactions_dates = ['membership_expire_date', 'transaction_date']
for c in transactions_dates:
    mask = transactions[c] > 100000000
    #transactions.loc[mask, c] = transactions[c].median()
    transactions.loc[mask, c] = 0

transactions['transaction_date_year'] = transactions['transaction_date'].apply(lambda x: int(str(x)[:4]))
transactions['transaction_date_month'] = transactions['transaction_date'].apply(lambda x: int(str(x)[4:6]))
transactions['transaction_date_day'] = transactions['transaction_date'].apply(lambda x: int(str(x)[-2:]))
transactions['membership_expire_date_year'] = transactions['membership_expire_date'].apply(lambda x: int(str(x)[:4]))
transactions['membership_expire_date_month'] = transactions['membership_expire_date'].apply(lambda x: int(str(x)[4:6]))
transactions['membership_expire_date_day'] = transactions['membership_expire_date'].apply(lambda x: int(str(x)[-2:]))

transactions['transaction_date_year'] = transactions['transaction_date_year'].astype(np.int16)
transactions['transaction_date_month'] = transactions['transaction_date_month'].astype(np.int8)
transactions['transaction_date_day'] = transactions['transaction_date_day'].astype(np.int8)
transactions['membership_expire_date_year'] = transactions['membership_expire_date_year'].astype(np.int16)
transactions['membership_expire_date_month'] = transactions['membership_expire_date_month'].astype(np.int8)
transactions['membership_expire_date_day'] = transactions['membership_expire_date_day'].astype(np.int8)

transactions = transactions.drop('transaction_date', 1)
transactions = transactions.drop('membership_expire_date', 1)

#d) user_logs.csv - consumption ?


#Merges
print ("2. Merges...")
print('transactions merge...')
X_train = pd.merge(train, transactions, how='left', on='msno')
X_test = pd.merge(test, transactions, how='left', on='msno')
print('user logs merge...')
X_train = pd.merge(X_train, user_logs, how='left', on='msno')
X_test = pd.merge(X_test, user_logs, how='left', on='msno')
print('members merge...')
X_train = pd.merge(X_train, members, how='left', on='msno')
X_test = pd.merge(X_test, members, how='left', on='msno')
print('gender map...')
X_train['gender'] = X_train['gender'].map(gender)
X_test['gender'] = X_train['gender'].map(gender)

#can be useful
X_train_members= pd.merge(train, members, how='left', on='msno')
X_test_members= pd.merge(test, members, how='left', on='msno')


#List of Columns:['msno', 'is_churn', 'trans_count', 'payment_method_id', 'payment_plan_days', 'plan_list_price', 'actual_amount_paid', 'is_auto_renew', 'transaction_date', 'membership_expire_date', 'is_cancel', 'logs_count', 'city', 'bd', 'gender', 'registered_via', 'registration_init_time']
#List of Categorical Features: ['payment_method_id', 'is_auto_renew', 'is_cancel', 'city', 'gender', 'registered_via']
#List of Numerical Features: ['trans_count', 'plan_list_price', 'actual_amount_paid', 'transaction_date', 'membership_expire_date', 'logs_count', 'registration_init_time']
#Other:  'payment_plan_days'(?), bd(?). (They can be seen as numerical but also as categories.
# TODO add: Discount, membership_duration, registration_duration.
# TODO check if we can get something from logs, analyse that data.

# ============================ 1. Clean the data matrix =================== #
# TODO use other methods to clean the data. delete the ones with full N/A?
# TODO eliminate outlayers (bd, etc...)

#Logs NaN are because there where no logs.
X_train['logs_count'] = X_train['logs_count'].fillna(0)
X_test['logs_count'] = X_test['logs_count'].fillna(0)

#columns with NaNs: ['city','bd','gender','registered_via','registration_init_year','registration_init_month','registration_init_day']

# Replace remaining NAs with median
if True:
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_test.median())

# Replace remainings Na with -1
X_train = X_train.fillna(-1)
X_test = X_test.fillna(-1)

# ========================= 3. Feature engineering ======================== #

# Features: 
#members (7): city | bd | gender | registered_via | 
#       registration_init_year | registration_init_month | registration_init_day

#transactions(9): payment_method_id | payment_plan_days | plan_list_price | 
#         actual_amount_paid | is_auto_renew | is_cancel | 
#        transaction_date_year | transaction_date_month | transaction_date_day

#user_logs: 

print("3. Feature engineering...")

# Create dummy features for categorical features
categorical_features = ['payment_method_id', 'city', 'registered_via']
binary_features = ['is_auto_renew', 'is_cancel', 'gender']
numerical_features = ['bd', 'payment_plan_days', 'plan_list_price', 'actual_amount_paid', 'transaction_date_year', 'transaction_date_month', 'transaction_date_day', 'membership_expire_date_year', 'membership_expire_date_month', 'membership_expire_date_day', 'logs_count', 'registration_init_year', 'registration_init_month', 'registration_init_day']
# TODO change bd and payment plat to categorical to see what happens
# TODO add feature when they are added

for c in categorical_features:
    X_train[c] = X_train[c].astype('category')
    X_test[c] = X_test[c].astype('category')

for c in binary_features:
    X_train[c] = X_train[c].astype('bool')
    X_test[c] = X_test[c].astype('bool')

# Split X_train into X_train and Y_train
Y_train = X_train['is_churn']
X_train = X_train.drop(['is_churn'], axis=1)

frames = [X_train, X_test]
X_combine = pd.concat(frames)
X_combine = pd.concat([X_combine, pd.get_dummies(X_combine.select_dtypes(
        include=['category']))], axis=1)

# test X_combine.select_dtypes(include=['category'])
X_combine = X_combine.drop(['payment_method_id', 'city', 'registered_via'], axis=1)

for c in numerical_features:
    if X_combine[c].std() > 0.00000001:
        X_combine[c] = (X_combine[c] - X_combine[c].mean()) / X_combine[c].std()

# TODO check if we can delete the one from before.
X_combine = X_combine.fillna(X_train.median())
X_combine = X_combine.fillna(-1)


#Analysis of the Data relevance

if False:
    #gender -> none
    
    #registered_via->relevant (almost no churn is city 19)
    churn_registered_via = pd.crosstab(X_train_members['registered_via'], X_train_members['is_churn'])
    churn_vs_registered_via_rate = churn_registered_via.div(churn_registered_via.sum(1).astype(float), axis=0) # normalize the value
    churn_vs_registered_via_rate.plot(kind='barh', stacked=True)
    # churn mostly in 4>3>9>7>13
    
    #city->relevant
    churn_vs_city = pd.crosstab(X_train_members['city'], X_train_members['is_churn'])
    churn_vs_city_rate = churn_vs_city.div(churn_vs_city.sum(1).astype(float),  axis=0) # normalize the value
    churn_vs_city_rate.plot(kind='bar', stacked=True)
    
    data = X_train_members.groupby('city').aggregate({'msno':'count'}).reset_index()
    ax = sns.barplot(x='city', y='msno', data=data) #max: city 13 > 5 > 4...
    
    #bd (==age)-> relevant
    sns.violinplot(x=X_train_members["is_churn"], y=X_train_members["bd"], data=X_train_members)


# ===================== 4. Model training and evaluation ================== #
#model = tree.DecisionTreeClassifier() # Decision tree
#model = ensemble.RandomForestClassifier(n_estimators=NUMBER_OF_ESTIMATORS) # Random forest
#model = linear_model.LogisticRegression() # Logistic regression

# =============================================================================

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

def build_embedding_network():
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
        
    model_payment_method_id = Sequential()
    model_payment_method_id.add(Embedding(37, 20, input_length=1))
    model_payment_method_id.add(Reshape(target_shape=(20,)))
    models.append(model_payment_method_id)

    model_city = Sequential()
    model_city.add(Embedding(21, 10, input_length=1))
    model_city.add(Reshape(target_shape=(10,)))
    models.append(model_city)

    model_registered_via = Sequential()
    model_registered_via.add(Embedding(5, 2, input_length=1))
    model_registered_via.add(Reshape(target_shape=(2,)))
    models.append(model_registered_via)

    model_rest = Sequential()
    model_rest.add(Dense(15, input_dim=17))
    models.append(model_rest)

    model = Sequential()
    model.add(Merge(models, mode='concat'))
    model.add(Dense(80, input_dim=20))
    model.add(Activation('relu'))
    model.add(Dropout(.35))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

print("4. Model training and evaluation...")

K = 8
runs_per_fold = 3
n_epochs = 15

cv_scores = []
full_val_preds = np.zeros(np.shape(X_train)[0])
y_preds = np.zeros((np.shape(X_test)[0],K))

kfold = StratifiedKFold(n_splits = K, 
                            random_state = 231, 
                            shuffle = True)

del X_train['msno']
del X_test['msno']

X_train = X_train.head(100000)
X_test = X_test.head(100000)
Y_train = Y_train.head(100000)

for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, Y_train)):

    X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()
    Y_train_f, y_val_f = Y_train[f_ind], Y_train[outf_ind]

    # intersection = pd.merge(X_val, X_train_f, how='inner', on=['id'])
    # print(intersection.shape)
    
    X_test_f = X_test.copy()
    
    # #upsampling adapted from kernel: 
    # #https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
    # pos = (pd.Series(Y_train_f == 1))
    
    # # Add positive examples
    # X_train_f = pd.concat([X_train_f, X_train_f.loc[pos]], axis=0)
    # Y_train_f = pd.concat([Y_train_f, Y_train_f.loc[pos]], axis=0)
    
    # Shuffle data
    idx = np.arange(len(X_train_f))
    np.random.shuffle(idx)
    X_train_f = X_train_f.iloc[idx]
    Y_train_f = Y_train_f.iloc[idx]
    
    #preprocessing
    proc_X_train_f, proc_X_val_f, proc_X_test_f = preproc(X_train_f, X_val_f, X_test_f)
    # proc_X_train_f = X_train_f
    # proc_X_val_f = X_val_f
    # proc_X_test_f = X_test_f
    
    #track of prediction for cv scores
    val_preds = 0
    
    for j in range(runs_per_fold):    
        NN = build_embedding_network()
        NN.fit(proc_X_train_f.values, Y_train_f.values, epochs=n_epochs, batch_size=4096, verbose=1)
   
        val_preds += NN.predict(proc_X_val_f.values)[:,0] / runs_per_fold
        y_preds[:,i] += NN.predict(proc_X_test_f[proc_X_train_f.columns] .values)[:,0] / runs_per_fold
        
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

# pred = pd.DataFrame({'id' : df_test.id, 
#                        'target' : y_pred_final},
#                        columns = ['id','target'])
# pred.to_csv('pred.csv', index=False)
