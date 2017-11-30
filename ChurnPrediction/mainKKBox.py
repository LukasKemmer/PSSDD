from numpy import *
set_printoptions(precision=3)
from matplotlib.pyplot import *
from scipy.linalg import inv
import pandas as pd
from collections import Counter
from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn import model_selection, tree, linear_model, feature_selection, decomposition, ensemble
from functions import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import seaborn as sns

def xgb_score(preds, X_train):
    labels = X_train.get_label()
    return 'log_loss', metrics.log_loss(labels, preds)

MAX_TRAINING = 1000000
MAX_LOGS = 100000
MAX_TRANSACTIONS = 1000000
MAX_TEST = 9074700
MAX_MEMBERS = 1000000
NUMBER_OF_ESTIMATORS = 100

PATH_TO_DATA = './churnData/'
PATH_TO_PREDICTIONS = './churnPredictions/'

# ============================ 0. Prepare the data matrix ========================== #

# Source: https://www.kaggle.com/the1owl/regressing-during-insomnia-0-21496

print("0. Readings...")

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

print("4. Model training and evaluation...")

model = XGBClassifier(max_depth=4, # Maximum tree depth for base learners
                      learning_rate=0.07, # Boosting learning rate (xgb’s “eta”)
                      n_estimators = 200, # Number of boosted trees to fit
                      silent = True, # Whether to print messages while running boosting
                      objective='binary:logistic', # learning objective
                      #booster='gbtree', # Which booster to use: gbtree, gblinear or dart
                      #n_jobs = 8, # Number of parallel threads
                      gamma=10, # Minimum loss reduction required to make a further partition on a leaf node of the tree
                      min_child_weight=6, # Minimum sum of instance weight(hessian) needed in a child
                      #max_delta_step=0, # Maximum delta step we allow each tree’s weight estimation to be
                      subsample=.8, # Subsample ratio of the training instance                      colsample_bytree=.8, # Subsample ratio of columns when constructing each tree
                      #colsample_bylevel=1, # Subsample ratio of columns for each split, in each level
                      reg_alpha=8, # L1 regularization term on weights
                      reg_lambda=1.3, # L2 regularization term on weights
                      scale_pos_weight=1.6) # Balancing of positive and negative weights
                      #base_score=0.5) # The initial prediction score of all instances, global bias

# =============================================================================
#cv

#fold = 5
#for i in range(fold):
#    params = {
#        'eta': 0.02, #use 0.002
#        'max_depth': 7,
#        'objective': 'binary:logistic',
#        'eval_metric': 'logloss',
#        'seed': i,
#        'silent': True
#    }
#    x1, x2, y1, y2 = model_selection.train_test_split(X_train[cols], Y_train['is_churn'], test_size=0.3, random_state=i)
#    watchlist = [(XGBClassifier.DMatrix(x1, y1), 'train'), (XGBClassifier.DMatrix(x2, y2), 'valid')]
#    model = XGBClassifier.train(params, XGBClassifier.DMatrix(x1, y1), 150,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50) #use 1500
#    if i != 0:
#        pred += model.predict(XGBClassifier.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
#    else:
#        pred = model.predict(XGBClassifier.DMatrix(X_test[cols]), ntree_limit=model.best_ntree_limit)
#pred /= fold
#X_test['is_churn'] = pred.clip(0.+1e-15, 1-1e-15)
#X_test[['msno','is_churn']].to_csv('submission.csv.gz', index=False, compression='gzip')

#first

X_train = X_combine.head(X_train.shape[0])
del X_train['msno']
X_test = X_combine.tail(X_test.shape[0])
# Make copies for X, Y to be used within CV
X = X_train.copy()
Y = Y_train.copy()

model = model.fit(X, Y)

# Predict probabilities
Y_probs = model.predict_proba(X_test[X.columns]) #if there is not enough data it can create a problem of the columns not matching
Y_probsCV=[median(Y_probs[:,1]) if x > 1 else x for x in Y_probs[:,1]] #still predictions > 1 (?)
Y_probsCV=[max(min(p, 1-10**(-15)), 10**(-15)) for p in Y_probsCV]


# Create output dataframe
results = pd.DataFrame({'msno': X_test['msno'], 'is_churn': Y_probsCV})

# Output results
results.to_csv(PATH_TO_PREDICTIONS + 'prediction.csv', columns=['msno', 'is_churn'], sep=',', index=False)

logloss2=-1/size(Y)*sum([y*log(p)+(1-y)*log(1-p) for y, p in zip(Y, Y_probsCV)])
print ('\n Logloss: ', logloss2)
