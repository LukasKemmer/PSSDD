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

MAX_TRAINING = 1000000
MAX_LOGS = 1000000
MAX_TRANSACTIONS = 1000000
MAX_TEST = 9074700
MAX_MEMBERS = 1000000
NUMBER_OF_ESTIMATORS = 100

PATH_TO_DATA = './../../churnData/'
PATH_TO_PREDICTIONS = './../../churnPredictions/'

# ============================ 0. Prepare the data matrix ========================== #

# Source: https://www.kaggle.com/the1owl/regressing-during-insomnia-0-21496
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
X_train = pd.merge(train, transactions, how='left', on='msno')
X_test = pd.merge(test, transactions, how='left', on='msno')
transactions=[]
print(X_train)

# Adds the number of user logs. For now only use this as the kernel showed that it was the only think useful.
# We could also see if 'number of recent logs' or thinks like that could help.
user_logs = pd.read_csv(PATH_TO_DATA + 'user_logs.csv', usecols=['msno'], nrows=MAX_LOGS)
user_logs = pd.concat((user_logs, pd.read_csv(PATH_TO_DATA + 'user_logs.csv', usecols=['msno'], nrows=MAX_LOGS)), axis=0, ignore_index=True).reset_index(drop=True)
user_logs = pd.DataFrame(user_logs['msno'].value_counts().reset_index())
user_logs.columns = ['msno','logs_count']
X_train = pd.merge(X_train, user_logs, how='left', on='msno')
X_test = pd.merge(X_test, user_logs, how='left', on='msno')
user_logs = []; print('user logs merge...')
print(X_train)

# adds the data of the members. Maps the genders to numbers.
members = pd.read_csv(PATH_TO_DATA + 'members_v2.csv', nrows=MAX_MEMBERS)
X_train = pd.merge(X_train, members, how='left', on='msno')
X_test = pd.merge(X_test, members, how='left', on='msno')
gender = {'male': 0, 'female': 1}
X_train['gender'] = X_train['gender'].map(gender)
X_test['gender'] = X_train['gender'].map(gender)
members = []; print('members merge...')

print(list(X_train))

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

# Replace remaining NAs with median
if True:
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_test.median())

# Replace remainings Na with -1
X_train = X_train.fillna(-1)
X_test = X_test.fillna(-1)

dates = ['membership_expire_date', 'transaction_date', 'registration_init_time']
for c in dates:
    mask = X_train[c] > 100000000
    print(X_train[c].max())
    print(X_train[c].max() > 100000000)
    X_train.loc[mask, c] = X_train[c].median()
    X_train.loc[mask, c] = 0
    X_test.loc[mask, c] = X_test[c].median()
    X_test.loc[mask, c] = 0

mask = X_train['bd'] < 0
X_train.loc[mask, c] = X_train[c].median()
mask = X_test['bd'] < 0
X_test.loc[mask, c] = X_test[c].median()

mask = X_train['bd'] > 100
X_train.loc[mask, c] = X_train[c].median()
mask = X_test['bd'] > 100
X_test.loc[mask, c] = X_test[c].median()

print(X_train.max())
print(X_train.min())

# ========================= 3. Feature engineering ======================== #

# Create dummy features for categorical features
categorical_features = ['payment_method_id', 'city', 'registered_via']
binary_features = ['is_auto_renew', 'is_cancel', 'gender']
numerical_features = ['bd', 'payment_plan_days', 'plan_list_price', 'actual_amount_paid', 'transaction_date', 'membership_expire_date', 'logs_count', 'registration_init_time']
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

# ===================== 4. Model training and evaluation ================== #
#model = tree.DecisionTreeClassifier() # Decision tree
model = ensemble.RandomForestClassifier(n_estimators=NUMBER_OF_ESTIMATORS) # Random forest
#model = linear_model.LogisticRegression() # Logistic regression

# =============================================================================
# model = XGBClassifier(max_depth=4, # Maximum tree depth for base learners
#                       learning_rate=0.07, # Boosting learning rate (xgb’s “eta”)
#                       n_estimators = 400, # Number of boosted trees to fit
#                       silent = True, # Whether to print messages while running boosting
#                       objective='binary:logistic', # learning objective
#                       #booster='gbtree', # Which booster to use: gbtree, gblinear or dart
#                       #n_jobs = 8, # Number of parallel threads
#                       gamma=10, # Minimum loss reduction required to make a further partition on a leaf node of the tree
#                       min_child_weight=6, # Minimum sum of instance weight(hessian) needed in a child
#                       #max_delta_step=0, # Maximum delta step we allow each tree’s weight estimation to be
#                       subsample=.8, # Subsample ratio of the training instance
#                       colsample_bytree=.8, # Subsample ratio of columns when constructing each tree
#                       #colsample_bylevel=1, # Subsample ratio of columns for each split, in each level
#                       reg_alpha=8, # L1 regularization term on weights
#                       reg_lambda=1.3, # L2 regularization term on weights
#                       scale_pos_weight=1.6) # Balancing of positive and negative weights
#                       #base_score=0.5) # The initial prediction score of all instances, global bias
#
# =============================================================================

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

logloss=-1/size(Y)*sum([y*log(p)+(1-y)*log(1-p) for y, p in zip(Y, Y_probsCV)])
print ('\n Logloss: ', logloss)