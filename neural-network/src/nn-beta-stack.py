import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import custom_object_scope
from keras import callbacks

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

DATA_X_TRAIN_PATH = '../input/x_train'
DATA_TEST_PATH = '../input/x_test_stack.csv'
embed_cols = []

def create_model(input_dim):
    model = Sequential()
    model.add(Dense(3, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    # model.add(Dense(20))
    # model.add(Activation('relu'))
    # model.add(Dropout(.15))
    # model.add(Dense(10))
    # model.add(Activation('relu'))
    # model.add(Dropout(.15))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    return model


def randomize(df0, df1 = None, axis=0):
    indices = np.arange(df0.shape[axis])
    np.random.shuffle(indices)
    
    df0 = df0.iloc[indices]
    if df1 is not None:
        df1 = df1.iloc[indices]

    return df0, df1


def balance(X_train, Y_train, proportion):
    one_indices = (Y_train["target"] == 1)
    zero_indices = (Y_train["target"] == 0)
    
    zero_count = int((Y_train.shape[0] / proportion) * (1. - proportion))
   
    X_zeros = X_train[zero_indices].head(zero_count)
    Y_zeros = Y_train[zero_indices].head(zero_count)

    X_ones = X_train[one_indices]
    Y_ones = Y_train[one_indices]
    
    return randomize(pd.concat([X_ones, X_zeros]), pd.concat([Y_ones, Y_zeros]))


def get_data():
    X_train_bea = pd.read_csv(DATA_X_TRAIN_PATH + "_bea.csv")
    X_train_diego = pd.read_csv(DATA_X_TRAIN_PATH + "_diego.csv")
    X_train_lukas = pd.read_csv(DATA_X_TRAIN_PATH + "_lukas.csv")
    X_train_laurin = pd.read_csv(DATA_X_TRAIN_PATH + "_laurin.csv")

    if not (X_train_bea.shape == X_train_diego.shape and X_train_bea.shape == X_train_lukas.shape and X_train_bea.shape == X_train_laurin.shape):
        print("Corrupt X_train files: Not same shape")

    if not (X_train_bea["target"].equals(X_train_diego["target"]) and X_train_bea["target"].equals(X_train_lukas["target"]) and X_train_bea["target"].equals(X_train_laurin["target"])):
        print("Corrupt X_train files: Not same targets")

    X_train_diego.drop("target", axis=1, inplace = True)
    X_train_lukas.drop("target", axis=1, inplace = True)
    X_train_laurin.drop("target", axis=1, inplace = True)

    r = X_train_bea.shape[0]

    X_train = X_train_bea.merge(X_train_diego, on="id").merge(X_train_lukas, on="id")#.join(X_train_laurin)

    if not r == X_train.shape[0]:
        print("Corrupt X_train files: Join failed")

    X_test = pd.read_csv(DATA_TEST_PATH)

    y_train = X_train.target
    X_train.drop ("target", axis=1, inplace = True)
    X_train.drop ("id", axis=1, inplace = True)

    return X_train, X_test, y_train


def preproc(X_train, X_val, X_test):
    input_list_train = []
    input_list_val = []
    input_list_test = []
    
    #the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i       
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)
     
    #the rest of the columns
    other_cols = [c for c in X_train.columns if (not c in embed_cols)]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)
    
    return input_list_train, input_list_val, input_list_test 


def makeOutputFile(pred_fun, test, subsFile):
    df_out = pd.DataFrame(index=test.index)
    y_pred = pred_fun( test )
    df_out['target'] = y_pred
    df_out.to_csv(subsFile, index_label="id")


def ginic(actual, pred):
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_c[-1] - (n + 1) / 2.0
    return giniSum / n


def gini_normalizedc(a, p):
    return ginic(a, p) / ginic(a, a)


def main():
    X_train, X_test, y_train = get_data()

    #network training
    K = 8
    runs_per_fold = 3
    n_epochs = 15

    cv_ginis = []
    full_val_preds = np.zeros(np.shape(X_train)[0])
    y_preds = np.zeros((np.shape(X_test)[0],K))

    kfold = StratifiedKFold(n_splits = K, 
                                random_state = 231, 
                                shuffle = True)  

    for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):

        X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()
        y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]
        
        X_test_f = X_test.copy()
        
        #upsampling adapted from kernel: 
        #https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
        pos = (pd.Series(y_train_f == 1))
        
        # Add positive examples
        X_train_f = pd.concat([X_train_f, X_train_f.loc[pos]], axis=0)
        y_train_f = pd.concat([y_train_f, y_train_f.loc[pos]], axis=0)
        
        # Shuffle data
        idx = np.arange(len(X_train_f))
        np.random.shuffle(idx)
        X_train_f = X_train_f.iloc[idx]
        y_train_f = y_train_f.iloc[idx]
        
        #preprocessing
        proc_X_train_f, proc_X_val_f, proc_X_test_f = preproc(X_train_f, X_val_f, X_test_f)
        
        #track oof prediction for cv scores
        val_preds = 0
        
        for j in range(runs_per_fold):
        
            NN = create_model(input_dim=X_train.shape[1])
            NN.fit(proc_X_train_f, y_train_f.values, epochs=n_epochs, batch_size=4096, verbose=1)
    
            val_preds += NN.predict(proc_X_val_f)[:,0] / runs_per_fold
            y_preds[:,i] += NN.predict(proc_X_test_f)[:,0] / runs_per_fold
            
        full_val_preds[outf_ind] += val_preds
            
        cv_gini = gini_normalizedc(y_val_f.values, val_preds)
        cv_ginis.append(cv_gini)
        print ('\nFold %i prediction cv gini: %.5f\n' %(i,cv_gini))
        
        print('Mean out of fold gini: %.5f' % np.mean(cv_ginis))
        print('Full validation gini: %.5f' % gini_normalizedc(y_train.values, full_val_preds))

        y_pred_final = np.mean(y_preds, axis=1)

        df_sub = pd.DataFrame({'id' : X_test.id, 
                            'target' : y_pred_final},
                            columns = ['id','target'])
        df_sub.to_csv('NN_EntityEmbed_10fold-sub.csv', index=False)

        pd.DataFrame(full_val_preds).to_csv('NN_EntityEmbed_10fold-val_preds.csv',index=False)


main()