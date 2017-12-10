import numpy as np
import pandas as pd
import theano

from callback import *

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
from sklearn.model_selection import train_test_split

DATA_X_TRAIN_PATH = '../input/x_train'
DATA_TEST_PATH = '../input/x_test_stack.csv'

def soft_AUC_theano(y_true, y_pred):
    # Extract 1s
    pos_pred_vr = y_pred[y_true.nonzero()]
    # Extract zeroes
    neg_pred_vr = y_pred[theano.tensor.eq(y_true, 0).nonzero()]
    # Broadcast the subtraction to give a matrix of differences  between pairs of observations.
    pred_diffs_vr = pos_pred_vr.dimshuffle(0, 'x') - neg_pred_vr.dimshuffle('x', 0)
    # Get signmoid of each pair.
    stats = theano.tensor.nnet.sigmoid(pred_diffs_vr * 2)
    # Take average and reverse sign
    return 1-theano.tensor.mean(stats) # as we want to minimise, and get this to zero

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

    model.compile(loss=soft_AUC_theano, metrics=[soft_AUC_theano], optimizer='adam')
    
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
    X_train_bea = pd.read_csv(DATA_X_TRAIN_PATH + "_bea.csv", index_col = "id")
    X_train_diego = pd.read_csv(DATA_X_TRAIN_PATH + "_diego.csv", index_col = "id")
    X_train_lukas = pd.read_csv(DATA_X_TRAIN_PATH + "_lukas.csv", index_col = "id")
    X_train_laurin = pd.read_csv(DATA_X_TRAIN_PATH + "_laurin.csv", index_col = "id")

    if not (X_train_bea.shape == X_train_diego.shape and X_train_bea.shape == X_train_lukas.shape and X_train_bea.shape == X_train_laurin.shape):
        print("Corrupt X_train files: Not same shape")

    if not (X_train_bea["target"].equals(X_train_diego["target"]) and X_train_bea["target"].equals(X_train_lukas["target"]) and X_train_bea["target"].equals(X_train_laurin["target"])):
        print("Corrupt X_train files: Not same targets")

    X_train_diego.drop("target", axis=1, inplace = True)
    X_train_lukas.drop("target", axis=1, inplace = True)
    X_train_laurin.drop("target", axis=1, inplace = True)

    r = X_train_bea.shape[0]

    X_train = X_train_bea.join(X_train_diego).join(X_train_lukas)#.join(X_train_laurin)

    if not r == X_train.shape[0]:
        print("Corrupt X_train files: Join failed")

    X_test = pd.read_csv(DATA_TEST_PATH)

    y_train = pd.DataFrame(index = X_train.index)
    y_train['target'] = X_train.loc[:,'target']
    X_train.drop ("target", axis=1, inplace = True)

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


def eval_gini(y_true, y_prob):
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    return 1 - 2 * gini / (ntrue * (n - ntrue))


def train_model( X_train, y_train, model, valSplit=0.15, epochs = 5, batch_size = 4096):
    return model.fit(x=np.array(X_train), y=np.array(y_train), validation_split=valSplit,
                        verbose=2, batch_size=batch_size, epochs=epochs)


def main():
    X_train, X_test, y_train = get_data()
    X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0)

    model = create_model( input_dim=X_train.shape[1])

    train_model(X_train, y_train, model, epochs=30)

    with custom_object_scope({'soft_AUC_theano': soft_AUC_theano}):
        pred_fun = lambda x: model.predict(np.array(x))

        y_pred = model.predict(np.array(X_val))

        y_pred = 1/y_pred
        print("GINI coeff: ", eval_gini(y_val.values.flatten(), y_pred.flatten()))
        print("\a\a\a")

    # y_pred = pred_fun(X_test)
    # makeOutputFile(pred_fun, X_test, "../output/pred.csv")


main()