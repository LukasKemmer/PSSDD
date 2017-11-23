import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import custom_object_scope
from keras import callbacks

from sklearn import preprocessing

import theano

from callback import *

DATA_TRAIN_PATH = '../input/train.csv'
DATA_TEST_PATH = '../input/test.csv'

featuresToDrop = [
    'ps_calc_10',
    'ps_calc_01',
    'ps_calc_02',
    'ps_calc_03',
    'ps_calc_13',
    'ps_calc_08',
    'ps_calc_07',
    'ps_calc_12',
    'ps_calc_04',
    'ps_calc_17_bin',
    'ps_car_10_cat',
    'ps_car_11_cat',
    'ps_calc_14',
    'ps_calc_11',
    'ps_calc_06',
    'ps_calc_16_bin',
    'ps_calc_19_bin',
    'ps_calc_20_bin',
    'ps_calc_15_bin',
    'ps_ind_11_bin',
    'ps_ind_10_bin'
]

# An analogue to AUC which takes the differences between each pair of true/false predictions
# and takes the average sigmoid of the differences to get a differentiable loss function.
# Based on code and ideas from https://github.com/Lasagne/Lasagne/issues/767
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


# Create the model.
def create_model_AUC(input_dim, first_layer_size, second_layer_size, third_layer_size, lr, l2reg, dropout):
    return create_model(input_dim, first_layer_size, second_layer_size, third_layer_size, lr, l2reg, dropout, "AUC")


def create_model_bce(input_dim, first_layer_size, second_layer_size, third_layer_size, lr, l2reg, dropout):
    return create_model(input_dim, first_layer_size, second_layer_size, third_layer_size, lr, l2reg, dropout, "crossentropy")


def create_model(input_dim, first_layer_size, second_layer_size, third_layer_size, lr, l2reg, dropout, mode="AUC"):
    print("Creating model with input dim ", input_dim)
    # likely to need tuning!
    reg = regularizers.l2(l2reg)

    model = Sequential()

    model.add(Dense(units=first_layer_size, kernel_initializer='lecun_normal', kernel_regularizer=reg, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(units=second_layer_size, kernel_initializer='lecun_normal', activation='relu', kernel_regularizer=reg))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(dropout))

    model.add(Dense(units=third_layer_size, kernel_initializer='lecun_normal', activation='relu', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(1, kernel_initializer='lecun_normal', activation='sigmoid'))

    # classifier.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['mae', 'accuracy'])
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    if (mode == "AUC"):
        model.compile(loss=soft_AUC_theano, metrics=[soft_AUC_theano], optimizer=opt)  # not sure whether to use metrics here?
    else:
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)  # not sure whether to use metrics here?
    return model


def train_model( X_train, y_train, model, valSplit=0.15, epochs = 5, batch_size = 4096):
    callbacksList = [AUC_SKlearn_callback(X_train, y_train, useCv = (valSplit > 0))]
    if (valSplit > 0):
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5,
                                                       verbose=0, mode='min')
        callbacksList.append( early_stopping )
    return model.fit(x=np.array(X_train), y=np.array(y_train),
                        callbacks=callbacksList, validation_split=valSplit,
                        verbose=2, batch_size=batch_size, epochs=epochs)


def scale_features(df_for_range, df_to_scale, columnsToScale):
    # Scale columnsToScale in df_to_scale
    columnsOut = list(map( (lambda x: x + "_scaled"), columnsToScale))
    for c, co in zip(columnsToScale, columnsOut):
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        print("scaling ", c ," to ",co)
        vals = df_for_range[c].values.reshape(-1, 1)
        scaler.fit(vals)
        df_to_scale[co]=scaler.transform(df_to_scale[c].values.reshape(-1,1))

    df_to_scale.drop (columnsToScale, axis=1, inplace = True)

    return df_to_scale


def one_hot(df, cols):
    # One hot cols requested, drop original cols, return df
    df = pd.concat([df, pd.get_dummies(df[cols], columns=cols)], axis=1)
    df.drop(cols, axis=1, inplace = True)
    return df


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


def split(df, proportion):
    head_count = int(df.shape[0] * proportion)
    tail_count = int(1.0 - head_count)

    return df.head(head_count), df.tail(tail_count)


def get_data():
    X_train = pd.read_csv(DATA_TRAIN_PATH, index_col = "id")
    randomize(X_train)

    X_test = pd.read_csv(DATA_TEST_PATH, index_col = "id")

    y_train = pd.DataFrame(index = X_train.index)
    y_train['target'] = X_train.loc[:,'target']
    X_train.drop ('target', axis=1, inplace = True)
    X_train.drop (featuresToDrop, axis=1, inplace = True)
    X_test.drop (featuresToDrop,axis=1, inplace = True)

    # car_11 is really a cat col
    X_train.rename(columns={'ps_car_11': 'ps_car_11a_cat'}, inplace=True)
    X_test.rename(columns={'ps_car_11': 'ps_car_11a_cat'}, inplace=True)

    cat_cols = [elem for elem in list(X_train.columns) if "cat" in elem]
    bin_cols = [elem for elem in list(X_train.columns) if "bin" in elem]
    other_cols = [elem for elem in list(X_train.columns) if elem not in bin_cols and elem not in cat_cols]

    # Scale numeric features in region of -1,1 using training set as the scaling range
    X_test = scale_features(X_train, X_test, columnsToScale=other_cols)
    X_train = scale_features(X_train, X_train, columnsToScale=other_cols)

    X_train = one_hot(X_train, cat_cols)
    X_test = one_hot(X_test, cat_cols)

    return X_train, X_test, y_train


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


def main():
    X_train, X_test, y_train = get_data()
    model = create_model( input_dim=X_train.shape[1],
                          first_layer_size=300,
                          second_layer_size=500,
                          third_layer_size=300,
                          lr=0.0001,
                          l2reg = 0.01,
                          dropout = 0.2,
                          mode="AUC")

    # X_train, X_val = split(X_train, 0.7)
    # y_train, y_val = split(y_train, 0.7)
    X_train, y_train = balance(X_train, y_train, 0.06)

    # X_train = X_train.head(20480)
    # y_train = y_train.head(20480)

    train_model(X_train, y_train, model, epochs=30)

    with custom_object_scope({'soft_AUC_theano': soft_AUC_theano}):
        pred_fun = lambda x: model.predict(np.array(x))

        # y_pred = pred_fun(X_val)
        # print("GINI coeff: ", eval_gini(y_val.values.flatten(), y_pred.flatten()))
        # print("\a\a\a")

        y_pred = pred_fun(X_test)
        makeOutputFile(pred_fun, X_test, "../output/pred.csv")

main()