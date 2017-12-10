import numpy as np
import pandas as pd

from keras import callbacks

from sklearn.metrics import roc_auc_score

import theano

# This callback records the SKLearn calculated AUC each round, for use by early stopping
# It also has slots where you can save down metadata or the model at useful points -
# for Kaggle kernel purposes I've commented these out
class AUC_SKlearn_callback(callbacks.Callback):
    
    def __init__(self, X_train, y_train, useCv = True):
        super(AUC_SKlearn_callback, self).__init__()
        self.bestAucCv = 0
        self.bestAucTrain = 0
        self.cvLosses = []
        self.bestCvLoss = 1,
        self.X_train = X_train
        self.y_train = y_train
        self.useCv = useCv


    def on_train_begin(self, logs={}):
        return
    
    
    def on_train_end(self, logs={}):
        return
    
    
    def on_epoch_begin(self, epoch, logs={}):
        return

    
    def on_epoch_end(self, epoch, logs={}):
        train_pred = self.model.predict(np.array(self.X_train))
        aucTrain = roc_auc_score(self.y_train, train_pred)
        print("SKLearn Train AUC score: " + str(aucTrain))

        if (self.bestAucTrain < aucTrain):
            self.bestAucTrain = aucTrain
            print ("Best SKlearn AUC training score so far")
            #**TODO: Add your own logging/saving/record keeping code here

        if (self.useCv):
            cv_pred = self.model.predict(self.validation_data[0])
            aucCv = roc_auc_score(self.validation_data[1], cv_pred)
            print ("SKLearn CV AUC score: " +  str(aucCv))

            if (self.bestAucCv < aucCv):
                # Great! New best *actual* CV AUC found (as opposed to the proxy AUC surface we are descending)
                print("Best SKLearn genuine AUC so far so saving model")
                self.bestAucCv = aucCv

                # **TODO: Add your own logging/model saving/record keeping code here.
                self.model.save("../output/best_auc_model.h5", overwrite=True)

            vl = logs.get('val_loss')
            if (self.bestCvLoss < vl):
                print("Best val loss on SoftAUC so far")
                #**TODO -  Add your own logging/saving/record keeping code here.
        return

    
    def on_batch_begin(self, batch, logs={}):
        return

    
    def on_batch_end(self, batch, logs={}):
        # logs include loss, and optionally acc( if accuracy monitoring is enabled).
        return