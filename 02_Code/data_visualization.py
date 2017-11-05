#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 19:08:41 2017

@author: lukaskemmer
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def visualize_features(X, bin_features, cat_features, cont_features):
    if bin_features:
        # Distribution within each feature (in % True)
        bin_cols = [c for c in X.columns if '_bin' in c]
        plt.figure()
        sns.barplot(x=X[bin_cols].sum()/X.shape[0], y=bin_cols)
        
    if cat_features:
        # Distribution of categorical data
        for col in [c for c in X.columns if '_cat' in c]:
            helper = X[col].value_counts()
            plt.figure()
            sns.barplot(x=helper.index, y=helper)
    
    if cont_features:
        # Distribution of continuous data
        for col in [c for c in X.drop(['id', 'target'], axis=1).columns 
                    if '_cat' not in c and '_bin' not in c]:
            plt.figure()
            sns.distplot(X[col], kde=False, norm_hist=True)          
            
def pair_plots(X_train, bin_features, cat_features, cont_features):
    if bin_features:
        # Create a pairplot for all binary features
        plot_data = X_train.filter(regex='(.*_bin|target)').iloc[:,:8].sample(
                n=500, replace=False)
        sns.pairplot(plot_data, hue='target', 
                     vars=plot_data.columns.drop('target'))
        
    if cat_features:    
        # Create a pairplot for all categorical features
        #plt.figure()
        plot_data = X_train.filter(regex='(.*_cat|target)').iloc[:,:8].sample(
                n=500, replace=False)
        sns.pairplot(plot_data, hue='target', vars=plot_data.columns.drop('target'))

    if cont_features:    
        # Create a pairplot for all remaining features
        plot_data = X_train.filter(regex='(\d$|target)').iloc[:,:8].sample(
                n=500, replace=False)
        sns.pairplot(plot_data, hue='target', 
                     vars=plot_data.columns.drop('target'))

            