#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 19:08:41 2017

@author: lukaskemmer
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
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
    
    
def visualize_features(X, data_title, bin_features=True, cat_features=True, 
                       cont_features=True):
    sns.set_context("notebook", font_scale=1)
    if bin_features:
        # Distribution within each feature (in % True)
        bin_cols = [c for c in X.columns if '_bin' in c]
        plt.figure(figsize=(8, 6), dpi=80)
        sns.barplot(x=bin_cols, y=X[bin_cols].sum()/X.shape[0], orient='v', 
                    color='b')
        plt.xticks(rotation=45)   
        plt.title("Distribution of binary features in: " + data_title)
        plt.show()
        
    if cat_features:
        # Distribution of categorical data
        fig = plt.figure(figsize=(16, 12), dpi=80)
        i=1
        for col in [c for c in X.columns if '_cat' in c]:
            helper = X[col].value_counts()
            fig.add_subplot(5,3,i)
            i += 1
            sns.barplot(x=helper.index, y=helper, color='b')
            plt.xticks(rotation=45)
            plt.title("Distribution of categorical features")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        fig.suptitle(data_title)
        plt.show()
        
    if cont_features:
        # Distribution of continuous data
        fig = plt.figure(figsize=(16, 12), dpi=80)
        i=1
        for col in [c for c in X.columns if '_cat' not in c and '_bin' not in c]:
            fig.add_subplot(6,5,i)
            i += 1
            data = X[col].dropna()
            sns.distplot(data, kde=False, norm_hist=True, color='b')
            plt.xticks(rotation=45, fontsize=6)
            plt.title("Distribution of continuous features", fontsize=6)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        fig.suptitle(data_title)
        plt.show()
            
def pair_plots(X, y, bin_features=True, cat_features=False, cont_features=False):
    sns.set_context("notebook", font_scale=0.5)
    X['target']=y
    if bin_features:
        # Create a pairplot for all binary features
        plot_data = X.filter(regex='(.*_bin|target)').iloc[:,:].sample(
                n=500, replace=False)
        sns.pairplot(plot_data, hue='target', palette="husl")
        
    if cat_features:    
        # Create a pairplot for all categorical features
        #plt.figure()
        plot_data = X.filter(regex='(.*_cat|target)').iloc[:,:8].sample(
                n=500, replace=False)
        sns.pairplot(plot_data, hue='target', vars=plot_data.columns.drop('target'))

    if cont_features:    
        # Create a pairplot for all remaining features
        plot_data = X.filter(regex='(\d$|target)').iloc[:,:8].sample(
                n=500, replace=False)
        sns.pairplot(plot_data, hue='target', 
                     vars=plot_data.columns.drop('target'))

            