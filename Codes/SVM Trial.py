#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:00:04 2019

@author: saileshmohanty
"""

import numpy as np
import pandas as pd

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn import decomposition

print('Load Train Data and Test Data.')
train_path = '/Users/saileshmohanty/Desktop/Documents/Algorithms Testing/Takeda Pharma/train.csv'
test_path  = '/Users/saileshmohanty/Desktop/Documents/Algorithms Testing/Takeda Pharma/test.csv'
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
unique_val_columns = df_train.T.apply(lambda x: x.nunique(), axis=1)
cols_to_remove = list(unique_val_columns[unique_val_columns==1].index)
df_train.drop(cols_to_remove,axis=1,inplace=True)
df_test.drop(cols_to_remove,axis=1,inplace=True)


target = df_train['Score']
trainID = df_train['ID']
df_train.drop(['ID','Score'],axis=1,inplace=True)
testID = df_test['ID']
df_test.drop('ID',axis=1,inplace=True)

pca = decomposition.PCA()
pca.fit(df_train)
train_t = pca.transform(df_train)


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                   ]

scores = 'r2'

print("# Tuning hyper-parameters for %s" % scores)
print()

clf = GridSearchCV(SVR(C=1), tuned_parameters, cv=5,
                       scoring='%s' % scores,verbose=100)
clf.fit(train_t, target)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
print()
