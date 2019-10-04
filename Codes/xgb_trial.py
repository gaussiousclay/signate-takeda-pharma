#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:42:57 2019

@author: saileshmohanty
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import pickle
import os
import gc
from xgboost import plot_importance
from matplotlib import pyplot



def fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name):
    
    model = xgb.XGBRegressor(max_depth=50,
                             n_estimators=999999,
                             colsample_bytree=0.316,
                             learning_rate=0.027,
                             subasample = 0.673,
                             min_child_weight = 9.56,
                             objective='reg:linear', 
                             n_jobs=-1)
     
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)], 
              verbose=0, 
              early_stopping_rounds=50)
              
    cv_val = model.predict(X_val)
    
    #Save XGBoost Model
    save_to = '{}{}_fold{}.dat'.format(xgb_path, name, counter+1)
    pickle.dump(model, open(save_to, "wb"))
    return cv_val

def train_stage(df_path,xgb_path):
    
    print('Load Train Data.')
    df_train = pd.read_csv(train_path)
    unique_val_columns = df_train.T.apply(lambda x: x.nunique(), axis=1)
    cols_to_remove = list(unique_val_columns[unique_val_columns==1].index)
    df_train.drop(cols_to_remove,axis=1,inplace=True)
    categorical_columns = list(unique_val_columns[unique_val_columns==2].index)
    for i in categorical_columns:
        arr = df_train[i].unique()
        if (arr[0] == 0  and arr[1]==1):
           df_train[i] = df_train[i].astype(str)
    df_train = pd.get_dummies(df_train)
    print('\nShape of Train Data: {}'.format(df_train.shape))
    
    y_df = np.array(df_train['Score'])                        
    df_ids = np.array(df_train.ID)                     
    df_train.drop(['ID', 'Score'], axis=1, inplace=True)
    
    xgb_cv_result = np.zeros(df_train.shape[0])
    
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    skf.get_n_splits(df_ids, y_df)
    
    print('\nModel Fitting...')
    for counter, ids in enumerate(skf.split(df_ids, y_df)):
        print('\nFold {}'.format(counter+1))
        X_fit, y_fit = df_train.values[ids[0]], y_df[ids[0]]
        X_val, y_val = df_train.values[ids[1]], y_df[ids[1]]
    
        print('XGBoost')
        xgb_cv_result[ids[1]] += fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name='xgb')
        
        del X_fit, X_val, y_fit, y_val
        gc.collect()
    
    rmse_xgb  = round(math.sqrt(mean_squared_error(y_df, xgb_cv_result)),4)
    print('XGBoost  VAL RMSE: {}'.format(rmse_xgb))
    
    return 0



if __name__ == '__main__':
    
    train_path = '/Users/saileshmohanty/Desktop/Documents/Algorithms Testing/Takeda Pharma/train.csv'
    test_path  = '/Users/saileshmohanty/Desktop/Documents/Algorithms Testing/Takeda Pharma/test.csv'
    
    xgb_path = '/Users/saileshmohanty/Desktop/Documents/Algorithms Testing/Takeda Pharma/exgb_trial_0/'

    train_stage(train_path, lgb_path, xgb_path, cb_path)
    #Stacking Stage
    print('Commencing Stacking and Prediction Stage.\n')
    stacking_and_prediction_stage(train_path, lgb_path, xgb_path, cb_path,test_path)
    
    print('\nDone.')

       