# -*- coding: utf-8 -*-
"""
Created on Tue Mar 04 10:24:11 2019

@author: SAILMOH
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
import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from xgboost import plot_importance
from matplotlib import pyplot
from pathlib import Path
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import r2_score
gc.enable()


def fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name):
    
    model = lgb.LGBMRegressor(max_depth=30,
                              n_estimators=999999,
                              learning_rate=0.016,
                              colsample_bytree=0.782,
                              subsample = 0.609,
                              min_child_weight = 46,
                              metric='rmse',
                              num_leaves = 10,
                              objective='regression', 
                              n_jobs=-1)
     
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)],
              verbose=100, 
              early_stopping_rounds=50)
                  
    cv_val = model.predict(X_val)
    
    #Save LightGBM Model
    save_to = '{}{}_fold{}.txt'.format(lgb_path, name, counter+1)
    model.booster_.save_model(save_to)
    return cv_val
    
    
def fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name):
    
    model = xgb.XGBRegressor(max_depth=30,
                             n_estimators=999999,
                             colsample_bytree=0.782,
                             learning_rate=0.016,
                             subsample = 0.609,
                             min_child_weight = 46,
                             num_leaves = 10,
                             objective='reg:linear', 
                             n_jobs=-1)
     
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)], 
              verbose=100, 
              early_stopping_rounds=50)
              
    cv_val = model.predict(X_val)
    
    #Save XGBoost Model
    save_to = '{}{}_fold{}.dat'.format(xgb_path, name, counter+1)
    pickle.dump(model, open(save_to, "wb"))
    return cv_val
    
    
def fit_cb(X_fit, y_fit, X_val, y_val, counter, cb_path, name):
    
    model = cb.CatBoostRegressor(iterations=999999,
                                  learning_rate=0.016,
                                  colsample_bylevel=0.782,
                                  max_depth = 15,
                                  objective="RMSE")
                                  
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)], 
              verbose=100, early_stopping_rounds=100)
              
    cv_val = model.predict(X_val)
    
    #Save Catboost Model          
    save_to = "{}{}_fold{}.mlmodel".format(cb_path, name, counter+1)
    model.save_model(save_to, format="coreml")
    
    return cv_val


def pre_process_train_test(train,test):
    one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
    one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]

    many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
    many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]

    big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

    cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols + one_value_cols_test))
    print('{} features are going to be dropped for being useless'.format(len(cols_to_drop)))

    train = train.drop(cols_to_drop, axis=1)
    test = test.drop(cols_to_drop, axis=1)

    print(f'Train dataset after removing garbage: {train.shape[0]} rows & {train.shape[1]} columns')
    print(f'Test dataset after removing garbage: {test.shape[0]} rows & {test.shape[1]} columns')

    train = train.replace(np.inf,999)
    test = test.replace(np.inf,999)
    scaler = preprocessing.MinMaxScaler()
    train = pd.DataFrame(scaler.fit_transform(train),columns=train.columns)
    test = pd.DataFrame(scaler.fit_transform(test),columns=train.columns)
    
    return train,test



def train_stage(train,target,trainid, lgb_path, xgb_path, cb_path):
    
    print('Load Train Data.')
    df_train = train
    df_target = target
    print('\nShape of Train Data: {}'.format(df_train.shape))
    
    y_df = df_target                     
    df_ids = np.array(trainid)                     
    
    lgb_cv_result = np.zeros(df_train.shape[0])
    xgb_cv_result = np.zeros(df_train.shape[0])
    cb_cv_result  = np.zeros(df_train.shape[0])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf.get_n_splits(df_ids, y_df)
    
    print('\nModel Fitting...')
    for counter, ids in enumerate(kf.split(df_ids, y_df)):
        print('\nFold {}'.format(counter+1))
        X_fit, y_fit = df_train.values[ids[0]], y_df[ids[0]]
        X_val, y_val = df_train.values[ids[1]], y_df[ids[1]]
    
        print('CatBoost')
        cb_cv_result[ids[1]]  += fit_cb(X_fit,  y_fit, X_val, y_val, counter, cb_path,  name='cb')
        
        print('LightGBM')
        lgb_cv_result[ids[1]] += fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name='lgb')
        print('XGBoost')
        xgb_cv_result[ids[1]] += fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name='xgb')
        
        del X_fit, X_val, y_fit, y_val
        gc.collect()
    
    
    rsq_lgb = round(r2_score(y_df,lgb_cv_result),4)
    rsq_xgb = round(r2_score(y_df,xgb_cv_result),4)
    rsq_cb = round(r2_score(y_df,cb_cv_result),4)
    rsq_mean = round(r2_score(y_df, (lgb_cv_result+xgb_cv_result+cb_cv_result)/3), 4)
    rsq_mean_lgb_cb = round(r2_score(y_df, (lgb_cv_result+cb_cv_result)/2), 4)
    print('\nLightGBM VAL RSQ: {}'.format(rsq_lgb))
    print('XGBoost  VAL RSQ: {}'.format(rsq_xgb))
    print('Catboost VAL RSQ: {}'.format(rsq_cb))
    print('Mean Catboost+LightGBM VAL RSQ: {}'.format(rsq_mean_lgb_cb))
    print('Mean XGBoost+Catboost+LightGBM, VAL RSQ: {}\n'.format(rsq_mean))
    
    return 0


"""    
    
def stacking_and_prediction_stage(train_path, lgb_path, xgb_path, cb_path,test_path):
   
    print('Load Train Data. for Stacking')
    df_train_features = pd.read_csv(train_path)
    df_train = pd.get_dummies(df_train_features,columns=['companyId','jobType','degree','major','industry'])
    df_target = pd.read_csv(target_path)
    df_train = pd.merge(df_train,df_target)
    print('\nShape of Train Data: {}'.format(df_train.shape))
    df_train.drop(['jobId','salary'], axis=1, inplace=True)
    
    lgb_models = sorted(os.listdir(lgb_path))
    xgb_models = sorted(os.listdir(xgb_path))
    cb_models  = sorted(os.listdir(cb_path))
    
    lgb_result = np.zeros(df_train.shape[0])
    xgb_result = np.zeros(df_train.shape[0])
    cb_result  = np.zeros(df_train.shape[0])
    
    print('\nMake predictions...\n')
    
    print('With LightGBM...')
    for m_name in lgb_models:
        #Load LightGBM Model
        model = lgb.Booster(model_file='{}{}'.format(lgb_path, m_name))
        lgb_result += model.predict(df_train.values)
        
    print('With XGBoost...')    
    for m_name in xgb_models:
        #Load XGBOOST Model
        model = pickle.load(open('{}{}'.format(xgb_path, m_name), "rb"))
        xgb_result += model.predict(df_train.values)
    
    print('With CatBoost...')        
    for m_name in cb_models:
        #Load Catboost Model
        model = cb.CatBoostRegressor()
        model = model.load_model('{}{}'.format(cb_path, m_name), format = 'coreml')
        cb_result += model.predict(df_train.values)
    
    lgb_result /= len(lgb_models)
    xgb_result /= len(xgb_models)
    cb_result  /= len(cb_models)
    
    
    submission = pd.DataFrame()
    submission['target'] = df_target['salary']
    submission['weighted_ensemble'] = (0.3*lgb_result)+(0.4*xgb_result)+(0.3*cb_result)
    submission['simple_ensemble'] = (lgb_result + cb_result + xgb_result)/3
    submission['xgb_result'] = xgb_result
    submission['lgb_result'] = lgb_result
    submission['cb_result'] = cb_result
    print('\nSimple Ensemble VAL RMSE: {}'.format(round(math.sqrt(mean_squared_error(df_target['salary'], submission['simple_ensemble'])),4)))
    print('\nWeighted Ensemble VAL RMSE: {}'.format(round(math.sqrt(mean_squared_error(df_target['salary'], submission['weighted_ensemble'])),4)))
    print('\nLightGBM VAL RMSE: {}'.format(round(math.sqrt(mean_squared_error(df_target['salary'], submission['lgb_result'])),4)))
    print('\nXGBOOST VAL RMSE: {}'.format(round(math.sqrt(mean_squared_error(df_target['salary'], submission['xgb_result'])),4)))
    print('\nCATBOOST VAL RMSE: {}'.format(round(math.sqrt(mean_squared_error(df_target['salary'], submission['cb_result'])),4)))
    submission.to_csv('train_submission_of_ensembles.csv')
    
    print('Load Model Predictions')
    print('\nShape of Predictions Data Data: {}'.format(submission.shape))
    print('Stacking Using H2O Deep Learning')
    
    h2o.init(ip="localhost", port=54321)
    train = h2o.H2OFrame(submission)
    train,valid,test = train.split_frame(ratios=[0.7, 0.15], seed=42)
    y = 'target'
    X = list(train.columns)
    X.remove(y)
    

    # Train the Model
    print('Training Deep Learning Model')
    dl_model =  H2ODeepLearningEstimator(training_frame=train,
                  validation_frame=valid,
                  stopping_rounds=10,
                  stopping_tolerance=0.0005,
                  epochs = 10000,
                  adaptive_rate = True,  
                  stopping_metric="rmse",
                  hidden=[256,256,256],      
                  balance_classes= False,
                  standardize = True,  
                  loss = "absolute",
                  activation =  'RectifierWithDropout',
                  input_dropout_ratio =  0.05,
                  l1 = 0.00001,
                  l2 = 0.00001,
                  max_w2 = 10.0,
                  hidden_dropout_ratios = [0.01,0.02,0.03])
    
    dl_model.train(X,y,train)
    print('Deep Learning Model Performance on Train and Validation')
    dl_model
    print('Deep Learning Model Performance on Test Partition')
    dl_model.model_performance(test)
    
    dl_model_df = dl_model.score_history()
    plt.plot(dl_model_df['training_rmse'], label="training_rmse")
    plt.plot(dl_model_df['validation_rmse'], label="validation_rmse")
    plt.title("Stacking Deep Learner (Tuned)")
    plt.legend();
    
    print('Beginning Prediction')
    print('Load Test Data')
    df_test_features = pd.read_csv(test_path)
    df_test = pd.get_dummies(df_test_features,columns=['companyId','jobType','degree','major','industry'])
    test_id = df_test.jobId
    print('\nShape of Test Data: {}'.format(df_test.shape))
    df_test.drop(['jobId'], axis=1, inplace=True)
   
    lgb_models = sorted(os.listdir(lgb_path))
    xgb_models = sorted(os.listdir(xgb_path))
    cb_models  = sorted(os.listdir(cb_path))
    
    lgb_test_result = np.zeros(df_test.shape[0])
    xgb_test_result = np.zeros(df_test.shape[0])
    cb_test_result  = np.zeros(df_test.shape[0])
    
    print('\nMake predictions...\n')
    
    print('With LightGBM...')
    for m_name in lgb_models:
        #Load LightGBM Model
        model = lgb.Booster(model_file='{}{}'.format(lgb_path, m_name))
        lgb_test_result += model.predict(df_test.values)
        
    print('With XGBoost...')    
    for m_name in xgb_models:
        #Load XGBOOST Model
        model = pickle.load(open('{}{}'.format(xgb_path, m_name), "rb"))
        xgb_test_result += model.predict(df_test.values)
    
    print('With CatBoost...')        
    for m_name in cb_models:
        #Load Catboost Model
        model = cb.CatBoostRegressor()
        model = model.load_model('{}{}'.format(cb_path, m_name), format = 'coreml')
        cb_test_result += model.predict(df_test.values)
    
    lgb_test_result /= len(lgb_models)
    xgb_test_result /= len(xgb_models)
    cb_test_result  /= len(cb_models)
    
    
    submission_test = pd.DataFrame()
    submission_test['weighted_ensemble'] = (0.3*lgb_test_result)+(0.4*xgb_test_result)+(0.3*cb_test_result)
    submission_test['simple_ensemble'] = (lgb_test_result + cb_test_result + xgb_test_result)/3
    submission_test['xgb_result'] = xgb_test_result
    submission_test['lgb_result'] = lgb_test_result
    submission_test['cb_result'] = cb_test_result
    
    test_submission = h2o.H2OFrame(submission_test)
    pred = dl_model.predict(test_submission).as_data_frame(use_pandas=True)
    submission_test = pd.DataFrame()
    submission_test['jobId'] = test_id
    submission_test['salary'] = pred
    submission_test.to_csv('D:\Analytics Repository\Indeed/test_salaries.csv',index=False)
    
    #Load XGBOOST Model
    #model = pickle.load(open('{}{}'.format(xgb_path,'xgb_fold5.dat'), "rb"))
    #feat_importances = pd.Series(model.feature_importances_, index=df_train.columns)
    #feat_importances.nlargest(10).plot(kind='barh')
    #feat_importances.nsmallest(10).plot(kind='barh')
    #results = model.evals_result()
    #epochs = len(results['validation_0']['rmse'])
    #x_axis = range(0, epochs)
    # plot log loss
    #fig, ax = pyplot.subplots()
    #ax.plot(x_axis, results['validation_0']['rmse'], label='Validation')
    #ax.legend()
    #pyplot.ylabel('RMSE')
    #pyplot.title('XGBoost RMSE')
    #pyplot.show()    
    return 0


"""


    
if __name__ == '__main__':
    
    
    folder_path = str(Path().absolute())
    lgb_path = folder_path+'/Models/lgb_models/'
    xgb_path = folder_path+'/Models/xgb_models/'
    cb_path = folder_path+'/Models/cb_models/'
    #Create dir for models if they dont exist already
    #os.mkdir(lgb_path)
    #os.mkdir(xgb_path)
    #os.mkdir(cb_path)
    train_path = folder_path+'/train.csv'
    test_path  = folder_path+'/test.csv'
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    target = train['Score']
    trainid = train['ID']
    testid = test['ID']
    train.drop(['Score','ID'],axis=1,inplace=True)
    test.drop(['ID'],axis=1,inplace=True)
    train,test = pre_process_train_test(train,test)

    train_stage(train,target,trainid,lgb_path, xgb_path, cb_path)
    #Stacking Stage
    print('Commencing Stacking and Prediction Stage.\n')
    stacking_and_prediction_stage(train_path, lgb_path, xgb_path, cb_path,test_path)
    
    print('\nDone.')
    


