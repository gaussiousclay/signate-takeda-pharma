# -*- coding: utf-8 -*-
"""
Created on Tue Mar 04 10:24:11 2019

@author: SAILMOH
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
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
gc.enable()


def fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name):
    
    model = lgb.LGBMRegressor(max_depth=-1,
                              n_estimators=999999,
                              learning_rate=0.005,
                              colsample_bytree=0.4,
                              num_leaves=5,
                              metric='rmse',
                              objective='regression', 
                              n_jobs=-1)
     
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)],
              verbose=0, 
              early_stopping_rounds=50)
                  
    cv_val = model.predict(X_val)
    
    #Save LightGBM Model
    save_to = '{}{}_fold{}.txt'.format(lgb_path, name, counter+1)
    model.booster_.save_model(save_to)
    return cv_val
    
    
def fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name):
    
    model = xgb.XGBRegressor(max_depth=10,
                             n_estimators=999999,
                             colsample_bytree=0.731,
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
    
    
def fit_cb(X_fit, y_fit, X_val, y_val, counter, cb_path, name):
    
    model = cb.CatBoostRegressor(iterations=999999,
                                  learning_rate=0.005,
                                  colsample_bylevel=0.03,
                                  objective="RMSE")
                                  
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)], 
              verbose=0, early_stopping_rounds=100)
              
    cv_val = model.predict(X_val)
    
    #Save Catboost Model          
    save_to = "{}{}_fold{}.mlmodel".format(cb_path, name, counter+1)
    model.save_model(save_to, format="coreml")
    
    return cv_val


def train_stage(df_path, lgb_path, xgb_path, cb_path):
    
    print('Load Train Data.')
    df_train_features = pd.read_csv(train_path)
    df_train = pd.get_dummies(df_train_features,columns=['companyId','jobType','degree','major','industry'])
    df_target = pd.read_csv(target_path)
    df_train = pd.merge(df_train,df_target)
    print('\nShape of Train Data: {}'.format(df_train.shape))
    
    y_df = np.array(df_train['salary'])                        
    df_ids = np.array(df_train.jobId)                     
    df_train.drop(['jobId', 'salary'], axis=1, inplace=True)
    
    lgb_cv_result = np.zeros(df_train.shape[0])
    xgb_cv_result = np.zeros(df_train.shape[0])
    cb_cv_result  = np.zeros(df_train.shape[0])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf.get_n_splits(df_ids, y_df)
    
    print('\nModel Fitting...')
    for counter, ids in enumerate(skf.split(df_ids, y_df)):
        print('\nFold {}'.format(counter+1))
        X_fit, y_fit = df_train.values[ids[0]], y_df[ids[0]]
        X_val, y_val = df_train.values[ids[1]], y_df[ids[1]]
    
        print('LightGBM')
        lgb_cv_result[ids[1]] += fit_lgb(X_fit, y_fit, X_val, y_val, counter, lgb_path, name='lgb')
        print('XGBoost')
        xgb_cv_result[ids[1]] += fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name='xgb')
        print('CatBoost')
        cb_cv_result[ids[1]]  += fit_cb(X_fit,  y_fit, X_val, y_val, counter, cb_path,  name='cb')
        
        del X_fit, X_val, y_fit, y_val
        gc.collect()
    
    rmse_lgb  = round(math.sqrt(mean_squared_error(y_df, lgb_cv_result)),4)
    rmse_xgb  = round(math.sqrt(mean_squared_error(y_df, xgb_cv_result)),4)
    rmse_cb   = round(math.sqrt(mean_squared_error(y_df, cb_cv_result)), 4)
    rmse_mean = round(math.sqrt(mean_squared_error(y_df, (lgb_cv_result+xgb_cv_result+cb_cv_result)/3)), 4)
    rmse_mean_lgb_cb = round(math.sqrt(mean_squared_error(y_df, (lgb_cv_result+cb_cv_result)/2)), 4)
    print('\nLightGBM VAL RMSE: {}'.format(rmse_lgb))
    print('XGBoost  VAL RMSE: {}'.format(rmse_xgb))
    print('Catboost VAL RMSE: {}'.format(rmse_cb))
    print('Mean Catboost+LightGBM VAL RMSE: {}'.format(rmse_mean_lgb_cb))
    print('Mean XGBoost+Catboost+LightGBM, VAL RMSE: {}\n'.format(rmse_mean))
    
    return 0
    
    
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



    
if __name__ == '__main__':
    
    train_path = 'D:\Analytics Repository\Indeed/train_features.csv'
    target_path = 'D:\Analytics Repository\Indeed/train_salaries.csv'
    test_path  = 'D:\Analytics Repository\Indeed/test_features.csv'
    
    lgb_path = 'D:\Analytics Repository\Indeed\lgb_models_stack/'
    xgb_path = 'D:\Analytics Repository\Indeed\exgb_models_stack/'
    cb_path = 'D:\Analytics Repository\Indeed\cb_models_stack/'
 
    #Create dir for models
    #os.mkdir(lgb_path)
    #os.mkdir(xgb_path)
    #os.mkdir(cb_path)
    #os.mkdir(xgb_stacking_path)

    train_stage(train_path, lgb_path, xgb_path, cb_path)
    #Stacking Stage
    print('Commencing Stacking and Prediction Stage.\n')
    stacking_and_prediction_stage(train_path, lgb_path, xgb_path, cb_path,test_path)
    
    print('\nDone.')
    


