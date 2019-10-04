##Ensembles
library(Metrics)
library(Hmisc)
library(xgboost)
library(checkmate)
library(mlr)
library(e1071)
library(randomForest)

preprocess_data = function(df_train)
{
  cols_to_remove = c()
  count=1
  for(i in 1:ncol(df_train))
  {
    if(length(unique((df_train[,i])))==1)
    {
      cols_to_remove[count] = colnames(df_train)[i]
      count=count+1
    }
  }
  
  count=1
  cols_to_categorical = c()
  for(i in 1:ncol(df_train))
  {
    if(length(unique((df_train[,i])))==2)
    {
      if(unique(df_train[,i])[1]==0 && unique(df_train[,i])[2]==1)
      {
        cols_to_categorical[count] = colnames(df_train)[i]
        count=count+1
      }
    }
  }
  
  cols_to_edit = NULL
  cols_to_edit[[1]] = cols_to_categorical
  cols_to_edit[[2]] = cols_to_remove
  return(cols_to_edit)
}

get_xgb_models = function()
{
  model_dir = 'Models/'
  file_list = list.files(model_dir,'XGB')
  models = NULL
  for (i in 1:length(file_list))
  {
    models[[i]] = xgb.load(paste0(model_dir,'/',file_list[i]))
    
  }
  return(models)  
}

evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  mean_labels = mean(labels)
  err = as.numeric(1-(sum((labels-preds)^2)/sum((labels-mean_labels)^2)))
  return(list(metric = "error", value = err))
}

df_train = read.csv('train.csv',stringsAsFactors = F)
df_test = read.csv('test.csv',stringsAsFactors = F)

cols_to_edit = preprocess_data(df_train)

df_train = df_train[,!colnames(df_train)%in% cols_to_edit[[2]]]
df_test = df_test[,!colnames(df_test)%in% cols_to_edit[[2]]]

#spurious_cols = c()
#count = 1

#for(i in cols_to_edit[[1]])
#{
#  df_train[,i] = as.factor(df_train[,i])
#  df_test[,i] = as.factor(df_test[,i])
#  if(length(unique(df_train[,i])) != length(unique(df_test[,i])))
#  {
#    spurious_cols[count] = i
#    count = count+1
#  }
#}

df_test$Score = 0
testId = df_test$ID
df_train$ID = df_test$ID = NULL
labels = df_train$Score 
ts_label =  df_test$Score
train = df_train[,c(2:3752)]
test = df_test[,c(1:3751)]
new_tr = model.matrix(~.+0,data = train) 
new_ts = model.matrix(~.+0,data = test)
dtrain = xgb.DMatrix(data = new_tr,label = labels)
dtest = xgb.DMatrix(data = new_ts,label=ts_label)

models = get_xgb_models()
ensemble_train = data.frame(Actual = labels)
ensemble_test = data.frame(Actual = ts_label)
for (i in (1:length(models)))
{
  ensemble_train[[paste0('preds_',i)]] = predict(models[[i]],dtrain)
  ensemble_test[[paste0('preds_',i)]] = predict(models[[i]],dtest)
}


ensemble_train$log_preds1 = log(ensemble_train$preds_1)
ensemble_train$log_preds2 = log(ensemble_train$preds_2)
ensemble_train$sqrt_preds1 = sqrt(ensemble_train$preds_1)
ensemble_train$sqrt_preds2 = sqrt(ensemble_train$preds_2)
ensemble_train$sqr_preds1 = (ensemble_train$preds_1**2)
ensemble_train$sqr_preds2 = (ensemble_train$preds_2**2)
ensemble_train$tanh_preds1 = tanh(ensemble_train$preds_1)
ensemble_train$tanh_preds2 = tanh(ensemble_train$preds_2)
ensemble_train$sigm_preds1 = sigmoid(ensemble_train$preds_1)
ensemble_train$sigm_preds2 = sigmoid(ensemble_train$preds_2)

ensemble_test$log_preds1 = log(ensemble_test$preds_1)
ensemble_test$log_preds2 = log(ensemble_test$preds_2)
ensemble_test$sqrt_preds1 = sqrt(ensemble_test$preds_1)
ensemble_test$sqrt_preds2 = sqrt(ensemble_test$preds_2)
ensemble_test$sqr_preds1 = (ensemble_test$preds_1**2)
ensemble_test$sqr_preds2 = (ensemble_test$preds_2**2)
ensemble_test$tanh_preds1 = tanh(ensemble_test$preds_1)
ensemble_test$tanh_preds2 = tanh(ensemble_test$preds_2)
ensemble_test$sigm_preds1 = sigmoid(ensemble_test$preds_1)
ensemble_test$sigm_preds2 = sigmoid(ensemble_test$preds_2)

ensemble_train[is.na(ensemble_train)] = 0
ensemble_test[is.na(ensemble_test)] = 0


trainTask = makeRegrTask(data = ensemble_train, target = "Actual")
trainTask = createDummyFeatures(trainTask)
testTask = makeRegrTask(data = ensemble_test, target = "Actual")
testTask = createDummyFeatures(testTask)

set.seed(123)
lrn = makeLearner("regr.xgboost")
lrn$par.vals = list(
  print_every_n = 50,
  objective = "reg:linear",
  eval_metric = evalerror,
  early_stopping_rounds = 100,
  maximize = T
)

lrn = makeImputeWrapper(lrn, classes = list(numeric = imputeMedian(), integer = imputeMedian()))

ps = makeParamSet(
  makeIntegerParam("nrounds",lower=600,upper=1500),
  makeNumericParam("eta", lower = 0.01, upper = 0.03),
  makeNumericParam("colsample_bytree", lower = 0.4, upper = 0.8),
  makeNumericParam("subsample", lower = 0.4, upper = 0.8),
  makeIntegerParam("min_child_weight", lower = 1, upper = 200),
  makeIntegerParam("max_depth",lower=30,upper=40),
  makeIntegerParam("gamma",lower=1,upper=3)
  
)

rdesc = makeResampleDesc("CV", iters = 3L)
ctrl =  makeTuneControlRandom(maxit = 59)
res = tuneParams(lrn, task = trainTask, resampling = rdesc, par.set = ps, control = ctrl, measures = rsq)
res
lrn = setHyperPars(lrn, par.vals = res$x)
cv = crossval(lrn, trainTask, iter = 3, measures = rsq, show.info = TRUE)

tr = train(lrn, trainTask)
pred = predict(tr, testTask)
submission = data.frame(testId,pred$data$response)
write.csv(submission, "ens_stacking_xgb2_xgb3.csv", row.names = FALSE)

