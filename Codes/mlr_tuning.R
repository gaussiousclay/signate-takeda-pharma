library(Metrics)
library(Hmisc)
library(xgboost)
library(checkmate)
library(mlr) 

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

trainTask = makeRegrTask(data = df_train, target = "Score")
trainTask = createDummyFeatures(trainTask)
testTask = makeRegrTask(data = df_test, target = "Score")
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
  makeIntegerParam("nrounds",lower=600,upper=1000),
  makeNumericParam("eta", lower = 0.01, upper = 0.03),
  makeNumericParam("colsample_bytree", lower = 0.4, upper = 0.8),
  makeNumericParam("subsample", lower = 0.4, upper = 0.8),
  makeIntegerParam("min_child_weight", lower = 1, upper = 200),
  makeIntegerParam("max_depth",lower=37,upper=37),
  makeIntegerParam("gamma",lower=1,upper=3)
  
)

rdesc = makeResampleDesc("CV", iters = 3L)
ctrl =  makeTuneControlRandom(maxit = 10)
res = tuneParams(lrn, task = trainTask, resampling = rdesc, par.set = ps, control = ctrl, measures = rsq)
res
lrn = setHyperPars(lrn, par.vals = res$x)
cv = crossval(lrn, trainTask, iter = 3, measures = rsq, show.info = TRUE)

tr = train(lrn, trainTask)
pred = predict(tr, testTask)
submission = data.frame(Id = testId)
submission$Response = pred$data$response
write.csv(submission, "xgboost_1.csv", row.names = FALSE)


##Variable Importance
labels = df_train$Score 
ts_label =  df_test$Score
train = df_train[,c(2:3752)]
test = df_test[,c(1:3751)]
new_tr = model.matrix(~.+0,data = train) 
new_ts = model.matrix(~.+0,data = test)
dtrain = xgb.DMatrix(data = new_tr,label = labels)
dtest = xgb.DMatrix(data = new_ts,label=ts_label)
params = list(booster = "gbtree", 
              objective = "reg:linear",
              eval_metric = evalerror,
              maximize = T,
              eta=0.016, 
              gamma=1, 
              max_depth=30, 
              min_child_weight=46, 
              subsample=0.609, 
              colsample_bytree=0.782)

xgbcv = xgb.cv(params = params, 
               data = dtrain,
               print_every_n = 10,
               early_stopping_rounds = 20,
               nrounds = 2000,
               maximize = T,
               nfold = 5, 
               showsd = T)

max(xgbcv$test.error.mean)
xgb1 = xgb.train (params = params, 
                  data = dtrain, 
                  nrounds = 79, 
                  watchlist = list(val=dtest,train=dtrain), 
                  print.every.n = 10, 
                  early.stop.round = 10, 
                  maximize = T , 
                  eval_metric = evalerror)
