# Code to analyse Santander customer prediction 

# Extract data from train.csv

cusTrain <- read.table(file = "C:/sbairagi_backup/Work/Content/Projects/santander-customer-transaction-prediction/train.csv", sep = ",", header = T)
head(cusTrain)
summary(cusTrain)

# [Placeholder] feature-wise analysis

# scale data from col 3
varScaled <- as.data.frame(scale(cusTrain[,-1:-2]))
summary(varScaled)

# combine col 1&2 w scaled data
cusTrainScaled <- cbind(cusTrain[,1:2], varScaled)
head(cusTrainScaled)
hist(cusTrainScaled$target)

# data preparation by partitioning 70:30 testing:training

install.packages("caret")
library(caret)

set.seed(321)
intrain <- createDataPartition(y = cusTrainScaled$target, p = 0.7, list = FALSE)
# ?createDataPartition part of caret package
training<- cusTrainScaled[intrain,]
hist(training$target)
length(training$target)
length(which(training$target==0))
length(which(training$target==1))

testing <- cusTrainScaled[-intrain,]
hist(testing$target)
length(testing$target)
length(which(testing$target==0))
length(which(testing$target==1))

# model preparation. Exclude ID column from dataset
trainingExID <- training[,-1]
head(trainingExID)

# model preparation using Logistic regression
model1 <- glm(trainingExID$target~., data = trainingExID, family = binomial(link = 'logit'))
summary(model1)
model1

# prediction on test data
pred1 <- predict(model1, testing[,-1:-2])
head(pred1) # gives prediction probabilities
pred1 <- as.numeric(pred1 > 0.5)
summary(pred1)
hist(pred1)

# model performance

plot.roc(testing$target, pred1)
auc(roc(testing$target, pred1)) # provides AUC value which is ~0.5879

# ---------------------------------------------------------------------------------------------------
  
# model preparation using XGBoost
# use XGBoost for model building
install.packages("xgboost")
library(xgboost)
?xgboost

#model2 <- xgboost(data = as.matrix(training[,-1:-2]), label = as.matrix(training$target), max_depth = 2, eta = 1, nthread = 6, nrounds = 3000, maximize = T, print_every_n = 100, objective = "binary:logistic")

# preparing XGB matrix
dtrain <- xgb.DMatrix(data = as.matrix(training[,-1:-2]), label = as.matrix(training$target))
# parameters
params <- list(booster = "gbtree",
               objective = "binary:logistic",
               eta=0.02,
               #gamma=80,
               max_depth=2,
               min_child_weight=1, 
               subsample=0.5,
               colsample_bytree=0.1,
               scale_pos_weight = round(sum(!training$target) / sum(training$target), 2))
set.seed(123)
xgbcv <- xgb.cv(params = params, 
                data = dtrain, 
                nrounds = 30000, 
                nfold = 5,
                showsd = F, 
                stratified = T, 
                print_every_n = 100, 
                early_stopping_rounds = 500, 
                maximize = T,
                metrics = "auc")

cat(paste("Best iteration:", xgbcv$best_iteration))

set.seed(123)
xgb_model <- xgb.train(
  params = params, 
  data = dtrain, 
  nrounds = xgbcv$best_iteration, 
  print_every_n = 100, 
  maximize = T,
  eval_metric = "auc")

#view variable importance plot
imp_mat <- xgb.importance(feature_names = colnames(training[,-1:-2]), model = xgb_model)
xgb.plot.importance(importance_matrix = imp_mat[1:30])

#summary(model2)
#model2

# prediction on test data
pred2 <- predict(xgb_model, as.matrix(testing[,-1:-2]))
head(pred2) # gives prediction probabilities
pred2 <- as.numeric(pred2 > 0.5)
summary(pred2)
length(which(pred2 == 1))
length(which(pred2 == 0))
hist(pred2)

# model performance

plot.roc(testing$target, pred2)
auc(roc(testing$target, pred2)) # provides AUC value which is ~0.8149

