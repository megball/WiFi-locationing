# Title: C3T3_caret_script

# Last update: 9.14.19

# File: C3T3_caret_script_r1.R
# Project name: Multiple models for WiFi locationing dataset


###############
# Project Notes
###############

# Summarize project: Using WAPs to locate indoor positioning

# Summarize top model and/or filtered dataset
# The top model was C50Fit3 used with trainset_v2.
                                                                                                                                                                                                                                                                                                                                                                                              


###############
# Housekeeping
###############

# Clear objects if necessary
rm(list = ls())

# get working directory
getwd()
? getwd  # get help
# set working directory
setwd("C:/Users/megan/OneDrive/Data_Analytics/WiFi-locationing")
dir()


################
# Load packages
################

install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("dplyr")
install.packages("mlbench")
install.packages("Hmisc")
install.packages("randomForest")
install.packages("doParallel")
install.packages("drat", repos = "https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos = "http://dmlc.ml/drat/", type = "source")
install.packages("tidyverse")

library(caret)
library(corrplot)
library(readr)
library(dplyr)
library(mlbench)
library(Hmisc)
library(randomForest)
library(doParallel)
library(tidyverse)
require(xgboost)

#####################
# Parallel processing
#####################

#--- for Win ---#
detectCores()  # detect number of cores (2)
cl <- makeCluster(4)  # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()  # confirm number of cores being used by RStudio
# Stop Cluster. After performing your tasks, make sure to stop your cluster.

stopCluster(cl)
registerDoSEQ()



###############
# Import data
##############

#--- Load raw datasets ---#

## Load Train/Existing data (Dataset 1)
trainingData <-
  read.csv(
    "trainingData.csv",
    stringsAsFactors = FALSE,
    header = T
  )
class(trainingData)  # "data.frame"


## Load Predict/New data (Dataset 2) ---#

validationData <-
  read.csv(
    "validationData.csv",
    stringsAsFactors = FALSE,
    header = T
  )
class(validationData)  # "data.frame"


#--- Load preprocessed datasets that have been saved ---#
#read back in files



################
# Evaluate data
################

#--- Dataset 1 ---#

str(trainingData)  # 19937 obs. of  529 variables
#move unique columns to front for easy viewing
ordered_columns_leftside = c(
  'LONGITUDE',
  'LATITUDE',
  'FLOOR',
  'BUILDINGID',
  'SPACEID',
  'TIMESTAMP',
  'RELATIVEPOSITION',
  'USERID',
  'PHONEID'
)
trainingData = trainingData[c(ordered_columns_leftside,
                              setdiff(names(trainingData), ordered_columns_leftside))]

head(trainingData)
str(trainingData$USERID) #int
str(trainingData$LATITUDE) #num
str(trainingData$LONGITUDE) #num
str(trainingData$FLOOR) #int
str(trainingData$BUILDINGID) #int
str(trainingData$SPACEID) #int
str(trainingData$RELATIVEPOSITION) #int
str(trainingData$PHONEID) #int
str(trainingData$TIMESTAMP) #int

names(trainingData)
tail(trainingData)
summary(trainingData)
# plot
hist(trainingData$USERID)
hist(trainingData$WAP001)
hist(trainingData$LATITUDE)
hist(trainingData$LONGITUDE)
qqnorm(trainingData$LONGITUDE)
# check for missing values
anyNA(trainingData)
is.na(trainingData)
# remove or exclude missing values
na.omit(DatasetName$ColumnName) # Drops any rows with missing values and omits them forever.
na.exclude(DatasetName$ColumnName) # Drops any rows with missing values, but keeps track of where they were.

#--- Dataset 2 ---#
str(validationData)  # 1111 obs. of  530 variables

validationData = validationData[c(ordered_columns_leftside,
                                  setdiff(names(validationData), ordered_columns_leftside))]

names(validationData)
head(validationData)
tail(validationData)
summary(validationData)
# plot
hist(validationData$USERID)
hist(validationData$WAP001)
hist(validationData$LATITUDE)
hist(validationData$LONGITUDE)
qqnorm(validationData$LONGITUDE)
# check for missing values
anyNA(validationData)
is.na(validationData)
# remove or exclude missing values
#na.omit(DatasetName$ColumnName) # Drops any rows with missing values and omits them forever.
#na.exclude(DatasetName$ColumnName) # Drops any rows with missing values, but keeps track of where they were.


#############
# Preprocess
#############

#--- Dataset 1 ---#

# change data types
trainingData$USERID <- as.factor(trainingData$USERID)
trainingData$PHONEID <- as.factor(trainingData$PHONEID)

#create new ID for single location identifier
trainingData <- trainingData %>%
  mutate(LOCATION = paste(SPACEID, RELATIVEPOSITION, BUILDINGID, FLOOR, sep = '_'))

colnames(trainingData)
head(trainingData, 1)
trainingData$LOCATION <- as.factor(trainingData$LOCATION)
str(trainingData$LOCATION)
#Factor w/ 905 levels "1_1_1_0","1_2_1_0",..: 93 93 48 34 285 79 48 17 178 41 ...

#move to first column
ordered_columns_leftside1 = c('LOCATION')
trainingData = trainingData[c(ordered_columns_leftside1,
                              setdiff(names(trainingData), ordered_columns_leftside1))]

colnames(trainingData)

str(trainingData$USERID)
str(trainingData$BUILDINGID)
str(trainingData$RELATIVEPOSITION)
str(trainingData$PHONEID)
str(trainingData$SPACEID)


# handle missing values (if applicable)
#na.omit(ds$ColumnName)
#na.exclude(ds$ColumnName)
#ds$ColumnName[is.na(ds$ColumnName)] <- mean(ds$ColumnName,na.rm = TRUE)

? na.omit  # returns object if with incomplete cases removed
? na.exclude




#--- Dataset 2 ---#

# change data types
validationData$USERID <- as.factor(validationData$USERID)
validationData$PHONEID <- as.factor(validationData$PHONEID)

#create new ID for single location identifier
validationData <- validationData %>%
  mutate(LOCATION = paste(SPACEID, RELATIVEPOSITION, BUILDINGID, FLOOR, sep = '_'))

colnames(validationData)
validationData$LOCATION <- as.factor(validationData$LOCATION)
str(validationData$LOCATION)

#move to first column
ordered_columns_leftside1 = c('LOCATION')
validationData = validationData[c(ordered_columns_leftside1,
                                  setdiff(names(validationData), ordered_columns_leftside1))]



#################
# Feature removal
#################

#--- Dataset 1 ---#

# remove ID and obvious features
trainingData$X <- NULL
trainingData_v1 <- trainingData
trainingData_v1$USERID <- NULL   # remove ID
trainingData_v1$PHONEID <- NULL  # remove phone ID
trainingData_v1$TIMESTAMP <- NULL #remove timestamp
trainingData_v1$SPACEID <-
  NULL #remove space ID to avoid duplicate with new location
trainingData_v1$FLOOR <- NULL #remove FLOOR as it is in new dv
trainingData_v1$RELATIVEPOSITION <- NULL
trainingData_v2 <- trainingData_v1
trainingData_v2$LATITUDE <- NULL
trainingData_v2$LONGITUDE <- NULL

#create subsets for each building before removing from df
trainingData_bldg0 <- filter(trainingData_v1, BUILDINGID == "0")
trainingData_bldg0$BUILDINGID <- NULL
str(trainingData_bldg0)
#5249 obs of 523 variables
str(trainingData_bldg0$LOCATION)
#Factor w/ 905 levels "1_1_1_0","1_2_1_0",..: 285 29 144 158 103 88 208 198 186 173 ...

trainingData_bldg0v2 <- filter(trainingData_v2, BUILDINGID == "0")
trainingData_bldg0v2$BUILDINGID <- NULL
str(trainingData_bldg0v2)
#5249 obs of 521 variables
str(trainingData_bldg0v2$LOCATION)
#Factor w/ 905 levels "1_1_1_0","1_2_1_0",..: 285 29 144 158 103 88 208 198 186 173 ...

#drop down factor levels
trainingData_bldg0$LOCATION <-
  droplevels(trainingData_bldg0$LOCATION)
str(trainingData_bldg0$LOCATION)
#Factor w/ 259 levels "101_2_0_1","101_2_0_2",..: 78 4 31 35 21 17 51 47 43 39 ...

#drop down factor levels
trainingData_bldg0v2$LOCATION <-
  droplevels(trainingData_bldg0v2$LOCATION)
str(trainingData_bldg0v2$LOCATION)
#Factor w/ 259 levels "101_2_0_1","101_2_0_2",..: 78 4 31 35 21 17 51 47 43 39 ...

trainingData_bldg1 <- filter(trainingData_v1, BUILDINGID == "1")
trainingData_bldg1$BUILDINGID <- NULL
str(trainingData_bldg1)
#5196 obs of 523 variables

trainingData_bldg1v2 <- filter(trainingData_v2, BUILDINGID == "1")
trainingData_bldg1v2$BUILDINGID <- NULL

#drop down factor levels
trainingData_bldg1$LOCATION <-
  droplevels(trainingData_bldg1$LOCATION)
str(trainingData_bldg1$LOCATION)
# Factor w/ 243 levels "1_1_1_0","1_2_1_0",..: 40 40 24 18 35 24 12 66 21 27 ...

#drop down factor levels
trainingData_bldg1v2$LOCATION <-
  droplevels(trainingData_bldg1v2$LOCATION)
str(trainingData_bldg1v2$LOCATION)
# Factor w/ 243 levels "1_1_1_0","1_2_1_0",..: 40 40 24 18 35 24 12 66 21 27 ...

trainingData_bldg2 <- filter(trainingData_v1, BUILDINGID == "2")
trainingData_bldg2$BUILDINGID <- NULL
str(trainingData_bldg2)
#9492 obs of 523 variables

trainingData_bldg2v2 <- filter(trainingData_v2, BUILDINGID == "2")
trainingData_bldg2v2$BUILDINGID <- NULL
str(trainingData_bldg2v2)
#9492 obs of 521 variables

#drop down factor levels
trainingData_bldg2$LOCATION <-
  droplevels(trainingData_bldg2$LOCATION)
str(trainingData_bldg2$LOCATION)
#Factor w/ 403 levels "101_1_2_2","101_1_2_3",..: 398 399 397 396 395 394 393 390 392 389 ...

trainingData_bldg2v2$LOCATION <-
  droplevels(trainingData_bldg2v2$LOCATION)
str(trainingData_bldg2v2$LOCATION)
#Factor w/ 403 levels "101_1_2_2","101_1_2_3",..: 398 399 397 396 395 394 393 390 392 389 ...


trainingData_v1$BUILDINGID <- NULL #remove as it is in new dv
trainingData_v2$BUILDINGID <- NULL #remove as it is in new dv

str(trainingData_v1) # 19937 obs. of  523 variables

#check for zero var attributes & create new data set for each subset
zeroVarData <- nearZeroVar(trainingData_v1, saveMetrics = TRUE)
head(zeroVarData, 30)
zeroVarData <- which(zeroVarData$zeroVar == T)
zeroVarData

zeroVarDatabldg0 <-
  nearZeroVar(trainingData_bldg0, saveMetrics = TRUE)
#this creates a vector containing the list of observations that have zeroVar
zeroVarDatabldg0a <- which(zeroVarDatabldg0$zeroVar == T)
#this inspects the results
zeroVarDatabldg0a
# new dataset without zero var data
trainingData_bldg0nozv <- trainingData_bldg0[, -(zeroVarDatabldg0a)]
#203 variables
trainingData_bldg0v2 <- trainingData_bldg0v2[, -(zeroVarDatabldg0a)]
#202 variables

zeroVarDatabldg1 <-
  nearZeroVar(trainingData_bldg1, saveMetrics = TRUE)
#this creates a vector containing the list of observations that have zeroVar
zeroVarDatabldg1a <- which(zeroVarDatabldg1$zeroVar == T)
#this inspects the results
zeroVarDatabldg1a
# new dataset without zero var data
trainingData_bldg1nozv <- trainingData_bldg1[, -(zeroVarDatabldg1a)]
#210 variables
trainingData_bldg1v2 <- trainingData_bldg1v2[, -(zeroVarDatabldg1a)]
#210 variables

zeroVarDatabldg2 <-
  nearZeroVar(trainingData_bldg2, saveMetrics = TRUE)
#this creates a vector containing the list of observations that have zeroVar
zeroVarDatabldg2a <- which(zeroVarDatabldg2$zeroVar == T)
#this inspects the results
zeroVarDatabldg2a
# new dataset without zero var data
trainingData_bldg2nozv <- trainingData_bldg2[, -(zeroVarDatabldg2a)]
#206 variables
trainingData_bldg2v2 <- trainingData_bldg2v2[, -(zeroVarDatabldg2a)]
#206 variables

# remove based on Feature Selection (FS)


#--- Dataset 2 ---#

# remove ID and obvious features
validationData$X <- NULL
validationData_v1 <- validationData
validationData_v1$USERID <- NULL   # remove ID
validationData_v1$PHONEID <- NULL  # remove phone ID
validationData_v1$TIMESTAMP <- NULL #remove timestamp
validationData_v1$SPACEID <- NULL #remove space ID
validationData_v1$FLOOR <- NULL #remove FLOOR as it is in new dv
validationData_v1$RELATIVEPOSITION <- NULL
validationData_v1$BUILDINGID <- NULL #remove as it is in new dv
str(validationData_v1) # 1111 obs. of  523 variables

validationData_v2 <- validationData_v1
validationData_v2$LATITUDE <- NULL
validationData_v2$LONGITUDE <- NULL

#remove zero var attributes to match training set
validationDatanozv <- validationData_v1[, -(zeroVarData)]
validationData_v2 <- validationData_v2[, -(zeroVarData)]


###############
# Save datasets
###############

# after ALL preprocessing, save a new version of the dataset

write.csv(trainingData_v1, file = "trainingData_v1.csv")
write.csv(validationData_v1, file = "validationData_v1.csv")

write.csv(trainingData_v2, file = "trainingData_v2.csv")
write.csv(validationData_v2, file = "validationData_v2.csv")

write.csv(trainingData_bldg0v2, file = "trainingData_bldg0v2.csv")
write.csv(trainingData_bldg1v2, file = "trainingData_bldg1v2.csv")
write.csv(trainingData_bldg2v2, file = "trainingData_bldg2v2.csv")

################
# Sampling
################


# ---- Sampling ---- #

# create 30% sample
set.seed(998) # set random seed
trainingData30p <-
  trainingData_v1[sample(1:nrow(trainingData_v1), round(nrow(trainingData_v1) *
                                                          .3), replace = FALSE), ]
nrow(trainingData30p)
head(trainingData30p) # ensure randomness

# sample again after removing any features


##########################
# Feature Selection (FS)
##########################

# Three primary methods
# 1. Filtering
# 2. Wrapper methods (e.g., RFE caret)
# 3. Embedded methods (e.g., varImp)

###########
# Filtering
###########




############
# caret RFE
############

# lmFuncs - linear model
# rfFuncs - random forests
# nbFuncs - naive Bayes
# treebagFuncs - bagged trees


## ---- lm ---- ##


# define refControl using a linear model selection function (regression only)
#LMcontrol <- rfeControl(functions=lmFuncs, method="cv", number=10)
# run the RFE algorithm
#set.seed(7)
#LMresults <- rfe(trainingData_v1[,2:470], trainingData_v1[,1], sizes=c(2:470), rfeControl=LMcontrol)
#LMresults
# Note performance metrics.

# plot the results
#plot(LMresults, type=c("g", "o"))
# show predictors used
#predictors(LMresults)
# Note results.
#varImp(LMresults)
# Note results.

## ---- rf ---- ##

#takes too long to run

# define the control using a random forest selection function (regression or classification)
#RFcontrol <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
#set.seed(7)
#RFresults <- rfe(trainingData_v1[,2:524], trainingData_v1[,1], sizes=c(2:524), rfeControl=RFcontrol)
#RFresults

# plot the results
#plot(RFresults, type=c("g", "o"))
# show predictors used
#predictors(RFresults)
# Note results.
#varImp(RFresults)
# Note results.



##############################
# Variable Importance (varImp)
##############################

# varImp is evaluated in the model train/fit section


# ---- Conclusion ---- #

#



##################
# Train/test sets
##################

# set random seed
set.seed(123)
# create the training partition that is 75% of total obs
inTraining <-
  createDataPartition(trainingData_v1$LOCATION, p = 0.75, list = FALSE)
# create training/testing dataset
trainSet <- trainingData_v1[inTraining, ]
testSet <- trainingData_v1[-inTraining, ]
# verify number of obs
nrow(trainSet) # 15184
nrow(testSet)  # 4753
#check dv
str(trainSet$LOCATION)
#$ LOCATION        : Factor w/ 905 levels
str(testSet$LOCATION)
# Factor w/ 905 level


#create train/test sets for v2 df --- ***this is the data set used for best models****
# set random seed
set.seed(123)
# create the training partition that is 75% of total obs
inTraining_v2 <-
  createDataPartition(trainingData_v2$LOCATION, p = 0.75, list = FALSE)
# create training/testing dataset
trainSet_v2 <- trainingData_v2[inTraining_v2, ]
testSet_v2 <- trainingData_v2[-inTraining_v2, ]
# verify number of obs
nrow(trainSet_v2) # 15184
nrow(testSet_v2)  # 4753

#check dv
str(trainSet_v2$LOCATION)
#Factor w/ 905 levels
str(testSet_v2$LOCATION)
# Factor w/ 905 level

#create train/test sets for bldg 0
# set random seed
set.seed(123)
# create the training partition that is 75% of total obs
inTraining_bldg0v2 <-
  createDataPartition(trainingData_bldg0v2$LOCATION, p = 0.75, list = FALSE)
# create training/testing dataset
trainSet_bldg0v2 <- trainingData_bldg0v2[inTraining_bldg0v2, ]
testSet_bldg0v2 <- trainingData_bldg0v2[-inTraining_bldg0v2, ]
# verify number of obs
nrow(trainSet_bldg0v2) # 3996
nrow(testSet_bldg0v2)  # 1253

#check dv
str(trainSet_bldg0v2$LOCATION)
#Factor w/ 259 levels
str(testSet_bldg0v2$LOCATION)
# Factor w/ 259 level

#create train/test sets for bldg 1
# set random seed
set.seed(123)
# create the training partition that is 75% of total obs
inTraining_bldg1v2 <-
  createDataPartition(trainingData_bldg1v2$LOCATION, p = 0.75, list = FALSE)
# create training/testing dataset
trainSet_bldg1v2 <- trainingData_bldg1v2[inTraining_bldg1v2, ]
testSet_bldg1v2 <- trainingData_bldg1v2[-inTraining_bldg1v2, ]
# verify number of obs
nrow(trainSet_bldg1v2) # 3964
nrow(testSet_bldg1v2)  # 1232

#check dv
str(trainSet_bldg1v2$LOCATION)
#Factor w/ 243 levels
str(testSet_bldg1v2$LOCATION)
# Factor w/ 243 level

#create train/test sets for bldg 2
# set random seed
set.seed(123)
# create the training partition that is 75% of total obs
inTraining_bldg2v2 <-
  createDataPartition(trainingData_bldg2v2$LOCATION, p = 0.75, list = FALSE)
# create training/testing dataset
trainSet_bldg2v2 <- trainingData_bldg2v2[inTraining_bldg2v2, ]
testSet_bldg2v2 <- trainingData_bldg2v2[-inTraining_bldg2v2, ]
# verify number of obs
nrow(trainSet_bldg2v2) # 7224
nrow(testSet_bldg2v2)  # 2268

#check dv
str(trainSet_bldg2v2$LOCATION)
#Factor w/ 403 levels
str(testSet_bldg2v2$LOCATION)
# Factor w/ 403 level

################
# Train control
################

# set 10 fold cross validation
fitControl <-
  trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 1,
    allowParallel = TRUE
  )

#set 5 fold cross validation due to # of features
fitControl1 <-
  trainControl(
    method = "repeatedcv",
    number = 5,
    repeats = 1,
    allowParallel = TRUE
  )


##############
# Train model
##############


## ------- KNN ------- ##


#model using only WAP data, without zero variance WAP or long/lat
KNN3 <- train(
  LOCATION ~ . ,
  data = trainSet_v2,
  method = 'knn',
  preProcess = c('center', 'scale'),
  metric = 'Accuracy',
  trControl = fitControl,
  tuneLength = 3
)

#k  Accuracy   Kappa    
#5  0.5704474  0.5698359
#7  0.5434229  0.5427712
#9  0.5217109  0.5210275

#using data from bldg0 only to speed up models
KNNbldg0 <-
  train(
    LOCATION ~ . ,
    data = trainSet_bldg0v2,
    method = 'knn',
    preProcess = c('center', 'scale'),
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 3
  )

KNNbldg0
#Pre-processing: centered (196), scaled (196)
#Resampling: Cross-Validated (10 fold, repeated 1 times)
#Summary of sample sizes: 3592, 3582, 3615, 3595, 3604, 3602, ...
#Resampling results across tuning parameters:

#  k  Accuracy   Kappa
#5  0.4615140  0.4592707
#7  0.4316180  0.4292608
#9  0.3987072  0.3962228

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was k = 5.

#remove pre-processing, theoretically do not need to center or scale data as it is WAP only
KNNbldg0_1 <-
  train(
    LOCATION ~ . ,
    data = trainSet_bldg0v2,
    method = 'knn',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 3
  )

KNNbldg0_1
#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 1 times) 
#Summary of sample sizes: 3604, 3599, 3603, 3593, 3593, 3600, ... 
#Resampling results across tuning parameters:
  
#  k  Accuracy   Kappa    
#5  0.4937293  0.4915938
#7  0.4650998  0.4628567
#9  0.4460690  0.4437537

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was k = 5.

KNNbldg0v2 <-
  train(
    LOCATION ~ . ,
    data = trainSet_bldg0v2,
    method = 'knn',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneGrid = data.frame(k = c(1,2,3))
  )

#k  Accuracy   Kappa    
#1  0.5582849  0.5564176
#2  0.5184230  0.5163970
#3  0.5176651  0.5156404

#model for all data using settings for best model from bldg0, no pre-process, but k=1
KNNfinal <- train(
  LOCATION ~ . ,
  data = trainSet_v2,
  method = 'knn',
  metric = 'Accuracy',
  trControl = fitControl,
  tuneGrid = data.frame(k = c(1))
)

KNNfinal
#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 1 times) 
#Summary of sample sizes: 13671, 13655, 13684, 13670, 13669, 13664, ... 
#Resampling results:
  
#  Accuracy  Kappa    
#0.657836  0.6573543

#Tuning parameter 'k' was held constant at a value of 1

## ------- xgb ------- ##

#too computationally expensive; doesn't finish running
#xgb_fitv1 <- train(LOCATION  ~., data = trainSet_v2, method = "xgbTree",
#                trControl=fitControl,
#                tuneLength = 1)

#xgb_fitv1 <-
#  train( 
#    LOCATION  ~ .,
#    data = trainSet_bldg0v2,
#    method = "xgbTree",
#    trControl = fitControl,
#    tuneLength = 1
#  )


## ------- C5.0 ------- ##

#computationally expensive - does not complete within 12 hrs
#C50Fit1 <- train(LOCATION  ~., data = trainSet_v2, method = 'C5.0',
#                 preProcess = c('center','scale'),
#                 metric = 'Accuracy',
#                 trControl=fitControl,
#                 tuneLength = 3)

C50Fit1 <-
  train(
    LOCATION  ~ .,
    data = trainSet_bldg0v2,
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 3
  )

C50Fit1
#  model  winnow  trials  Accuracy   Kappa    
# rules  FALSE   20      0.6518586  0.6503680

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were trials = 20, model = rules and winnow = FALSE.

#match settings for best model on bldg 0 set
C50grid <- expand.grid(model = "rules", winnow = FALSE, trials = 20)

#train model on all data
C50Fit3 <-
  train(
    LOCATION  ~ .,
    data = trainSet_v2,
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneGrid = C50grid
  )

#  Accuracy   Kappa    
#0.7309119  0.7305313

#run same model but with other settings to see if accuracy/kappa improves

#did not finish in 12+ hours
#C50Fit4 <-
#  train(
#    LOCATION  ~ .,
#    data = trainSet_v2,
#    method = 'C5.0',
#    metric = 'Accuracy',
#    trControl = fitControl,
#    tuneLength = 5
#  )


## ------- SVM ------- ##

SVMFit1 <-
  train(
    LOCATION  ~ .,
    data = trainSet_bldg0v2,
    method = 'svmLinear2',
    trControl = fitControl
  )

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 1 times) 
#Summary of sample sizes: 3599, 3598, 3588, 3589, 3591, 3594, ... 
#Resampling results across tuning parameters:
  
#  cost  Accuracy   Kappa    
#0.25  0.5367121  0.5347636
#0.50  0.5367128  0.5347649
#1.00  0.5395146  0.5375799

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was cost = 1.

SVMFit2 <-
  train(
    LOCATION  ~ .,
    data = trainSet_bldg0v2,
    method = 'svmLinear2',
    trControl = fitControl,
    tuneGrid = data.frame(cost = c(0.01, 0.05, 0.1)
    ))

#cost  Accuracy   Kappa    
#0.01  0.5384315  0.5364686
#0.05  0.5417261  0.5397846
#0.10  0.5402067  0.5382650

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was cost = 0.05.

#same model settings but using all data
SVMFit3 <-
  train(
    LOCATION ~.,
    data = trainSet_v2,
    method = 'svmLinear2',
    trControl = fitControl,
    tuneGrid = data.frame(cost = c(0.05))
  )

#it appears some CV folds show zero variance so model cannot run. skipping this, does not work on complete dataset

#did not finish
#SVMFit3 <-
#  train(
#    LOCATION  ~ .,
#    data = trainSet_bldg0v2,
#    method = 'svmRadialCost',
#    trControl = fitControl,
#)


## ------- rf ------- ##

rfFit1 <-
  train(
    LOCATION ~ .,
    data = trainSet_bldg0v2,
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 2
  )

#mtry  Accuracy    Kappa      
#2   0.00902934  0.001020132
#196   0.69871261  0.697414888

#RF does not do well with many features


## ------- adabag ------- ##

adaGrid <- expand.grid(mfinal = 150, maxdepth = 10)

adaFit1 <-
  train(
    LOCATION ~ .,
    data = trainSet_bldg0v2,
    method = 'AdaBag',
    trControl = fitControl,
    tuneLength = 3
  )

#maxdepth  mfinal  Accuracy     Kappa       
#1          50     0.008009256  6.093065e-05
#1         100     0.008009256  6.093065e-05
#1         150     0.008254956  3.149923e-04
#2          50     0.020182481  1.292938e-02
#2         100     0.018541332  1.136104e-02
#2         150     0.017545342  1.027108e-02
#3          50     0.046806254  4.105290e-02
#3         100     0.047594184  4.159887e-02
#3         150     0.049897541  4.388179e-02

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were mfinal = 150 and maxdepth = 3.

adaFit2 <-
  train(
    LOCATION ~ .,
    data = trainSet_bldg0v2,
    method = 'AdaBag',
    trControl = fitControl,
    tuneGrid = adaGrid
  )

#Resampling results:

#  Accuracy   Kappa    
#0.4718443  0.4694463

#Tuning parameter 'mfinal' was held constant at a value of 150
#Tuning parameter 'maxdepth' was held constant at a value
#of 10

adaGrid2 <- expand.grid(mfinal = 150, maxdepth = c(9, 10, 11, 12))

adaFit2 <-
  train(
    LOCATION ~ .,
    data = trainSet_bldg0v2,
    method = 'AdaBag',
    trControl = fitControl,
    tuneGrid = adaGrid
  )

#Accuracy   Kappa    
#0.4799275  0.4775451

#Tuning parameter 'mfinal' was held constant at a value of 150
#Tuning parameter 'maxdepth' was held constant at a value
#of 10

adaFit2 <-
  train(
    LOCATION ~ .,
    data = trainSet_v2,
    method = 'AdaBag',
    trControl = fitControl,
    tuneGrid = adaGrid
  )


## ------- hdda ------- ##

hddaFit1 <-
  train(
    LOCATION ~ .,
    data = trainSet_bldg0v2,
    method = "hdda",
    trControl = fitControl,
  )

#threshold  Accuracy   Kappa    
#0.050      0.4048117  0.4022007
#0.175      0.4621507  0.4599049
#0.300      0.4604221  0.4582002

#Tuning parameter 'model' was held constant at a value of all
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were threshold = 0.175 and model = all.

hddaFit2 <-
  train(
    LOCATION ~ .,
    data = trainSet_bldg0v2,
    method = "hdda",
    trControl = fitControl,
    tuneLength = 5
  )

#  threshold  Accuracy   Kappa    
#0.0500     0.4163485  0.4138067
#0.1125     0.4418513  0.4394458
#0.1750     0.4595544  0.4572817
#0.2375     0.4547510  0.4524964
#0.3000     0.4565886  0.4543305


hddaFit3 <-
  train(
    LOCATION ~ .,
    data = trainSet_v2,
    method = "hdda",
    preProcess = c("center", "scale"),
    trControl = fitControl,
  )

#Error: Stopping
#In addition: Warning message:
#  In nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo,  :
#                            There were missing values in resampled performance measures.

anyNA(trainSet_v2)
#[1] FALSE

#it appears some CV folds show zero variance so model cannot run. skipping this, does not work on complete dataset


##--- Compare metrics ---##

ModelFitResults <- resamples(list(knn = KNNbldg0_1,  C50 = C50Fit1, SVM = SVMFit1, hdda = hddaFit2))
# output summary metrics for tuned models
summary(ModelFitResults)

#Models: knn, C50, SVM, hdda 
#Number of resamples: 10 

#Accuracy 
#           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#knn  0.4491315 0.4822157 0.4892147 0.4937293 0.5129816 0.5250000    0
#C50  0.6278481 0.6348101 0.6418680 0.6518586 0.6723972 0.6870229    0
#SVM  0.5063939 0.5283651 0.5423858 0.5395146 0.5503839 0.5675676    0
#hdda 0.4168798 0.4551114 0.4651292 0.4595544 0.4736460 0.4815725    0

#Kappa 
#          Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#knn  0.4467811 0.4800249 0.4870894 0.4915938 0.5109319 0.5229876    0
#C50  0.6263081 0.6332476 0.6403269 0.6503680 0.6709874 0.6856613    0
#SVM  0.5043676 0.5264139 0.5404526 0.5375799 0.5484807 0.5657509    0
#hdda 0.4144477 0.4528152 0.4629226 0.4572817 0.4714154 0.4793913    0


ModelFitResults_all <- resamples(list(knn = KNNfinal,  C50 = C50Fit3))

summary(ModelFitResults_all)

#Call:
#  summary.resamples(object = ModelFitResults_all)

#Models: knn, C50 
#Number of resamples: 10 

#Models: knn, C50 
#Number of resamples: 10 

#Accuracy 
#         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#knn 0.6485246 0.6530780 0.6557064 0.6566786 0.6587108 0.6688439    0
#C50 0.7155963 0.7233708 0.7300552 0.7309119 0.7379193 0.7479839    0

#Kappa 
#         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#knn 0.6480339 0.6525886 0.6552212 0.6561941 0.6582284 0.6683783    0
#C50 0.7151976 0.7229833 0.7296692 0.7305313 0.7375521 0.7476172    0


#--- Save/load top performing model ---#

saveRDS(C50Fit1, "C50Fit.rds")
saveRDS(C50Fit3, "C50Fit_all.rds")
# load and name model
C50Fit1 <- readRDS("C50Fit.rds")


############################
# Predict testSet/validation
############################

# predict with KNN
knnPred1 <- predict(KNNbldg0_1, testSet_bldg0v2)
#performace measurment
postResample(knnPred1, testSet_bldg0v2$LOCATION)
# Accuracy     Kappa 
#0.5083799 0.506364 

C50Pred <- predict(C50Fit1, testSet_bldg0v2)
postResample(C50Pred, testSet_bldg0v2$LOCATION)
#Accuracy     Kappa
#0.6584198 0.6570274 

C50ConfusionMatrix <- confusionMatrix(C50Pred, testSet_bldg0v2$LOCATION)
write.csv(as.table(C50ConfusionMatrix),file = "C50ConfusionMatrix.csv")

SVMPred1 <- predict(SVMFit1, testSet_bldg0v2)
postResample(SVMPred1, testSet_bldg0v2$LOCATION)
#Accuracy     Kappa 
#0.5299282 0.5280138 

hddaPred1 <- predict(hddaFit1, testSet_bldg0v2)
postResample(hddaPred1, testSet_bldg0v2$LOCATION)
#Accuracy     Kappa 
#0.4644852 0.4622885 

testSet_bldg0_final <- mutate(testSet_bldg0v2, knnPred1, C50Pred, SVMPred1)

write.csv(testSet_bldg0_final,file = "testSet_predictions_bldg0.csv")

knnPredall <- predict(KNNfinal, testSet_v2)
postResample(knnPredall, testSet_v2$LOCATION)
# Accuracy     Kappa 
#0.6557963 0.6553183 

C50Predall <- predict(C50Fit3, testSet_v2)
postResample(C50Predall, testSet_v2$LOCATION)
#Accuracy     Kappa 
#0.7567852 0.7564437  


testSet_v2_C50 <- mutate(testSet_v2, C50Predall, knnPredall)


write.csv(testSet_v2_C50,file = "testSet_predictions.csv")

###############################
# Predict new data (Dataset 2)
###############################

# predict with C50 - only model using entire data set
C50Pred_valid <- predict(C50Fit3, newdata = validationData_v2)

C50Pred_valid <- mutate(validationData, C50Pred_valid)
write.csv(C50Pred_valid,file = "validation_predictions.csv")
