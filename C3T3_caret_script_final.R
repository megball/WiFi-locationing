# Title: C3T3_caret_script

# Last update: 5.13.19

# File: C3T3_caret_script.R
# Project name: Multiple models for WiFi locationing dataset


###############
# Project Notes
###############

# Summarize project: Using WAPs to locate indoor positioning

# Summarize top model and/or filtered dataset
# The top model was model_name used with ds_name.



###############
# Housekeeping
###############

# Clear objects if necessary
rm(list = ls())

# get working directory
getwd()
? getwd  # get help
# set working directory
setwd()
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
cl <- makeCluster(2)  # select number of cores
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
    "C:/Users/Megan/Documents/Data Analytics/C3T3/UJIndoorLoc/trainingData.csv",
    stringsAsFactors = FALSE,
    header = T
  )
class(trainingData)  # "data.frame"


## Load Predict/New data (Dataset 2) ---#

validationData <-
  read.csv(
    "C:/Users/Megan/Documents/Data Analytics/C3T3/UJIndoorLoc/validationData.csv",
    stringsAsFactors = FALSE,
    header = T
  )
class(validationData)  # "data.frame"


#--- Load preprocessed datasets that have been saved ---#
#read back in files
trainingData_v1 <- read.csv("trainingData_v1.csv", header = T)
trainingData_v1$X <- NULL
validationData_v1 <- read.csv("validationData_v1.csv", header = T)
validationData_v1$X <- NULL
trainingDatanozv <- read.csv("trainingDatanozv.csv", header = T)
trainingDatanozv$X <- NULL
validationDatanozv <- read.csv("validationDatanozv.csv", header = T)
validationDatanozv$X <- NULL
trainingData_v1n <- read.csv("trainingData_v1n.csv", header = T)
trainingData_v1n$X <- NULL
validationData_v1n <- read.csv("validationData_v1n.csv", header = T)
validationData_v1n$X <- NULL
trainingData_v2 <- read.csv("trainingData_v2.csv", header = T)
trainingData_v2$X <- NULL
validationData_v2 <- read.csv("validationData_v2.csv", header = T)
validationData_v$X <- NULL


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
str(validationData)  # 1111 obs. of  529 variables

validationData = validationData[c(ordered_columns_leftside,
                                  setdiff(names(trainingData), ordered_columns_leftside))]

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
trainingData$BUILDINGID <- as.factor(trainingData$BUILDINGID)
trainingData$RELATIVEPOSITION <-
  as.factor(trainingData$RELATIVEPOSITION)
trainingData$PHONEID <- as.factor(trainingData$PHONEID)
trainingData$SPACEID <- as.integer(trainingData$SPACEID)
trainingData$FLOOR <- as.factor(trainingData$FLOOR)

#create new ID for single location identifier
trainingData <-
  cbind(
    trainingData,
    paste(
      trainingData$SPACEID,
      trainingData$RELATIVEPOSITION,
      trainingData$BUILDINGID,
      trainingData$FLOOR,
      sep = "_"
    )
  )
colnames(trainingData)
head(trainingData, 1)
colnames(trainingData)[530] <- "LOCATION"
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
validationData$BUILDINGID <- as.factor(validationData$BUILDINGID)
validationData$RELATIVEPOSITION <-
  as.factor(validationData$RELATIVEPOSITION)
validationData$PHONEID <- as.factor(validationData$PHONEID)
validationData$SPACEID <- as.integer(validationData$SPACEID)
validationData$FLOOR <- as.factor(validationData$FLOOR)

#create new ID for single location identifier
validationData <-
  cbind(
    validationData,
    paste(
      validationData$SPACEID,
      validationData$RELATIVEPOSITION,
      validationData$BUILDINGID,
      validationData$FLOOR,
      sep = "_"
    )
  )
colnames(validationData)
colnames(validationData)[530] <- "LOCATION"
str(validationData$LOCATION)
#Factor w/ 13 levels "0_0_0_0","0_0_0_1",..: 6 13 13 13 3 11 12 12 11 9 ...

#move to first column
ordered_columns_leftside1 = c('LOCATION')
validationData = validationData[c(ordered_columns_leftside1,
                                  setdiff(names(validationData), ordered_columns_leftside1))]



#################
# Feature removal
#################

#--- Dataset 1 ---#

# remove ID and obvious features
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
#5249 obs of 523 variables
str(trainingData_bldg0$LOCATION)
#Factor w/ 905 levels "1_1_1_0","1_2_1_0",..: 285 29 144 158 103 88 208 198 186 173 ...

trainingData_bldg0v2 <- filter(trainingData_v2, BUILDINGID == "0")
trainingData_bldg0v2$BUILDINGID <- NULL
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
#9492 obs of 523 variables

trainingData_bldg2v2 <- filter(trainingData_v2, BUILDINGID == "2")
trainingData_bldg2v2$BUILDINGID <- NULL
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
#197 variables

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
#204 variables

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
#165 variables

# remove based on Feature Selection (FS)

#create normalized data set for KNN
#normalize <- function(x) {
#  return ((x - min(x)) / (max(x) - min(x)))
#}

trainingData_v1n <-
  as.data.frame(lapply(trainingDatanozv[2:468], normalize))
trainingData_dv <- select(trainingDatanozv, "LOCATION")
#check for normalization
summary(trainingData_v1n$LATITUDE)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
#0.0000  0.2773  0.3927  0.4619  0.6796  1.0000

summary(trainingData_v1n$WAP001)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
#0.0000  1.0000  1.0000  0.9991  1.0000  1.0000

#add dv back to dataset
trainingData_v1n <- cbind(trainingData_v1n, trainingData_dv)
colnames(trainingData_v1n)


#--- Dataset 2 ---#

# remove ID and obvious features
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
validationDatanozv <- validationData_v1[, -(zeroVarData1)]
validationData_v2 <- validationData_v2[, -(zeroVarData1)]

validationData_v1n <-
  as.data.frame(lapply(validationDatanozv[2:468], normalize))
validationData_dv <- select(validationDatanozv, "LOCATION")
#check for normalization
summary(validationData_v1n$LATITUDE)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
#0.0000  0.3526  0.6211  0.5714  0.8129  1.0000

summary(validationData_v1n$WAP001)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
#0.0000  1.0000  1.0000  0.9929  1.0000  1.0000

#add dv back to dataset
validationData_v1n <- cbind(validationData_v1n, validationData_dv)
colnames(validationData_v1n)

###############
# Save datasets
###############

# after ALL preprocessing, save a new version of the dataset

write.csv(trainingData_v1, file = "trainingData_v1.csv")
write.csv(validationData_v1, file = "validationData_v1.csv")

write.csv(trainingDatanozv, file = "trainingDatanozv.csv")
write.csv(validationDatanozv, file = "validationDatanozv.csv")

write.csv(trainingData_v1n, file = "trainingData_v1n.csv")
write.csv(validationData_v1n, file = "validationData_v1n.csv")

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

#create train/test sets with no zero var data set
# set random seed
set.seed(123)
# create the training partition that is 75% of total obs
inTraining_nozv <-
  createDataPartition(trainingDatanozv$LOCATION, p = 0.75, list = FALSE)
# create training/testing dataset
trainSetnozv <- trainingDatanozv[inTraining_nozv, ]
testSetnozv <- trainingDatanozv[-inTraining_nozv, ]
# verify number of obs
nrow(trainSetnozv) # 15184
nrow(testSetnozv)  # 4753

#check dv
str(trainSetnozv$LOCATION)
#Factor w/ 905 levels
str(testSetnozv$LOCATION)
# Factor w/ 905 level

#create train/test sets with no zero var & normalized set
# set random seed
set.seed(123)
# create the training partition that is 75% of total obs
inTraining_v1n <-
  createDataPartition(trainingData_v1n$LOCATION, p = 0.75, list = FALSE)
# create training/testing dataset
trainSet_v1n <- trainingData_v1n[inTraining_v1n, ]
testSet_v1n <- trainingData_v1n[-inTraining_v1n, ]
# verify number of obs
nrow(trainSet_v1n) # 15184
nrow(testSet_v1n)  # 4753

#check dv
str(trainSet_v1n$LOCATION)
#Factor w/ 905 levels
str(testSet_v1n$LOCATION)
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
#5  0.5608522  0.5602251
#7  0.5281915  0.5275135
#9  0.5104214  0.5097186


#Accuracy   Kappa    
#0.5819965  0.5814015

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

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 1 times)
#Summary of sample sizes: 3603, 3594, 3595, 3602, 3604, 3597, ...
#Resampling results across tuning parameters:

#  k  Accuracy   Kappa
#5  0.4929646  0.4908672
#7  0.4631849  0.4609785
#9  0.4453320  0.4430566

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

#model for all data using settings for best model from bldg0, no pre-process, k=5
KNNfinal <- train(
  LOCATION ~ . ,
  data = trainSet_v2,
  method = 'knn',
  metric = 'Accuracy',
  trControl = fitControl,
  tuneGrid = data.frame(k = c(1))
)

#Accuracy  Kappa    
#0.644313  0.6438091

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

#  model  winnow  trials  Accuracy   Kappa    
#rules   TRUE   20      0.6655919  0.6641748

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were trials = 20, model = rules and winnow = TRUE.

#match settings for best model on bldg 0 set
C50grid <- expand.grid(model = "rules", winnow = TRUE, trials = 20)
  
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

# Accuracy   Kappa    
#0.7209598  0.7205643

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
#Summary of sample sizes: 3592, 3609, 3586, 3593, 3593, 3595, ... 
#Resampling results across tuning parameters:
  
#  cost  Accuracy   Kappa    
#0.25  0.5551738  0.5532898
#0.50  0.5536684  0.5517796
#1.00  0.5541725  0.5522858

SVMFit2 <-
  train(
    LOCATION  ~ .,
    data = trainSet_bldg0v2,
    method = 'svmLinear2',
    trControl = fitControl,
    tuneGrid = data.frame(cost = c(0.01, 0.05, 0.1)
  ))
  
#cost  Accuracy   Kappa    
#0.01  0.5526237  0.5507420
#0.05  0.5556759  0.5538038
#0.10  0.5559549  0.5540848

#The final value used for the model was cost = 0.1.

#same model settings but using all data
SVMFit3 <-
  train(
    LOCATION ~.,
    data = trainSet_v2,
    method = 'svmLinear2',
    trControl = fitControl,
    tuneGrid = data.frame(cost = c(0.1))
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

## ------- nb ------- ##

NBFit1 <-
  train(
    LOCATION  ~ .,
    data = trainSet_bldg0v2,
    method = 'naive_bayes',
    metric = 'Accuracy',
    trControl = fitControl
  )

#  usekernel  Accuracy   Kappa    
#FALSE      0.1045373  0.1014990
#TRUE      0.2469316  0.2428586

#NB not good model for numeric features

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
#0.050      0.4145099  0.4120958
#0.175      0.4826954  0.4805337
#0.300      0.4950491  0.4929210

#Tuning parameter 'model' was held constant at a value of all
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were threshold = 0.3 and model = all

hddaFit2 <-
  train(
    LOCATION ~ .,
    data = trainSet_bldg0v2,
    method = "hdda",
    trControl = fitControl,
    tuneLength = 5
  )

#threshold  Accuracy   Kappa    
#0.0500     0.4183381  0.4159439
#0.1125     0.4510682  0.4488101
#0.1750     0.4865957  0.4844402
#0.2375     0.4907990  0.4886480
#0.3000     0.4981710  0.4960469

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
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#knn  0.4694377 0.4790529 0.4878973 0.4929646 0.5057214 0.5313283    0
#C50  0.6335878 0.6430825 0.6534202 0.6655919 0.6873499 0.7167488    0
#SVM  0.5268542 0.5466319 0.5547213 0.5551738 0.5613945 0.5955335    0
#hdda 0.4632911 0.4823732 0.4980883 0.4981710 0.5073892 0.5427873    0

#Kappa 
#         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#knn  0.4672241 0.4769056 0.4857524 0.4908672 0.5036974 0.5294515    0
#C50  0.6320751 0.6415657 0.6519509 0.6641748 0.6860199 0.7155165    0
#SVM  0.5248152 0.5447001 0.5528611 0.5532898 0.5595273 0.5938429    0
#hdda 0.4610389 0.4801979 0.4959971 0.4960469 0.5052707 0.5408302    0

ModelFitResults_all <- resamples(list(knn = KNNfinal,  C50 = C50Fit3))

summary(ModelFitResults_all)

#Call:
#  summary.resamples(object = ModelFitResults_all)

#Models: knn, C50 
#Number of resamples: 10 

#Accuracy 
#         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#knn 0.6217105 0.6409245 0.6438588 0.6443130 0.6509218 0.6595461    0
#C50 0.6984231 0.7151050 0.7204541 0.7209598 0.7273642 0.7418301    0

#Kappa 
#         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#knn 0.6211778 0.6404201 0.6433559 0.6438091 0.6504241 0.6590635    0
#C50 0.6979999 0.7147029 0.7200575 0.7205643 0.7269763 0.7414670    0

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
#0.5035914 0.5015701 

C50Pred <- predict(C50Fit1, testSet_bldg0v2)
postResample(C50Pred, testSet_bldg0v2$LOCATION)
#Accuracy     Kappa
#0.6807662 0.6794693

C50ConfusionMatrix <- confusionMatrix(C50Pred, testSet_bldg0v2$LOCATION)
write.csv(as.table(C50ConfusionMatrix),file = "C50ConfusionMatrix.csv")

SVMPred1 <- predict(SVMFit1, testSet_bldg0v2)
postResample(SVMPred1, testSet_bldg0v2$LOCATION)
#Accuracy     Kappa 
#0.5586592 0.5568684 

hddaPred1 <- predict(hddaFit1, testSet_bldg0v2)
postResample(hddaPred1, testSet_bldg0v2$LOCATION)
#Accuracy     Kappa 
#0.4804469 0.4783410 

testSet_bldg0_final <- mutate(testSet_bldg0v2, knnPred1, C50Pred, SVMPred1)

write.csv(testSet_bldg0_final,file = "testSet_predictions_bldg0.csv")

knnPredall <- predict(KNNfinal, testSet_v2)
postResample(knnPredall, testSet_v2$LOCATION)
# Accuracy     Kappa 
#0.6492741 0.6487808 

C50Predall <- predict(C50Fit3, testSet_v2)
postResample(C50Predall, testSet_v2$LOCATION)
#Accuracy     Kappa 
#0.7304860 0.7301077 


testSet_v2_C50 <- mutate(testSet_v2, C50Predall, knnPredall)


write.csv(testSet_v2_C50,file = "testSet_predictions.csv")

###############################
# Predict new data (Dataset 2)
###############################

# predict with C50 - only model using entire data set
C50Pred_valid <- predict(C50Fit3, newdata = validationData_v2)

C50Pred_valid <- mutate(validationData, C50Pred_valid)
write.csv(C50Pred_valid,file = "validation_predictions.csv")
