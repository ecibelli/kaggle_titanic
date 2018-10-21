## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

## ----workspace, results='hide', message=FALSE, warning=FALSE-------------

library(randomForest)
library(ggplot2)
library(lattice)
library(RColorBrewer)
library(gridExtra)
library(dplyr)
library(mice)
library(scales)
library(caret)
library(ROSE)

# Working directory with data
wd = "C:/Users/esc642/Desktop/workbox/kaggle/titanic"
setwd(wd)

# Read in data
train = read.csv(sprintf("%s/train.csv", wd),
                 head = T, na.strings=c("", "NA"))
test = read.csv(sprintf("%s/test.csv", wd),
                head = T, na.strings=c("", "NA"))

# Combine for feature engineering purposes
fullData = bind_rows(train, test)

# Convert survival to factor
fullData$surviveFactor=as.factor(as.character(fullData$Survived))


## ----dataInspect---------------------------------------------------------

# How many built in features and how many passengers?
ncol(fullData)
nrow(fullData)

summary(fullData)


## ----survivalBySex, fig.width = 4, fig.height = 4, fig.align = "center"----

prop.table(xtabs(~Survived + Sex, data = fullData), 2)
mosaicplot(~Survived + Sex, data = fullData, col = c("red", "blue"),
           main = "Survival by gender")


## ----survivalByClass, fig.width = 4, fig.height = 4, fig.align = "center"----

prop.table(xtabs(~Survived + Pclass, data = fullData), 2)
mosaicplot(~Survived + Pclass, data = fullData, 
           col = brewer.pal(3, "YlGnBu"),
           main = "Survival by class")


## ----survivalByAge, fig.align = "center"---------------------------------

ggplot(data=subset(fullData, !is.na(Survived)), aes(x = Age)) +
  geom_density(aes(fill = as.factor(Survived),
                   group = as.factor(Survived)), alpha = 0.3) +
                 theme_bw() +
  ggtitle("Survival by age")

## ----survivalByFare, fig.align = "center"--------------------------------

ggplot(data=subset(fullData, !is.na(Survived)), aes(x = log(Fare))) +
  geom_density(aes(fill = as.factor(Survived),
                   group = as.factor(Survived)), alpha = 0.3) +
                 theme_bw() +
  ggtitle("Survival by ticket fare")

## ----survivalByTravelComp, fig.width = 11, fig.height = 4, fig.align = "center"----

# Siblings and spouses
ssPlot = ggplot(data=subset(fullData, !is.na(Survived)), aes(as.factor(SibSp))) +
         geom_bar(aes(group = as.factor(Survived),
                      fill = as.factor(Survived),
                      color = as.factor(Survived)),
                  position = "dodge", alpha = 0.5) +
  theme_bw() +
  ggtitle("Survival by siblings/spouses")

# Parents and children
pcPlot = ggplot(data=subset(fullData, !is.na(Survived)), aes(as.factor(Parch))) +
         geom_bar(aes(group = as.factor(Survived),
                      fill = as.factor(Survived),
                      color = as.factor(Survived)),
                  position = "dodge", alpha = 0.5) +
  theme_bw() +
  ggtitle("Survival by parents/children")

grid.arrange(ssPlot, pcPlot, ncol = 2)


## ----survivalByEmbarked, fig.align = "center"----------------------------
# Siblings and spouses
ggplot(data=subset(fullData, !is.na(Survived)), aes(as.factor(Embarked))) +
         geom_bar(aes(group = as.factor(Survived),
                      fill = as.factor(Survived),
                      color = as.factor(Survived)),
                  position = "dodge", alpha = 0.5) +
  theme_bw() +
  ggtitle("Survival by embarkment location")

xtabs(~Embarked + Survived, data = fullData)


## ----titleEngineer, fig.align="center"-----------------------------------

# First, cut name between comma and period
fullData$title =  gsub(".*,(.*)\\..*", "\\1", as.character(fullData$Name))
fullData$title = gsub(" ", "", fullData$title)
fullData$title = gsub("\\.", "", fullData$title)

unique(fullData$title)

# Fix two
fullData[fullData$Name == "Rothschild, Mrs. Martin (Elizabeth L. Barrett)",]$title = "Mrs"
fullData[fullData$title == "theCountess",]$title = "Countess"

# Collapse French and English titles
fullData[fullData$title == "Mlle",]$title = "Miss"
fullData[fullData$title == "Mme",]$title = "Mrs"

# Condense unusual titles (N < 10)
fullData$title2 = fullData$title
rareTitles = names(which(table(fullData$title) < 10))
fullData[fullData$title2 %in% rareTitles,]$title2 = "other"

unique(rareTitles)

# Our five titles
table(fullData$title2)

# Do they match up with age as expected?
boxplot(Age ~ title2, data = fullData)


## ----famSizeEng, fig.align = "center"------------------------------------

fullData$familySize = fullData$Parch + fullData$SibSp

ggplot(data = subset(fullData, !is.na(Survived)), aes(as.factor(familySize))) +
         geom_bar(aes(group = as.factor(Survived),
                      fill = as.factor(Survived),
                      color = as.factor(Survived)),
                  position = "dodge", alpha = 0.5) +
  theme_bw() +
  ggtitle("Survival by family size")


## ----cabinRoomCount, fig.width = 11, fig.height = 4, fig.align = "center"----

# Inspect to see what we have
unique(fullData$Cabin)

# Most cases of spaces seem to be multiple rooms, but there are examples on F deck that seem to be followed by a number with E or G. Googling a few of these passangers suggests that those are in fact on F-deck, so we'll first adjust to ensure that that is represented. 

fullData$origCabin = fullData$Cabin
fullData$Cabin = gsub("F E", "F", fullData$Cabin)
fullData$Cabin = gsub("F G", "F", fullData$Cabin)

# Now we can count the number of rooms
fullData$roomCount = sapply(strsplit(fullData$Cabin, " "), length)

# Pull name of deck
fullData$Deck = gsub("\\d", "", fullData$Cabin)
# Restrict to first character to cover cases of multiple rooms
fullData$Deck = substr(fullData$Deck, 1, 1)


# Look at survival by deck and number of rooms
numRoomPlot = ggplot(data=subset(fullData, !is.na(Survived)),
                     aes(as.factor(roomCount))) +
         geom_bar(aes(group = as.factor(Survived),
                      fill = as.factor(Survived),
                      color = as.factor(Survived)),
                  position = "dodge", alpha = 0.5) +
  theme_bw() +
  ggtitle("Survival by number of rooms")

deckPlot = ggplot(data=subset(fullData, !is.na(Survived)),
                  aes(as.factor(Deck))) +
         geom_bar(aes(group = as.factor(Survived),
                      fill = as.factor(Survived),
                      color = as.factor(Survived)),
                  position = "dodge", alpha = 0.5) +
  theme_bw() +
  ggtitle("Survival by cabin deck")

grid.arrange(numRoomPlot, deckPlot, ncol = 2)


## ----fareNA--------------------------------------------------------------

fullData[which(is.na(fullData$Fare)),]

# We don't have cabin or deck information for tihs passenger, but we do know he was in third class and departed from Southhampton. Let's look at the faare values of passengers in that class/embarkment location.

boxplot(fullData[fullData$Embarked == "S" & fullData$Pclass == 3,]$Fare)
mean(fullData[fullData$Embarked == "S" & fullData$Pclass == 3,]$Fare,
     na.rm=T)
median(fullData[fullData$Embarked == "S" & fullData$Pclass == 3,]$Fare,
       na.rm=T)

# There are some outliers, so the median might be a safer choice here.
fullData[which(is.na(fullData$Fare)),]$Fare = median(
  fullData[fullData$Embarked == "S" & fullData$Pclass == 3,]$Fare,
       na.rm=T)


## ----embarkNA------------------------------------------------------------

# Find the missing rows
which(is.na(fullData$Embarked))

# check info on these people
fullData[c(62, 830),]

# Interestingly, both shared a cabin, have the same ticket, and paid the same price. So they likely traveled together, and we can likely assume the same embarked value for both.

# Did they travel with relatives? Check for shared surnames
grep("Icard", fullData$Name)
grep("Stone", fullData$Name) # False alarm: a different Stone

# Makes sense; family size for both is 0.

# Let's see if we can find simliar fares in the data set for first class passengers.
xtabs(~Embarked + Pclass,
              data = fullData[fullData$Fare > 70 & fullData$Fare < 90,])


# Could be either Cherbourg or Southhampton. Slightly more consistency in this range in Cherbourg, so let's set that as the value for these two passengers.

fullData[c(62, 830),]$Embarked = "C"


## ----ageNA, fig.align = 'center'-----------------------------------------

# First ID those values that are and are not missing, so we can sanity-check after imputation.
hasAge = which(!is.na(fullData$Age))
missAge = which(is.na(fullData$Age))

# Save data as a backup
fullSaver = fullData

# Convert relevant factors to factor to play nice with imputation
factor_vars <- c("Pclass", "Sex", "title")
fullData[factor_vars] <- lapply(fullData[factor_vars], function(x) as.factor(x))

# Use multiple imputation to generate predicted values for age
# Only using features that are not too granular (e.g. name, ticket)
# 5 imputations, default method (pmm) should be ok for a dataset this size
ageMI = mice(fullData[,c("Age", "Pclass", "Sex", "SibSp", "Parch", "title")],
             m = 5)

# Get all imputations from the first "round" for now
miOutput <- complete(ageMI,1 )

# Add these values back to the dataset
fullData$Age = miOutput$Age

# Double-check that the non-NA data hasn't changed
which(fullData[hasAge,]$Age != fullSaver[hasAge,]$Age)


# # Check to see if predicted Age values fall in a reasonable range
plotData = fullData # new dataset to hold some useful plotting features
plotData$ageImputed = "actual"
plotData[missAge,]$ageImputed = "imputed"

ggplot(plotData, aes(x = "", y = Age)) + 
  geom_boxplot(alpha = 0.3) +
  geom_point(aes(y = Age, colour = factor(ageImputed), 
                 shape = factor(ageImputed)
                 ), alpha = 0.4,
             position = "jitter") +
  theme_bw() 


## ----dropCols------------------------------------------------------------
# Save data again
fullSaver2 = fullData

# Drop columns
fullData$Name = NULL
fullData$Ticket = NULL
fullData$Cabin = NULL
fullData$title = NULL
fullData$origCabin = NULL
fullData$surviveFactor = NULL
fullData$Deck = NULL

## ----scaleDF-------------------------------------------------------------

# Hold aside binary and categorical variables
dataBin = fullData[, c('PassengerId', 'Pclass', 'Survived', 'Sex', 'Embarked', 
                      'title2')]

binCols = which(colnames(fullData) %in% colnames(dataBin))

# Separate out just continuous variables
dataCont = fullData[,-binCols]

# Scale these
dataCont2 = data.frame(lapply(dataCont, function(x) rescale(x)))

# Merge data back together
fullData = cbind(dataBin, dataCont2)


## ----profHot-------------------------------------------------------------

# Sex is binary - just convert to numeric
fullData$Sex = ifelse(fullData$Sex == "female", 1, 0)

# Convert title and Embarked to factor, then one-hot
fullData$title2 = as.factor(as.character(fullData$title2))
fullData$Embarked = as.factor(as.character(fullData$Embarked))

fullSaver3 = fullData
fullDummy = dummyVars(~., data = fullData)
fullData <- as.data.frame(predict(fullDummy, newdata = fullData))

## ----split---------------------------------------------------------------

# Split by passenger IDs from the original files
trainSet = fullData[fullData$PassengerId %in% train$PassengerId,]
testSet = fullData[fullData$PassengerId %in% test$PassengerId,]


# Drop Survived from test set (all NAs)
testSet$Survived = NULL

# Further split train data into a train-train and a test-train set.
trainTrainN = round(nrow(trainSet)*.75, 0) # how much is 75% of the data?
trainTrainSet = trainSet[sample(nrow(trainSet), trainTrainN), ] 
trainTestSet = trainSet[!(trainSet$PassengerId %in% trainTrainSet$PassengerId),]

# Remove PassengerID from training sets
trainTrainSaver = trainTrainSet
trainTestSaver = trainTestSet
trainTrainSet$PassengerId = NULL
trainTestSet$PassengerId = NULL


## ----baseModel-----------------------------------------------------------

# Fit basic model on trainTrainset
model1 <- randomForest(factor(Survived) ~ .,  
              data = trainTrainSet)


# Check accuracy on our training set
predTrainDF1 = trainTrainSet
predTrainDF1$pred = model1$predicted
xtabs(~pred + Survived, data = predTrainDF1)


## ----baseModelPred-------------------------------------------------------

model1Pred = predict(model1, newdata = trainTestSet)

# Append the predictions to the test data and look at performance
predTestDF = trainTestSet
predTestDF$predModel1 = model1Pred
predTestDF$accModel1 = ifelse(predTestDF$pred == predTestDF$Survived, 1, 0)

xtabs(~predModel1 + Survived, data = predTestDF)

## ----model2Setup---------------------------------------------------------

# Oversample - bring up to 820 so survived N = died N
overTrainDF = ovun.sample(Survived ~ ., data = trainTrainSet, 
                            method = "over", N= 820)$data
table(overTrainDF$Survived)

# Undersample - bring down to 516 to survived N = died N
underTrainDF = ovun.sample(Survived ~ ., data = trainTrainSet,
                            method = "under", N = 516)$data
table(underTrainDF$Survived)


# ROSE algorithm: predictively generate underrepresented class
roseTrainDF = ROSE(Survived ~ ., data = trainTrainSet, p = 0.5)$data
table(roseTrainDF$Survived)

## ----model2Fit-----------------------------------------------------------

modelOver <- randomForest(factor(Survived) ~ .,  
              data = overTrainDF)

modelUnder <- randomForest(factor(Survived) ~ .,  
              data = underTrainDF)

modelRose <- randomForest(factor(Survived) ~ .,  
              data = roseTrainDF)


# Predict data for each
modelOverPred = predict(modelOver, newdata = trainTestSet)
modelUnderPred = predict(modelUnder, newdata = trainTestSet)
modelRosePred = predict(modelRose, newdata = trainTestSet)

# Append these results to our original table for comparing labels and predictions

predTestDF$predModel2Over = modelOverPred
predTestDF$predModel2Under = modelUnderPred
predTestDF$predModel2Rose = modelRosePred

# Save all confusion matrices
confOrig = xtabs(~predModel1 + Survived, data = predTestDF)
confOver = xtabs(~predModel2Over + Survived, data = predTestDF)
confUnder = xtabs(~predModel2Under + Survived, data = predTestDF)
confRose = xtabs(~predModel2Rose + Survived, data = predTestDF)

confOrig
confOver
confUnder
confRose

## ----accRecallPrec-------------------------------------------------------

accDF = data.frame(c("orig", "over", "under", "rose"))
colnames(accDF) = "model"
accDF$acc = ""
accDF$recall = ""
accDF$prec = ""

for (i in 1:nrow(accDF)){
  if(accDF$model[i] == "orig") {
    targetConf = confOrig 
     } else if (accDF$model[i] == "over") {
       targetConf = confOver 
       } else if (accDF$model[i] == "under") {
         targetConf = confUnder 
       } else {
           targetConf = confRose}
accDF$acc[i] = round((targetConf[1,1] + targetConf[2,2])/sum(targetConf), 3)
accDF$prec[i] = round(targetConf[1,1]/(targetConf[1,1] + targetConf[1,2]), 3)
accDF$recall[i] = round(targetConf[1,1]/(targetConf[1,1] + targetConf[2,1]), 3)
}
        
accDF

## ----rfTune--------------------------------------------------------------

tuneFeatures = trainTrainSet
tuneFeatures$Survived = NULL

# search for best mtry (number of variables available for splitting at each node)
# allow 500 trees, demand improvement of at least 1e-5 for search to continue
# step mtry by stepFactor
bestmtry <- tuneRF(y = factor(trainTrainSet$Survived), 
                   x = tuneFeatures, 
                   nTreeTry = 500,
                   stepFactor=1.5, improve=1e-5)

print(bestmtry)
plot(bestmtry)

# Extract best value
bestmtryDF = data.frame(bestmtry)
bestM = bestmtryDF[which(bestmtryDF$OOBError == min(bestmtryDF$OOBError)),]$mtry

# Refit model with best mtry
modelTune <- randomForest(factor(Survived) ~ .,  
              data = underTrainDF, mtry = bestM)

# Predict
modelTunePred = predict(modelTune, newdata = trainTestSet)
predTestDF$predModel3Tune = modelTunePred
confTune = xtabs(~predModel3Tune + Survived, data = predTestDF)

# Add to table
tuneAcc = round((confTune[1,1] + confTune[2,2])/sum(confTune), 3)
tunePrec = round(confTune[1,1]/(confTune[1,1] + confTune[1,2]), 3)
tuneRecall = round(confTune[1,1]/(confTune[1,1] + confTune[2,1]), 3)

accDF$model = as.character(accDF$model)
accDF[5,] = c("tune", tuneAcc, tunePrec, tuneRecall)

accDF


## ----var1Import, fig.align = "center"------------------------------------

# Get importance
importance    <- importance(modelTune)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

importSort = varImportance[order(-varImportance$Importance),]
importSort

## ----trueTest------------------------------------------------------------

# Drop PassengerID from testSet
testSet$PassengerId = NULL

# Predict
modelTunePred = predict(modelTune, newdata = testSet)
finalPreds = data.frame(names(modelTunePred), modelTunePred)
colnames(finalPreds) = c("PassengerId", "Survived")

# Save to csv
write.csv(finalPreds, sprintf("%s/finalPreds_20oct2018.csv", wd), row.names = FALSE)


