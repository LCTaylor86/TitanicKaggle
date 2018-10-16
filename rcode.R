
####Loading libraries used for data cleaning/mining/wraggling
  library(dplyr)
  library(tidyverse)
  library(forcats)
  library(stringr)
  library(caTools)
  library(Hmisc)
  library(tidyr)
  
####Loading libraries used for data assessment/visualizations
  library(DT)
  library(data.table)
  library(pander)
  library(ggplot2)
  library(scales)
  library(grid)
  library(gridExtra)
  library(corrplot)
  library(VIM) 
  library(knitr)
  library(vcd)
  library(caret)

####Loading libraries used for modeling
  library(xgboost)
  library(MLmetrics)
  library('randomForest') 
  library('rpart')
  library('rpart.plot')
  library('car')
  library('e1071')
  library(vcd)
  library(ROCR)
  library(pROC)
  library(VIM)
  library(glmnet) 
  library(caret) 
  library(party)
  library(Matrix)
  library(mice)
  library(spFSR)

##Loading Trainging and Test Sets
  train <- read.csv("train.csv", stringsAsFactors = F, na.strings = c("NA", ""))
  test <- read.csv("test.csv", stringsAsFactors = F, na.strings = c("NA", ""))

##Combinded Train and Test for exploration purposes
  test$Survived <- NA
  all <- rbind(train, test)

# check data
  glimpse(all)  ## general summary

  datacheck <- t(sapply(all, function(x) rbind(class = class(x),
                                           n = length(x),
                                           mean = ifelse(is.numeric(x), round(mean(x, na.rm = TRUE), 2), NA),
                                           median = ifelse(is.numeric(x), median(x, na.rm = TRUE), NA), 
                                           min = ifelse(is.numeric(x), min(x, na.rm = TRUE), NA),
                                           max = ifelse(is.numeric(x), max(x, na.rm = TRUE), NA),
                                           uniqueValues =  length(unique(x)), 
                                           countMissing = sum(is.na(x)),  
                                           percentMissing = round(sum(is.na(x))/length(x)*100, 2))))

  colnames(datacheck) <- c('class', 'n', 'mean', 'median', 'min', 'max', 'uniqueValues', 'countMissing', 'percentMissing')

## recode categorical varaibles to factors/ordinal
  all$Sex <- as.factor(all$Sex)
  all$Survived <- as.factor(all$Survived)
  all$Pclass <- as.ordered(all$Pclass) #because Pclass is ordinal

## Creating New Variables that may help
### Title
Title <- all%>%
  mutate(Title = gsub("^.*, (.*?)\\..*$", "\\1", Name), 
         Title = as.factor(case_when(Title %in% c("Mlle", "Ms", "Lady", "Dona") ~ "Miss",
                           Title == "Mme"  ~ "Mrs", 
                           !(Title %in% c('Master', 'Miss', 'Mr', 'Mrs')) ~ "Rare Title", 
                           TRUE ~ as.character(Title)))) 

all <-merge(all, Title)

rm(Title)

## Family Size
FamilySize <- all %>%
    mutate(Fsize = SibSp + Parch + 1,
           FamilySized = as.factor(case_when(Fsize == 1 ~ 'Single',
                                           Fsize < 5 & Fsize >= 2 ~ 'Small',
                                           Fsize >= 5 ~ 'Big')))

all <-merge(all, FamilySize)
rm(FamilySize)

## Ticket Size
tickets <- all %>%
  group_by(Ticket) %>%
  summarise(Tsize = n()) %>%
  mutate(GroupSize = as.factor(case_when(Tsize == 1 ~ 'Single', 
                                         Tsize < 5 & Tsize >= 2 ~ 'Small', 
                                         Tsize >= 5 ~ 'Big')))

all <-merge(all, tickets)
rm(tickets)

## Cabin Imputation for missing (replacing NAs with imaginary Deck U, and keeping only the first letter of ech Cabin (Deck))
all$CabinImp <- substring(all$Cabin, 1, 1)
all$CabinNum <- case_when(all$CabinImp == 'A' ~ 1, 
                          all$CabinImp == 'B' ~ 2, 
                          all$CabinImp == 'C' ~ 3, 
                          all$CabinImp == 'D' ~ 4, 
                          all$CabinImp == 'E' ~ 5, 
                          all$CabinImp == 'F' ~ 6, 
                          all$CabinImp == 'G' ~ 7,
                          all$CabinImp == 'T' ~ 8)

mice_out <- mice(select(all, CabinNum, Pclass, Title, FamilySized, GroupSize, Age,Fare,Sex,Embarked),method = 'cart')
mice_output <- complete(mice_out)
all$CabinImp <- as.factor(mice_output$Cabin)

## Replacing missing fare with mean fare. Replacing missing Embarked with most common code (becasue there is so little missing here, I choose to 
## to replace with mean and most commen)
all <- all %>%
  mutate(Fare = ifelse(is.na(Fare), median(all$Fare, na.rm=TRUE), Fare),
         Embarked = as.factor(ifelse(is.na(Embarked), 'S', Embarked)))

#predicting Age with Linear Regression
set.seed(12000)
AgeLM <- lm(Age ~ Pclass + Sex + Title + FamilySized + GroupSize, data=all[!is.na(all$Age),])
summary(AgeLM)
all$AgeLM <- predict(AgeLM, all)
all$Age[is.na(all$Age)] <- all$AgeLM[is.na(all$Age)]
rm(AgeLM)

groupedAge <- as.factor(case_when(all$Age <=15 ~ "Child", 
                                  all$Age >15 & all$Age <= 50 ~ "Adult",
                                  all$Age > 50 ~ "Senior"))

all$groupedAge <- groupedAge
rm(groupedAge)

## Counting TicketSurvivors
TicketSurvivors <- all %>%
  group_by(Ticket) %>%
  summarise(Tsize = length(Survived),
            NumNA = sum(is.na(Survived)),
            SumSurvived = sum(as.numeric(Survived)-1, na.rm=T)) %>%
  mutate(AnySurvivors = as.factor(case_when(Tsize == 1 ~ 'other', 
                                  Tsize >= 2 & SumSurvived >=1 ~ 'survivors in group', 
                                  TRUE ~ 'other')))

all <-merge(all, TicketSurvivors[,c("Ticket", "AnySurvivors")])
rm(TicketSurvivors)

#### Plotting a few different things ###

p0 <- ggplot(all[!is.na(all$Survived),], aes(x = Survived, fill = Survived)) +
  geom_bar(stat='count', position='dodge') +
  labs(x = "Survived") +
  geom_text(stat='count', aes(label=..count..), vjust = 1.5, position = position_dodge(.9), size = 3) +
  theme(legend.position="none")

p2 <- ggplot(all[!is.na(all$Survived),], aes(x = Sex, fill = Survived)) +
  geom_bar(stat='count', position='dodge') +
  labs(x = 'Gender') +
  geom_text(stat='count', aes(label=..count..), vjust = 1.5, position = position_dodge(.9), size = 3) +
  theme(legend.position="none")

p3 <- ggplot(all, aes(x=CabinNum, fill=Survived)) +
  geom_bar(stat='count', position='dodge') + 
  labs(x="Cabin") +
  geom_text(stat='count', aes(label=..count..), vjust = 1.5, position = position_dodge(.9), size = 3) +
  theme(legend.position="none")

p4 <- ggplot(all[!is.na(all$Survived),], aes(x = Pclass, fill = Survived)) +
  geom_bar(stat='count', position='dodge') +
  labs(x = 'Pclass') +
  geom_text(stat='count', aes(label=..count..), vjust = 1.5, position = position_dodge(.9), size = 3) +
  theme(legend.position="none")

p5 <- ggplot(all[!is.na(all$Survived),], aes(x = FamilySized, fill = Survived)) +
  geom_bar(stat='count', position='dodge') +
  labs(x = 'FamilySize') +
  geom_text(stat='count', aes(label=..count..), vjust = 1.5, position = position_dodge(.9), size = 3) +
  theme(legend.position="none")

p6 <- ggplot(all[!is.na(all$Survived),], aes(x = GroupSize, fill = Survived)) +
  geom_bar(stat='count', position='dodge') +
  labs(x = 'GroupSize') +
  geom_text(stat='count', aes(label=..count..), vjust = 1.5, position = position_dodge(.9), size = 3) +
  theme(legend.position="none")

p7 <- ggplot(all[!is.na(all$Survived),], aes(x = Title, fill = Survived)) +
  geom_bar(stat='count', position='dodge') +
  labs(x = 'Title') +
  geom_text(stat='count', aes(label=..count..), vjust = 1.5, position = position_dodge(.9), size = 3) +
  theme(legend.position="none")

p8 <- ggplot(all[!is.na(all$Survived),], aes(x = Embarked, fill = Survived)) +
  geom_bar(stat='count', position='dodge') +
  labs(x = 'Embarked') +
  geom_text(stat='count', aes(label=..count..), vjust = 1.5, position = position_dodge(.9), size = 3) +
  theme(legend.position="none")

grid.arrange(p0, p2, p3, p4, p5, p6, p7, p8, ncol=2)

ageplot<- ggplot(all[!is.na(all$Age),], aes(x = Age))+
  geom_histogram(aes(y = ..density..), fill = "red") +
  stat_function(fun = dnorm, args = with(all[!is.na(all$Age),], c(mean = mean(Age), sd = sd(Age))))+
  scale_x_continuous("Age") +
  theme_grey(base_size = 12) +
  theme(legend.position="none")

AgeLMplot <- ggplot(all[!is.na(all$AgeLM),], aes(x = AgeLM))+
  geom_histogram(aes(y = ..density..), fill = "red") +
  stat_function(fun = dnorm, args = with(all[!is.na(all$AgeLM),], c(mean = mean(AgeLM), sd = sd(AgeLM))))+
  scale_x_continuous("AgeLM") +
  theme_grey(base_size = 12) +
  theme(legend.position="none")

grid.arrange(ageplot, AgeLMplot, nrow=1)

corr_data <- all %>%
   filter(!is.na(Survived))%>%
   select(-Ticket, -PassengerId, -Name, -Cabin, -CabinNum, -AgeLM, -Age)%>%
   mutate(Survived = as.numeric(Survived), 
          Pclass = as.numeric(Pclass), 
          Sex = as.numeric(Sex), 
          groupedAge = as.numeric(groupedAge), 
          SibSp, 
          Parch, 
          Fare, 
          Embarked = as.numeric(Embarked), 
          Title = as.numeric(Title), 
          Fsize, 
          FamilySized = as.numeric(FamilySized), 
          Tsize, 
          GroupSize = as.numeric(GroupSize), 
          CabinImp = as.numeric(CabinImp), 
          AnySurvivors = as.numeric(AnySurvivors))
  
mcorr_data <- cor(corr_data)

corrplot(mcorr_data,method="circle")

#cleaning up
#splitting data into train and test set again
trainClean <- all[!is.na(all$Survived),]
testClean <- all[is.na(all$Survived),]

trainClean$Survived<-as.character(trainClean$Survived)
testClean$Survived<-as.character(testClean$Survived)
testClean$Survived <- 1

###determining variable imporatnce
knnWrapper    <- makeLearner("classif.xgboost") 
corr_data$Survived<-as.factor(corr_data$Survived)
classifTask   <- makeClassifTask(data = corr_data, target = 'Survived')
perf.measure  <- acc

set.seed(123)
spsaMod <- spFeatureSelection(
  task = classifTask,
  wrapper = knnWrapper,
  measure = perf.measure ,
  num.features.selected = 0,
  iters.max = 25,
  num.cores = 1)

getImportance(spsaMod)
bestMod <- getBestModel(spsaMod)


###Variable Selection
VariableSelect <- c("Survived", "Pclass", "Title", 'groupedAge', "Sex",  "Fare", "FamilySized", "AnySurvivors")

trainSelect <- trainClean[,VariableSelect]
testSelect <- testClean[,VariableSelect]

## Spliting data cross validation purposes
ind=createDataPartition(trainClean$Survived,times=1,p=0.8,list=FALSE)
train_val=trainClean[ind,VariableSelect]
test_val=trainClean[-ind,VariableSelect]

##################
### Approach 1 ###
##################

#### XGBoosted Model
train1.spare <- sparse.model.matrix(Survived~., data = trainSelect)
test1.spare <- sparse.model.matrix(Survived~., data = testSelect)

##Determining the best Parameters for xgboost
best_param = list()
best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0

for (iter in 1:100) {
  param <- list(objective = "binary:logistic",
                eval_metric = "logloss",
                max_depth = sample(1:20, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.3), 
                lambda= runif(1, 0.0, 0.3), 
                alpha= runif(1, 0.0, 0.3),
                subsample = runif(1, .1, .9),
                colsample_bytree = runif(1, .1, .9), 
                min_child_weight = sample(0:50, 1),
                max_delta_step = sample(0:50, 1)
  )
  cv.nround = 1000
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=train1.spare, label= trainSelect$Survived, params = param, nthread=6, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = T, early_stop_round=10, maximize=FALSE)
  
  min_logloss = min(mdcv$evaluation_log[,test_logloss_mean])
  min_logloss_index = which.min(mdcv$evaluation_log[,test_logloss_mean])
  
  if (min_logloss < best_logloss) {
    best_logloss = min_logloss
    best_logloss_index = min_logloss_index
    best_seednumber = seed.number
    best_param = param
  }
}

nround = best_logloss_index

set.seed(best_seednumber)
md <- xgboost(data=train1.spare, label= trainSelect$Survived, params=best_param, nrounds=nround, nthread=6)

xgb.importance(model = md)


#### Determine Accurary
xg_prediction_accuracy <- predict(md, train1.spare)
xg_prediction_accuracy <- ifelse(xg_prediction_accuracy >= 0.51,1,0)
confusionMatrix(factor(xg_prediction_accuracy),factor(trainSelect$Survived))

xg_prediction <- predict(md,test1.spare)
xg_prediction <- ifelse(xg_prediction >= 0.51,1,0)
xg_Survived <- as.numeric(as.character(xg_prediction))

#### Support Vector Machines

set.seed(best_seednumber)

#### First attempt
svm_model<-svm(factor(Survived)~.,
               data = trainSelect)
svm_model
#### Determine Accurary
svm_prediction_accuracy <- predict(svm_model,
                                   data = trainSelect)
confusionMatrix(svm_prediction_accuracy,as.factor(trainSelect$Survived))
#### Tune parameters
svm_tune <- tune(svm, factor(Survived)~.,
                 data = trainSelect, 
                 kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
print(svm_tune)
#### Rerun with new Pararmeters
svm_model_after_tune <- svm(factor(Survived)~.,
                            data = trainSelect,
                            type='C-classification',
                            kernel="radial",
                            cost=1, 
                            gamma=0.5)
svm_model_after_tune
# Check Accuracy again
svm_prediction_accuracy2 <- predict(svm_model_after_tune,
                                   data = trainSelect)
confusionMatrix(svm_prediction_accuracy2,as.factor(trainSelect$Survived))

#### Predict Test
svm_prediction <- predict(svm_model_after_tune,
                          newdata = testSelect)
svm_Survived <- as.numeric(as.character(svm_prediction))

#### Random Forest

set.seed(1234)
crf_model <- cforest(factor(Survived)~.,
                     data = trainSelect,
                     controls=cforest_unbiased(ntree=1500, mtry=3))
crf_model
#### Determine Accurary
crf_prediction_accuracy <- predict(crf_model,
                                   newdata = trainSelect)
confusionMatrix(crf_prediction_accuracy,as.factor(trainSelect$Survived))

### Run test prediction
crf_prediction <- predict(crf_model, 
                          newdata = testSelect, 
                          OOB=FALSE, 
                          type = "response")
crf_Survived <- as.numeric(as.character(crf_prediction))

#### Combined the Three outputs and take the most commen result
Survived <- xg_Survived + svm_Survived + crf_Survived
Survived[Survived < 2] <- 0
Survived[Survived >= 2] <- 1
Survived <- data.frame(PassengerId = testClean$PassengerId,Survived = Survived)

### Write submission file
write.csv(Survived,file = 'Survived.csv',row.names = FALSE)

##################
### Approach 2 ###
##################

### Random Forest ###

set.seed(1234)
caret_matrix <- train(x=trainSelect[,-1],
                      y=trainSelect[,1], 
                      data=trainSelect, 
                      method='rf', 
                      trControl=trainControl(method="cv"))

rf_caret_prediction_accuracy <- predict(caret_matrix,
                                         data = trainSelect)
confusionMatrix(rf_caret_prediction_accuracy,as.factor(trainSelect$Survived))

#using the model to make Survival predictions on the test set
solution_rf <- predict(caret_matrix, 
                       testSelect)
RF_survived <- as.numeric(solution_rf)-1

#### Support Vector Machines #####

set.seed(1234)
caret_svm <- train(Survived~ ., 
                   data=trainSelect, 
                   method='svmRadial', 
                   preProcess= c('center', 'scale'), 
                   trControl=trainControl(method="cv", number=5))

#### Determine Accurary
svm_caret_prediction_accuracy <- predict(caret_svm,
                                   data = trainSelect)
confusionMatrix(svm_caret_prediction_accuracy,as.factor(trainSelect$Survived))

#using the model to make Survival predictions on the test set
solution_svm <- predict(caret_svm, 
                        testSelect)
SVM_Survived2<- as.numeric(solution_svm)-1

#### Boosted Model #####

set.seed(1234)
caret_boost <- train(Survived~ ., 
                     data=trainSelect, 
                     method='gbm', 
                     preProcess= c('center', 'scale'), 
                     trControl=trainControl(method="cv", number=7), 
                     verbose=FALSE)

boosted_prediction_accuracy <- predict(caret_boost,
                                   newdata = trainSelect)
confusionMatrix(boosted_prediction_accuracy,as.factor(trainSelect$Survived))

#using the model to make Survival predictions on the test set
solution_boost <- predict(caret_boost, 
                          testSelect)
Boost_survived <- as.numeric(solution_boost)-1

#adding the three model predictions to test dataframe

Survived2 <- RF_survived + SVM_Survived2 + Boost_survived
Survived2[Survived2 < 2] <- 0
Survived2[Survived2 >= 2] <- 1
Survived2 <- data.frame(PassengerId = testClean$PassengerId,Survived = Survived2)

#writing final submission file
write.csv(Survived2, file = 'Titanic_select.csv', row.names = F)

#compose correlations plot
corrplot.mixed(cor(do.call("rbind", list(RF_survived, SVM_Survived2, Boost_survived))), order="hclust", tl.col="black")


