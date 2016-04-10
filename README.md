---
title: "Practical Machine Learning Project course"
author: "Riad Darawish"
date: "Saturday, April 10, 2016"
output: html_document
---

---

### Executive Summary

  *This report summarizes the analysis conducted on Weight Lifting Exercises dataset collected from accelerometers attached to belt, forearm, arm and dumbbell of 6 participants. The goal is to predict the fashion in which the exercise is performed. Three tree-based methods are examined in this procedure; Decision-Tree, Bagging and Random Forest. The tree-based predicting methods are preferred in this practice than the classical linear methods because the relationship between the features and the response is highly non-linear and complex relationship.*

  *The dataset used in this project is provided by* [research group on Human Activity Recognition](http://groupware.les.inf.puc-rio.br/har).
  
  **caret** and **randomForest** *R packages are used to test the three machine learning algorithms. The results revealed that random forest method out-performs decision tree and bagging in prediction accuracy. The test error rate achieved by random forest is 0.0038 compared to 0.24 and 0.0145 for decision tree and bagging, respectively.*
  
------
### Prerequisites R libraries:
The R packages used in this analysis are ggplot2, tree, ggplot2, corrplot, rpart, rpart.plot, caret,randomForest,reshape2, rattle, partykit and quietly.

```{r "chunkname01" , echo=TRUE, results="hide", warning =FALSE, message=FALSE}
    library(ggplot2)
    library(tree)
    library(reshape2)
    require(ggplot2, quietly = T)
    require(corrplot, quietly = T)
    require(rpart, quietly = T)
    require(rpart.plot, quietly = T)
    require(caret, quietly = T)
    require(randomForest, quietly = T)
    require(rattle, quietly = T)
    require(partykit, quietly = T)
```

### Dataset retrieval, Cleansing and partitioning
  The training dataset can be downloaded from [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and testing dataset from [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv). The training dataset comes with 19622 observations and 160 variables.
  Both training and testing datasets are passed through a cleansing process to remove the variables with 90% NA contents as well as meaningless variables that have nothing to do in predicting the outcome variable "classe". Factor variables with numeric contents are also converted to numeric type as these factors have more than 32 levels. This conversion is required because tree and random forest complementation in R has a hard limit of 32-levels for categorical variable.
  
1. Downloading and loading the datasets  
    ```{r "chunkname1" ,echo=TRUE,warning=FALSE, message=FALSE}
    trainingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    testingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    trainingFile <- "./dataSets/pml-training.csv"
    testingFile <- "./dataSets/pml-testing.csv"
    if (!file.exists("./dataSets")){
      dir.create("./dataSets")
    }
    if(!file.exists(trainingFile)){
      download.file(url= trainingURL, destfile =trainingFile, method = "curl" )
    }
     if(!file.exists(testingFile)){
      download.file(url= testingURL, destfile =testingFile, method = "curl" )
    }
        
    pml.training <- read.csv(trainingFile,header = TRUE)
    pml.testing <- read.csv(testingFile,header = TRUE)
    ```


  2. Removing the variables that have 90% of their contents are missing values "NA". After this step, the number of the variables are reduced to  93.
  
    ```{r "chunkname2" ,echo=TRUE,warning=FALSE,  message=FALSE}
    pml.training.n <- pml.training[,colSums(is.na(pml.training)) < (nrow(pml.training) * 0.9)]
    pml.testing.n <- pml.testing[,colnames(pml.training.n)[!is.element(colnames(pml.training.n),"classe")]]
    ```
    
    ```{r "chunkname3" ,echo=TRUE,warning=FALSE,  message=FALSE}
    dim(pml.training.n)
    dim(pml.testing.n)
    ```
    
  3. Removing meaningless variables that have nothing to do in predicting the outcome "classe". This reduced the number of variables to 66.
    ```{r "chunkname4" ,echo=TRUE,warning=FALSE,  message=FALSE}
    
    pml.training.n.set <- subset(pml.training.n, select=-c(X,user_name,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window        ,num_window, kurtosis_yaw_belt,skewness_yaw_belt,max_yaw_belt,min_yaw_belt,amplitude_yaw_belt,kurtosis_yaw_dumbbell,skewness_yaw_dumbbell,amplitude_yaw_dumbbell,amplitude_yaw_dumbbell,skewness_yaw_forearm,amplitude_yaw_forearm,min_yaw_forearm,max_yaw_forearm,skewness_pitch_forearm,skewness_roll_forearm,kurtosis_yaw_forearm,kurtosis_picth_forearm,kurtosis_roll_forearm,max_yaw_dumbbell,min_yaw_dumbbell))
    
    
    pml.testing.n.set <- subset(pml.testing.n, select=-c(X,user_name,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window,num_window, kurtosis_yaw_belt,skewness_yaw_belt,max_yaw_belt,min_yaw_belt,amplitude_yaw_belt,kurtosis_yaw_dumbbell,skewness_yaw_dumbbell,amplitude_yaw_dumbbell,amplitude_yaw_dumbbell,skewness_yaw_forearm,amplitude_yaw_forearm,min_yaw_forearm,max_yaw_forearm,skewness_pitch_forearm,skewness_roll_forearm,kurtosis_yaw_forearm,kurtosis_picth_forearm,kurtosis_roll_forearm,max_yaw_dumbbell,min_yaw_dumbbell))
    
    ```

  4. Variables type conversion:
    Factor variables with numeric contents are converted to numeric type as these factors have more than 32 levels. This conversion is required because tree and random forest implementation in R has a hard limit of 32-levels for categorical variable.
```{r "chunkname5" ,echo=FALSE , warning=FALSE,  message=FALSE, include= FALSE }    
    pml.training.n.set$kurtosis_roll_belt=as.numeric(pml.training.n.set$kurtosis_roll_belt)
    pml.training.n.set$kurtosis_picth_belt=as.numeric(pml.training.n.set$kurtosis_picth_belt)
    pml.training.n.set$skewness_roll_belt=as.numeric(pml.training.n.set$skewness_roll_belt)
    pml.training.n.set$skewness_roll_belt.1=as.numeric(pml.training.n.set$skewness_roll_belt.1)
    pml.training.n.set$kurtosis_roll_arm=as.numeric(pml.training.n.set$kurtosis_roll_arm)
    pml.training.n.set$kurtosis_picth_arm=as.numeric(pml.training.n.set$kurtosis_picth_arm)
    pml.training.n.set$kurtosis_yaw_arm=as.numeric(pml.training.n.set$kurtosis_yaw_arm)
    pml.training.n.set$skewness_roll_arm=as.numeric(pml.training.n.set$skewness_roll_arm)
    pml.training.n.set$skewness_pitch_arm=as.numeric(pml.training.n.set$skewness_pitch_arm)
    pml.training.n.set$skewness_yaw_arm=as.numeric(pml.training.n.set$skewness_yaw_arm)
    pml.training.n.set$kurtosis_roll_dumbbell=as.numeric(pml.training.n.set$kurtosis_roll_dumbbell)
    pml.training.n.set$kurtosis_picth_dumbbell=as.numeric(pml.training.n.set$kurtosis_picth_dumbbell)
    pml.training.n.set$skewness_roll_dumbbell=as.numeric(pml.training.n.set$skewness_roll_dumbbell)
    pml.training.n.set$skewness_pitch_dumbbell=as.numeric(pml.training.n.set$skewness_pitch_dumbbell)
    pml.testing.n.set$kurtosis_roll_belt=as.numeric(pml.testing.n.set$kurtosis_roll_belt)
    pml.testing.n.set$kurtosis_picth_belt=as.numeric(pml.testing.n.set$kurtosis_picth_belt)
    pml.testing.n.set$skewness_roll_belt=as.numeric(pml.testing.n.set$skewness_roll_belt)
    pml.testing.n.set$skewness_roll_belt.1=as.numeric(pml.testing.n.set$skewness_roll_belt.1)
    pml.testing.n.set$kurtosis_roll_arm=as.numeric(pml.testing.n.set$kurtosis_roll_arm)
    pml.testing.n.set$kurtosis_picth_arm=as.numeric(pml.testing.n.set$kurtosis_picth_arm)
    pml.testing.n.set$kurtosis_yaw_arm=as.numeric(pml.testing.n.set$kurtosis_yaw_arm)
    pml.testing.n.set$skewness_roll_arm=as.numeric(pml.testing.n.set$skewness_roll_arm)
    pml.testing.n.set$skewness_pitch_arm=as.numeric(pml.testing.n.set$skewness_pitch_arm)
    pml.testing.n.set$skewness_yaw_arm=as.numeric(pml.testing.n.set$skewness_yaw_arm)
    pml.testing.n.set$kurtosis_roll_dumbbell=as.numeric(pml.testing.n.set$kurtosis_roll_dumbbell)
    pml.testing.n.set$kurtosis_picth_dumbbell=as.numeric(pml.testing.n.set$kurtosis_picth_dumbbell)
    pml.testing.n.set$skewness_roll_dumbbell=as.numeric(pml.testing.n.set$skewness_roll_dumbbell)
    pml.testing.n.set$skewness_pitch_dumbbell=as.numeric(pml.testing.n.set$skewness_pitch_dumbbell)
    ```
      
    ```{r "chunkname6" ,echo=TRUE,warning=FALSE ,  message=FALSE}
    dim(pml.training.n.set)
    dim(pml.testing.n.set)
    ```
    
### Data slicing:
**createDataPartition()** function from **caret** package is used to slice the training dataset into 80% (training) to build the model and 20% for validation (testing).

```{r "chunkname7" ,echo=TRUE, warning=FALSE , message=FALSE}
set.seed(88338)
inTrain <- createDataPartition(pml.training.n.set$classe, p= .8, list=F)
training <- pml.training.n.set[inTrain,]
testing <- pml.training.n.set[-inTrain,]
``` 

### First Model: Decision Tree with pruning
**train()** function from **caret** package is used to construct a classification tree to predict "classe" variable using all features. The final model consists of 44 terminal nodes. The most important indicator of **classe** appears to be roll_belt. A repeated 10-fold cross-validation is conducted to find the best value among 30  values of the cost-complexity parameter that is used to determine the best tree depth. In order to estimate the out-of-sample error, another 10-fold cross-validation is conducted. This produced 0.23 as an estimate of the Out-of-sample error. The overall accuracy obtained by the final model is 0.7612 (0.239 miss classification error rate). 

```{r "chunkname8" ,echo=FALSE ,warning=FALSE , message=FALSE, include=FALSE}
    set.seed(1)
    folds <- createFolds(y=training$classe, k=10, list=TRUE, returnTrain=FALSE)
    overAllAccuracy <- NULL
    cvCtrl <- trainControl(method="repeatedcv", number=10, repeats="3", classProbs = TRUE)
	   
	  #for ( i in 1:10) {
    #rpart.pml <- train(classe ~ ., data =training[folds[[i]],], method="rpart", tuneLength = 30, trControl = cvCtrl)
	 
    #pred.rpart <- predict(rpart.pml$finalModel, newdata= testing, type="class")
    #cfm <- confusionMatrix(pred.rpart, testing$classe)
	  #overAllAccuracy <- append(overAllAccuracy,cfm$overall[1])
	  #}
    
    #saveRDS(overAllAccuracy, "overAllAccuracy.rds")
    overAllAccuracy <- readRDS("overAllAccuracy.rds")
	
    #rpart.pml <- train(classe ~ ., data =training, method="rpart", tuneLength = 30, trControl = cvCtrl)
    #saveRDS(rpart.pml, "rpart.pml.rds")
    rpart.pml <- readRDS("rpart.pml.rds")

    #print(rpart.pml$finalModel)
    #party.pml <- as.party(rpart.pml$finalModel)
    #plot(party.pml)
    pred.rpart <- predict(rpart.pml$finalModel, newdata= testing, type="class")
```    

```{r "chunkname81" ,echo=TRUE ,warning=FALSE , message=FALSE}
    1-mean( overAllAccuracy)
    confusionMatrix(pred.rpart, testing$classe)
``` 


```{r "chunkname91" ,echo=FALSE,warning=FALSE,fig.height=20, fig.width=20, dev='svg' , message=FALSE}
   # fancyRpartPlot(rpart.pml$finalModel)
```

### Second Model: Bagging
In the second model, bagging is applied to the decision tree. The model is trained on 500 different bootstrapped training datasets. Bagging has impressive improvement on the overall prediction accuracy. The overall accuracy obtained when applying the model on the validation dataset is 98.5%. **randomForest()** function from **randomForest** package is used to build bagging model. This is because bagging is especial case of random forest. In the former each split in a trees is considered, a random sample of m predictors is chosen as a split candidates from the full set of p predictors. Instead of doing cross-validation, the model gives an out-of-bag error which is considered as an estimate for the test error rate. The out-of-bag is obtained from this model is 1.26%.


```{r "chunkname10" ,echo=FALSE,warning=FALSE, dev='svg' , include=FALSE , message=FALSE}
    set.seed(1)
    ###bagCtrl <- bagControl(fit= ctreeBag$fit, predict = ctreeBag$pred, aggregate = ctreeBag$aggregate)
    ###bagging.pml <- bag(training[, ! colnames(training) %in% c("classe") ], training$classe, B = 10, oob= TRUE, bagControl = bagCtrl)
    
    ###pred.bagging <- predict(bagging.pml, testing)
    ###confusionMatrix(pred.bagging, testing$classe)
    ###cvCtrl <- trainControl(method="repeatedcv", number=10, repeats="3", classProbs = TRUE)
    ###bagging.pml <- train(classe ~ ., data = training ,  method="rf" , tuneGrid = data.frame(.mtry = ncol(training) -1 ) , trControl = trainControl (method = "oob") )
     #bagging.pml <- randomForest(classe ~ . , data = training, ntree = 500, replace = TRUE , importance = TRUE, mtry = ncol(training) -1 )
     #saveRDS(bagging.pml,"bagging.pml.rds")
    
      bagging.pml <- readRDS("bagging.pml.rds")
      pred.bagging <- predict(bagging.pml, newdata= testing, type="class")
```


```{r "chunkname100" ,echo=TRUE,warning=FALSE, dev='svg' ,  message=FALSE}
      confusionMatrix(pred.bagging, testing$classe)
```


### Third Model: Random Forests
Random forest achieved the most impressive results. With 500-tree random forest model, the overall accuracy obtained on the validation dataset is 99.62%. As in bagging, out-of-bag error is used to estimate the test error rate which is the model provides out-of-bag estimate which is equal 0.43%. 

```{r "chunkname14" ,echo=FALSE,warning=FALSE, dev='svg' , include=FALSE , message=FALSE}
     set.seed(1)
     #randomForest.pml = randomForest(classe ~ . , data=training)
     #saveRDS(randomForest.pml , "randomForest.pml.rds")
     randomForest.pml <- readRDS("randomForest.pml.rds")
     pred.randomForest <- predict(randomForest.pml, newdata= testing, type="class")
```

```{r "chunkname141" ,echo=TRUE,warning=FALSE, dev='svg' , message=FALSE}
   confusionMatrix(pred.randomForest, testing$classe)
```

### Compare Error rates between the three models
Bagging and random forest results for the training dataset. The test error (green and purple) is shown as a function of B, the number of bootstrapped training sets used. Random forests were applied with m =√p. The dashed line indicates the test error resulting from a single classification tree. The red and blue traces show the OOB error, which in this case is considerably lower.


```{r "chunkname15" ,echo=FALSE,warning=FALSE, dev='svg' , include=FALSE , message=FALSE}
  #set.seed(1)
  #oob.bagging <- NULL
  #validation.err.bagging <- NULL
	#oob.rf <- NULL
	#validation.err.rf <- NULL
	 
	#for( i in 1:29) { 
	#bagging.pml <- randomForest(classe ~ . , data = training, ntree = i, replace = TRUE , importance = TRUE, mtry = ncol(training) -1 )
	#oob.bagging <- append(oob.bagging,mean(predict(bagging.pml) != training$classe, na.rm = T))
	 
   #pred.bagging <- predict(bagging.pml, newdata= testing, type="class")
  #cm.bagging <- confusionMatrix(pred.bagging, testing$classe)
	#validation.err.bagging <- append(validation.err.bagging, 1- cm.bagging$overall[1])
	  
   #randomForest.pml = randomForest(classe ~ . , data=training, ntree = i, replace=TRUE)
  	#oob.rf <- append(oob.rf,mean(predict(randomForest.pml) != training$classe, , na.rm = T))
	 
	 #pred.randomForest <- predict(randomForest.pml, newdata= testing, type="class")
    #cm.rf <- confusionMatrix(pred.randomForest, testing$classe)
     #validation.err.rf <- append(validation.err.rf,1 - cm.rf$overall[1])
	 #}
 #modelsDifferences <- data.frame(oob.bagging = oob.bagging, validation.err.bagging = validation.err.bagging , oob.rf = oob.rf , validation.err.rf = validation.err.rf
    #modelsDifferences = saveRDS(modelsDifferences,  file = "modelsDifferences.rds")
    #modelsDifferences <- readRDS("modelsDifferences.rds")
    #modelsDifferences <- data.frame(oob.bagging = oob.bagging.rds, validation.err.bagging = readRDS("validation.err.bagging.rds"), oob.rf = oob.rf.rds, validation.err.rf = readRDS("validation.err.rf.rds"))
    modelsDifferences <- readRDS("modelsDifferences.rds")
    modelsDifferences.melt <- melt(modelsDifferences)
    modelsDifferences.melt$no.tree <- c(1:423,1:423,1:423,1:423)
    
```

```{r "chunkname151" ,echo=TRUE,warning=FALSE, message=FALSE, fig.height=5, fig.width=8, dev='svg'}
    ggplot(data=modelsDifferences.melt, aes(x = no.tree, y = value, color=variable)) + geom_line() +
      geom_hline(aes(yintercept=0.1756), color="black", linetype = "dashed") + 
      xlab("Number of Trees") + ylab("Error") +
      scale_color_discrete(name = "Error") +
      theme(legend.background = element_rect(colour ="black") ) 
    
```

### Importance of the variables in radnom forest model
```{r "chunkname17" ,echo=TRUE,warning=FALSE, fig.height=5, fig.width=8, dev='svg', message=FALSE}
    varImpPlot(randomForest.pml)
```   

### References 
**Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.**

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz45Ri7LMlk  
