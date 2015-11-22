# Practical Machine Learning Assignment
Dominic Tran  
21 November 2015  
## Synopsis
This report details the model developed which predicts the manner in which barbell lifts were performed, using the [Weight Lifting Exercise Dataset](http://groupware.les.inf.puc-rio.br/har#weight_lifting_exercises).  The training data are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and the test data are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).

## Data

```r
library(caret)
```

Download the datasets and remove the first 7 columns: X, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window as data for these columns are not sensor measurements and are not used in the prediction model.


```r
read.data <- function(data.file) {
  data.read <- read.csv(data.file, na.strings = c("NA","#DIV/0!", ""), stringsAsFactors = FALSE) 
  data.read[,-(1:7)]
}

train.read <- read.data("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
test.read <- read.data("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
```

Exploratory analysis on the data showed that there are lots of NAs.  The data is subset by discarding columns in which NAs fill up 95% or more.


```r
# count number of NAs in each col
na.counts <- apply(train.read, 2, function(x) sum(is.na(x)))

# retain cols with less than 95% of NAs
keep.cols <- which(na.counts < (nrow(train.read) * 0.95))
keep.train.read <- train.read[, keep.cols]
keep.test.read <- test.read[, keep.cols]
```

No near zero-variance predictors (using *{caret}*__nearZeroVar__) are found.  Only a small number of predictors with pairwise correlations greater than 0.90 (using *{caret}*__findCorrelation__) are found, so no further subsetting is performed.

Twenty five percent of the training data is retained for cross-validation.


```r
set.seed(11)
in.train <- createDataPartition(keep.train.read$classe, p = 3/4, list = FALSE)

col.classe <- which(names(keep.train.read) == "classe")
train.data <- keep.train.read[in.train, -col.classe]
train.classe <- keep.train.read[in.train, col.classe]
validate.data <- keep.train.read[-in.train, -col.classe]
validate.classe <- keep.train.read[-in.train, col.classe]

test.data <- keep.test.read[, -which(names(keep.test.read) == "problem_id")]
```

## Model
As this is a classification problem, *Random Forest* was picked and fitted using *{caret}* default paramters.


```r
model.fit <- train(x = train.data, y = train.classe, method = "rf")
```

```
## Warning: package 'randomForest' was built under R version 3.2.2
```


```r
model.fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.65%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4179    3    2    0    1 0.001433692
## B   24 2817    5    2    0 0.010884831
## C    0   12 2546    9    0 0.008180756
## D    0    0   22 2387    3 0.010364842
## E    0    1    6    5 2694 0.004434590
```

As can be seen in the output above, the OOB estimate of error rate is 0.65%.


```r
pred.classe <- predict(model.fit, newdata = validate.data)
confusionMatrix(validate.classe, pred.classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    8  938    2    1    0
##          C    0    3  849    3    0
##          D    0    0    8  795    1
##          E    0    1    2    1  897
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9939          
##                  95% CI : (0.9913, 0.9959)
##     No Information Rate : 0.2861          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9923          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9943   0.9958   0.9861   0.9938   0.9989
## Specificity            1.0000   0.9972   0.9985   0.9978   0.9990
## Pos Pred Value         1.0000   0.9884   0.9930   0.9888   0.9956
## Neg Pred Value         0.9977   0.9990   0.9970   0.9988   0.9998
## Prevalence             0.2861   0.1921   0.1756   0.1631   0.1831
## Detection Rate         0.2845   0.1913   0.1731   0.1621   0.1829
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9971   0.9965   0.9923   0.9958   0.9989
```

The estimated accuracy of the model is 99.39% (error rate is 0.61%).

Finally, prediction is performed on test data as follows.


```r
test.classe <- predict(model.fit, newdata = test.data)
```


