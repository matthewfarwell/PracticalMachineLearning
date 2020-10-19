---
title: "Weight lifting exercise prediction"
author: "Matthew Farwell"
output:
  html_document:
    keep_md: true
#  pdf_document:
#    keep_md: true
documentclass: article
classoption: a4paper
subparagraph: yes
geometry: margin=2cm
header-includes: |
  \usepackage{titlesec}
  \titlespacing{\section}{0pt}{12pt plus 2pt minus 1pt}{0pt plus 1pt minus 1pt}
  \titlespacing{\subsection}{0pt}{12pt plus 2pt minus 1pt}{0pt plus 1pt minus 1pt}
  \titlespacing{\subsubsection}{0pt}{12pt plus 2pt minus 1pt}{0pt plus 1pt minus 1pt}
---

# Overview

This project predicts the manner in which weight lifting exercises were done. This project was done for the [Practical Machine Learning](https://www.coursera.org/learn/practical-machine-learning) from Coursera.

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways (the classe in the dataset). More information is available from the [Human Activity Recognition website](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset). 

The goal of the project was to predict the classes for 20 new rows in the testing set.

# Conclusion

- The values of the classes were correctly predicted. The estimated accuracy of the model was 99%.



# Data Preparation





We read the training and testing data


```r
csvTraining <- read.csv("data/pml-training.csv")
csvTesting <- read.csv("data/pml-testing.csv")

dim(csvTraining)
```

```
## [1] 19622   160
```

```r
unique(csvTraining$user_name)
```

```
## [1] "carlitos" "pedro"    "adelmo"   "charles"  "eurico"   "jeremy"
```

```r
unique(csvTraining$classe)
```

```
## [1] "A" "B" "C" "D" "E"
```

We need to remove various fields, because they will not be useful in the predictions. Indeed, the timestamps may well negatively influence the training. The exercises for the training data may be grouped (all exercises of a certain type may have been performed together). The testing data may not have been created at the same time.


```r
columnsToRemove <- c("X", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")
```

We can see that some of the columns have a majority of NAs, for instance *avg_roll_forearm*:


```r
sum(is.na(csvTraining$avg_roll_forearm))
```

```
## [1] 19216
```

There are ``19622`` rows altogether. The columns which have NAs (or blanks) have missing data for 19216 rows. If we were to include these in our training (with whatever value were imputed or selected), then these would unduly influence our results. We remove the columns where there are some NAs or blanks. This in turn will simplify the training and subsequently the model.


```r
columnsWithNAs <- sapply(names(csvTraining), function(x) { sum(is.na(csvTraining[[x]])) > 0 })
columnsWithBlanks <- sapply(names(csvTraining), function(x) { sum(csvTraining[[x]] == "") > 0 })

columnNamesWithBlanksOrNAs <- names(csvTraining)[columnsWithNAs | columnsWithBlanks]
```

This eliminates 100 columns.

We modify the training and test data set to only include the relevant columns. We also make the *user_name* and *classe* factors.


```r
training <- as.data.frame(csvTraining %>%
        mutate(user_name=as.factor(user_name)) %>%
        mutate(classe=as.factor(classe)) %>%
        select(-all_of(columnsToRemove)) %>%
        select(-all_of(columnNamesWithBlanksOrNAs)))

testing <- csvTesting %>%
        mutate(user_name=as.factor(user_name)) %>%
        select(-all_of(columnsToRemove)) %>%
        select(-all_of(columnNamesWithBlanksOrNAs))

dim(training)
```

```
## [1] 19622    54
```

```r
dim(testing)
```

```
## [1] 20 54
```

We are left with 54 columns.

# Training

We train the model. For categorical prediction, we will use a random forest with repeated cross validation.

Repeated k-fold cross-validation has the benefit of improving the estimate of the mean model performance at the cost of fitting and evaluating many more models.


```r
trControl <- caret::trainControl(method='repeatedcv', 
                        number=4,
                        repeats=2,
                        verboseIter=TRUE)

modelFit <- caret::train(classe~., method="rf", data=training, trControl=trControl)
```

```
## + Fold1.Rep1: mtry= 2 
## - Fold1.Rep1: mtry= 2 
## + Fold1.Rep1: mtry=29 
## - Fold1.Rep1: mtry=29 
## + Fold1.Rep1: mtry=57 
## - Fold1.Rep1: mtry=57 
## + Fold2.Rep1: mtry= 2 
## - Fold2.Rep1: mtry= 2 
## + Fold2.Rep1: mtry=29 
## - Fold2.Rep1: mtry=29 
## + Fold2.Rep1: mtry=57 
## - Fold2.Rep1: mtry=57 
## + Fold3.Rep1: mtry= 2 
## - Fold3.Rep1: mtry= 2 
## + Fold3.Rep1: mtry=29 
## - Fold3.Rep1: mtry=29 
## + Fold3.Rep1: mtry=57 
## - Fold3.Rep1: mtry=57 
## + Fold4.Rep1: mtry= 2 
## - Fold4.Rep1: mtry= 2 
## + Fold4.Rep1: mtry=29 
## - Fold4.Rep1: mtry=29 
## + Fold4.Rep1: mtry=57 
## - Fold4.Rep1: mtry=57 
## + Fold1.Rep2: mtry= 2 
## - Fold1.Rep2: mtry= 2 
## + Fold1.Rep2: mtry=29 
## - Fold1.Rep2: mtry=29 
## + Fold1.Rep2: mtry=57 
## - Fold1.Rep2: mtry=57 
## + Fold2.Rep2: mtry= 2 
## - Fold2.Rep2: mtry= 2 
## + Fold2.Rep2: mtry=29 
## - Fold2.Rep2: mtry=29 
## + Fold2.Rep2: mtry=57 
## - Fold2.Rep2: mtry=57 
## + Fold3.Rep2: mtry= 2 
## - Fold3.Rep2: mtry= 2 
## + Fold3.Rep2: mtry=29 
## - Fold3.Rep2: mtry=29 
## + Fold3.Rep2: mtry=57 
## - Fold3.Rep2: mtry=57 
## + Fold4.Rep2: mtry= 2 
## - Fold4.Rep2: mtry= 2 
## + Fold4.Rep2: mtry=29 
## - Fold4.Rep2: mtry=29 
## + Fold4.Rep2: mtry=57 
## - Fold4.Rep2: mtry=57 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 29 on full training set
```

# Model

We summarise the model, including the predicted accuracy of the model.


```r
print(modelFit)
```

```
## Random Forest 
## 
## 19622 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold, repeated 2 times) 
## Summary of sample sizes: 14716, 14716, 14717, 14717, 14715, 14716, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9937315  0.9920701
##   29    0.9940881  0.9925217
##   57    0.9883039  0.9852048
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 29.
```

# Prediction

We predict the classes using the testing data.


```r
predictedClasses <- predict(modelFit, newdata=testing)
```

Here are the predictions for the testing data set:


Table: Predictions

|problem_id |Predicted class |
|:----------|:---------------|
|1          |B               |
|2          |A               |
|3          |B               |
|4          |A               |
|5          |A               |
|6          |E               |
|7          |D               |
|8          |B               |
|9          |A               |
|10         |A               |
|11         |B               |
|12         |C               |
|13         |B               |
|14         |A               |
|15         |E               |
|16         |E               |
|17         |A               |
|18         |B               |
|19         |B               |
|20         |B               |

\newpage

\newpage
# Appendix 1: Data description

**Weight Lifting Exercises Dataset**

## Description

This human activity recognition research has traditionally focused on discriminating between different activities, i.e. to predict "which" activity was performed at a specific point in time (like with the Daily Living Activities dataset above). The approach we propose for the Weight Lifting Exercises dataset is to investigate "how (well)" an activity was performed by the wearer. The "how (well)" investigation has only received little attention so far, even though it potentially provides useful information for a large variety of applications,such as sports training.

In this work (see the paper) we first define quality of execution and investigate three aspects that pertain to qualitative activity recognition: the problem of specifying correct execution, the automatic and robust detection of execution mistakes, and how to provide feedback on the quality of execution to the user. We tried out an on-body sensing approach (dataset here), but also an "ambient sensing approach" (by using Microsoft Kinect - dataset still unavailable)

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).


## Source

[Human Activity Recognition](http://groupware.les.inf.puc-rio.br/har)
[Qualitative Activity Recognition of Weight Lifting Exercises](http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201)

## References

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 

# Appendix 2: Session Info


```r
sessionInfo()
```

```
## R version 4.0.2 (2020-06-22)
## Platform: x86_64-pc-linux-gnu (64-bit)
## Running under: Ubuntu 20.04.1 LTS
## 
## Matrix products: default
## BLAS:   /usr/lib/x86_64-linux-gnu/blas/libblas.so.3.9.0
## LAPACK: /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.9.0
## 
## locale:
##  [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C              
##  [3] LC_TIME=en_GB.UTF-8        LC_COLLATE=en_US.UTF-8    
##  [5] LC_MONETARY=en_GB.UTF-8    LC_MESSAGES=en_US.UTF-8   
##  [7] LC_PAPER=en_GB.UTF-8       LC_NAME=C                 
##  [9] LC_ADDRESS=C               LC_TELEPHONE=C            
## [11] LC_MEASUREMENT=en_GB.UTF-8 LC_IDENTIFICATION=C       
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
##  [1] gbm_2.1.8           randomForest_4.6-14 rattle_5.4.0       
##  [4] bitops_1.0-6        tibble_3.0.3        rpart_4.1-15       
##  [7] e1071_1.7-3         caret_6.0-86        lattice_0.20-41    
## [10] dplyr_1.0.0         plyr_1.8.6          knitr_1.29         
## [13] ggplot2_3.3.2      
## 
## loaded via a namespace (and not attached):
##  [1] tidyselect_1.1.0     xfun_0.15            purrr_0.3.4         
##  [4] reshape2_1.4.4       splines_4.0.2        colorspace_1.4-1    
##  [7] vctrs_0.3.4          generics_0.0.2       htmltools_0.5.0     
## [10] stats4_4.0.2         yaml_2.2.1           survival_3.1-12     
## [13] prodlim_2019.11.13   rlang_0.4.7          ModelMetrics_1.2.2.2
## [16] pillar_1.4.6         glue_1.4.2           withr_2.2.0         
## [19] foreach_1.5.0        lifecycle_0.2.0      lava_1.6.7          
## [22] stringr_1.4.0        timeDate_3043.102    munsell_0.5.0       
## [25] gtable_0.3.0         recipes_0.1.13       codetools_0.2-16    
## [28] evaluate_0.14        class_7.3-17         highr_0.8           
## [31] Rcpp_1.0.5           scales_1.1.1         ipred_0.9-9         
## [34] digest_0.6.25        stringi_1.4.6        grid_4.0.2          
## [37] tools_4.0.2          magrittr_1.5         crayon_1.3.4        
## [40] pkgconfig_2.0.3      ellipsis_0.3.1       MASS_7.3-53         
## [43] Matrix_1.2-18        data.table_1.12.8    pROC_1.16.2         
## [46] lubridate_1.7.9      gower_0.2.2          rmarkdown_2.3       
## [49] iterators_1.0.12     R6_2.4.1             nnet_7.3-14         
## [52] nlme_3.1-149         compiler_4.0.2
```


