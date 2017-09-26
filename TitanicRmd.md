Titanic
================
Narcel Reedus
September 14, 2017

Summary
=======

The maiden voyage of the Titanic; made popular by James Cameron's epic 1997 film, is one of the most infamous tragedies in human history. The "unsinkable" oceanliner struck an iceberg and sank to the bottom of the Atlantic killing 1,502 of its 2,224 passengers and crew.

Today, with the help of the ship's manifest, data analysts and data scientists from around the globe are able to discover numerous insights by appyling machine learning algorithms to predict who lived and who ultimately died onboard the RMS Titanic.

My goal here is to use feature engineering to viusalize the distinguishing factors between the passengers more likely to survive the shipwreck from those who ultimately perished.

Analysis
========

Load dplyr and stringr libraries

``` r
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(stringr)
library(printr)
library(pander)
library(knitr)
library(httpuv)
library(caTools)
```

The first step is to become intimately familiar with the data. Even though there are only 891 observations and 12 variables, there are many insights hidden within each data point.

We begin by loading the titanic\_train.csv data. Assign dataset to train. View data and view the structure of the data with str.

``` r
titanic_train <- read.csv("C:/Users/narce/OneDrive/Documents/GitHub/Titanic/titanic_train.csv")
train <- titanic_train
View(train)
str(train)
```

    ## 'data.frame':    891 obs. of  12 variables:
    ##  $ PassengerId: int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ Survived   : int  0 1 1 1 0 0 0 0 1 1 ...
    ##  $ Pclass     : int  3 1 3 1 3 3 1 3 3 2 ...
    ##  $ Name       : Factor w/ 891 levels "Abbing, Mr. Anthony",..: 109 191 358 277 16 559 520 629 417 581 ...
    ##  $ Sex        : Factor w/ 2 levels "female","male": 2 1 1 1 2 2 2 2 1 1 ...
    ##  $ Age        : num  22 38 26 35 35 NA 54 2 27 14 ...
    ##  $ SibSp      : int  1 1 0 1 0 0 0 3 0 1 ...
    ##  $ Parch      : int  0 0 0 0 0 0 0 1 2 0 ...
    ##  $ Ticket     : Factor w/ 681 levels "110152","110413",..: 524 597 670 50 473 276 86 396 345 133 ...
    ##  $ Fare       : num  7.25 71.28 7.92 53.1 8.05 ...
    ##  $ Cabin      : Factor w/ 148 levels "","A10","A14",..: 1 83 1 57 1 1 131 1 1 1 ...
    ##  $ Embarked   : Factor w/ 4 levels "","C","Q","S": 4 2 4 4 4 3 4 4 4 2 ...

Variables
=========

**PassengerID** The PassengerID varible simply counts the number of observations and in not other way directly connects to actual passengers onboard the Titanic. I will not use the PassengerID in this project.

**Survived** Survived is a binary variable and equates to whether the passenger survived or perished. Survived is hugely important and will serve as as our dependent variable.

**Pclass** The Pclass variable translate well as a look into the socio-economic status of each passenger. 1 = first class or Upper deck, 2 = second class or Middle deck, and 3 = third class or lower deck. I will certainly take advange of this independent variable classification as I build new features.

**Name** Name is a string variable that presents this dataset with a bit of a challenge. There are a number of titles included within the name variable that we can extract and attempt to assign to gender. For instance, Jonkheer is an adult male but Miss is used to describe unwed adult women and female children.

**Sex** The Sex Variable translates to gender.

**Age** The Age variable is important but the Age value is missing for several passengers. I will impute the Age for the missing values.

**SibSp** SipSp lumps siblings (brother, sister, stepbrother, stepsister) together with Spouses (husbands and wives)

**Parch** The Parch variable combines Parents travelling with children.

**Ticket** The ticket variable provides a ticket number. However, some passengers do not have a ticket number or share a ticket number with their family.

**Fare** The Fare variable relates to the price passengers' paid to travel on the Titanic. There are a number of passengers that did not pay a Fare.

**Cabin** The cabin variable has a lot of missing values and may not play an inportant role in this analysis.

**Embarked** The Embarked variable correlates to where passengers bordered the ship: C = Cherbourg, Q = Queenstown, S = Southampton

Filter for passengers that paid zero fare - assign to zero.fare

``` r
zero.fare <- train %>%
  filter(Fare == 0.0)
zero.fare
```

|  PassengerId|  Survived|  Pclass| Name                             | Sex  |  Age|  SibSp|  Parch| Ticket |  Fare| Cabin | Embarked |
|------------:|---------:|-------:|:---------------------------------|:-----|----:|------:|------:|:-------|-----:|:------|:---------|
|          180|         0|       3| Leonard, Mr. Lionel              | male |   36|      0|      0| LINE   |     0|       | S        |
|          264|         0|       1| Harrison, Mr. William            | male |   40|      0|      0| 112059 |     0| B94   | S        |
|          272|         1|       3| Tornquist, Mr. William Henry     | male |   25|      0|      0| LINE   |     0|       | S        |
|          278|         0|       2| Parkes, Mr. Francis "Frank"      | male |   NA|      0|      0| 239853 |     0|       | S        |
|          303|         0|       3| Johnson, Mr. William Cahoone Jr  | male |   19|      0|      0| LINE   |     0|       | S        |
|          414|         0|       2| Cunningham, Mr. Alfred Fleming   | male |   NA|      0|      0| 239853 |     0|       | S        |
|          467|         0|       2| Campbell, Mr. William            | male |   NA|      0|      0| 239853 |     0|       | S        |
|          482|         0|       2| Frost, Mr. Anthony Wood "Archie" | male |   NA|      0|      0| 239854 |     0|       | S        |
|          598|         0|       3| Johnson, Mr. Alfred              | male |   49|      0|      0| LINE   |     0|       | S        |
|          634|         0|       1| Parr, Mr. William Henry Marsh    | male |   NA|      0|      0| 112052 |     0|       | S        |
|          675|         0|       2| Watson, Mr. Ennis Hastings       | male |   NA|      0|      0| 239856 |     0|       | S        |
|          733|         0|       2| Knight, Mr. Robert J             | male |   NA|      0|      0| 239855 |     0|       | S        |
|          807|         0|       1| Andrews, Mr. Thomas Jr           | male |   39|      0|      0| 112050 |     0| A36   | S        |
|          816|         0|       1| Fry, Mr. Richard                 | male |   NA|      0|      0| 112058 |     0| B102  | S        |
|          823|         0|       1| Reuchlin, Jonkheer. John George  | male |   38|      0|      0| 19972  |     0|       | S        |
