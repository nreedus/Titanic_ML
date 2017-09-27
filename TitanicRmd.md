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

**PassengerID**

The PassengerID varible simply counts the number of observations and in not other way directly connects to actual passengers onboard the Titanic. I will not use the PassengerID in this project.

**Survived**

Survived is a binary variable and equates to whether the passenger survived or perished. Survived is hugely important and will serve as as our dependent variable.

**Pclass**

The Pclass variable translate well as a look into the socio-economic status of each passenger. 1 = first class or Upper deck, 2 = second class or Middle deck, and 3 = third class or lower deck. I will certainly take advange of this independent variable classification as I build new features.

**Name**

Name is a string variable that presents this dataset with a bit of a challenge. There are a number of titles included within the name variable that we can extract and attempt to assign to gender. For instance, Jonkheer is an adult male but Miss is used to describe unwed adult women and female children.

**Sex**

The Sex Variable translates to gender.

**Age**

The Age variable is important but the Age value is missing for several passengers. I will impute the Age for the missing values.

**SibSp**

SipSp lumps siblings (brother, sister, stepbrother, stepsister) together with Spouses (husbands and wives)

**Parch**

The Parch variable combines Parents travelling with children.

**Ticket**

The ticket variable provides a ticket number. However, some passengers do not have a ticket number or share a ticket number with their family.

**Fare**

The Fare variable relates to the price passengers' paid to travel on the Titanic. There are a number of passengers that did not pay a Fare.

**Cabin**

The cabin variable has a lot of missing values and may not play an inportant role in this analysis.

**Embarked**

The Embarked variable correlates to where passengers bordered the ship: C = Cherbourg, Q = Queenstown, S = Southampton

============================================================

There are a number of passengers that appear to not have paid a fare. I filter the Fare variable for 0.0 to take a closer look at this anomaly. I assign the results to zero.fare

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

I take the zero.fare anomaly a step further by grouping zero.class by Pclass and then summarizing. The results are six passengers in first class, five passengers in first class, four passengers in third class paid zero fare.

``` r
zero.fare.pclass <- zero.fare %>%
  group_by(Pclass) %>%
  summarize(Total = n()) %>%
  arrange(desc(Total))
zero.fare.pclass
```

|  Pclass|  Total|
|-------:|------:|
|       2|      6|
|       1|      5|
|       3|      4|

The next phase of this analysis to the get a better understanding of the survivors by gender, denoted here as Sex. My first step is to seperate adult males and females from the male and female adolecents on the ship. I used regex function str\_extract to extract passengers by title (Mr. Mrs. Miss. Rev...) by locating the word just before the (.) period in the Name variable and then create a new feature called Title.

``` r
train <- train %>%
mutate(Title = str_extract(Name, "[a-zA-Z0-9]+\\."))
table(train$Title)
```

|  Capt.|  Col.|  Countess.|  Don.|  Dr.|  Jonkheer.|  Lady.|  Major.|  Master.|  Miss.|  Mlle.|  Mme.|  Mr.|  Mrs.|  Ms.|  Rev.|  Sir.|
|------:|-----:|----------:|-----:|----:|----------:|------:|-------:|--------:|------:|------:|-----:|----:|-----:|----:|-----:|-----:|
|      1|     2|          1|     1|    7|          1|      1|       2|       40|    182|      2|     1|  517|   125|    1|     6|     1|

Now I create a new data.frame (titles.lookup) that contains the title variable and a New.Title variable condensed by gender (Mr. Mrs. Miss, and Master). My goal is to line up the adult male titles to Mr., married adult female titles to Mrs., adolescent females and unwed adult females as Miss., and adolescent boys as Master.

``` r
titles.lookup <- data.frame(Title = c("Mr.", "Capt.", "Col.", "Don.", "Dr.", 
                                      "Jonkheer.", "Major", "Rev.", "Sir", 
                                      "Mrs.", "Dana.", "Lady.", "Mme.", 
                                      "Countess.", "Miss.", "Mlle.", "Ms.", 
                                      "Master."), 
                            New.Title = c(rep("Mr.", 9),
                                          rep("Mrs.", 5),
                                          rep("Miss.", 3),
                                          "Master."),
                            stringsAsFactors = FALSE)
View(titles.lookup)
knitr::kable(titles.lookup)
```

| Title     | New.Title |
|:----------|:----------|
| Mr.       | Mr.       |
| Capt.     | Mr.       |
| Col.      | Mr.       |
| Don.      | Mr.       |
| Dr.       | Mr.       |
| Jonkheer. | Mr.       |
| Major     | Mr.       |
| Rev.      | Mr.       |
| Sir       | Mr.       |
| Mrs.      | Mrs.      |
| Dana.     | Mrs.      |
| Lady.     | Mrs.      |
| Mme.      | Mrs.      |
| Countess. | Mrs.      |
| Miss.     | Miss.     |
| Mlle.     | Miss.     |
| Ms.       | Miss.     |
| Master.   | Master.   |

Currently there are 18 different titles in the Titanic dataset. To arrcurately predict survivors by Sex I need to simplfy both the Titles and Sex variables. I will use the New.Titles table to left\_join (Mr. Mrs. Miss. and Master) to match the titles.lookup data.frame. This will create a more condensed Title variable that more accurately identifies the Sex and Title of each passenger.

``` r
train <- train %>%
  left_join(titles.lookup, by = "Title")
```

``` r
train <- train %>%
  mutate(Title = New.Title) %>%
  select(-New.Title)
```

There may be an error in reassigning titles by Sex. To mke sure that male and female passengers have the correct title I will filter females male titles and males with female titles.

``` r
train %>%
  filter((Sex == "female" & (Title == "Mr." | Title == "Master.")) |
         (Sex == "male" & (Title == "Mrs." | Title == "Miss.")))
```

|  PassengerId|  Survived|  Pclass| Name                        | Sex    |  Age|  SibSp|  Parch| Ticket |     Fare| Cabin | Embarked | Title |
|------------:|---------:|-------:|:----------------------------|:-------|----:|------:|------:|:-------|--------:|:------|:---------|:------|
|          797|         1|       1| Leader, Dr. Alice (Farnham) | female |   49|      0|      0| 17465  |  25.9292| D17   | S        | Mr.   |

I found one woman with the title of Dr. Since all the other titles with Dr. are male I will change this female passengers' title to Mrs.

``` r
train$Title[train$PassengerId == 797] <- "Mrs."
```

There are numerous male passengers with zero fare. I generate summary stats for passengers with the title "Mr." by Fare and Pclass (min, max, mean, median, var, SD,IQR).

``` r
mr.fare.stats <- train %>%
  filter(Title == "Mr.") %>%
  group_by(Pclass) %>%
  summarize(Fare.Min = min(Fare),
            Fare.Max = max(Fare),
            Fare.Mean = mean(Fare),
            Fare.Median = median(Fare),
            Fare.Var = var(Fare),
            Fare.SD = sd(Fare),
            Fare.IQR = IQR(Fare))
mr.fare.stats
```

|  Pclass|  Fare.Min|  Fare.Max|  Fare.Mean|  Fare.Median|   Fare.Var|   Fare.SD|  Fare.IQR|
|-------:|---------:|---------:|----------:|------------:|----------:|---------:|---------:|
|       1|         0|  512.3292|   66.67414|      39.8625|  6212.1977|  78.81750|  49.56670|
|       2|         0|   73.5000|   19.05412|      13.0000|   230.8936|  15.19518|  14.50000|
|       3|         0|   69.5500|   11.29976|       7.8958|   109.0203|  10.44128|   0.91665|

In case I need to come back to passengers that had zero fare, I will create a binary table tracking feature to identify zero fare passengers from paid passengers if ifelse statment - 0 = Y, Else N.

``` r
train$Fare.Zero <- ifelse(train$Fare == 0.0, "Y", "N")
```

In order to impute the missing Fare values I must find the median fare by Pclass. I will create a lookup table for zero fare values using filter, group\_by, and summarise and then assign it to zero.fare.lookup.

``` r
zero.fare.lookup <- train %>%
  filter(Title == "Mr.") %>%
  group_by(Pclass, Title) %>%
  summarise(New.Fare = median(Fare))
  
zero.fare.lookup
```

    ## # A tibble: 3 x 3
    ## # Groups:   Pclass [?]
    ##   Pclass Title New.Fare
    ##    <int> <chr>    <dbl>
    ## 1      1   Mr.  39.8625
    ## 2      2   Mr.  13.0000
    ## 3      3   Mr.   7.8958

``` r
knitr::kable(zero.fare.lookup)
```

|  Pclass| Title |  New.Fare|
|-------:|:------|---------:|
|       1| Mr.   |   39.8625|
|       2| Mr.   |   13.0000|
|       3| Mr.   |    7.8958|

Now that I have determined the median fare for first class as 39.86, second class 13.00, and third class 7.89, I can impute those values into the zero fare values by left\_joining the zero.fare.lookup data.frame to the train dataset. This will replace zero fares with the median value per Pclass.

``` r
train <- train %>%
  left_join(zero.fare.lookup, by = c("Pclass", "Title")) %>%
  mutate(fare = ifelse(Fare == 0.0, New.Fare, Fare)) %>%
  select(-New.Fare)
```

I will generate summary stats based on age that will be helpful in imputing more missing values and create new insights of male survivors by age, title and Pclass.

``` r
age.stats <- train %>%
  group_by(Pclass, Title) %>%
  summarize(Age.Min = min(Age, na.rm = TRUE),
            Age.Max = max(Age, na.rm = TRUE),
            Age.Mean = mean(Age, na.rm = TRUE),
            Age.Median = median(Age, na.rm = TRUE),
            Age.Var = var(Age, na.rm = TRUE),
            Age.SD = sd(Age, na.rm = TRUE),
            Age.IQR = IQR(Age, na.rm = TRUE)) %>%
  arrange(Title, Pclass)

age.stats
```

    ## # A tibble: 13 x 9
    ## # Groups:   Pclass [3]
    ##    Pclass   Title Age.Min Age.Max  Age.Mean Age.Median    Age.Var
    ##     <int>   <chr>   <dbl>   <dbl>     <dbl>      <dbl>      <dbl>
    ##  1      1 Master.    0.92      11  5.306667        4.0  26.682133
    ##  2      2 Master.    0.67       8  2.258889        1.0   5.487936
    ##  3      3 Master.    0.42      12  5.350833        4.0  12.914017
    ##  4      1   Miss.    2.00      63 29.744681       30.0 159.498612
    ##  5      2   Miss.    2.00      50 22.560606       24.0 174.246212
    ##  6      3   Miss.    0.75      45 16.123188       18.0  94.037910
    ##  7      1     Mr.   17.00      80 42.184211       40.0 199.138578
    ##  8      2     Mr.   16.00      70 33.588889       31.0 150.261673
    ##  9      3     Mr.   11.00      74 28.724891       26.0 110.059948
    ## 10      1    Mrs.   17.00      62 40.631579       41.5 155.049787
    ## 11      2    Mrs.   14.00      57 33.682927       32.0 106.471951
    ## 12      3    Mrs.   15.00      63 33.515152       31.0 100.632576
    ## 13      1    <NA>   45.00      52 48.666667       49.0  12.333333
    ## # ... with 2 more variables: Age.SD <dbl>, Age.IQR <dbl>

As I did before, I will create a binary table tracking feature to identify missing Age values with an if ifelse statment - 0 = Y, Else N.

``` r
train$Age.Missing <- ifelse(is.na(train$Age), "Y", "N")
```

This is a lookup table for the Age values selecting Pclass, Title, Age.Mean, Age.Median.

``` r
age.lookup <- age.stats %>%
  select(Pclass, Title, Age.Mean, Age.Median)
```

I will impute missing ages by using this lookup table.

``` r
train <- train %>%
  left_join(age.lookup, by = c("Pclass", "Title")) %>%
  mutate(Age = ifelse(Age.Missing == "Y",
                      ifelse(Title == "Miss.", Age.Median, Age.Mean),
                      Age)) %>%
  select(-Age.Median, -Age.Mean)
```

For missing fare values by age and sex I will create a Ticket-based feature using Group\_by ticket, summarise group.count, Avg fare = max fare / n(), sum of Female.Count, ratio of the n() of males in ticket count / number of people on that ticket.

``` r
ticket.lookup <- train %>%
  group_by(Ticket) %>%
  summarise(Group.Count = n(),
            Avg.Fare = max(Fare) / n(),
            Female.Count = sum(Sex == "female"),
            Male.Count = sum(Sex == "male"),
            Child.Count = sum(Age < 18),
            Elderly.Count = sum(Age > 54.0),
            Female.Ratio = sum(Sex == "female") / n(),
            Male.Ratio = sum(Sex == "Male") / n(),
            Child.Ratio = sum(Age < 18) / n(),
            Elderly.Ratio = sum(Age > 54.0) / n(),
            Female.Child.Ratio = (sum(Age < 18) +
                                  sum(Sex == "female" & Age >=18)) / n(),
            Min.Age = min(Age),
            Max.Age = max(Age))

ticket.lookup
```

| Ticket             |  Group.Count|    Avg.Fare|  Female.Count|  Male.Count|  Child.Count|  Elderly.Count|  Female.Ratio|  Male.Ratio|  Child.Ratio|  Elderly.Ratio|  Female.Child.Ratio|    Min.Age|    Max.Age|
|:-------------------|------------:|-----------:|-------------:|-----------:|------------:|--------------:|-------------:|-----------:|------------:|--------------:|-------------------:|----------:|----------:|
| 110152             |            3|   28.833333|             3|           0|            1|              0|     1.0000000|           0|    0.3333333|      0.0000000|           1.0000000|  16.000000|  33.000000|
| 110413             |            3|   26.550000|             2|           1|            0|              0|     0.6666667|           0|    0.0000000|      0.0000000|           0.6666667|  18.000000|  52.000000|
| 110465             |            2|   26.000000|             0|           2|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  47.000000|
| 110564             |            1|   26.550000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.000000|  28.000000|
| 110813             |            1|   75.250000|             1|           0|            0|              1|     1.0000000|           0|    0.0000000|      1.0000000|           1.0000000|  60.000000|  60.000000|
| 111240             |            1|   33.500000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  61.000000|  61.000000|
| 111320             |            1|   38.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  47.000000|  47.000000|
| 111361             |            2|   28.989600|             2|           0|            1|              0|     1.0000000|           0|    0.5000000|      0.0000000|           1.0000000|  16.000000|  44.000000|
| 111369             |            1|   30.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  26.000000|  26.000000|
| 111426             |            1|   26.550000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  35.000000|  35.000000|
| 111427             |            1|   26.550000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| 111428             |            1|   26.550000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  45.000000|  45.000000|
| 112050             |            1|    0.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  39.000000|  39.000000|
| 112052             |            1|    0.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| 112053             |            1|   30.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  19.000000|  19.000000|
| 112058             |            1|    0.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| 112059             |            1|    0.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  40.000000|  40.000000|
| 112277             |            1|   31.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  40.000000|  40.000000|
| 112379             |            1|   39.600000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| 113028             |            1|   26.550000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| 113043             |            1|   28.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  45.500000|  45.500000|
| 113050             |            1|   26.550000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  45.000000|  45.000000|
| 113051             |            1|   27.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  30.000000|  30.000000|
| 113055             |            1|   26.550000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  51.000000|  51.000000|
| 113056             |            1|   26.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| 113059             |            1|   47.100000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.000000|  28.000000|
| 113501             |            1|   30.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  29.000000|  29.000000|
| 113503             |            1|  211.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  27.000000|  27.000000|
| 113505             |            2|   27.500000|             2|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  22.000000|  40.631579|
| 113509             |            1|   61.979200|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  65.000000|  65.000000|
| 113510             |            1|   35.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| 113514             |            1|   26.550000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  62.000000|  62.000000|
| 113572             |            2|   40.000000|             2|           0|            0|              1|     1.0000000|           0|    0.0000000|      0.5000000|           1.0000000|  38.000000|  62.000000|
| 113760             |            4|   30.000000|             2|           2|            2|              0|     0.5000000|           0|    0.5000000|      0.0000000|           0.7500000|  11.000000|  36.000000|
| 113767             |            1|   50.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| 113773             |            1|   53.100000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  19.000000|
| 113776             |            2|   33.300000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  22.000000|  29.000000|
| 113781             |            4|   37.887500|             3|           1|            2|              0|     0.7500000|           0|    0.5000000|      0.0000000|           1.0000000|   0.920000|  25.000000|
| 113783             |            1|   26.550000|             1|           0|            0|              1|     1.0000000|           0|    0.0000000|      1.0000000|           1.0000000|  58.000000|  58.000000|
| 113784             |            1|   35.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  45.000000|  45.000000|
| 113786             |            1|   30.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  52.000000|  52.000000|
| 113787             |            1|   30.500000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  55.000000|  55.000000|
| 113788             |            1|   35.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.000000|  28.000000|
| 113789             |            2|   26.000000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  35.000000|  42.000000|
| 113792             |            1|   26.550000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  56.000000|  56.000000|
| 113794             |            1|   26.550000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  34.000000|  34.000000|
| 113796             |            1|   42.400000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| 113798             |            2|   15.500000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  30.000000|  42.184210|
| 113800             |            1|   26.550000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  60.000000|  60.000000|
| 113803             |            2|   26.550000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  35.000000|  37.000000|
| 113804             |            1|   30.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  27.000000|  27.000000|
| 113806             |            2|   26.550000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  27.000000|  33.000000|
| 113807             |            1|   26.550000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  62.000000|  62.000000|
| 11668              |            2|   10.500000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  27.000000|  29.000000|
| 11751              |            2|   26.277100|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  37.000000|  47.000000|
| 11752              |            1|   26.283300|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  19.000000|  19.000000|
| 11753              |            1|   52.554200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.000000|  42.000000|
| 11755              |            1|   39.600000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  48.000000|  48.000000|
| 11765              |            1|   55.441700|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  25.000000|  25.000000|
| 11767              |            2|   41.579150|             2|           0|            0|              1|     1.0000000|           0|    0.0000000|      0.5000000|           1.0000000|  24.000000|  56.000000|
| 11769              |            1|   51.479200|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  53.000000|  53.000000|
| 11771              |            1|   29.700000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  58.000000|  58.000000|
| 11774              |            1|   29.700000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| 11813              |            1|   76.291700|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  32.000000|  32.000000|
| 11967              |            2|   45.539600|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  19.000000|  25.000000|
| 12233              |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  34.000000|  34.000000|
| 12460              |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 12749              |            2|   46.750000|             2|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  30.000000|  52.000000|
| 13049              |            1|   40.125000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  36.000000|  36.000000|
| 13213              |            1|   35.500000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  56.000000|  56.000000|
| 13214              |            1|   30.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  32.000000|  32.000000|
| 13502              |            3|   25.986100|             3|           0|            0|              1|     1.0000000|           0|    0.0000000|      0.3333333|           1.0000000|  21.000000|  63.000000|
| 13507              |            2|   27.950000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  39.000000|  50.000000|
| 13509              |            1|   26.550000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  65.000000|  65.000000|
| 13567              |            1|   79.200000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  60.000000|  60.000000|
| 13568              |            1|   49.500000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  22.000000|  22.000000|
| 14311              |            1|    7.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 14312              |            1|    7.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 14313              |            1|    7.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 14973              |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  22.000000|  22.000000|
| 1601               |            7|    8.070829|             0|           7|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  26.000000|  32.000000|
| 16966              |            2|   67.250000|             2|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  40.000000|  41.000000|
| 16988              |            1|   30.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| 17421              |            4|   27.720825|             2|           2|            1|              0|     0.5000000|           0|    0.2500000|      0.0000000|           0.7500000|  17.000000|  49.000000|
| 17453              |            2|   44.552100|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  40.631579|  49.000000|
| 17463              |            1|   51.862500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  54.000000|  54.000000|
| 17464              |            1|   51.862500|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  40.631579|  40.631579|
| 17465              |            1|   25.929200|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  49.000000|  49.000000|
| 17466              |            1|   25.929200|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  48.000000|  48.000000|
| 17474              |            2|   28.500000|             1|           1|            1|              0|     0.5000000|           0|    0.5000000|      0.0000000|           0.5000000|  17.000000|  31.000000|
| 17764              |            1|   30.695800|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  56.000000|  56.000000|
| 19877              |            2|   39.425000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  26.000000|  36.000000|
| 19928              |            2|   45.000000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  33.000000|  44.000000|
| 19943              |            2|   45.000000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  35.000000|  38.000000|
| 19947              |            1|   35.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| 19950              |            4|   65.750000|             2|           2|            0|              1|     0.5000000|           0|    0.0000000|      0.2500000|           0.5000000|  19.000000|  64.000000|
| 19952              |            1|   26.550000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  48.000000|  48.000000|
| 19972              |            1|    0.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  38.000000|  38.000000|
| 19988              |            1|   30.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| 19996              |            2|   26.000000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  40.631579|  48.000000|
| 2003               |            1|   26.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  28.000000|  28.000000|
| 211536             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  27.000000|  27.000000|
| 21440              |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  51.000000|  51.000000|
| 218629             |            1|   13.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.000000|  28.000000|
| 219533             |            1|   12.350000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  57.000000|  57.000000|
| 220367             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  27.000000|  27.000000|
| 220845             |            2|   32.500000|             2|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  24.000000|  48.000000|
| 2223               |            1|    8.300000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  18.000000|  18.000000|
| 223596             |            1|   13.500000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  45.000000|  45.000000|
| 226593             |            1|   12.350000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  24.000000|  24.000000|
| 226875             |            1|   26.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  36.000000|  36.000000|
| 228414             |            1|   26.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  29.000000|  29.000000|
| 229236             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  36.000000|  36.000000|
| 230080             |            3|    8.666667|             0|           3|            2|              0|     0.0000000|           0|    0.6666667|      0.0000000|           0.6666667|   2.000000|  36.500000|
| 230136             |            2|   19.500000|             1|           1|            2|              0|     0.5000000|           0|    1.0000000|      0.0000000|           1.0000000|   1.000000|   4.000000|
| 230433             |            2|   13.000000|             2|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  25.000000|  50.000000|
| 230434             |            1|   13.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  28.000000|  28.000000|
| 231919             |            2|   11.500000|             2|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  34.000000|
| 231945             |            1|   11.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  18.000000|  18.000000|
| 233639             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  23.000000|  23.000000|
| 233866             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  24.000000|  24.000000|
| 234360             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  48.000000|  48.000000|
| 234604             |            1|   13.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  32.000000|  32.000000|
| 234686             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  25.000000|  25.000000|
| 234818             |            1|   12.350000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  30.000000|  30.000000|
| 236171             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  18.000000|  18.000000|
| 236852             |            1|   13.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  42.000000|  42.000000|
| 236853             |            1|   26.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  25.000000|  25.000000|
| 237442             |            1|   13.500000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  59.000000|  59.000000|
| 237565             |            1|   15.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  47.000000|  47.000000|
| 237668             |            1|   13.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  28.000000|  28.000000|
| 237671             |            1|   13.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  38.000000|  38.000000|
| 237736             |            2|   15.035400|             1|           1|            1|              0|     0.5000000|           0|    0.5000000|      0.0000000|           0.5000000|  14.000000|  32.500000|
| 237789             |            1|   30.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  25.000000|  25.000000|
| 237798             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.000000|  42.000000|
| 239853             |            3|    0.000000|             0|           3|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  33.588889|  33.588889|
| 239854             |            1|    0.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  33.588889|  33.588889|
| 239855             |            1|    0.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  33.588889|  33.588889|
| 239856             |            1|    0.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  33.588889|  33.588889|
| 239865             |            2|   13.000000|             0|           2|            1|              0|     0.0000000|           0|    0.5000000|      0.0000000|           0.5000000|  16.000000|  35.000000|
| 240929             |            1|   12.650000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  28.000000|  28.000000|
| 24160              |            3|   70.445833|             3|           0|            1|              0|     1.0000000|           0|    0.3333333|      0.0000000|           1.0000000|  15.000000|  43.000000|
| 243847             |            2|   13.500000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  24.000000|  42.000000|
| 243880             |            1|   13.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  34.000000|  34.000000|
| 244252             |            2|   13.000000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  44.000000|  54.000000|
| 244270             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  31.000000|  31.000000|
| 244278             |            1|   10.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  23.000000|  23.000000|
| 244310             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.000000|  42.000000|
| 244358             |            1|   26.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  27.000000|  27.000000|
| 244361             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  25.000000|  25.000000|
| 244367             |            2|   13.000000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  24.000000|  34.000000|
| 244373             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  33.588889|  33.588889|
| 248698             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  34.000000|  34.000000|
| 248706             |            1|   16.000000|             1|           0|            0|              1|     1.0000000|           0|    0.0000000|      1.0000000|           1.0000000|  55.000000|  55.000000|
| 248723             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  39.000000|  39.000000|
| 248727             |            3|   11.000000|             2|           1|            1|              0|     0.6666667|           0|    0.3333333|      0.0000000|           0.6666667|   6.000000|  28.000000|
| 248731             |            1|   13.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  52.000000|  52.000000|
| 248733             |            1|   13.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  24.000000|  24.000000|
| 248738             |            2|   14.500000|             1|           1|            1|              0|     0.5000000|           0|    0.5000000|      0.0000000|           1.0000000|   0.830000|  22.000000|
| 248740             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.000000|  28.000000|
| 248747             |            1|   13.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  24.000000|  24.000000|
| 250643             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  50.000000|  50.000000|
| 250644             |            2|    9.750000|             2|           0|            1|              0|     1.0000000|           0|    0.5000000|      0.0000000|           1.0000000|  13.000000|  41.000000|
| 250646             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  30.000000|  30.000000|
| 250647             |            2|    6.500000|             0|           2|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  34.000000|  52.000000|
| 250648             |            1|   13.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  30.000000|  30.000000|
| 250649             |            2|    7.250000|             1|           1|            1|              0|     0.5000000|           0|    0.5000000|      0.0000000|           1.0000000|   0.670000|  24.000000|
| 250651             |            1|   26.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  26.000000|  26.000000|
| 250652             |            1|   13.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 250653             |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  30.000000|  30.000000|
| 250655             |            2|   13.000000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  19.000000|  39.000000|
| 2620               |            1|    7.225000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  22.000000|  22.000000|
| 2623               |            1|    7.225000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  40.000000|  40.000000|
| 2624               |            1|    7.225000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 2625               |            1|    8.516700|             0|           1|            1|              0|     0.0000000|           0|    1.0000000|      0.0000000|           1.0000000|   0.420000|   0.420000|
| 2626               |            1|    7.229200|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  33.515152|  33.515152|
| 2627               |            2|    7.229150|             1|           1|            1|              0|     0.5000000|           0|    0.5000000|      0.0000000|           0.5000000|  17.000000|  28.724891|
| 2628               |            1|    7.225000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  45.500000|  45.500000|
| 2629               |            1|    7.229200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 2631               |            1|    7.225000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 26360              |            2|   13.000000|             2|           0|            1|              0|     1.0000000|           0|    0.5000000|      0.0000000|           1.0000000|   2.000000|  33.000000|
| 2641               |            1|    7.229200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 2647               |            1|    7.225000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 2648               |            1|    4.012500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  20.000000|  20.000000|
| 2649               |            1|    7.225000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  33.515152|  33.515152|
| 2650               |            1|   15.245800|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  29.000000|  29.000000|
| 2651               |            2|    5.620850|             1|           1|            2|              0|     0.5000000|           0|    1.0000000|      0.0000000|           1.0000000|  12.000000|  14.000000|
| 2653               |            2|    7.870850|             1|           1|            1|              0|     0.5000000|           0|    0.5000000|      0.0000000|           0.5000000|   1.000000|  20.000000|
| 2659               |            2|    7.227100|             1|           1|            1|              0|     0.5000000|           0|    0.5000000|      0.0000000|           0.5000000|  15.000000|  27.000000|
| 2661               |            2|    7.622900|             0|           2|            2|              0|     0.0000000|           0|    1.0000000|      0.0000000|           1.0000000|   5.350833|   5.350833|
| 2662               |            1|   21.679200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 2663               |            1|    7.229200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  20.000000|  20.000000|
| 2664               |            1|    7.225000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 2665               |            2|    7.227100|             2|           0|            1|              0|     1.0000000|           0|    0.5000000|      0.0000000|           1.0000000|  14.500000|  18.000000|
| 2666               |            4|    4.814575|             4|           0|            3|              0|     1.0000000|           0|    0.7500000|      0.0000000|           1.0000000|   0.750000|  24.000000|
| 2667               |            1|    7.225000|             1|           0|            1|              0|     1.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  15.000000|  15.000000|
| 2668               |            2|   11.179150|             2|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  33.515152|
| 2669               |            1|    7.229200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  22.000000|  22.000000|
| 26707              |            1|   26.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  44.000000|  44.000000|
| 2671               |            1|    7.229200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 2672               |            1|    7.225000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  25.000000|  25.000000|
| 2674               |            1|    7.225000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 2677               |            1|    7.229200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 2678               |            2|    7.622900|             2|           0|            1|              0|     1.0000000|           0|    0.5000000|      0.0000000|           1.0000000|   9.000000|  33.515152|
| 2680               |            1|   14.454200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  26.000000|  26.000000|
| 2683               |            1|    6.437500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  34.500000|  34.500000|
| 2685               |            1|    7.229200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  30.000000|  30.000000|
| 2686               |            1|    7.229200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 2687               |            1|    7.229200|             1|           0|            1|              0|     1.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  13.000000|  13.000000|
| 2689               |            1|   14.458300|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  33.515152|  33.515152|
| 2690               |            1|    7.229200|             0|           1|            1|              0|     0.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  17.000000|  17.000000|
| 2691               |            2|    7.227100|             2|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  45.000000|
| 2693               |            1|    7.229200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  23.500000|  23.500000|
| 2694               |            1|    7.225000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  30.000000|  30.000000|
| 2695               |            1|    7.229200|             0|           1|            1|              0|     0.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  15.000000|  15.000000|
| 2697               |            1|    7.229200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.500000|  28.500000|
| 2699               |            2|    9.393750|             0|           2|            1|              0|     0.0000000|           0|    0.5000000|      0.0000000|           0.5000000|  11.000000|  26.000000|
| 2700               |            1|    7.229200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 27042              |            1|   30.000000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  80.000000|  80.000000|
| 27267              |            1|   13.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  32.500000|  32.500000|
| 27849              |            1|   13.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  36.000000|  36.000000|
| 28134              |            1|   11.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  21.000000|  21.000000|
| 28206              |            1|   10.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  35.000000|  35.000000|
| 28213              |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  39.000000|  39.000000|
| 28220              |            1|   32.500000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  34.000000|  34.000000|
| 28228              |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  30.000000|  30.000000|
| 28403              |            2|   13.000000|             0|           2|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  46.000000|  54.000000|
| 28424              |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  19.000000|
| 28425              |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  23.000000|  23.000000|
| 28551              |            1|   13.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  36.000000|  36.000000|
| 28664              |            1|   21.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  34.000000|  34.000000|
| 28665              |            1|   10.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  19.000000|
| 29011              |            1|   14.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  54.000000|  54.000000|
| 2908               |            2|   13.000000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  19.000000|  32.000000|
| 29103              |            1|   23.000000|             1|           0|            1|              0|     1.0000000|           0|    1.0000000|      0.0000000|           1.0000000|   4.000000|   4.000000|
| 29104              |            1|   11.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  23.000000|  23.000000|
| 29105              |            1|   23.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  54.000000|  54.000000|
| 29106              |            3|    6.250000|             1|           2|            2|              0|     0.3333333|           0|    0.6666667|      0.0000000|           1.0000000|   0.830000|  24.000000|
| 29108              |            1|   11.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  18.000000|  18.000000|
| 2926               |            1|   26.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  29.000000|  29.000000|
| 29750              |            2|   19.500000|             1|           1|            0|              1|     0.5000000|           0|    0.0000000|      0.5000000|           0.5000000|  40.000000|  60.000000|
| 29751              |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  23.000000|  23.000000|
| 3101264            |            1|    6.495800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  34.000000|  34.000000|
| 3101265            |            1|    7.495800|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 3101267            |            1|    6.495800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  18.000000|  18.000000|
| 3101276            |            1|    7.925000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  37.000000|  37.000000|
| 3101277            |            1|    7.925000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.000000|  28.000000|
| 3101278            |            2|    7.925000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  32.000000|  33.000000|
| 3101281            |            1|    7.925000|             1|           0|            1|              0|     1.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  17.000000|  17.000000|
| 3101295            |            6|    6.614583|             1|           5|            5|              0|     0.1666667|           0|    0.8333333|      0.0000000|           1.0000000|   1.000000|  41.000000|
| 3101296            |            1|    7.925000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  39.000000|  39.000000|
| 3101298            |            1|   12.287500|             1|           0|            1|              0|     1.0000000|           0|    1.0000000|      0.0000000|           1.0000000|   2.000000|   2.000000|
| 31027              |            2|   10.500000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  30.000000|  34.000000|
| 31028              |            1|   10.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  26.000000|  26.000000|
| 312991             |            1|    7.775000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 312992             |            1|    7.775000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  21.000000|  21.000000|
| 312993             |            1|    7.775000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 31418              |            1|   13.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  40.000000|  40.000000|
| 315037             |            1|    8.662500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 315082             |            1|    7.875000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  29.000000|  29.000000|
| 315084             |            1|    8.662500|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  30.000000|  30.000000|
| 315086             |            1|    8.662500|             0|           1|            1|              0|     0.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  17.000000|  17.000000|
| 315088             |            1|    8.662500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.000000|  42.000000|
| 315089             |            1|    8.662500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  38.000000|  38.000000|
| 315090             |            1|    8.662500|             0|           1|            1|              0|     0.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  17.000000|  17.000000|
| 315093             |            1|    8.662500|             0|           1|            1|              0|     0.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  17.000000|  17.000000|
| 315094             |            1|    8.662500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  20.000000|  20.000000|
| 315096             |            1|    8.662500|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  20.000000|  20.000000|
| 315097             |            1|    8.662500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  21.000000|  21.000000|
| 315098             |            1|    8.662500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  27.000000|  27.000000|
| 315151             |            1|    8.662500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  26.000000|  26.000000|
| 315153             |            1|   22.025000|             1|           0|            1|              0|     1.0000000|           0|    1.0000000|      0.0000000|           1.0000000|   4.000000|   4.000000|
| 323592             |            1|    7.250000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 323951             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  19.000000|
| 324669             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  22.000000|  22.000000|
| 330877             |            1|    8.458300|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 330909             |            1|    7.629200|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 330919             |            1|    7.829200|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 330923             |            1|    8.029200|             1|           0|            1|              0|     1.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  15.000000|  15.000000|
| 330931             |            1|    7.879200|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 330932             |            1|    7.787500|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 330935             |            1|    8.137500|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 330958             |            1|    7.879200|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  19.000000|  19.000000|
| 330959             |            1|    7.879200|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 330979             |            1|    7.829200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 330980             |            1|    7.879200|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 334912             |            1|    7.733300|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 335097             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  31.000000|  31.000000|
| 335677             |            1|    7.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 33638              |            1|   81.858300|             0|           1|            1|              0|     0.0000000|           0|    1.0000000|      0.0000000|           1.0000000|   4.000000|   4.000000|
| 336439             |            1|    7.750000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  65.000000|  65.000000|
| 3411               |            1|    8.712500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 341826             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  26.000000|  26.000000|
| 34218              |            1|   10.500000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  27.000000|  27.000000|
| 342826             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  24.500000|  24.500000|
| 343095             |            1|    8.050000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  33.515152|  33.515152|
| 343120             |            1|    7.650000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  21.000000|  21.000000|
| 343275             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  24.000000|  24.000000|
| 343276             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  29.000000|  29.000000|
| 345364             |            1|    6.237500|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  61.000000|  61.000000|
| 345572             |            1|   17.400000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  36.000000|  36.000000|
| 345763             |            1|   18.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  31.000000|  31.000000|
| 345764             |            2|    9.000000|             1|           1|            1|              0|     0.5000000|           0|    0.5000000|      0.0000000|           1.0000000|  16.000000|  18.000000|
| 345765             |            1|    9.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  47.000000|  47.000000|
| 345767             |            1|    9.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  22.000000|  22.000000|
| 345769             |            1|    9.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  20.000000|  20.000000|
| 345770             |            1|    9.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.000000|  28.000000|
| 345773             |            3|    8.050000|             2|           1|            1|              0|     0.6666667|           0|    0.3333333|      0.0000000|           0.6666667|  10.000000|  36.000000|
| 345774             |            1|    9.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  30.000000|  30.000000|
| 345777             |            1|    9.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 345778             |            1|    9.500000|             0|           1|            1|              0|     0.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  16.000000|  16.000000|
| 345779             |            1|    9.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  29.000000|  29.000000|
| 345780             |            1|    9.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  33.000000|  33.000000|
| 345781             |            1|    9.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  24.000000|  24.000000|
| 345783             |            1|    9.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.000000|  28.000000|
| 3460               |            1|    7.045800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  29.000000|  29.000000|
| 347054             |            2|    5.231250|             2|           0|            1|              0|     1.0000000|           0|    0.5000000|      0.0000000|           1.0000000|   2.000000|  29.000000|
| 347060             |            1|    7.775000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  74.000000|  74.000000|
| 347061             |            1|    6.975000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  45.000000|  45.000000|
| 347062             |            1|    7.775000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  33.000000|  33.000000|
| 347063             |            1|    7.775000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  31.000000|  31.000000|
| 347064             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  51.000000|  51.000000|
| 347067             |            1|    7.775000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  29.000000|  29.000000|
| 347068             |            1|    7.775000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  26.000000|  26.000000|
| 347069             |            1|    7.775000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  19.000000|
| 347071             |            1|    7.775000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  25.000000|  25.000000|
| 347073             |            1|    7.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  45.000000|  45.000000|
| 347074             |            1|    7.775000|             0|           1|            1|              0|     0.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  16.000000|  16.000000|
| 347076             |            1|    7.775000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  25.000000|  25.000000|
| 347077             |            4|    7.846875|             2|           2|            3|              0|     0.5000000|           0|    0.7500000|      0.0000000|           1.0000000|   3.000000|  38.000000|
| 347078             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  18.000000|  18.000000|
| 347080             |            2|    7.200000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  28.000000|  34.000000|
| 347081             |            1|    7.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  22.000000|  22.000000|
| 347082             |            7|    4.467857|             5|           2|            5|              0|     0.7142857|           0|    0.7142857|      0.0000000|           0.8571429|   2.000000|  39.000000|
| 347083             |            1|    7.775000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  25.000000|  25.000000|
| 347085             |            1|    7.775000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  22.000000|  22.000000|
| 347087             |            1|    7.775000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 347088             |            6|    4.650000|             3|           3|            4|              0|     0.5000000|           0|    0.6666667|      0.0000000|           0.8333333|   2.000000|  45.000000|
| 347089             |            1|    6.975000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  27.000000|  27.000000|
| 3474               |            1|    7.887500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  26.000000|  26.000000|
| 347464             |            1|    7.854200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.000000|  28.000000|
| 347466             |            1|    7.854200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  20.000000|  20.000000|
| 347468             |            1|    7.854200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  23.000000|  23.000000|
| 347470             |            1|    7.854200|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  26.000000|  26.000000|
| 347742             |            3|    3.711100|             2|           1|            2|              0|     0.6666667|           0|    0.6666667|      0.0000000|           1.0000000|   1.000000|  27.000000|
| 347743             |            1|    7.054200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  51.000000|  51.000000|
| 348121             |            1|    7.650000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.000000|  42.000000|
| 348123             |            1|    7.650000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  25.000000|  25.000000|
| 348124             |            1|    7.650000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  19.000000|
| 349201             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 349203             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  25.000000|  25.000000|
| 349204             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  23.000000|  23.000000|
| 349205             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  19.000000|
| 349206             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  22.000000|  22.000000|
| 349207             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.000000|  28.000000|
| 349208             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 349209             |            1|    7.495800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  24.000000|  24.000000|
| 349210             |            1|    7.495800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  36.000000|  36.000000|
| 349212             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  19.000000|
| 349213             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  35.000000|  35.000000|
| 349214             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 349215             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 349216             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 349217             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 349218             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 349219             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  27.000000|  27.000000|
| 349221             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 349222             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 349223             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 349224             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  26.000000|  26.000000|
| 349225             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 349227             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 349228             |            1|   10.170800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  19.000000|
| 349231             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  19.000000|
| 349233             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  24.000000|  24.000000|
| 349234             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 349236             |            1|    8.850000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  24.000000|  24.000000|
| 349237             |            2|    8.900000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  18.000000|  25.000000|
| 349239             |            1|    8.662500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  33.000000|  33.000000|
| 349240             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  29.000000|  29.000000|
| 349241             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  33.000000|  33.000000|
| 349242             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  32.000000|  32.000000|
| 349243             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.000000|  28.000000|
| 349244             |            1|    8.683300|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  31.000000|  31.000000|
| 349245             |            1|    7.895800|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  28.000000|  28.000000|
| 349246             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  30.000000|  30.000000|
| 349247             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  36.000000|  36.000000|
| 349248             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  26.000000|  26.000000|
| 349249             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  38.000000|  38.000000|
| 349251             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  40.000000|  40.000000|
| 349252             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  22.000000|  22.000000|
| 349253             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 349254             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 349256             |            1|   13.416700|             1|           0|            1|              0|     1.0000000|           0|    1.0000000|      0.0000000|           1.0000000|   4.000000|   4.000000|
| 349257             |            1|    7.895800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  33.000000|  33.000000|
| 349909             |            4|    5.268750|             3|           1|            3|              0|     0.7500000|           0|    0.7500000|      0.0000000|           1.0000000|   2.000000|  29.000000|
| 349910             |            1|   15.550000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  36.000000|  36.000000|
| 349912             |            1|    7.775000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  18.000000|  18.000000|
| 350025             |            1|    7.854200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  26.000000|  26.000000|
| 350026             |            1|   14.108300|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  41.000000|  41.000000|
| 350029             |            1|    7.854200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  21.000000|  21.000000|
| 350034             |            1|    7.795800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  21.000000|  21.000000|
| 350035             |            1|    7.795800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  24.000000|  24.000000|
| 350036             |            1|    7.795800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  18.000000|  18.000000|
| 350042             |            1|    7.795800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.000000|  28.000000|
| 350043             |            1|    7.795800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  27.000000|  27.000000|
| 350046             |            1|    7.854200|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  19.000000|  19.000000|
| 350047             |            1|    7.854200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  48.000000|  48.000000|
| 350048             |            1|    7.054200|             0|           1|            1|              0|     0.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  17.000000|  17.000000|
| 350050             |            1|    7.854200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  20.000000|  20.000000|
| 350052             |            1|    7.795800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  22.000000|  22.000000|
| 350060             |            1|    7.520800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  22.000000|  22.000000|
| 350404             |            1|    7.854200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  18.000000|  18.000000|
| 350406             |            1|    7.854200|             1|           0|            1|              0|     1.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  14.000000|  14.000000|
| 350407             |            1|    7.854200|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  31.000000|  31.000000|
| 350417             |            1|    7.854200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  32.000000|  32.000000|
| 35273              |            3|   37.758333|             2|           1|            0|              1|     0.6666667|           0|    0.0000000|      0.3333333|           0.6666667|  23.000000|  58.000000|
| 35281              |            2|   38.643750|             0|           2|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  21.000000|  54.000000|
| 35851              |            1|    7.733300|             1|           0|            1|              0|     1.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  16.000000|  16.000000|
| 35852              |            1|    7.733300|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 358585             |            2|    7.250000|             0|           2|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  28.724891|
| 36209              |            1|    7.725000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 362316             |            1|    7.250000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 363291             |            3|    6.841667|             1|           2|            1|              0|     0.3333333|           0|    0.3333333|      0.0000000|           0.6666667|   9.000000|  33.000000|
| 363294             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  34.000000|  34.000000|
| 363592             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  44.000000|  44.000000|
| 364498             |            1|   14.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 364499             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  30.500000|  30.500000|
| 364500             |            1|    7.250000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  59.000000|  59.000000|
| 364506             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  34.000000|  34.000000|
| 364511             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  44.000000|  44.000000|
| 364512             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  35.000000|  35.000000|
| 364516             |            2|    6.237500|             2|           0|            1|              0|     1.0000000|           0|    0.5000000|      0.0000000|           1.0000000|   5.000000|  30.000000|
| 364846             |            1|    7.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  21.000000|  21.000000|
| 364848             |            1|    7.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 364849             |            2|    7.750000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  32.000000|  40.000000|
| 364850             |            1|    7.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  30.500000|  30.500000|
| 364851             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 365222             |            1|    6.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  19.000000|
| 365226             |            1|    6.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 36568              |            1|   15.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 367226             |            2|   11.625000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  18.000000|  28.724891|
| 367228             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 367229             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 367230             |            2|    7.750000|             2|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 367231             |            1|    7.750000|             1|           0|            1|              0|     1.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  16.000000|  16.000000|
| 367232             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  40.500000|  40.500000|
| 367655             |            1|    7.729200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 368323             |            1|    6.950000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 36864              |            1|    7.741700|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  25.000000|  25.000000|
| 36865              |            1|    7.737500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 36866              |            1|    7.737500|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 368703             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 36928              |            2|   82.433350|             2|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  31.000000|  45.000000|
| 36947              |            2|   39.133350|             2|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  52.000000|  54.000000|
| 36963              |            1|   32.320800|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  61.000000|  61.000000|
| 36967              |            1|   34.020800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  47.000000|  47.000000|
| 36973              |            2|   41.737500|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  35.000000|  45.000000|
| 370129             |            2|   10.106250|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  18.000000|  41.000000|
| 370365             |            2|    7.750000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  28.724891|  33.515152|
| 370369             |            1|    7.750000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  70.500000|  70.500000|
| 370370             |            1|    7.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 370371             |            1|   15.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 370372             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 370373             |            1|    7.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  22.000000|  22.000000|
| 370375             |            1|    7.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 370376             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  32.000000|  32.000000|
| 370377             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 371060             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 371110             |            3|    8.050000|             1|           2|            0|              0|     0.3333333|           0|    0.0000000|      0.0000000|           0.3333333|  18.000000|  28.724891|
| 371362             |            1|   16.100000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  44.000000|  44.000000|
| 372622             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 373450             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  35.000000|  35.000000|
| 374746             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 374887             |            1|    7.250000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  25.000000|  25.000000|
| 374910             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 376564             |            2|    8.050000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  28.724891|  33.515152|
| 376566             |            1|   16.100000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  24.000000|  24.000000|
| 382649             |            1|    7.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 382651             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  29.000000|  29.000000|
| 382652             |            5|    5.825000|             1|           4|            4|              0|     0.2000000|           0|    0.8000000|      0.0000000|           1.0000000|   2.000000|  39.000000|
| 383121             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 384461             |            1|    7.750000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 386525             |            1|   16.100000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  33.515152|  33.515152|
| 392091             |            1|    9.350000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 392092             |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 392096             |            2|    6.237500|             1|           1|            1|              0|     0.5000000|           0|    0.5000000|      0.0000000|           1.0000000|   6.000000|  27.000000|
| 394140             |            1|    6.858300|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 4133               |            4|    6.366675|             3|           1|            1|              0|     0.7500000|           0|    0.2500000|      0.0000000|           1.0000000|   5.350833|  18.000000|
| 4134               |            1|    9.587500|             1|           0|            0|              1|     1.0000000|           0|    0.0000000|      1.0000000|           1.0000000|  63.000000|  63.000000|
| 4135               |            1|    9.587500|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  37.000000|  37.000000|
| 4136               |            1|    9.825000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  20.000000|  20.000000|
| 4137               |            1|    9.825000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  21.000000|  21.000000|
| 4138               |            1|    9.841700|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| 4579               |            1|    8.404200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.000000|  42.000000|
| 54636              |            2|    8.050000|             0|           2|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  21.000000|  28.500000|
| 5727               |            1|   25.587500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  47.000000|  47.000000|
| 65303              |            1|   19.966700|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 65304              |            1|   19.966700|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 65306              |            1|    8.112500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| 6563               |            1|    9.225000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  20.000000|  20.000000|
| 693                |            1|   26.000000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  64.000000|  64.000000|
| 695                |            1|    5.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  33.000000|  33.000000|
| 7267               |            1|    9.225000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  23.000000|  23.000000|
| 7534               |            2|    4.922900|             0|           2|            1|              0|     0.0000000|           0|    0.5000000|      0.0000000|           0.5000000|  16.000000|  20.000000|
| 7540               |            1|    8.654200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  33.000000|  33.000000|
| 7545               |            1|    9.483300|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  29.000000|  29.000000|
| 7546               |            1|    9.475000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  40.000000|  40.000000|
| 7552               |            1|   10.516700|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  22.000000|  22.000000|
| 7553               |            1|    9.837500|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  22.000000|  22.000000|
| 7598               |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  45.000000|  45.000000|
| 8471               |            1|    8.362500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  32.000000|  32.000000|
| 8475               |            1|    8.433300|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  21.000000|  21.000000|
| 9234               |            1|    7.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| A./5. 2152         |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  21.000000|  21.000000|
| A./5. 3235         |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| A.5. 11206         |            1|    8.050000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  55.500000|  55.500000|
| A.5. 18509         |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  30.000000|  30.000000|
| A/4 45380          |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  21.000000|  21.000000|
| A/4 48871          |            2|   12.075000|             0|           2|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  24.000000|  39.000000|
| A/4. 20589         |            1|    8.050000|             0|           1|            1|              0|     0.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  16.000000|  16.000000|
| A/4. 34244         |            1|    7.250000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| A/4. 39886         |            1|    7.800000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  21.000000|  21.000000|
| A/5 21171          |            1|    7.250000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  22.000000|  22.000000|
| A/5 21172          |            1|    7.250000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  22.000000|  22.000000|
| A/5 21173          |            1|    7.250000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  20.500000|  20.500000|
| A/5 21174          |            1|    7.250000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  22.000000|  22.000000|
| A/5 2466           |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| A/5 2817           |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| A/5 3536           |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  43.000000|  43.000000|
| A/5 3540           |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  18.000000|  18.000000|
| A/5 3594           |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  50.000000|  50.000000|
| A/5 3902           |            1|    7.250000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  47.000000|  47.000000|
| A/5. 10482         |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  19.000000|
| A/5. 13032         |            1|    7.733300|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  21.000000|  21.000000|
| A/5. 2151          |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  20.000000|  20.000000|
| A/5. 3336          |            2|    8.050000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  26.000000|  30.000000|
| A/5. 3337          |            1|   14.500000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  47.000000|  47.000000|
| A/5. 851           |            1|   14.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  40.500000|  40.500000|
| A/S 2816           |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| A4. 54510          |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| C 17369            |            1|    7.141700|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  24.000000|  24.000000|
| C 4001             |            1|   22.525000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.000000|  28.000000|
| C 7075             |            1|    6.450000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  43.000000|  43.000000|
| C 7076             |            1|    7.250000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  30.000000|  30.000000|
| C 7077             |            1|    7.250000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  22.000000|  22.000000|
| C.A. 17248         |            1|   10.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  36.000000|  36.000000|
| C.A. 18723         |            1|   10.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  31.000000|  31.000000|
| C.A. 2315          |            2|   10.287500|             0|           2|            1|              0|     0.0000000|           0|    0.5000000|      0.0000000|           0.5000000|   1.000000|  26.000000|
| C.A. 24579         |            1|   10.500000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  66.000000|  66.000000|
| C.A. 24580         |            1|   10.500000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  70.000000|  70.000000|
| C.A. 2673          |            2|   10.125000|             1|           1|            1|              0|     0.5000000|           0|    0.5000000|      0.0000000|           1.0000000|  16.000000|  35.000000|
| C.A. 29178         |            1|   13.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  25.000000|  25.000000|
| C.A. 29395         |            1|   10.500000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  29.000000|  29.000000|
| C.A. 29566         |            1|   10.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  24.000000|  24.000000|
| C.A. 31026         |            1|   10.500000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  21.000000|  21.000000|
| C.A. 31921         |            3|    8.750000|             2|           1|            1|              0|     0.6666667|           0|    0.3333333|      0.0000000|           0.6666667|   8.000000|  31.000000|
| C.A. 33111         |            1|   10.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  32.000000|  32.000000|
| C.A. 33112         |            2|   18.375000|             0|           2|            1|              0|     0.0000000|           0|    0.5000000|      0.0000000|           0.5000000|   8.000000|  19.000000|
| C.A. 33595         |            1|   15.750000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  40.000000|  40.000000|
| C.A. 34260         |            1|   10.500000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  34.000000|  34.000000|
| C.A. 34651         |            3|    9.250000|             2|           1|            1|              0|     0.6666667|           0|    0.3333333|      0.0000000|           0.6666667|   5.000000|  36.000000|
| C.A. 37671         |            2|    7.950000|             0|           2|            2|              0|     0.0000000|           0|    1.0000000|      0.0000000|           1.0000000|   3.000000|   9.000000|
| C.A. 5547          |            1|    7.550000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.000000|  42.000000|
| C.A. 6212          |            1|   15.100000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| C.A./SOTON 34068   |            1|   10.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.000000|  28.000000|
| CA 2144            |            6|    7.816667|             2|           4|            5|              0|     0.3333333|           0|    0.8333333|      0.0000000|           1.0000000|   1.000000|  43.000000|
| CA. 2314           |            1|    7.550000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  23.000000|  23.000000|
| CA. 2343           |            7|    9.935714|             3|           4|            1|              0|     0.4285714|           0|    0.1428571|      0.0000000|           0.5714286|   5.350833|  28.724891|
| F.C. 12750         |            1|   52.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  31.000000|  31.000000|
| F.C.C. 13528       |            1|   21.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  35.000000|  35.000000|
| F.C.C. 13529       |            3|    8.750000|             2|           1|            1|              0|     0.6666667|           0|    0.3333333|      0.0000000|           0.6666667|   7.000000|  45.000000|
| F.C.C. 13531       |            1|   10.500000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  50.000000|  50.000000|
| Fa 265302          |            1|    7.312500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| LINE               |            4|    0.000000|             0|           4|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  49.000000|
| P/PP 3381          |            2|   12.000000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  28.000000|  30.000000|
| PC 17318           |            1|   25.925000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| PC 17473           |            1|   26.287500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  36.000000|  36.000000|
| PC 17474           |            1|   26.387500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  36.000000|  36.000000|
| PC 17475           |            1|   26.287500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  35.000000|  35.000000|
| PC 17476           |            1|   26.287500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.000000|  42.000000|
| PC 17477           |            2|   34.650000|             2|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  24.000000|  24.000000|
| PC 17482           |            1|   49.504200|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  24.000000|  24.000000|
| PC 17483           |            1|  221.779200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| PC 17485           |            2|   28.464600|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  30.000000|  49.000000|
| PC 17558           |            2|  123.760400|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  24.000000|  50.000000|
| PC 17569           |            2|   73.260400|             2|           0|            0|              1|     1.0000000|           0|    0.0000000|      0.5000000|           1.0000000|  40.631579|  58.000000|
| PC 17572           |            3|   25.576400|             1|           2|            0|              0|     0.3333333|           0|    0.0000000|      0.0000000|           0.3333333|  27.000000|  49.000000|
| PC 17582           |            3|   51.154167|             2|           1|            0|              1|     0.6666667|           0|    0.0000000|      0.3333333|           0.6666667|  38.000000|  58.000000|
| PC 17585           |            1|   79.200000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  40.631579|  40.631579|
| PC 17590           |            1|   50.495800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  31.000000|  31.000000|
| PC 17592           |            1|   39.400000|             1|           0|            1|              0|     1.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  16.000000|  16.000000|
| PC 17593           |            2|   39.600000|             0|           2|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  24.000000|  46.000000|
| PC 17595           |            1|   28.712500|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  50.000000|  50.000000|
| PC 17596           |            1|   29.700000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  37.000000|  37.000000|
| PC 17597           |            1|   61.379200|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  51.000000|  51.000000|
| PC 17599           |            1|   71.283300|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  38.000000|  38.000000|
| PC 17600           |            1|   30.695800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| PC 17601           |            1|   27.720800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  40.000000|  40.000000|
| PC 17603           |            1|   59.400000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  54.000000|  54.000000|
| PC 17604           |            2|   41.085400|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  28.000000|  40.631579|
| PC 17605           |            1|   27.720800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| PC 17608           |            2|  131.187500|             2|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  21.000000|
| PC 17609           |            1|   49.504200|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  71.000000|  71.000000|
| PC 17610           |            1|   27.720800|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  44.000000|  44.000000|
| PC 17611           |            2|   66.825000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  40.631579|  50.000000|
| PC 17612           |            1|   27.720800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  42.184210|  42.184210|
| PC 17754           |            1|   34.654200|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  71.000000|  71.000000|
| PC 17755           |            3|  170.776400|             1|           2|            0|              0|     0.3333333|           0|    0.0000000|      0.0000000|           0.3333333|  35.000000|  36.000000|
| PC 17756           |            1|   83.158300|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  39.000000|  39.000000|
| PC 17757           |            4|   56.881250|             3|           1|            0|              0|     0.7500000|           0|    0.0000000|      0.0000000|           0.7500000|  18.000000|  42.184210|
| PC 17758           |            2|   54.450000|             1|           1|            1|              0|     0.5000000|           0|    0.5000000|      0.0000000|           0.5000000|  17.000000|  18.000000|
| PC 17759           |            1|   63.358300|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  23.000000|  23.000000|
| PC 17760           |            3|   45.211100|             2|           1|            0|              0|     0.6666667|           0|    0.0000000|      0.0000000|           0.6666667|  22.000000|  36.000000|
| PC 17761           |            2|   53.212500|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  30.000000|  50.000000|
| PP 4348            |            1|    9.350000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  22.000000|  22.000000|
| PP 9549            |            2|    8.350000|             2|           0|            1|              0|     1.0000000|           0|    0.5000000|      0.0000000|           1.0000000|   4.000000|  24.000000|
| S.C./A.4. 23567    |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| S.C./PARIS 2079    |            2|   18.502100|             0|           2|            1|              0|     0.0000000|           0|    0.5000000|      0.0000000|           0.5000000|   1.000000|  31.000000|
| S.O./P.P. 3        |            2|    5.250000|             1|           1|            1|              1|     0.5000000|           0|    0.5000000|      0.5000000|           1.0000000|  16.000000|  57.000000|
| S.O./P.P. 751      |            1|    7.550000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| S.O.C. 14879       |            5|   14.700000|             0|           5|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  18.000000|  32.000000|
| S.O.P. 1166        |            1|   12.525000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  51.000000|  51.000000|
| S.P. 3464          |            1|    8.158300|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  19.000000|
| S.W./PP 752        |            1|   10.500000|             0|           1|            0|              1|     0.0000000|           0|    0.0000000|      1.0000000|           0.0000000|  62.000000|  62.000000|
| SC 1748            |            1|   12.000000|             1|           0|            1|              0|     1.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  17.000000|  17.000000|
| SC/AH 29037        |            1|   26.000000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  37.000000|  37.000000|
| SC/AH 3085         |            1|   26.000000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  42.000000|  42.000000|
| SC/AH Basle 541    |            1|   13.791700|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  23.000000|  23.000000|
| SC/Paris 2123      |            3|   13.859733|             2|           1|            1|              0|     0.6666667|           0|    0.3333333|      0.0000000|           0.6666667|   3.000000|  25.000000|
| SC/PARIS 2131      |            1|   15.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  33.588889|  33.588889|
| SC/PARIS 2133      |            1|   15.045800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  23.000000|  23.000000|
| SC/PARIS 2146      |            1|   13.862500|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  33.588889|  33.588889|
| SC/PARIS 2149      |            1|   13.858300|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  27.000000|  27.000000|
| SC/Paris 2163      |            1|   12.875000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  36.000000|  36.000000|
| SC/PARIS 2167      |            1|   27.720800|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  29.000000|  29.000000|
| SCO/W 1585         |            1|   12.275000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  33.000000|  33.000000|
| SO/C 14885         |            1|   10.500000|             1|           0|            1|              0|     1.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  17.000000|  17.000000|
| SOTON/O.Q. 3101305 |            1|    7.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| SOTON/O.Q. 3101306 |            1|    7.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  38.000000|  38.000000|
| SOTON/O.Q. 3101307 |            1|    7.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  20.000000|  20.000000|
| SOTON/O.Q. 3101310 |            1|    7.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  35.000000|  35.000000|
| SOTON/O.Q. 3101311 |            1|    7.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  24.000000|  24.000000|
| SOTON/O.Q. 3101312 |            1|    7.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  25.000000|  25.000000|
| SOTON/O.Q. 392078  |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  32.000000|  32.000000|
| SOTON/O.Q. 392087  |            1|    8.050000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| SOTON/O2 3101272   |            1|    7.125000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  41.000000|  41.000000|
| SOTON/O2 3101287   |            1|    7.925000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  20.000000|  20.000000|
| SOTON/OQ 3101316   |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| SOTON/OQ 3101317   |            1|    7.250000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  21.000000|  21.000000|
| SOTON/OQ 392076    |            1|    7.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  25.000000|  25.000000|
| SOTON/OQ 392082    |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| SOTON/OQ 392086    |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  28.724891|  28.724891|
| SOTON/OQ 392089    |            1|    8.050000|             0|           1|            1|              0|     0.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  16.000000|  16.000000|
| SOTON/OQ 392090    |            1|    8.050000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  30.000000|  30.000000|
| STON/O 2. 3101269  |            1|    7.925000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  44.000000|  44.000000|
| STON/O 2. 3101273  |            1|    7.125000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  35.000000|  35.000000|
| STON/O 2. 3101274  |            1|    7.125000|             0|           1|            1|              0|     0.0000000|           0|    1.0000000|      0.0000000|           1.0000000|  17.000000|  17.000000|
| STON/O 2. 3101275  |            1|    7.125000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  22.000000|  22.000000|
| STON/O 2. 3101280  |            1|    7.925000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  21.000000|  21.000000|
| STON/O 2. 3101285  |            1|    7.925000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  20.000000|  20.000000|
| STON/O 2. 3101286  |            1|    7.925000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  32.000000|  32.000000|
| STON/O 2. 3101288  |            1|    7.925000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  31.000000|  31.000000|
| STON/O 2. 3101289  |            1|    7.925000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  39.000000|  39.000000|
| STON/O 2. 3101292  |            1|    7.925000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  32.000000|  32.000000|
| STON/O 2. 3101293  |            1|    7.925000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  32.000000|  32.000000|
| STON/O 2. 3101294  |            1|    7.925000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  21.000000|  21.000000|
| STON/O2. 3101271   |            1|    7.925000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  25.000000|  25.000000|
| STON/O2. 3101279   |            2|    7.925000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  24.000000|  28.000000|
| STON/O2. 3101282   |            1|    7.925000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  26.000000|  26.000000|
| STON/O2. 3101283   |            1|    7.925000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  27.000000|  27.000000|
| STON/O2. 3101290   |            1|    7.925000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  23.000000|  23.000000|
| SW/PP 751          |            1|   10.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  19.000000|  19.000000|
| W./C. 14258        |            1|   10.500000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  50.000000|  50.000000|
| W./C. 14263        |            1|   10.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  29.000000|  29.000000|
| W./C. 6607         |            2|   11.725000|             1|           1|            0|              0|     0.5000000|           0|    0.0000000|      0.0000000|           0.5000000|  18.000000|  28.724891|
| W./C. 6608         |            4|    8.593750|             3|           1|            2|              0|     0.7500000|           0|    0.5000000|      0.0000000|           1.0000000|   9.000000|  48.000000|
| W./C. 6609         |            1|    7.550000|             1|           0|            0|              0|     1.0000000|           0|    0.0000000|      0.0000000|           1.0000000|  18.000000|  18.000000|
| W.E.P. 5734        |            1|   61.175000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  46.000000|  46.000000|
| W/C 14208          |            1|   10.500000|             0|           1|            0|              0|     0.0000000|           0|    0.0000000|      0.0000000|           0.0000000|  30.000000|  30.000000|
| WE/P 5735          |            2|   35.500000|             1|           1|            0|              1|     0.5000000|           0|    0.0000000|      0.5000000|           0.5000000|  36.000000|  70.000000|
