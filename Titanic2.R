# Load packages

library(dplyr)
library(tidyr)
library(ggplot2)

# Load data

titanic_train <- read_csv("C:/Users/narce/OneDrive/Documents/GitHub/Titanic/titanic_train.csv")

# Assign titanic_train dataset to train

train <- titanic_train

# Review data with view, summary and str

View(train)
summary(train)
str(train)

# Create a subset of all Fares that had a zero fare

zero.fare <- train %>%
  filter(Fare == 0.0)
View(zero.fare)

# Create a subset of the number of suvivors 

survivors <- train %>% 
  filter(Survived == 1) %>%
  summarise(Survived = n())

# Create a subset of the number of women that survived

survivors.female <- train %>%
  filter(Survivors == 1) %>%
  filer(sex == female) %>%
  summarise(survivors.female = n())



