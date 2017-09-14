# Titanic_ML

# Load dplyr and stringr libraries

library(dplyr)
library(stringr)

# Load titanic_train.csv data

titanic_train <- read_csv("C:/Users/narce/OneDrive/Documents/GitHub/Titanic/titanic_train.csv")

# Assign dataset to train

train <- titanic_train

# View data

# View the structure of the data with str

# Filter for passengers that paid zero fare - assign to zero.fare

# Calculate the total fare by Pclass - assign zero.fare to zero.fare.pclass - use groupby, summarise & arrange

# Extract passengers by title (Mr. Mrs. Miss. Rev.) - use str_extract and regex

# Show table of train/title column

# Map exisitng tiles to new title list - create new data.frame and assign it to titles.lookup. Create new vectors by title from (title column). Then define new gender titles (Mr. Mrs...)

# View titles.lookup

# Replace Titles with values in the lookup table using the join.left function

# Replace the New.Tile variable with Tile then delete the New.Title variable. 

train <- train %>%
  mutate(Title = New.Title) %>%
  select(-New.Title)

# Find all gender errors - by filtering females with male titles and males with female titles - use new title names

train %>%
  filter((Sex == "female" & (Title == "Mr." | Title == "Master.")) |
         (Sex == "male" & (Title == "Mrs." | Title == "Miss.")))

# Change title from "Dr." to "Mrs." using PassengerID

train$Title[train$PassengerId == 797] <- "Mrs."

# Generate summary stats for passengers with the title "Mr." by Fare and Pclass (min, max, mean, median, var, SD,IQR)

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

# Creating tracking feature for fare variable and assign to fare.zero and if ifelse statment - 0 = Y, Else N 

train$Fare.Zero <- ifelse(train$Fare == 0.0, "Y", "N")

View(train)

# Create lookup table for zero fare values using filter, group_by, and summarise - assign to zero.fare.lookup

zero.fare.lookup <- train %>%
  filter(Title == "Mr.") %>%
  group_by(Pclass, Title) %>%
  summarise(New.Fare = median(Fare))

# Impute zero fares using the lookup table and left_join. Replace zero fares with the median value per Pclass

train <- train %>%
  left_join(zero.fare.lookup, by = c("Pclass", "Title")) %>%
  mutate(fare = ifelse(Fare == 0.0, New.Fare, Fare)) %>%
  select(-New.Fare)

# Take a closer look at the age variable

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

# Creating tracking feature for fare variable and assign to fare.zero and if ifelse statment - 0 = Y, Else N 

train$Age.Missing <- ifelse(is.na(train$Age), "Y", "N")

# Create lookup table for age variable with missing values using select

age.lookup <- age.stats %>%
  select(Pclass, Title, Age.Mean, Age.Median)

# Impute missing ages by using a lookup table

train <- train %>%
  left_join(age.lookup, by = c("Pclass", "Title")) %>%
  mutate(Age = ifelse(Age.Missing == "Y",
                      ifelse(Title == "Miss.", Age.Median, Age.Mean),
                      Age)) %>%
  select(-Age.Median, -Age.Mean)

# Create Ticket-based features - Group_by ticket, summarise group.count, Avg fare = max fare / n(), sum of Female.Count, ratio of the n() of males in ticket count / number of people on that ticket.

ticket.lookup <- train %>%
  group_by(Ticket) %>%
  summarise(Group.Count = n(),
            Ave.Fare = max(Fare) / n(),
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

# Double check the work

ticket.lookup %>% filter(Ticket == "3101295")

View(train %>% filter(Ticket == "3101295"))


# Populate train data with ticket lookup table

train <- train %>%
  left_join(ticket.lookup, by = "Ticket")
View(train %>% filter(Ticket == "3101295"))

# Load install the ggplot2 library - load ggplot2
library(ggplot2)

# Create factors for Survived and Pclass

train$Survived <- as.factor(train$Survived)
train$Pclass <- as.factor(train$Pclass)

# Create a subset based on passengers traveling with children

tickets.children <- train %>%
  filter(Child.Count > 0)

# Visualize the with children subset

ggplot(tickets.children, aes(x = Pclass, fill = Survived)) + 
  theme_bw() +
  geom_bar() +
  facet_wrap(~ Title) +
  labs(y = "Count of Passengers",
       title = "Survival Rates for Ticket Group Traveling with Children")

# Create subset based on passengers traveling without children

tickets.no.children <- train %>%
  filter(Child.Count == 0)

# Visualize the no children subset

ggplot(tickets.no.children, aes(x = Pclass, fill = Survived)) +
  theme_bw() +
  geom_bar() +
  facet_wrap(~ Title) +
  labs(y = "Count of Passengers",
title = "Survival Rates for Ticket Groups Traveling without Chil
