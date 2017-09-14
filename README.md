# Titanic_ML

#### Load dplyr and stringr libraries

#### Load titanic_train.csv data

#### Assign dataset to train

#### View data

#### View the structure of the data with str

#### Filter for passengers that paid zero fare - assign to zero.fare

#### Calculate the total fare by Pclass - assign zero.fare to zero.fare.pclass - use groupby, summarise & arrange

#### Extract passengers by title (Mr. Mrs. Miss. Rev.) - use str_extract and regex

#### Show table of train/title column

#### Map exisitng tiles to new title list - create new data.frame and assign it to titles.lookup. Create new vectors by title from (title column). Then define new gender titles (Mr. Mrs...)

#### View titles.lookup

#### Replace Titles with values in the lookup table using the join.left function

#### Replace the New.Tile variable with Tile then delete the New.Title variable. 

#### Find all gender errors - by filtering females with male titles and males with female titles - use new title names

#### Change title from "Dr." to "Mrs." using PassengerID

#### Generate summary stats for passengers with the title "Mr." by Fare and Pclass (min, max, mean, median, var, SD,IQR)

#### Creating tracking feature for fare variable and assign to fare.zero and if ifelse statment - 0 = Y, Else N 

#### Create lookup table for zero fare values using filter, group_by, and summarise - assign to zero.fare.lookup

#### Impute zero fares using the lookup table and left_join. Replace zero fares with the median value per Pclass

#### Take a closer look at the age variable

#### Creating tracking feature for fare variable and assign to fare.zero and if ifelse statment - 0 = Y, Else N 

#### Create lookup table for age variable with missing values using select

#### Impute missing ages by using a lookup table

#### Create Ticket-based features - Group_by ticket, summarise group.count, Avg fare = max fare / n(), sum of Female.Count, ratio of the n() of males in ticket count / number of people on that ticket.

#### Double check the work

#### Populate train data with ticket lookup table

#### Load install the ggplot2 library - load ggplot2

#### Create factors for Survived and Pclass

#### Create a subset based on passengers traveling with children

#### Visualize the with children subset

#### Create subset based on passengers traveling without children

#### Visualize the no children subset
