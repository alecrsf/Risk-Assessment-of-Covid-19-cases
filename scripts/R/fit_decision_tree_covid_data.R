# Load packages

library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(caret)

#### Code assumes that the data set is loaded as a dataframe into the variable covid_data ####

library(tidyverse)

setwd("~/Desktop/ML Systems for Data Science/Project")

covid_data <- readxl::read_excel("data/CFR_data_04_04_20.xlsx")

#remove country and total deaths
covid_data <- covid_data %>%
	select(-Country, -'Total deaths', -'Climate zones')

#### Fit a tree
mydt <- rpart(CFR~.,  
							data = covid_data, 
							method = "anova",
							minsplit = 5)

mydt$control

mydt$variable.importance
##### Print summary of the tree

summary(mydt)

#### Plot tree

httpgd::hgd()
httpgd::hgd_browse()
fancyRpartPlot(mydt, caption=NULL)

#### Accuracy of the fitted tree

predicted_mydt <- predict(mydt)
RMSE(predicted_mydt, covid_data$CFR)

