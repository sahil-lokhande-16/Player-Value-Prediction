# Install required packages
install.packages(c("readxl", "ggplot2", "tidyverse", "caret", "car", "scales",
                   "lmtest", "rpart", "randomForest", "xgboost", "e1071"))

# Load required libraries
library(readxl)
library(ggplot2)
library(caret)
library(car)
library(tidyverse)
library(dplyr)
library(lmtest)
library(rpart)
library(randomForest)
library(xgboost)
library(e1071)

# Load the dataset
playerData <- read.csv(file.choose())  # select fifa_players.csv file

# Display summary of the dataset
summary(playerData)

# Disable scientific notation
options(scipen = 999)

# Count missing values in each column
na_counts <- colSums(sapply(playerData, is.na))

# Summarize NA, NaN, and blank counts
result <- playerData %>%
  summarise_all(list(
    NA_Count = ~sum(is.na(.)),
    NaN_Count = ~sum(is.nan(.)),
    Blank_Count = ~sum(. == "", na.rm = TRUE)
  ))

# View the result
print(result)

# Filter out rows with NA, NaN, or blank values in 'value_euro'
player_data_clean <- playerData %>%
  filter(!is.na(value_euro) & !is.nan(value_euro) & value_euro != "")

# Box plot of age
ggplot(data = player_data_clean) +
  geom_boxplot(mapping = aes(x = age)) +
  labs(title = "Box plot of age", x = "Age")

# Remove players older than 35
player_data_v2 <- player_data_clean %>%
  filter(age <= 35)

# Box plot of age after filtering
ggplot(data = player_data_v2) +
  geom_boxplot(mapping = aes(x = age)) +
  labs(title = "Box plot of age (Filtered)", x = "Age")

# Scatter plot with regression line
ggplot(data = player_data_v2) +
  geom_point(mapping = aes(x = overall_rating, y = value_euro)) + 
  geom_smooth(method = "lm", formula = y ~ x, mapping = aes(x = overall_rating, y = value_euro)) +
  scale_y_continuous(name = "Value (Euro)", labels = scales::comma) +
  labs(title = "Price by Rating", x = "Overall Rating")

# Box plot of player value based on overall rating
ggplot(player_data_v2, aes(x = factor(overall_rating), y = value_euro)) + 
  geom_boxplot(fill = "antiquewhite4") + 
  scale_y_continuous(name = "Value (Euro)", labels = scales::comma) +
  labs(title = "Player Value Based on Overall Rating", x = 'Overall Rating', y = 'Value (Euro)')

# Specify columns to remove
columns_to_remove <- c("national_team", "national_rating", "national_team_position", "national_jersey_number")

# Create a subset of player_data_v3 excluding specified columns
player_data_v3 <- player_data_v2 %>%
  select(-one_of(columns_to_remove))

# Replace missing values in 'wage_euro' with the mean of the column
player_data_v3$wage_euro[is.na(player_data_v3$wage_euro) | 
                       is.nan(player_data_v3$wage_euro) | 
                       player_data_v3$wage_euro == ""] <- mean(player_data_v3$wage_euro, na.rm = TRUE)

# Replace missing values in 'release_clause_euro' with the mean of the column
player_data_v3$release_clause_euro[is.na(player_data_v3$release_clause_euro) | 
                                 is.nan(player_data_v3$release_clause_euro) | 
                                 player_data_v3$release_clause_euro == ""] <- mean(player_data_v3$release_clause_euro, na.rm = TRUE)

# Convert character columns to factors
player_data_v3 <- player_data_v3 %>%
  mutate_if(is.character, as.factor)

# Select numeric columns for correlation analysis
numeric_cols <- sapply(player_data_v3, is.numeric)
numeric_data <- player_data_v3[, numeric_cols]

# Compute correlations with 'value_euro'
correlations <- cor(numeric_data)[, "value_euro"]

# Print correlations
print(correlations)


# Set seed for reproducibility
set.seed(40405123)

# Split data into training and testing sets
index <- createDataPartition(player_data_v3$value_euro, p = 0.8, list = FALSE)
train <- player_data_v3[index,]
test <- player_data_v3[-index,]

formula <- value_euro ~ age + overall_rating + potential + 
  wage_euro + weak_foot + skill_moves + crossing + finishing + heading_accuracy + 
  short_passing + volleys + dribbling + curve + freekick_accuracy + 
  long_passing + ball_control + acceleration + sprint_speed + agility + 
  reactions + balance + shot_power + jumping + stamina + strength + long_shots +
  aggression + interceptions + positioning + vision + penalties + composure + 
  marking + standing_tackle + sliding_tackle

############################################### Multiple linear regression model
lr_model <- lm(formula = formula, data = train)

# Predict values for the test set
prediction <- predict(lr_model, test)

# Evaluate lr_model performance
lr_model_performance <- postResample(prediction, test$value_euro)
print(lr_model_performance)

####################################################### Decision Tree Regression
tree_model <- rpart(formula, data = train, method = "anova")
tree_pred <- predict(tree_model, test)
tree_performance <- postResample(tree_pred, test$value_euro)
print(tree_performance)

####################################################### Random Forest Regression
rf_model <- randomForest(formula, data = train)
rf_pred <- predict(rf_model, test)
rf_performance <- postResample(rf_pred, test$value_euro)
print(rf_performance)

################################################### Gradient Boosting Regression
# Define predictors (features) and target variable
x_train <- as.matrix(train[, c("age", "overall_rating", "potential", "wage_euro", 
                               "weak_foot", "skill_moves", "crossing", "finishing", 
                               "heading_accuracy", "short_passing", "volleys", 
                               "dribbling", "curve", "freekick_accuracy", "long_passing", 
                               "ball_control", "acceleration", "sprint_speed", "agility", 
                               "reactions", "balance", "shot_power", "jumping", 
                               "stamina", "strength", "long_shots","aggression", 
                               "interceptions", "positioning", "vision", "penalties", 
                               "composure", "marking", "standing_tackle", "sliding_tackle")])
y_train <- train$value_euro

x_test <- as.matrix(test[, c("age", "overall_rating", "potential", "wage_euro", 
                             "weak_foot", "skill_moves", "crossing", "finishing", 
                             "heading_accuracy", "short_passing", "volleys", 
                             "dribbling", "curve", "freekick_accuracy", "long_passing", 
                             "ball_control", "acceleration", "sprint_speed", "agility", 
                             "reactions", "balance", "shot_power", "jumping", 
                             "stamina", "strength", "long_shots","aggression", 
                             "interceptions", "positioning", "vision", "penalties", 
                             "composure", "marking", "standing_tackle", "sliding_tackle")])
y_test <- test$value_euro
# Prepare the data
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest <- xgb.DMatrix(data = x_test, label = y_test)

params <- list(booster = "gbtree", objective = "reg:squarederror", eta = 0.1, max_depth = 6)
xgb_model <- xgb.train(params, dtrain, nrounds = 100)
xgb_pred <- predict(xgb_model, dtest)
xgb_performance <- postResample(xgb_pred, y_test)
print(xgb_performance)

###################################################### Support Vector Regression
svr_model <- svm(formula, data = train)
svr_pred <- predict(svr_model, test)
svr_performance <- postResample(svr_pred, test$value_euro)
print(svr_performance)

######################################################################## Testing
################################################################################
# Load the new testing data
testing_data <- read_excel(file.choose()) # select testing.xlsx file

# View the structure of the new data
str(testing_data)

# Ensure that all necessary columns are present in the new data and preprocess the data
required_columns <- c("age", "overall_rating", "potential", "wage_euro", "weak_foot", 
                      "skill_moves", "crossing", "finishing", "heading_accuracy", 
                      "short_passing", "volleys", "dribbling", "curve", "freekick_accuracy", 
                      "long_passing", "ball_control", "acceleration", "sprint_speed", "agility", 
                      "reactions", "balance", "shot_power", "jumping", "stamina", "strength", 
                      "long_shots", "aggression", "interceptions", "positioning", "vision", 
                      "penalties", "composure", "marking", "standing_tackle", "sliding_tackle")

# Check for missing required columns and add them if necessary
missing_columns <- setdiff(required_columns, colnames(testing_data))
for (col in missing_columns) {
  testing_data[[col]] <- NA
}

# Convert character columns to factors
testing_data <- testing_data %>%
  mutate_if(is.character, as.factor)

# Handle missing values in wage_euro and release_clause_euro
testing_data$wage_euro[is.na(testing_data$wage_euro) | 
                         is.nan(testing_data$wage_euro) | 
                         testing_data$wage_euro == ""] <- mean(train$wage_euro, na.rm = TRUE)

# Fill missing values in other required columns with the mean of the corresponding columns in the training data
for (col in required_columns) {
  if (any(is.na(testing_data[[col]]) | is.nan(testing_data[[col]]))) {
    testing_data[[col]][is.na(testing_data[[col]]) | is.nan(testing_data[[col]])] <- mean(train[[col]], na.rm = TRUE)
  }
}

# Ensure that the column types in testing_data match those in the training data
for (col in required_columns) {
  if (is.factor(train[[col]]) && !is.factor(testing_data[[col]])) {
    testing_data[[col]] <- as.factor(testing_data[[col]])
  } else if (is.numeric(train[[col]]) && !is.numeric(testing_data[[col]])) {
    testing_data[[col]] <- as.numeric(testing_data[[col]])
  }
}

# Use the SVR model to make predictions on the new data
svr_predictions <- predict(rf_model, testing_data[required_columns])

# Print the predictions
print(svr_predictions)

# Add the predictions to the dataset
testing_data$predicted_value_euro <- svr_predictions

library(openxlsx)  # Library to write to Excel files

# Export the dataset to a new Excel file
write.xlsx(testing_data, file.choose()) #save to new file

# Confirm export success
print("Predictions have been added and the dataset has been exported successfully.")
