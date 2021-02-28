# Job Change of Data Scientists: Project Overview
* Predict if a data scientist candidate will work for a company
* Cleaned over 20,000 data extracted from Kaggle
* Optimized Logistic Regression, K-Nearest Neighbours, Decission-Tree Classifier and Random Forest Classifier using GridsearchCV to reach the best model.
## Code and Resources 
* **Python Version:** 3.8.5
* **Pckages:** pandas, numpy, sklearn, matplotlib, seaborn.
* **Data Source:** https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists
## Data Cleaning
* Filled most of the missing values in categorical data with the mode
* Simplified some categorical values
* Deleted special characters from experience to convert it to int64
* Put the company size in the same format 
* Deleted special characters from old-new-job (years since last and new job) to be int64
## Exploratory data analysis
I looked at target variable (if people are interested in change job or not) within each categorical data and stracted the following insights

Distribution of trainig hours with people interested in change job and that who aren't
![Distribution training](https://github.com/ismael-lopezb/employee_class_project/blob/main/trainigh.png)

Distribution of cadidates' experience in years
![Distribution experience](https://github.com/ismael-lopezb/employee_class_project/blob/main/experience.png)

Interest of changing job by gender, university enrollment, education level and company size
![Categories](https://github.com/ismael-lopezb/employee_class_project/blob/main/categories.png)

## Model Building 
First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%

I tried four different models and evaluated them using f1 score because the data is imbalanced and the false negatives are equally important as false positives

The four differnt models are:
* **Logistic Regression:** - basic for any classification model
* **K-Nearest Neighbours:** - the one i thought will perform the best
* **Decission-Tree Classifier:** - decission trees usually perform well
* **Random Forest Classifier:** - it's populat in regression problems

## Model performance
In this case K-Nearest Neighbours best performed on the test and validation sets in terms of f1 score but Decission-Tree classifier got a better accuracy and almost the same f1.
* **Decission-Tree Classifier:** F1 Score = 0.396, Accuracy = 75.9%
* **K-Nearest Neighbours:** F1 Score = 0.400, Accuracy = 70.0%
* **Logistic Regression:** F1 Score = 0.363, Accuracy = 76.6%
* **Random Forest Classifier:** F1 Score = 0.290, Accuracy = 76.5%

Decission-Tree Classifier is the model with best F1 Score and Accuracy.
