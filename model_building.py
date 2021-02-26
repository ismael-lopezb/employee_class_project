#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:10:33 2021

@author: ismaellopezbahena
"""
#import pandas 
import pandas as pd
import numpy as np
#read data cleaned and describe
df = pd.read_csv('data_cleaned.csv')
df.describe()
df.info()
#in this case we don't want the city column because city development index is enough
df=df.drop(['city'], axis=1)
#get dummies to pass actegorical data into numerical
df_model = pd.get_dummies(df)
#now we want our data in X for variables and y to target (the one we want to predict)
X = df_model.drop(['target'], axis=1).values
y = df_model['target'].values

# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create instance of standard scaler
scaler = StandardScaler()

scaler.fit(X) # Everything but target variable 

# Use scaler object to do a transform columns
X = scaler.transform(X) # perform

#now we split our data in train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#let's model

#logistic regression
from sklearn.linear_model import LogisticRegression

# Create instance of model
lreg = LogisticRegression()

# Pass training data into model
lreg.fit(X_train, y_train)

#predict 
y_pred_lreg = lreg.predict(X_test)

# Score It
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score 

# Confusion Matrix
print('Logistic Regression')
print('\n')
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_lreg))
print('--'*40)

# Classification Report
print('Classification Report')
print(classification_report(y_test,y_pred_lreg))

# Accuracy
print('--'*40)
logreg_f1 = f1_score(y_test, y_pred_lreg) 
print('F1 Score', logreg_f1)

#we have all negative values so let's see if cross validation improve it
cv_acc = cross_val_score(lreg, X_train, y_train, scoring='f1', cv=5)
print(np.mean(cv_acc))


#KNN
from sklearn.neighbors import KNeighborsClassifier

# Create instance of model
knn = KNeighborsClassifier(n_neighbors = 5)

# Fit to training data
knn.fit(X_train,y_train)

#predict 
y_pred_knn = knn.predict(X_test)

# Score it
print('K-Nearest Neighbors (KNN)')
print('k = 5')
print('\n')
# Confusion Matrix
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred_knn))

# Classification Report
print('--'*40)
print('Classification Report')
print(classification_report(y_test, y_pred_knn))

# Accuracy
print('--'*40)
knn_f1 = f1_score(y_test, y_pred_knn)
print('F1 Score',knn_f1)

#hypermarameter tunning with GridSearchCV
from sklearn.model_selection import GridSearchCV 

lreg_grid = {'C':np.logspace(-5, 8, 15)}
lreg_cv = GridSearchCV(lreg, lreg_grid, cv=5, scoring='f1')
lreg_cv.fit(X_train, y_train)
print('logistic regression best params', lreg_cv.best_params_)
print('logistic regression best score', lreg_cv.best_score_)

knn_grid = {'n_neighbors': np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_grid, cv=5, scoring='f1')
knn_cv.fit(X_train, y_train)
print('KNN best params', knn_cv.best_params_)
print('KNN best score', knn_cv.best_score_)

#Decision tree classifier
# Import model
from sklearn.tree import DecisionTreeClassifier

# Create model object
dtree = DecisionTreeClassifier()

# Fit to training sets
dtree.fit(X_train,y_train)

#predict 
y_pred_dtree = dtree.predict(X_test)

# Score It
print('Decision Tree')
# Confusion Matrix
print('\n')
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_dtree))

# Classification Report
print('--'*40)
print('Classification Report',classification_report(y_test, y_pred_dtree))

# Accuracy
print('--'*40)
dtree_f1 = f1_score(y_test, y_pred_dtree)
print('Accuracy',dtree_f1)

#hypermarameter tunning with GridSearchCV
dtree_grid = {'criterion':['gini', 'entropy'], 'max_depth':[3, 4, 5, 6], 'min_samples_leaf':[0.04,0.06,0.08],
              'max_features':[0.2, 0.4, 0.6, 0.8]}
dtree_cv = GridSearchCV(dtree, dtree_grid, cv=5, scoring='f1')
dtree_cv.fit(X_train, y_train)
print('Desission tree best params', dtree_cv.best_params_)
print('Desisscion tree best score', dtree_cv.best_score_)

#random forest classifier
# Import model object
from sklearn.ensemble import RandomForestClassifier

# Create model object
rfc = RandomForestClassifier(n_estimators = 200)

# Fit model to training data
rfc.fit(X_train,y_train)

#Predict 
y_pred_rfc = rfc.predict(X_test)

# Score It
print('Random Forest')
# Confusion matrix
print('\n')
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_rfc))

# Classification report
print('--'*40)
print('Classification Report')
print(classification_report(y_test, y_pred_rfc))

# Accuracy
print('--'*40)
rf_f1 = f1_score(y_test, y_pred_rfc)
print('Accuracy', rf_f1)

#hypermarameter tunning with GridSearchCV
rfc_grid = {'n_estimators': [200, 500, 700],
            'max_features': ['sqrt', 'log2'],
            'max_depth' : [4,5,7],
            'criterion' :['gini', 'entropy']}
#{'n_estimators':[300,500,800], 'max_depth':[4, 5, 6], 'min_samples_leaf':[0.04,0.06,0.08],
#            'max_features':[0.2, 0.4, 0.6, 0.8]}
rfc = RandomForestClassifier()
rfc_cv = GridSearchCV(rfc, rfc_grid, cv=5, scoring='f1')
rfc_cv.fit(X_train, y_train)
print('Random forest classifier best params', rfc_cv.best_params_)
print('Random forest classifier best score', rfc_cv.best_score_)

#let's now use the 4 models for a Voting Classifier
from sklearn.ensemble import VotingClassifier
classifiers = {('Logistic Regression', lreg_cv.best_estimator_),
               ('K Nearest Neighbours', knn_cv.best_estimator_),
               ('Classification Tree', dtree_cv.best_estimator_),
               ('Random Forest Classifier', rfc_cv.best_estimator_)}
vc = VotingClassifier(estimators=classifiers)
vc.fit(X_train, y_train)
y_pred_vc = vc.predict(X_test)
print('Voting Classifier F1', f1_score(y_test, y_pred_vc))

#the final results
y_pred_lreg = lreg_cv.best_estimator_.predict(X_test)
y_pred_knn = knn_cv.best_estimator_.predict(X_test)
y_pred_dtree = dtree_cv.best_estimator_.predict(X_test)
y_pred_rfc = rfc_cv.best_estimator_.predict(X_test)
print('Tuned Logistic Regression f1 score:',f1_score(y_test, y_pred_lreg),
      'accuracy:', accuracy_score(y_test, y_pred_lreg))
print('Tuned K-Nearest Neighbours f1 score:',f1_score(y_test, y_pred_knn),
      'accuracy:', accuracy_score(y_test, y_pred_knn))
print('Tuned Decission Tree Classifier f1 score:',f1_score(y_test, y_pred_dtree),
      'accuracy:', accuracy_score(y_test, y_pred_dtree))
print('Tuned Random Forest Classifier f1 score:',f1_score(y_test, y_pred_rfc),
      'accuracy:', accuracy_score(y_test, y_pred_rfc))
