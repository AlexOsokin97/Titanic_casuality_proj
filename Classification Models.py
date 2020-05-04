# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:18:19 2020

@author: Alexander
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load our train data and test data
df_train = pd.read_csv('df_train_new.csv')

#choosing relevant columns
df_model = df_train[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

#create dummy variables
df_dums = pd.get_dummies(df_model)

#creating the dependent(y) and independent(X) variables
X = df_dums.drop('Survived', axis=1).values
y = df_dums['Survived'].values

#creating train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)

#applying feature scalling using standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#Scalling the Age column(1) and the Fare column(4) of our training set
X_train[:,[1,4]] = sc.fit_transform(X_train[:,[1,4]])

#Scalling the Age column(1) and the Fare column(4) of our test set
X_test[:,[1,4]] = sc.transform(X_test[:,[1,4]])

#using logistic regression and fitting it to x and y training sets
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Predicting the test set result
y_predict = classifier.predict(X_test)

#Creating a confusion matrix
from sklearn.metrics import confusion_matrix
classifier_cm = confusion_matrix(y_test, y_predict)

