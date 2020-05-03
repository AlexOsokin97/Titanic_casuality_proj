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
df_test = pd.read_csv('test.csv')

#clean the test data
df_test.drop('Cabin', axis=1, inplace=True)
df_test['Embarked'] = df_test['Embarked'].apply(lambda x: 'Queenstown' if x =='Q' else x).apply(lambda x: 'Cherbourg' if x=='C' else x).apply(lambda x: 'Southampton' if x=='S' else x)

#choosing relevant columns
df_model = df_train[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

#create dums
df_dums = pd.get_dummies(df_model)

#creating the dependent(y) and independent(X) variables
X = df_dums.drop('Survived', axis=1).values
y = df_dums['Survived'].values

#creating train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#applying feature scalling using standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#Scalling the Age column(1) and the Fare column(4) of our training set
X_train[:,1:2] = sc.fit_transform(X_train[:,1:2])
X_train[:,4:5] = sc.fit_transform(X_train[:,4:5])

#Scalling the Age column(1) and the Fare column(4) of our test set
X_test[:,1:2] = sc.transform(X_test[:,1:2])
X_test[:,4:5] = sc.transform(X_test[:,4:5])

#Logistic Regression using sklearn
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
x_predict = clf.predict(X_test)
lr_score = clf.score(X_test, y_train)





