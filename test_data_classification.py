# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:25:26 2020

@author: Alex
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing our test data
df_test = pd.read_csv('test.csv')

#preparing our data
df_test.set_index('PassengerId', inplace=True)
df_test.drop('Cabin',axis=1 , inplace=True)
df_test['Embarked'] = df_test['Embarked'].apply(lambda x: 'Queenstown' if x=='Q' else x).apply(lambda x: 'Southampton' if x=='S' else x).apply(lambda x: 'Cherbourg' if x=='C' else x)

#dropping nan age values from our data
df_test.dropna(axis=0, inplace=True)

#creating the model df
df_model = df_test.drop(['Name', 'Ticket'], axis=1)

#creating dummy variables
df_dum = pd.get_dummies(df_model)

#creating our tasting data for our model
X_test = df_dum.values

#applying standarization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_test[:,[1,4]] = sc.fit_transform(X_test[:,[1,4]])

#loading in our trained SVM model
import pickle
loaded_model = pickle.load(open('finalized_SVM_model.sav', 'rb'))

Survived_prediction = loaded_model.predict(X_test)

df_test.insert(0, 'Survived_prediction', Survived_prediction)

df_test.to_csv('C:/Users/User/Documents/GitHub/titanic_casualties_proj/df_TESTED.csv')
