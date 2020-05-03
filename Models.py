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

#clean the test data alittle
df_test.drop('Cabin', axis=1, inplace=True)
df_test['Embarked'] = df_test['Embarked'].apply(lambda x: 'Queenstown' if x =='Q' else x).apply(lambda x: 'Cherbourg' if x=='C' else x).apply(lambda x: 'Southampton' if x=='S' else x)

#choosing relevant columns
df_model = df_train[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

#get dummy data
df_dum = pd.get_dummies(df_model)

