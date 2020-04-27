# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:45:34 2020

@author: User
"""

import pandas as pd

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

#GOAL: check if passenger survived based on his pclass info, sex, age
#fill out the missing ages using pclass, sex and avg passenger age
#change the embarked column info to C = Cherbourg, Q = Queenstown, S = Southampton
#dropping the "cabin" column because too much information is missing

###df_train cleaning

#for each Pclass(3) find the mean of female age and male age.
#for each missing data(nan) in 'Age' fill in age according to
#the sex type and it's mean

df_train_pclass = df_train[['Pclass', 'Sex','Age']]

#pclass = 1
filt = df_train_pclass[(df_train_pclass['Pclass']==1) & (df_train_pclass['Sex'] == 'female')]
female_1class_mean = int(filt['Age'].mean())

filt = df_train_pclass[(df_train_pclass['Pclass']==1) & (df_train_pclass['Sex'] == 'male')]
male_1class_mean = int(filt['Age'].mean())

#pclass = 2
filt = df_train_pclass[(df_train_pclass['Pclass']==2) & (df_train_pclass['Sex'] == 'female')]
female_2class_mean = int(filt['Age'].mean())

filt = df_train_pclass[(df_train_pclass['Pclass']==2) & (df_train_pclass['Sex'] == 'male')]
male_2class_mean = int(filt['Age'].mean())

#pclass = 3
filt = df_train_pclass[(df_train_pclass['Pclass']==3) & (df_train_pclass['Sex'] == 'female')]
female_3class_mean = int(filt['Age'].mean())

filt = df_train_pclass[(df_train_pclass['Pclass']==3) & (df_train_pclass['Sex'] == 'male')]
male_3class_mean = int(filt['Age'].mean())

#fill the missing Age data 
df = df_train.copy()

#changing all na values in the 'Age' column to -1
df['Age'] = df['Age'].apply(lambda x: -1 if pd.isna(x) else x)
df['Age'].loc[(df['Sex'] == 'female') & (df['Pclass'] == 1)].apply(lambda x: female_1class_mean if x==-1 else x)
