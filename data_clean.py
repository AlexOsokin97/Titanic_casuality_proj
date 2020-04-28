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

#changing embarked location to the full name
df['Embarked'] = df['Embarked'].apply(lambda x: 'Cherbourg' if x=='C' else x)
df['Embarked'] = df['Embarked'].apply(lambda x: 'Queenstown' if x=='Q' else x)
df['Embarked'] = df['Embarked'].apply(lambda x: 'Southampton' if x=='S' else x)
df['Embarked'] = df['Embarked'].apply(lambda x: 'Unknown' if pd.isna(x) else x)

#creating 3 new data frames, each represents different class (1,2,3) to ease the 
#data exploration and visualization process
df_pclass1 = df[df['Pclass']==1].copy()
df_pclass2 = df[df['Pclass']==2].copy()
df_pclass3 = df[df['Pclass']==3].copy()

#filling missing age data of first class df:
df_pclass1['Age'] = df_pclass1[['Sex','Age']].apply(lambda x: female_1class_mean if x['Sex']=='female' and x['Age']==-1 else x['Age'], axis=1)
df_pclass1['Age'] = df_pclass1[['Sex','Age']].apply(lambda x: male_1class_mean if x['Sex']=='male' and x['Age']==-1 else x['Age'],axis=1)

#filling missing age data of second class df:
df_pclass2['Age'] = df_pclass2[['Sex','Age']].apply(lambda x: female_2class_mean if x['Sex']=='female' and x['Age']==-1 else x['Age'], axis=1)
df_pclass2['Age'] = df_pclass2[['Sex','Age']].apply(lambda x: male_2class_mean if x['Sex']=='male' and x['Age']==-1 else x['Age'],axis=1)

#filling missing age data of second class df:
df_pclass3['Age'] = df_pclass3[['Sex','Age']].apply(lambda x: female_3class_mean if x['Sex']=='female' and x['Age']==-1 else x['Age'], axis=1)
df_pclass3['Age'] = df_pclass3[['Sex','Age']].apply(lambda x: male_3class_mean if x['Sex']=='male' and x['Age']==-1 else x['Age'],axis=1)

#saving the 3 new dataframes for further use:
df_pclass1.to_csv('C:/Users/Alexander/Documents/GitHub/titanic_casualties_proj/pclass1_data.csv')
df_pclass2.to_csv('C:/Users/Alexander/Documents/GitHub/titanic_casualties_proj/pclass2_data.csv')
df_pclass3.to_csv('C:/Users/Alexander/Documents/GitHub/titanic_casualties_proj/pclass3_data.csv')








