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
import numpy as np
import matplotlib.pyplot as plt

#for each Pclass(3) find the mean of female age and male age.
#for each missing data(nan) in 'Age' fill in age according to
#the sex type and it's mean

df_train_pclass = df_train[['Pclass', 'Sex','Age']] #new dataframe with relevent columns
#s_sex = df_train['Sex'].apply(lambda x: 1 if x.lower() == 'male' else 0) #transfroming 'male' & 'female' to 1 & 0
df = df_train_pclass.copy()
#df = df.assign(Sex = s_sex)

#time to use or copy of df_train_pclass dataframe:
#We need to get a certain pclass group(1,2,3), and find the age mean of the male group(1) and female group(0)

#creating dataframes with the specific pclass value:
df_pclas_first = df[df['Pclass']==1]
df_pclass_second = df[df['Pclass']==2]
df_pclass_third = df[df['Pclass'] ==3]

#finding the mean of Age of each pclass df according to sex value(male/female)




#dummy = pd.get_dummies(df_train_pclass['Sex']) #setting dummy variable
#df_train_pclass = pd.concat([df_train_pclass, dummy], axis=1)
#df_train_pclass.drop(['Sex'], axis=1, inplace=True)

        

