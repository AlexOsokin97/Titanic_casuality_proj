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

class_type = df_train['Pclass']
age = df_train['Age']

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(class_type, age)
plt.show()