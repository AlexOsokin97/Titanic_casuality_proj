# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 01:59:25 2020

@author: Alexander
"""

import pandas as pd

df1 = pd.read_csv('unfixed_pclass_dfs/pclass1_data.csv')

df1.set_index('PassengerId', inplace=True)
df1.drop(columns=['Unnamed: 0'], inplace=True)

df1.to_csv('C:/Users/Alexander/Documents/GitHub/titanic_casualties_proj/pclass1_data_.csv')

#-------------------------------------------------------------------------------------------

df2 = pd.read_csv('unfixed_pclass_dfs/pclass2_data.csv')

df2.set_index('PassengerId', inplace=True)
df2.drop(columns=['Unnamed: 0'], inplace=True)

df2.to_csv('C:/Users/Alexander/Documents/GitHub/titanic_casualties_proj/pclass2_data_.csv')

#-------------------------------------------------------------------------------------------

df3 = pd.read_csv('unfixed_pclass_dfs/pclass3_data.csv')

df3.set_index('PassengerId', inplace=True)
df3.drop(columns=['Unnamed: 0'], inplace=True)

df3.to_csv('C:/Users/Alexander/Documents/GitHub/titanic_casualties_proj/pclass3_data_.csv')