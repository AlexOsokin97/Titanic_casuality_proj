# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:19:45 2020

@author: Alexander
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing our training data
df_train = pd.read_csv('df_train_new.csv')

#preparing the data
df_train.set_index('PassengerId', inplace=True)