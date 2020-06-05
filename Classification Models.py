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

#choosing relevant columns and cleaning
df_model = df_train[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
df_model = df_model.loc[df_model['Embarked'] !='Unknown']

#create dummy variables
df_dums = pd.get_dummies(df_model)

#creating the dependent(y) and independent(X) variables
X = df_dums.drop(['Survived','Sex_male', 'Embarked_Southampton'], axis=1).values
y = df_dums['Survived'].values

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

#creating train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

######################################GradientBoostingClassifier###########################################

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, 
                                 n_estimators=1000, criterion='friedman_mse')

gbc.fit(X_train, y_train)

y_pred_gb = gbc.predict(X_test)
cm_gb = confusion_matrix(y_test, y_pred_gb)
print(cm_gb)
print("Accuracy: {}%".format(accuracy_score(y_test, y_pred_gb)))


gb_accuracies = cross_val_score(gbc, X = X_train, y = y_train, cv=7)
print("The Avg Accuracy of cross-val-score: {}%".format(gb_accuracies.mean()*100))

#########################################XGBClassifier#########################################

from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, y_train)

y_pred_XGB = xgb.predict(X_test)
cm_XGB = confusion_matrix(y_test, y_pred_XGB)
print(cm_XGB)
print("Accuracy: {}%".format(accuracy_score(y_test, y_pred_XGB)))

XGB_accuracies = cross_val_score(xgb, X = X_train, y = y_train, cv=7)
print("The Avg Accuracy of cross-val-score: {} %".format(XGB_accuracies.mean()*100))

#######################################RandomForestClassifier########################################

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=160,criterion='entropy')

rfc.fit(X_train, y_train)

y_pred_rfc = rfc.predict(X_test)
cm_rfc = confusion_matrix(y_test, y_pred_rfc)
print(cm_rfc)
print("Accuracy: {}%".format(accuracy_score(y_test, y_pred_rfc)))

rfc_accuracies = cross_val_score(rfc, X = X_train, y = y_train, cv=7)
print("The Avg Accuracy of cross-val-score: {} %".format(rfc_accuracies.mean()*100))

###################################SupportVectorMachine####################################################
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train[:, [1,4]] = sc.fit_transform(X_train[:, [1,4]])
X_test[:, [1,4]] = sc.transform(X_test[:, [1,4]])

from sklearn.svm import SVC

svc = SVC(kernel='rbf')

svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)
cm_svc = confusion_matrix(y_test, y_pred_svc)
print(cm_svc)
print("Accuracy: {}%".format(accuracy_score(y_test, y_pred_svc)))

svc_accuracies = cross_val_score(svc, X = X_train, y = y_train, cv=7)
print("The Avg Accuracy of cross-val-score: {} %".format(svc_accuracies.mean()*100))

###########################################ModelSaving#################################################

import pickle
file_name = "SVC_model.sav"
pickle.dump(svc, open(file_name, 'wb'))










