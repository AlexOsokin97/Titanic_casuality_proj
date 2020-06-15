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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score

#creating train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

######################################GradientBoostingClassifier###########################################
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=42)
print("Avg GradientBoosting accuracy: ", np.mean(cross_val_score(gbc, X_train, y_train, cv=10)), "%")

gbc_params = [{'loss':('deviance','exponential'), 'learning_rate': (0.001, 0.01, 0.1), 'n_estimators': range(300,1500,300),
               'criterion':('friedman_mse', 'mse', 'mae'), 'min_samples_split': range(2,14,4), 'min_samples_leaf': range(2,20,6),
               'max_features': ('auto', 'sqrt', 'log2', None)}]

gbc_gs = GridSearchCV(gbc, gbc_params, scoring='accuracy', cv=3)

gbc_gs.fit(X_train, y_train)

gbc_score = gbc_gs.best_score_
gbc_param = gbc_gs.best_params_
gbc_estimator = gbc_gs.best_estimator_

print("Best Score: ", gbc_gs.best_score_)
print("Best parameters: ", gbc_gs.best_params_)
print("Best model(estimator): ", gbc_gs.best_estimator_)

#########################################XGBClassifier#########################################

from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=42)
print("Avg XGBoosting accuracy: ", np.mean(cross_val_score(xgb, X_train, y_train, cv=10)), "%")

xgb_params = [{'eta': (0.05,0.1,0.3,0.5), 'gamma':(0, 4, 8, 12), 'max_depth':(3,6,9), 
               'subsample':(0.5, 1), 'lambda':(0,1,2), 'alpha':(0,1,2), 
               'tree_method':('auto','exact'), 
               'grow_policy':('depthwise','lossguide'),'max_leaves':(0,8,16)}]

xgb_gs = GridSearchCV(xgb, xgb_params, scoring='accuracy', cv=3)

xgb_gs.fit(X_train, y_train)

print("Best Score: ", xgb_gs.best_score_)
print("Best parameters: ", xgb_gs.best_params_)
print("Best model(estimator): ", xgb_gs.best_estimator_)
#######################################RandomForestClassifier########################################

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42)

print("Avg RandomForest accuracy: ", np.mean(cross_val_score(xgb, X_train, y_train, cv=10)), "%")

rfc_params = [{'n_estimators':(100,900,200), 'criterion':('gini', 'entropy'),
               'max_depth':(3,6,9,None), 'min_samples_split':(2,4,6),
               'max_features':('auto','sqrt','log2'), 'max_leaf_nodes':(8,16,32)}]

rfc_gs = GridSearchCV(rfc, rfc_params, scoring='accuracy', cv=3)

rfc_gs.fit(X_train, y_train)

print("Best Score: ", rfc_gs.best_score_)
print("Best parameters: ", rfc_gs.best_params_)
print("Best model(estimator): ", rfc_gs.best_estimator_)
###################################SupportVectorMachine####################################################
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train[:, [1,4]] = sc.fit_transform(X_train[:, [1,4]])
X_test[:, [1,4]] = sc.transform(X_test[:, [1,4]])

from sklearn.svm import SVC

svc = SVC(random_state=42)
print("Avg SVM accuracy", np.mean(cross_val_score(svc, X_train, y_train, cv=10)), "%")

svc_params = [{'C':(0.5, 1.0, 2.0), 'kernel':('poly','rbf','sigmoid'), 
               'degree':range(3,9,3), 'gamma':('scale','auto')}]

svc_gs = GridSearchCV(svc, svc_params, scoring='accuracy', cv=3)

svc_gs.fit(X_train, y_train)

print("Best Score: ", svc_gs.best_score_)
print("Best parameters: ", svc_gs.best_params_)
print("Best model(estimator): ", svc_gs.best_estimator_)
###########################################ModelSaving#################################################











