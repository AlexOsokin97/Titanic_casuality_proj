# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:18:19 2020

@author: Alexander
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

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

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import plot_confusion_matrix

#creating train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#function for model saving
def save_model(model, name):
    filename = name+".sav"
    pickle.dump(model, open(filename, 'wb'))
    
#function for calculating model performance
def model_performance(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    accuracy = loaded_model.score(X_test, y_test)
    plot_confusion_matrix(loaded_model, X_test, y_test)
    return accuracy

######################################LogisticRegressionClassifier############################################
from sklearn.linear_model import LogisticRegression

lrc = LogisticRegression(max_iter=2000)
print("Avg LogisticRegression accuracy: ", np.mean(cross_val_score(lrc, X_train, y_train, cv=3)), "%")

lrc_params = [{'C':(1.0, 2.0, 3.0), 'fit_intercept':(True, False),'solver':('newton-cg', 'lbfgs', 'sag'),
               'multi_class':('auto', 'ovr', 'multinomial'), 'max_iter':range(6000,8000,500)}]

lrc_gs = GridSearchCV(lrc, lrc_params, scoring='accuracy', cv=3)

lrc_gs.fit(X_train, y_train)

lrc_estimator = lrc_gs.best_estimator_

save_model(lrc_estimator, 'lrc_estimator')
model_performance('lrc_estimator.sav')
#########################################XGBClassifier########################################################
from xgboost import XGBClassifier

xgb = XGBClassifier()
print("Avg XGBoosting accuracy: ", np.mean(cross_val_score(xgb, X_train, y_train, cv=3)), "%")

xgb_params = [{'eta': (0.05,0.1,0.3,0.5), 'gamma':(0, 4, 8, 12), 'max_depth':(3,6,9),'subsample':(0.5, 1), 
               'lambda':(0,1,2), 'alpha':(0,1,2),'tree_method':('auto','exact'), 'learning_rate':(0.001,0.01,0.1),
               'grow_policy':('depthwise','lossguide'),'max_leaves':(0,8,16), 'n_estimators':range(100,1000,100)}]

xgb_gs = GridSearchCV(xgb, xgb_params, scoring='accuracy', cv=3)

xgb_gs.fit(X_train, y_train)

xgb_estimator = xgb_gs.best_estimator_

print("Best Score: ", xgb_gs.best_score_)
print("Best parameters: ", xgb_gs.best_params_)
print("Best model(estimator): ", xgb_gs.best_estimator_)

save_model(xgb_estimator, 'xgb_estimator')
model_performance('xgb_estimator.sav')
#########################################GaussianNaiveBaysClassifier###########################################
from sklearn.naive_bayes import GaussianNB

nbc = GaussianNB()
print("Avg RandomForest accuracy: ", np.mean(cross_val_score(nbc, X_train, y_train, cv=3)), "%")

nbc.fit(X_train, y_train)

save_model(nbc, 'nbc')
model_performance('nbc.sav')
###################################SupportVectorMachine########################################################
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train[:, [1,4]] = sc.fit_transform(X_train[:, [1,4]])
X_test[:, [1,4]] = sc.transform(X_test[:, [1,4]])

from sklearn.svm import SVC

svc = SVC()
print("Avg SVM accuracy", np.mean(cross_val_score(svc, X_train, y_train, cv=3)), "%")

svc_params = [{'C':(0.5, 1.0, 2.0), 'kernel':('poly','rbf','sigmoid'), 
               'degree':range(3,9,3), 'gamma':('scale','auto')}]

svc_gs = GridSearchCV(svc, svc_params, scoring='accuracy', cv=3)

svc_gs.fit(X_train, y_train)

svc_estimator = svc_gs.best_estimator_

print("Best Score: ", svc_gs.best_score_)
print("Best parameters: ", svc_gs.best_params_)
print("Best model(estimator): ", svc_gs.best_estimator_)

save_model(svc_estimator, 'svc_estimator')
model_performance('svc_estimator.sav')


    










