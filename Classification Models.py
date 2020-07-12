# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:18:19 2020

@author: Alexander
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

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

#creating train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#Functions:
def cross_score(estimator, training_features, training_label, scoring, cv):
    """
    this function does cross validation and returns accuracy and performance time

    Parameters
    ----------
    estimator : (sklearn.modeltype)
        sklearn/other api model.
    training_features : (numpy array)
        DESCRIPTION.
    training_label : (numpy array)
        DESCRIPTION.
    scoring : (string)
        sklearn api scoring metric for classification.
    cv : (integer)
        K amount of folds.

    Returns
    -------
    accuracy : (float)

    train_time : (float)
    
    """
    training_start = time.perf_counter()
    accuracy = np.mean(cross_val_score(estimator, X=training_features, y=training_label, scoring=scoring, cv=cv))
    training_end = time.perf_counter()
    train_time = training_end-training_start
    return accuracy, train_time

def best_hyper_params(estimator,training_features, training_label, params, scoring):
    """
    this function does parameter tuning using grid search

    Parameters
    ----------
    estimator : (sklearn.modeltype)
        sklearn/other api model.
    training_features : (numpy array)
        X_train.
    training_label : (numpy array)
        y_train.
    params : (list)
        a dictionary with model's hyper parameters.
    scoring : (string)
        sklearn api scoring metric for classification.

    Returns
    -------
    best_find : (object)
        returns a model with the best hyper parameters.
    best_score : (float)
        returns the score of the best model.
    search_time : (float)
        searching time.

    """
    search = GridSearchCV(estimator, param_grid=params, scoring=scoring, cv=3, n_jobs=-1)
    search_start = time.perf_counter()
    search.fit(training_features, training_label)
    search_end = time.perf_counter()
    best_find = search.best_estimator_
    best_score = search.best_score_
    search_time = search_end - search_start
    return best_find, best_score, search_time

def model_performance(estimator, test_features, test_label):
    """
    this function evaluates the performance of the model on the testing set

    Parameters
    ----------
    estimator : (object)
        an already tuned model and fitted to the training data (use the estimator returned by best_hyper_params function).
    test_features : (numpy array)
        X_test.
    test_label : (numpy array)
        y_test.

    Returns
    -------
    conf_matrix : (img)
        returns an image of a confusion matrix.
    accuracy : (float)
        model's accuracy.
    precision : (float)
        model's precision accuracy.

    """
    predicts = estimator.predict(test_features)
    conf_matrix = plot_confusion_matrix(estimator, X=test_features, y_true=test_label)
    accuracy = accuracy_score(y_true=test_label, y_pred=predicts)
    return conf_matrix, accuracy
    
######################################XGBoost Classifier############################################

xg = XGBClassifier(random_state=1)

xg_accuracy, xg_train_time = cross_score(xg, X_train, y_train, 'accuracy', 10)

xg_params = [{'max_depth':(3,6), 'learning_rate':(0.01, 0.03, 0.1, 0.3), 'n_estimators':range(50,250,100), 'reg_alpha':(0,0.3,0.6,0.8)}]

xg_best_esti, xg_best_score, xg_search_time = best_hyper_params(xg, X_train, y_train, xg_params, 'accuracy')

xg_cm, xg_pred_accuracy = model_performance(xg_best_esti, X_test, y_test) 

#########################################KNN########################################################

knc = KNeighborsClassifier()

knc_accuracy, knc_train_time = cross_score(knc, X_train, y_train, 'accuracy', 10)

knc_params = [{'n_neighbors': (2,4,22,24,42,44), 'weights':('uniform', 'distance'),'algorithm':('ball_tree', 'kd_tree', 'brute'), 
               'leaf_size':range(30,90,30),'p':(1,2)}]

knc_best_esti, knc_best_score, knc_search_time = best_hyper_params(knc, X_train, y_train, knc_params, 'accuracy')

knc_cm, knc_pred_accuracy = model_performance(knc_best_esti, X_test, y_test) 

#########################################RandomForest Classifier###########################################

rfc = RandomForestClassifier(random_state=2)

rfc_accuracy, rfc_train_time = cross_score(rfc, X_train, y_train, 'accuracy', 10)

rfc_params = [{'n_estimators':range(100,400,100), 'criterion':('gini', 'entropy'), 'max_depth': (3,6,9), 'min_samples_split':(2,8,16), 
               'max_features':('sqrt', 'log2', 'None')}]

rfc_best_esti, rfc_best_score, rfc_search_time = best_hyper_params(rfc, X_train, y_train, rfc_params, 'accuracy')

rfc_cm, rfc_pred_accuracy = model_performance(rfc_best_esti, X_test, y_test) 

###################################SupportVectorMachine########################################################

svc = SVC(random_state=3)

scaler = MinMaxScaler()
X_train_sclaed = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svc_accuracy, svc_train_time = cross_score(svc, X_train_sclaed, y_train, 'accuracy', 10)

svc_params = [{'C':(1.0, 2.0, 11.0, 12.0, 21.0, 22.0), 'kernel':('rbf','sigmoid'), 'gamma':('scale','auto')}]

svc_best_esti, svc_best_score, svc_search_time = best_hyper_params(svc, X_train_sclaed, y_train, svc_params, 'accuracy')

svc_cm, svc_pred_accuracy = model_performance(svc_best_esti, X_test_scaled, y_test) 




    










