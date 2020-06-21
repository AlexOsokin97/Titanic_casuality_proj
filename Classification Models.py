# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:18:19 2020

@author: Alexander
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
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

#function for model saving
def save_model(model, name):
    filename = name+".sav"
    pickle.dump(model, open(filename, 'wb'))
    
#function for calculating model performance (Scaled Data)
def model_performance(model, filename):
    if model == 'Support Vector Machine':
        loaded_model = pickle.load(open(filename, 'rb'))
        accuracy = loaded_model.score(scaled_X_test, y_test)
        plot_confusion_matrix(loaded_model, scaled_X_test, y_test)
        return accuracy
    else:
        loaded_model = pickle.load(open(filename, 'rb'))
        accuracy = loaded_model.score(X_test, y_test)
        plot_confusion_matrix(loaded_model, X_test, y_test)
        return accuracy

#function for feature scaling (Standardization)
def data_scaler(train_features, test_features, train_cols = [], test_cols = []):
    sc = StandardScaler()
    train_features[: , train_cols] = sc.fit_transform(train_features[: , train_cols])
    test_features[: , test_cols] = sc.transform(test_features[: , test_cols])
    return train_features, test_features
    
######################################LogisticRegressionClassifier############################################

lrc = LogisticRegression(max_iter=2000)
print("Avg LogisticRegression accuracy: ", np.mean(cross_val_score(lrc, X_train, y_train, cv=3)), "%")

lrc_params = [{'C':(1.0, 2.0, 3.0), 'fit_intercept':(True, False),'solver':('newton-cg', 'lbfgs', 'sag'),
               'multi_class':('auto', 'ovr', 'multinomial'), 'max_iter':range(6000,8000,500)}]

lrc_gs = GridSearchCV(lrc, lrc_params, scoring='accuracy', cv=3)

lrc_gs.fit(X_train, y_train)

lrc_estimator = lrc_gs.best_estimator_

save_model(lrc_estimator, 'lrc_estimator')
model_performance('LogisticRegression', 'lrc_estimator.sav')

#########################################KNN########################################################

knc = KNeighborsClassifier()
print("Avg KNeighborsClassifier accuracy: ", np.mean(cross_val_score(knc, X_train, y_train, cv=3)), "%")

knc_params = [{'n_neighbors': range(10,50,10), 'weights':('uniform', 'distance'), 
               'algorithm':('ball_tree', 'kd_tree', 'brute', 'auto'), 'leaf_size':range(30,90,20),
               'p':(1,2)}]

knc_gs = GridSearchCV(knc, knc_params, scoring='accuracy', cv=3)

knc_gs.fit(X_train, y_train)

knc_estimator = knc_gs.best_estimator_

save_model(knc_estimator, 'knc_estimator')
model_performance('K-NearestNeighbors','knc_estimator.sav')

#########################################GaussianNaiveBaysClassifier###########################################

nbc = GaussianNB()
print("Avg Gaussian Naive Bayes accuracy: ", np.mean(cross_val_score(nbc, X_train, y_train, cv=3)), "%")

nbc.fit(X_train, y_train)

save_model(nbc, 'nbc')
model_performance('Gaussian Naive Bayes','nbc.sav')

###################################SupportVectorMachine########################################################

scaled_X_train, scaled_X_test = data_scaler(X_train, X_test, [1,4], [1,4])

svc = SVC()
print("Avg SVM accuracy", np.mean(cross_val_score(svc, scaled_X_train, y_train, cv=3)), "%")

svc_params = [{'C':(0.5, 1.0, 2.0), 'kernel':('poly','rbf','sigmoid'), 
               'degree':range(3,9,3), 'gamma':('scale','auto')}]

svc_gs = GridSearchCV(svc, svc_params, scoring='accuracy', cv=3)

svc_gs.fit(scaled_X_train, y_train)

svc_estimator = svc_gs.best_estimator_

save_model(svc_estimator, 'svc_estimator')
model_performance('Support Vector Machine','svc_estimator.sav')


    










