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
df_model.drop(df_model.loc[df_model['Embarked']=='Unknown'].index, inplace=True)

#create dummy variables
df_dums = pd.get_dummies(df_model)

#creating the dependent(y) and independent(X) variables
X = df_dums.drop(['Survived'], axis=1).values
y = df_dums['Survived'].values

#creating train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)

#applying feature scalling using standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#Scalling the Age column(1) and the Fare column(4) of our training set
X_train[:,[1,4]] = sc.fit_transform(X_train[:,[1,4]])

#Scalling the Age column(1) and the Fare column(4) of our test set
X_test[:,[1,4]] = sc.transform(X_test[:,[1,4]])

#using logistic regression and fitting it to x and y training sets
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Predicting the test set result
y_predict = classifier.predict(X_test)

#Creating a confusion matrix
from sklearn.metrics import confusion_matrix
classifier_cm = confusion_matrix(y_test, y_predict)

#finding the accuarcy score of the logistic regression:
from sklearn.metrics import accuracy_score
classifier_score = accuracy_score(y_test, y_predict)

#using svm and fitting it to x and y training sets
#using kernel='poly'
from sklearn.svm import SVC
svm_clf_poly = SVC(kernel='poly', degree=3, gamma='auto')
svm_clf_poly.fit(X_train, y_train)

#predicting with the svm model
svmP_y_predict = svm_clf_poly.predict(X_test) 

#creating confusion matrix for the svm model
svmP_cm = confusion_matrix(y_test,svmP_y_predict)

#finding the accuarcy of the svm model
svmP_score = accuracy_score(y_test, svmP_y_predict)

#using svm and fitting it to x and y training sets
#using kernel='sigmoid'
svm_clf_sigmoid = SVC(kernel='sigmoid', gamma='scale')
svm_clf_sigmoid.fit(X_train, y_train)

#predicting with the svm model
svmS_y_predict = svm_clf_poly.predict(X_test) 

#creating confusion matrix for the svm model
svmS_cm = confusion_matrix(y_test,svmS_y_predict)

#finding the accuarcy of the svm model
svmS_score = accuracy_score(y_test, svmS_y_predict)

#using randomforestclassifier and fitting it to x and y training sets
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=200, criterion="entropy")
rf_classifier.fit(X_train, y_train)

#predicting with randomforestclassifier model
rfc_y_predict = rf_classifier.predict(X_test)

#creating confusion matrix for the svm model
rfc_cm = confusion_matrix(y_test,rfc_y_predict)

#finding the accuarcy of the svm model
rfc_score = accuracy_score(y_test,rfc_y_predict)

#using KNN and fitting it to x and y training sets
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=8, metric='minkowski',p=2)
knn_classifier.fit(X_train, y_train)

#predicting with KNN model
knn_y_predict = knn_classifier.predict(X_test)

#creating confusion matrix for the svm model
knn_cm = confusion_matrix(y_test,knn_y_predict)

#finding the accuarcy of the svm model
knn_score = accuracy_score(y_test,knn_y_predict)

#saving our best preformed model: SVM
import pickle
filename = 'finalized_SVM_model.sav'
pickle.dump(svm_clf_poly, open(filename, 'wb'))
