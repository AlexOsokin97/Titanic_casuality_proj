# Titanic Survival Estimator: Project Overview #
* **Was the survival of some Titanic's passangers a mere coincidence? Or there were certain conditions that had major**
**role in deciding the passanger's fate?**
* **Used over 1000 samples of Titanic passanger's information from Kaggle**
* **Cleaned the original datasets and created new ones and applied them in data analysis and model training**
* **Built models using Logistic Regression, SVM, RandomForest Classifier and K-NN algorithms to predict if a passanger Survived/Died and  compared their preformance and used the best one to fill the missing "Survived" values**
* **used standardization to eliminate bias/variance influence on the classification algorithms**

## File Description:
* ***Data Analysis [Directory]:*** Contains the datasets used for data analysis and the jupyter notebook file
* ***Original_DF's [Directory]:*** Contains the original test and train data sets downloaded from kaggle
* ***Classification Models [Python File]:*** Contains the trained machine learning classification algorithms 
* ***Complete_df [CSV File]:*** The full complete titanic passenger data set
* ***data_clean [Python File]:*** Contains the cleaning code of the 'train' dataset
* ***df_TESTED [CSV File]:*** The 'test.csv' dataset after predicting and adding the 'Survived' column
* ***df_training_new [CSV File]:*** New trainig dataset created after cleaning
* ***finalized_SVM_model [SAV File]:*** The trained classification model
* ***test_data_classification [Python File]:*** Application of the classification model on the test dataset

## Code and Resources Used:
* ***Python Version:*** 3.8.2
* ***Original Data Set:*** <https://www.kaggle.com/c/titanic/data>
* ***Packages:*** pandas, numpy, matplotlib, seaborn, sklearn, pickle
* ***IDES Used:*** Anaconda, Spyder, Jupyter-Notebook
* ***Saving and Loading ML models:*** <https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/>
* ***Titanic_Proj_Example:*** <https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8>

## Youtube:
* ***Youtube:*** Videos and explainations from Ken Jee who is a data scientist. You can look up his channel [Here](https://www.youtube.com/channel/UCiT9RITQ9PW6BhXK0y2jaeg)

## EDA:
**I looked at the distributions of the data for numerical and categorical data. Made plots that describe the dataset and made it easier to find correlation between data. Here are some examples:**

![alt text][plot1]

[plot1]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/Data_Analysis/corrHeatmap.png "CorrHeatmap"
