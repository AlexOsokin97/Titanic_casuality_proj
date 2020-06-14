# Titanic Passenger Survival Estimator: Project Overview #
**Was the survival of some Titanic's passengers a mere coincidence? Or there were certain conditions that had major role in deciding the passanger's fate?**

***As we all know Titanic was one of the most shocking world tragedy of the 19th century. The Titanic was the largest passenger ship ever built at the time which had around 4000 passengers on board including crew members, engineers, staff and common folk. That ship was in-fact so big that it looked unsinkable but, on April 14th, 1912 the unthinkable happened. The Titanic hit an iceberg and sunk in the early morning of April 15th, 1912. That incident had many casualties from little children to wealthy business men but, it also had many survivors. So as a Data Scientist I decided to research that tragedy and find out WHO had survived and WHY.***

***I managed to get a dataset of the Titanic's passengers with over 1000 passengers. Unfortunately, some data was missing and the dataset wasn't properly orginized. By using many mathematical, statistical and programming techniques I cleaned the dataset and made it readable. 
Also, I looked up and used other techniques which were better/worse than the original.***

***After the cleaning process I used statistical techniques to visualize the data and find correlation between different features in the dataset but most importantly to find which features correlate with the passengers' survival.***

***In the end, I applied Machine Learning algorithms to check if the data is sufficient enough for the algorithms to classify whether a passenger survived or did not survive so that in the future if new passenger information is found we could use the model to check if that passenger survived or not.***

## Data Cleaning & Visualization:
***Getting the necessary datasets is one thing but, making them readable, useful and useable is another thing. This is where the data cleaning and remodeling process comes in.***

***Most of the changes that I had done to the dataset are simple such as: changing column names, column data types and removing columns with too much missing data and unnecessary. The more advanced technique I used was when I filled the missing values of the 'Age' column where I used the mean age of all samples while taking into consideration the passenger's gender and traveling class***

***After the data was cleaned, made readable and useful I did exploratory data analysis to dive deeper into the dataset and find correlations between different features and most importantly which features correlate with the passenger's survival. Here are some examples from my data exploration***

![alt text][plot1] ![alt text][plot2] ![alt text][plot4]


[plot1]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/Data_Analysis/corrHeatmap.png "CorrHeatmap"
[plot2]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/Data_Analysis/MaleFemaleSurvived.png "MaleFemaleSurvived"
[plot4]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/Data_Analysis/fig.png "Survivals/Deaths in each gender "

***In the end, I made few more changes to the dataset such as removing more unnecessary columns and transfroming categorical columns to binary to make the dataset ready for Machine Learning algorithms usage***

## Model Building & Performance:
**I wanted to create the best model which will be able to predict whether a passanger Survived or Died based on most of the passenger's info: Gender, Age, Travel Class, Had Children/Spouces, Had Parents/Siblings, Fare. As a result, I decided to use 4 different classification algorithms. I split the data into training and testing set using train_test_split function and applied cross_val_score function which creates multiple training sets and one test set, applies the algorithm to each training set and checks the performance of the algorithm with the accuracy_score function. I ranked the performance of each algorithm in a descending order**

* **Support Vector Machine (rbf): 82.25% Average Accuracy**
* **Gradient Boosting Classifier: 81.85% Average Accuracy**
* **Random Forest Classifier (160 estimators): 81.57% Average Accuracy** 
* **XGBoost Classifier: 80.44% Average Accuracy**

## Conclusion:
**After analyzing the data with graphs, plots and applying machine learning which predicted the passenger's fate (Survived/Died) I can cinfidantly say that a passanger's survival was most of the time not coincidential and had many influencers from being female or male, traveling in the first, second or third class or even the fare amount that was paid. Those who traveled in the first and second class had more chances of survial than those who traveled in the third class. Female passengers had higher survival chances than male passangers as most of the victims were males.**

**In conclusion, Titanic was a great tragedy and had taken many lives. But, I believe by studying these kinds of incidents and applying scientific study to them we can prevent future disasters as this one.**

***-Alexander Osokin***

## File Description:
* ***Data Analysis [Directory]:*** Contains the datasets used for data analysis and the jupyter notebook file
* ***Original_DF's [Directory]:*** Contains the original test and train data sets downloaded from kaggle
* ***Classification Models [Python File]:*** Contains the trained machine learning classification algorithms 
* ***Complete_df [CSV File]:*** The full complete titanic passenger data set
* ***data_clean [Python File]:*** Contains the cleaning code of the 'train' dataset
* ***df_training_new [CSV File]:*** New trainig dataset created after cleaning

## Code and Resources Used:
* ***Python Version:*** 3.8.2
* ***Original Data Set:*** <https://www.kaggle.com/c/titanic/data>
* ***Packages:*** pandas, numpy, matplotlib, seaborn, sklearn, pickle
* ***IDES Used:*** Anaconda, Spyder, Jupyter-Notebook
* ***Saving and Loading ML models:*** <https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/>
* ***Titanic_Proj_Example:*** <https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8>
* ***Youtube:*** Videos and explainations from Ken Jee who is a data scientist. You can look up his channel [Here](https://www.youtube.com/channel/UCiT9RITQ9PW6BhXK0y2jaeg)
