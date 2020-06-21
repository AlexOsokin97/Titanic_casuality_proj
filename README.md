# Titanic Passenger Survival Estimator: Project Overview #
**Was the survival of some Titanic's passengers a mere coincidence? Or there were certain conditions that had major role in deciding the passanger's fate?**

***As we all know Titanic was one of the most shocking world tragedy of the 19th century. The Titanic was the largest passenger ship ever built at the time which had around 4000 passengers on board including crew members, engineers, staff and common folk. That ship was in-fact so big that it looked unsinkable but, on April 14th, 1912 the unthinkable happened. The Titanic hit an iceberg and sunk in the early morning of April 15th, 1912. That incident had many casualties from little children to wealthy business men but, it also had many survivors. So as a Data Scientist I decided to research that tragedy and find out WHO had survived and WHY.***

***I managed to get a dataset of the Titanic's passengers with over 1000 passengers. Unfortunately, some data was missing and the dataset wasn't properly orginized. By using many mathematical, statistical and programming techniques I cleaned the dataset and made it readable. 
Also, I looked up and used other techniques which were better/worse than the original.***

***After the cleaning process I used statistical techniques to visualize the data and find correlation between different features in the dataset but most importantly to find which features correlate with the passengers' survival.***

***In the end, I applied Machine Learning algorithms to check if the data is sufficient enough for the algorithms to classify whether a passenger survived or did not survive so that in the future if new passenger information is found we could use the model to check if that passenger survived or not.***

## Data Cleaning & Visualization:
***Getting the necessary datasets is one thing but, making it readable, useful and useable is another thing. This is where the data cleaning and remodeling process came in.***

***Most of the changes that I had done to the dataset are simple such as: changing column names, column data types and removing columns with too much missing data and unnecessary. The more advanced technique I used was when I filled the missing values of the 'Age' column where I used the mean age of all samples while taking into consideration the passenger's gender and traveling class***

***After the data was cleaned, made readable and useful I did exploratory data analysis to dive deeper into the dataset and find correlations between different features and most importantly which features correlate with the passenger's survival. Here are some examples from my data exploration***

![alt text][plot2] ![alt text][plot4]

[plot2]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/Data_Analysis/MaleFemaleSurvived.png "MaleFemaleSurvived"
[plot4]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/Data_Analysis/fig.png "Survivals/Deaths in each gender "

***In the end, I made few more changes to the dataset such as removing more unnecessary columns and transfroming categorical columns to binary to make the dataset ready for Machine Learning algorithms usage***

## Model Building & Performance:
***I was intrested to find out if there is a function that could predict if a passenger survived/not survived based on the given data so, I decided to apply Machine Learning classification algorithms to the dataset to see if they successfuly managed to generalize the data and make accurate predictions. The following algorithms were used:***

* **Logistic Regression:** Linear based model used for binary classification. Chose it because there could be only 2 predicted values (survived/not survived also can be written as 1/0).

* **KNeighborsClassifier:** I chose this algorithm because after the exploritory data analysis phase, I saw that most of the survivors had specific features which differ from the non survivors features. As a result, I assumed that whenever a new data point is plugged in, it would be closer to the correct neighboring points and would be classified correctly

* **Naive Bayes Classifier:** The Gaussian Naive Bays algorithm uses given tree branches to compute possible probabilities. In other words, it allows the quick classification of (in my case) whether the passeneger survived or not survived. In addition, the dataset is relatively small and therefore, the algorithm could show high performance

* **Support Vector Machine:** The algorithm uses the help of hyperplanes to classify groups of data. Because the goal of the algorithm was to classify whether a passenger survived or not survived (1 or 0 which is binary classification) this algorithm could show high performance In addition, it is able to work with infinite dimensions.

**The next step was applying the following techniques to make the learning process for the algorithms easier, faster and qualitative:**

* **train_test_split:** I used this technique in order to split my dataset into 2 fractions. The first largest fraction is the 'training set' which would include 80% of the whole dataset and the second smallest fraction is the 'testing set' which would include 20% from the whole dataset. I used this technique because I wanted to check how succeful my models were by applying them on data they had never seen before.

* **cross_val_score:** I used this method because I wanted to check if my model's high/low performance is not accidental. I checked it by splitting my training data to 3 small datasets where 1 of 3 is used as testing data and fitting the model to each split variation. In the end I took the mean accuracy score of all the accuracies scores. This method removed the probabilty that the model's performance was accidental and also gave me a general idea of the quality of my dataset. Each model achieved above 80% accuracy.

* **grid_search_cv:** This method is very powerful for model selection but also takes some time to execute. Because the models I used have many parameters that you can change and apply it wouldn't be smart to apply a naive model with its' default parameters. So, by applying grid search I gave the models to experiment on all different parameters and pick the best model which had the best score.

* **standardization:** I applied standardization in order to scale the data and make the learning process easier and faster. I used this method only when I applied The support vector machine algorithm. I did not use normilization because the distribution of features varied and was not normal.

**Moving onwards, I applied the trained models on new unseen data and evaluated their performance using the accuracy and confusion matrix metrics. Each model performed differently and had different accuracies.**

#### Logistic Regression:
![alt text][plot1]


#### KNeighbors Classifier:
![alt text][plot3]


#### Naive Bayes Classifier:
![alt text][plot5]

#### Support Vector Machine:
![alt text][plot6]




[plot1 =256x256]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/models_%26_cms/Logistic_Regression_cm.png "LRCM"
[plot3]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/models_%26_cms/K-NearestNeighbors_cm.png "KNCCM"
[plot5]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/models_%26_cms/Gaussian_Naive_Bayes_cm.png "NBCCM"
[plot6]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/models_%26_cms/Support_Vector_Machine.png "SVMCM"
**Lastly, I decided to choose (model) as the classification algorithm for this case because (...)**

## Conclusion:


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
