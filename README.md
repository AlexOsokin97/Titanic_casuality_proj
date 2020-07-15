# Titanic Passenger Survival Estimator: Project Overview #
**Was the survival of some Titanic's passengers a mere coincidence? Or there were certain conditions that had major role in deciding the passenger's fate?**

***As we all know Titanic was one of the most shocking world tragedies of the 19th century. The Titanic was the largest passenger ship ever built at the time which had around 4000 passengers on board including crew members, engineers, staff and common folk. That ship was in-fact so big that it looked unsinkable but, on April 14th, 1912 the unthinkable happened. The Titanic hit an iceberg and sunk in the early morning of April 15th, 1912. That incident had many casualties from little children to wealthy business men but, it also had many survivors. So as a Data Scientist I decided to research that tragedy and find out WHO had survived and WHY.***

***I managed to get a dataset of the Titanic's passengers with over 1000 passengers. Unfortunately, some data was missing and the dataset wasn't properly organized. By using many mathematical, statistical and programming techniques I cleaned the dataset and made it readable. 
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

***In the end, I made few more changes to the dataset such as removing more unnecessary columns and transforming categorical columns to binary to make the dataset ready for Machine Learning algorithms usage***

* **According to the data set, if you are a woman you had 80% survival chance and if you are a man you had 18% survival rate** 

* **According to the first plot we can clearly classify between the survivors and non survivors**

* **If I classify everyone as non-survivor, I would get 60% accuracy and if I classify everyone as a survivor, I'll get 40% accuracy.**

## Model Building & Performance:
***I decided to use Machine Learning to see if a model was able to achieve a high classification accuracy. The following algorithms were used:***

* **XGBoost Classifier: XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. I chose this algorithm because it is fast, outliers have less impact on it and uses less computation resources** [https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d]

* **K-NN Classifier: An algorithm which stores all given cases and classifies them to K amount of neighbors based on a similarity measure. I chose this algorithm because it is very simple to understand and implement** [https://www.saedsayad.com/k_nearest_neighbors.htm#:~:text=K%20nearest%20neighbors%20is%20a,as%20a%20non%2Dparametric%20technique.]

* **RandomForest Classifier: RandomForests or random decision forests are an ensemble learning method that operate by constructing a multitude of decision trees at training time. I chose this algorithm because it is powerful and accurate on many problems** 

* **Support Vector Machine:** The algorithm uses the help of hyperplanes to classify groups of data. I chose this algorithm because it may give high performance due to its ability to work with infinite dimensions

**The next step was applying the following techniques to make the learning process for the algorithms easier, faster and qualitative:**

* **train_test_split:** I used this technique in order to split my dataset into 2 fractions. The first largest fraction is the 'training set' which would include 80% of the whole dataset and the second smallest fraction is the 'testing set' which would include 20% from the whole dataset. I used this technique because I wanted to check how successful my models were by applying them on data they had never seen before.

* **cross_val_score:** I used this method because I wanted to check if my model's high/low performance is not accidental. I checked it by splitting my training data to 3 small datasets where 1 of 3 is used as validation data and fitting the model to each split variation. In the end I took the mean accuracy score of all the accuracies. This method lowered the probability that the model's performance was accidental and also gave me a general idea of the quality of my dataset. Each model achieved above 80% accuracy.

* **grid_search_cv:** I used this method for model's hyper parameter tuning. With this method I was able to get better classification results

* **MinMaxScaler: parameter scaling function which scales the parameters between 0,1. I used this function only when I used the SVC algorithm because it is advised to scale the data between 0,1 in order to achieve best performance**

**Moving onwards, I applied the trained models on new unseen data and evaluated their performance using the accuracy and confusion matrix metrics. Each model performed differently and had different accuracies.**

### XGBoost Classifier:
![alt text][plot1]

* ***Accuracy on testing_set: 80%*** 

### K-NN Classifier:
![alt text][plot3]

* ***Accuracy on testing_set: 72%*** 

### RandomForest Classifier:
![alt text][plot5]

* ***Accuracy on testing_set: 79%*** 

### Support Vector Machine:
![alt text][plot6]

* ***Accuracy on testing_set: 82%***

[plot1]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/cms/xgb_cm.png "xgbcm"
[plot3]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/cms/knc_cm.png "knccm"
[plot5]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/cms/rfc_cm.png "rfccm"
[plot6]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/cms/svc_cm.png "svmcm"

### Conclusion:
***I used 4 different algorithms and each of them had unique advantage which could show high performance. After training and testing 3 of the algorithms had very good performance. The best performance was shown by the Support Vector Machine algorithm achieving 82%. The reason for the algorithm's good performance I assumed was because I scaled the data thus, lowered big numerical differences in the dataset and lowered outlier influence.***

## File Description:
* ***Data Analysis [Directory]:*** Contains the datasets used for data analysis and the jupyter notebook file
* ***Original_DF's [Directory]:*** Contains the original test and train data sets downloaded from kaggle
* ***cms [Directory]:*** Contains images of each model's confusion matrix
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
