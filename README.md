# Titanic Survival Estimator: Project Overview #
* **Was the survival of some Titanic's passangers a mere coincidence? Or there were certain conditions that had major**
**role in deciding the passanger's fate?**
* **Used over 1000 samples of Titanic passanger's information from Kaggle**
* **Cleaned the original datasets and created new ones and applied them in data analysis and model training**
* **Built models using Logistic Regression, SVM, RandomForest Classifier and K-NN algorithms to predict if a passanger Survived/Died and  compared their preformance and used the best one to fill the missing "Survived" values**
* 

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

## Data Cleaning:
**After downloading the traing and test datasets I analyzed them in-order to get a quick overview and look for missing data and prepare it for model training. After analyzation I made the following changes:**
* Changed the Embarked location name from the first letter of the location to the full name.
* Dropped the Cabin column for having too much missing data
* Filled the missing values in Age column by calculating the age mean while taking the passenger's travel class and gender in consideration
* Dropped the Name column because it couldn't be used in my model training
* Made the PassengerId column as the dataset index and thus getting rid of it too
* Transformed the Sex column into numerical data 1s and 0s for each gender 
* Created dummy variables for each categorical data in the dataset as preparation for the model training and testing

## EDA:
**I looked at the distributions of the data for numerical and categorical data. Made plots that describe the dataset and made it easier to find correlation between data. Here are some examples:**

![alt text][plot1] ![alt text][plot2]
![alt text][plot3] ![alt text][plot4]


[plot1]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/Data_Analysis/corrHeatmap.png "CorrHeatmap"
[plot2]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/Data_Analysis/MaleFemaleSurvived.png "MaleFemaleSurvived"
[plot3]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/Data_Analysis/grid.png "Survivals/Casualties in classes"
[plot4]: https://github.com/AlexOsokin97/titanic_casualties_proj/blob/master/Data_Analysis/fig.png "Survivals/Deaths in each gender "

## Model Building:
**I decided to create a model which will be able to predict whether a passanger Survived or Died based on most of the passenger's info: Gender, Age, Travel Class, Had Children/Spouces, Had Parents/Siblings, Fare**
I transformed all of the categorical data which was not numerical to numerical, created dummy variables for each categorical data which was used and applied standardization in order to reduce the chances of my models being affected by high variance/bias and thus making it less accurate
**The algorithms I used were the following:**
* **Logistic Regression** - basic and easy to apply algorithm for classification
* **Support Vector Machine** - More complex algorithm, because I had many features I thought it might be a good fit. Also applied two different kernels: polynomial and sigmoid.
* **Random Forest** - More complex algorithm, Again because I had many features I thought it might be a good fit. Also applied different tree amounts: 50 ,100, 200.
* **K-Nearest Neighbor** - Was intrested to see the preformance of this algorithm

## Model Performance:
Although all the classification model's accuarcy was almost the same, the Support Vector Machine with the polynomial kernel had the best performance:

* ***Support Vector Machine (polynomial): Accuarcy- 0.815 (81.5%)*** 
* ***Support Vector Machine (sigmoid): Accuarcy- 0.814 (81.4%)*** 
* ***K- Nearest Neighbor: Accuarcy- 0.800 (80%)***
* ***Logistic Regression: Accuarcy- 0.790 (79%)*** 
* ***Random Forest Classification: Accuarcy- 0.765 (76.5%)*** 


