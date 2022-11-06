# Credit_Risk_Analysis

## Overview of the analysis: 
##### The purpose of this analysis is to use machine learning to perform data cleaning on the Credit Risk data set and use resampling techniques to understand the quality of the data. This is done through splitting the data into training and testing and making loan status the target variable within the data. The performance of the data is tested by naive random oversample, SMOTE oversampling to compare the algorithms along with undersampling, combination (over and under sampling),    ,  . 
---
## Results:

#### The main steps are as follows:
* ##### 1. View the count of the target classes using Counter from the collections library.
* ##### 2. Use the resampled data to train a logistic regression model.
* ##### 3. Calculate the balanced accuracy score from sklearn.metrics.
* ##### 4. Print the confusion matrix from sklearn.metrics.
* ##### 5. Generate a classication report using the imbalanced_classification_report from imbalanced-learn.
###### Precision = TP/(TP + FP)
#### The original split data that shows the counts for low and high risk loans:

![image](https://user-images.githubusercontent.com/105329532/200195823-3a840180-6594-4989-9789-7492532a8bbe.png) 

---
### Naive Random Oversampling 
* ##### The Balance Acurracy Score:  The score is at about 65%
* ##### Precision: The precision score is calculated at 0.0097 which is barely 1%.
![image](https://user-images.githubusercontent.com/105329532/200196031-6aba69ac-9ee7-4edb-ad15-6dd5cd788bd7.png)

---
### SMOOT Oversampling 
* ##### The Balance Acurracy Score:   The score is at about 64% which very closely resembles the accuracy resembled using random sampling for the the algorithm.
* ##### Precision: The precison score is also the same as the score from random oversampling (.0097)
![image](https://user-images.githubusercontent.com/105329532/200196091-6e97b849-860e-4e4d-96ad-ccdf7b63ca68.png)


### Undersampling 
* ##### The Balance Acurracy Score:  The accuracy score is at almost 58% which is a lower score in comparison to oversampling the data
* ##### Precision: The precion score is .0081 which is also lower in comparison to oversampling.
![image](https://user-images.githubusercontent.com/105329532/200196159-e59c6388-6363-4c11-b666-3ecb4732bc88.png)

---
### Combination Sampling 
* ##### The Balance Acurracy Score: The accuracy score closeloy resembles the score shown for undersamply (about 58%) 
* ##### Precision: The precision score is the same as the oversampling precion score at .0097
![image](https://user-images.githubusercontent.com/105329532/200196478-d65c97c8-1e36-46b7-8d50-271e2006492c.png)


---
### Balanced Random Forest Classifier 
* ##### The Balance Acurracy Score: The accuracy was shown to be at about 75%
* ##### Precision: The precision score is at abot 3% which is a higher score than both the over and under sampling techniques.
![image](https://user-images.githubusercontent.com/105329532/200196712-e78e5672-fdaa-4720-9f1f-49d8e8696a6a.png)

---
### Easy Ensemble Classifier 
* ##### The Balance Acurracy Score: The accuracy score is shown at 93%, which is the highest percentage out of all of the algorithm resampling techniques. 
* ##### Precision: The precision score is at abot 8.8% which is the highest score out og all of the techniques.
![image](https://user-images.githubusercontent.com/105329532/200196890-f9b8ec97-f2e0-49ed-a1a1-f55416e89214.png)

---
### Summary: 
##### The lowest performing models were mainly the undersampling and oversampling models. Both of theses techniques produced results that revealed low accuracy and precision within the data's algorithm . The combination method which is a mix between under and over sampling data resulted in results that were similar to those techniques. It did not produce high scores for the data accuracy and precision; the accuracy resembled the undersampling and the preciosion score resemble that of the oversampled data. The second highest scoring model was the Balanced Random Forest Classifier with 75% accuracy which is a pretty good range to be in,but it was not as high as the Easy Ensemble Classifier. I would reccomend using the Easy Ensemle Classifier because it had an accuracy score of 93% which is goob because it has a high accuracy, but not too much to the point of overfitting the dataset. The precsion score was also the highest using this model which means that the positive classification of the data is the most reliable using this technique. 











