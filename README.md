import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# loading the dataset
credit_card_data = pd.read_csv('creditcard.csv')
# printing the head rows of dataset
credit_card_data.head()
# printing last rows of the dataset
credit_card_data.tail()
# getting the dataset info
credit_card_data.info()
# checking the null values
credit_card_data.isnull().sum()
# checking the distributions of legit transactions and fraudlent transactions
credit_card_data['Class'].value_counts()
# this is highly unbalanced dataset
# 0--> Normal transaction
# 1--> Fraudlent transaction
# Separating the data for analysis
legit = credit_card_data[credit_card_data.Class==0]
fraud = credit_card_data[credit_card_data.Class==1]
print(legit.shape)
print(fraud.shape)
# statistical measures of data
legit.Amount.describe()
fraud.Amount.describe()
# compare the values for both transactions
credit_card_data.groupby('Class').mean()
# under sampling
# Build a sample dataset containing similar distributions of normal transaction and fraudlent transactions.

legit_sample = legit.sample(n=492)
# concatenating two dataframes
new_dataset = pd.concat([legit_sample , fraud] , axis=0)
new_dataset.head()
new_dataset.tail()
new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()
# splitting the data into Features and Targets
X=new_dataset.drop(columns='Class' , axis=1)
Y=new_dataset['Class']
print(X)
print(Y)
# split the data into training data and testing data
X_train , X_test , Y_train , Y_test=train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape , X_train.shape , X_test.shape)
# Model training 
# Logistic regression model
model = LogisticRegression()
# training the Logistic Regression Model with training data
model.fit(X_train , Y_train)
# Model Evaluation
# Accuracy Score
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction , Y_train)
print("Accuracy on training data : " , training_data_accuracy)
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction , Y_test)
print("Accuracy score on test data : " , test_data_accuracy)
