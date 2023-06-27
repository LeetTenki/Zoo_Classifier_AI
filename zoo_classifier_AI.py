# Importing methods and classes that are going to be used
import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import accuracy_score


# Reading zoo_dataset.xlsx file
read_file = pd.read_excel(r'zoo_dataset.xlsx', sheet_name='Sheet1')

# Converting to zoo_dataset.csv
read_file.to_csv(r'zoo_dataset.csv', index=None, header=True)

# Reading zoo_dataset.csv file
Animals_data=pd.read_csv('zoo_dataset.csv')

# Getting X values, excluding first and last columns using 'iloc' function.
x=Animals_data.iloc[:, 1:17].values

# Displaying X values
print('X Values: \n', x)

# Getting Y values, which is the last column using 'iloc' function.
y=Animals_data.iloc[:, 17:18].values

# Displaying Y values
print('\nY Values: \n', y)


# Spliting the dataset for training and testing
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=4)


# Training model using Gaussian Naive Bayes
gsNB =  GaussianNB()
gsNB.fit(x_train, y_train.ravel())
print('\n',gsNB)


# Testing the success of learning
gs_pred=gsNB.predict(x_test) 
print('Our Gaussian predict: ',gs_pred)

# Training model using Bernoulli Naive Bayes
bernNB = BernoulliNB()
bernNB.fit(x_train, y_train.ravel())
print('\n',bernNB)


# Testing the success of learning
bn_pred=bernNB.predict(x_test)
print('Our Bernoulli predict: ',bn_pred)

# Print the Gaussian accuracy score
print('\nGaussian Model Accuracy : ', accuracy_score(y_test,gs_pred))


# Print the Bernoulli accuracy score
print('Bernoulli Model Accuracy : ', accuracy_score(y_test,bn_pred))
