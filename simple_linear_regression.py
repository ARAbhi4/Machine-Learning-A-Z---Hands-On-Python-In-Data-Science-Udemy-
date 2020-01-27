#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:14:32 2020

@author: xd
"""

# Simple Linear Regression

# IMPORTING THE LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset

dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:,:1].values

y = dataset.iloc[:,1].values


# Splitting the dataset into the training set and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 1/3, random_state = 0)


"""# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)
"""

# Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

# Predicting the Test set result

y_pred = regressor.predict(x_test)

# Visualising the Training set result

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train,regressor.predict(x_train) , color = 'blue')
plt.title('Salary vs Experience (Traning set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set result

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train,regressor.predict(x_train) , color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()






