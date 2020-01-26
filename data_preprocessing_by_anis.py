#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 16:59:43 2020

@author: Anis
"""

# Importing the dataset

import pandas as pd

dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:,:3].values
y = dataset.iloc[:,3].values


# Missing data

import numpy as np
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(np.nan, strategy = 'mean')
x[:,1:3] = imputer.fit_transform(x[:,1:3])


# Categorical data 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder = LabelEncoder()
x[:,0] = labelencoder.fit_transform(x[:,0])

onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])], remainder = 'passthrough')
x = onehotencoder.fit_transform(x)

y = labelencoder.fit_transform(y)


# Splitting the Dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


# Feature Scalling

from sklearn.preprocessing import StandardScaler

standardscaler = StandardScaler()
x_train = standardscaler.fit_transform(x_train)
x_test = standardscaler.transform(x_test)




