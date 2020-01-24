#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:23:53 2020

@author: Anis
"""

# IMPORTING THE LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset

dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :3].values

y = dataset.iloc[:,3].values


# Taking care of missing data

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(np.nan, strategy = 'mean')

imputer = imputer.fit(x[:,1:3])

x[:,1:3] = imputer.transform(x[:,1:3])


# Encoding categorical data

from sklearn.preprocessing import LabelEncoder , OneHotEncoder

from sklearn.compose import ColumnTransformer

labelencoder_x = LabelEncoder()

x[:,0] = labelencoder_x.fit_transform(x[:,0])

onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],remainder='passthrough')

x = onehotencoder.fit_transform(x)

labelencoder_y = LabelEncoder()

y = labelencoder_x.fit_transform(y)


# Splitting the dataset into the training set and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)







