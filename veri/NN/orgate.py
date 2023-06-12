# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 17:29:42 2022

@author: OmerAtalay
"""


import numpy as np

table=np.array([[0,0],[0,1],[1,0],[1,1]])

#and
#logic=np.array([[0],[0],[0],[1]])

#or
logic=np.array([[0],[1],[1],[1]])

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))

# Adding the output layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Adding the output layer
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
# Fitting the ANN to the Training set
classifier.fit(table, logic, batch_size = 4, epochs = 3000)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(np.array([[0,0],[0,1],[1,0],[1,1]]))
y_pred = (y_pred > 0.5)