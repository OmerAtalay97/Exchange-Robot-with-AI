# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:52:43 2022

@author: OmerAtalay
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dense,Dropout,BatchNormalization,Activation
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import SGD







def getdata(data):
    
    file = open(data, "r")
    f= file.readlines()

    newList= []
    
    for line in f:
        newList.append(line[1:-2])
    
    sonList=[]
    i=0
    for x in range(len(newList)-1):
        a=newList[x].split(",")
        sonList.append(a)
        i=i+1
    return sonList

data=getdata("test.txt")
df = pd.DataFrame(data=data)
df1=pd.to_numeric(df[0])
df2=pd.to_numeric(df[1])
df3=pd.to_numeric(df[2])

df=pd.DataFrame(data=df1)
df.insert(loc=1, column=1, value=df2)
df.insert(loc=2, column=2, value=df3)


trained=df.iloc[0:60,:-1]
y_data=df.iloc[0:60,2]

model = Sequential()
model.add(Dense(2, input_dim=2, input_shape=(2,),activation='relu'))
model.add(Dense(8, activation='sigmoid'))
# model.add(BatchNormalization())
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trained, y_data, epochs=220, batch_size=2)

_, accuracy = model.evaluate(trained, y_data)
print('Accuracy: %.2f' % (accuracy*100))
print(model.summary())

output=model.predict(np.array([[0,1]]))
print(output)
