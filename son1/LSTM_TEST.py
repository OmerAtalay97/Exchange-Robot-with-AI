# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 20:17:38 2022

@author: OmerAtalay
"""
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dense,Dropout,BatchNormalization,Activation
from keras.layers.core import Activation
from keras.layers.core import Dropout
#from keras.utils.np_utils import to_categorical

from tensorflow.keras.optimizers import SGD
model = keras.models.load_model("DOHOL_train")
model.summary()


def general_normalizer(column):
    maximum_value=max(column)
    minumum_value=min(column)
    normalized_data=[]
    for r in column:
        new_r=(r-minumum_value)/(maximum_value-minumum_value)
        normalized_data.append(new_r)
    return normalized_data

def date_normalizer(date):
    #date='11.02.2022 09:30:03'
    dates=date.split(" ")
    date=dates[0]
    hours=dates[1]
    
    dates=date.split(".")
    
    days=int(dates[0])
    month=int(dates[1])
    
    dateseconds=(days*24+(month*30*24))*3600
    
    hours=hours.split(":")
    
    hour=int(hours[0])
    minute=int(hours[1])
    second=int(hours[2])
    
    output=dateseconds+(hour*3600)+(minute*60)+second
    return output
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

def testdata(Model,girdi):
    #normalized_input=general_normalizer(girdi)
    output=Model.predict(girdi)
    for i in output:
        if i==0:
            print("AL")
        elif i == 1:
            print("SAT")
        else:
            print("BEKLE")
data=getdata("DOHOL.txt")
df = pd.DataFrame(data=data)

df1=pd.to_numeric(df[1])
df11= pd.to_numeric(df[11])

price=df1
situation=[]


for i in range(0,len(price)):
    try:
        current_price=price[i]
        after_price=price[i+1]
        if current_price==after_price:
            situation.append(1) #Bekle
        elif current_price>after_price:
            situation.append(0.5) #SAT
        else:
            situation.append(0) #AL
    except:
        situation.append(1)
situation = pd.Series(situation)
df.insert(loc=20, column=20, value=situation)


for w in range(0,len(df)):
    df[0][w]=date_normalizer(df[0][w])
for c in range(0,len(df.columns)-1):
    counter=0
    for r in df[c]:
        
        df[c][counter]=float(r)
        
        counter+=1
    df[c]=general_normalizer(df[c])
    

train_data=df.iloc[:,0:19]
y_data=df[20]


anaTrain=train_data.values
anaTest=y_data.values
ilkshiftTrain=train_data.shift(1, axis = 0).values
ikincishiftTrain=train_data.shift(2, axis = 0).values
ucuncushiftTrain=train_data.shift(3, axis = 0).values




temp=np.zeros(train_data.shape + (3,))
temp[:, :, 0]=anaTrain
temp[:, :, 1]=ilkshiftTrain
temp[:, :, 2]=ikincishiftTrain
train_datam=temp[3:,:,:]

result=model.predict(train_datam)

result=general_normalizer(result)
result = np.asarray(result)

result[result >= 0.7  ] = 1
result[(result < 0.7) & (result > 0.3)] = 0.5
result[result <= 0.3] = 0


real_result=y_data.shift(-3)
real_result=real_result[:-3]
real_result=real_result.values
plt.plot(real_result) 

plt.plot(result)
plt.show()










