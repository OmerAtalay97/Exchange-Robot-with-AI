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

from keras.layers import LSTM
from keras.layers.embeddings import Embedding



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


def general_normalizer(column):
    maximum_value=max(column)
    minumum_value=min(column)
    normalized_data=[]
    for r in column:
        new_r=(r-minumum_value)/(maximum_value-minumum_value)
        normalized_data.append(new_r)
    return normalized_data


def testdata(Model,girdi):
    #normalized_input=general_normalizer(girdi)
    output=Model.predict(girdi)
    for i in output:
        if i>0.66:
            print("SAT")
        elif 0.66>=i>0.33:
            print("Bekle")
        else:
            print("AL")
            
"------------------------------------------------------------------------------"            
            
data=getdata("qew.txt")
df = pd.DataFrame(data=data)

df9=pd.to_numeric(df[9])
df11= pd.to_numeric(df[11])

 

buy=[]    
         
for b in data:
        
    #b5 = float(b[1]) * float(b[2])*0.75
    b4 = float(b[3]) * float(b[4])*0.80
    b3 = float(b[5]) * float(b[6])*0.85
    b2 = float(b[7]) * float(b[8])*0.90
    b1 = float(b[9]) * float(b[10])*0.95
    sum = b1 + b2 + b3 + b4# + b5
    buy.append(sum)

sell=[]
for s in data:
    
    s1 = float(s[11]) * float(s[12])*0.95
    s2 = float(s[13]) * float(s[14])*0.90
    s3 = float(s[15]) * float(s[16])*0.85
    s4 = float(s[17]) * float(s[18])*0.80
    # s5 = float(s[19]) * float(s[20])*0.75
    sum = s1 + s2 + s3 + s4# + s5
    sell.append(sum)




price=(df9)
situation=[]
for i in range(0,len(price)):
    try:
        current_price=price[i]
        after_price=price[i+1200]
        if current_price==after_price:
            situation.append(0.5) #Bekle
        elif current_price>after_price:
            situation.append(1) #SAT
        else:
            situation.append(0) #AL
    except:
        situation.append(0.5)
        

son=pd.DataFrame(data=df[0])
son.insert(loc=1, column=1, value=buy)
son.insert(loc=2, column=2, value=sell)
son.insert(loc=3, column=3, value=situation)
        


# situation = pd.Series(situation)
# df.insert(loc=20, column=20, value=situation)


for w in range(0,len(son)):
    son[0][w]=date_normalizer(son[0][w])
for c in range(0,len(son.columns)-1):
    counter=0
    for r in son[c]:
        
        son[c][counter]=float(r)
        
        counter+=1
    son[c]=general_normalizer(son[c])
    
trained=son.iloc[0:17000,:-1]
y_data=son.iloc[0:17000,3]

embedding_vecor_length = 32
model = Sequential()
model.add(Dense(3, input_dim=(3), activation='relu'))
model.add(Dense(16, activation='relu'))
# model.add(BatchNormalization())
model.add(Embedding(5000, 32, input_length=500))
# model.add(LSTM(16))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trained, y_data, epochs=100, batch_size=50)

_, accuracy = model.evaluate(trained, y_data)
print('Accuracy: %.2f' % ((accuracy)*100))
# print(model.summary())

results = testdata(model,np.array([[0.7514873828899679	,0.5981984910845682	,0.6799156455237485]]))
print(results)




















# x=df.iloc[:,0:19]
# y=df[20]        

# seq_model=Sequential()
# seq_model.add(Dense(8,input_shape=(19, ),activation= "relu"))
# seq_model.add(Dense(4,activation="relu"))
# seq_model.add(Dense(1,activation="sigmoid"))
# seq_model.summary()

# layer1=x(shape=(19, ))
# layer2=Dense(8, activation="relu")(layer1)
# layer3=Dense(4, activation="relu")(layer2)
# output=Dense(1, activation="sigmoid")(layer3)
# func_model=Model(inputs=layer1,outputs=y)
# func_model.summary()


























            
            
            
            
            