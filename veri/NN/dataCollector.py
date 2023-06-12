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
"-----------------------------------------------------------------------------"

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

"-----------------------------------------------------------------------------"

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

"-----------------------------------------------------------------------------"


def general_normalizer(column):
    maximum_value=max(column)
    minumum_value=min(column)
    normalized_data=[]
    for r in column:
        new_r=(r-minumum_value)/(maximum_value-minumum_value)
        normalized_data.append(new_r)
    return normalized_data


"-----------------------------------------------------------------------------"
# data=getdata("a.txt")  

# buy=[]    
         
# for b in data:
        
#     b5 = float(b[1]) * float(b[2])*0.75
#     b4 = float(b[3]) * float(b[4])*0.80
#     b3 = float(b[5]) * float(b[6])*0.85
#     b2 = float(b[7]) * float(b[8])*0.90
#     b1 = float(b[9]) * float(b[10])*0.95
#     sum = b1 + b2 + b3 + b4 + b5
#     buy.append(sum)

# sell=[]
# for s in data:
    
#     s1 = float(s[11]) * float(s[12])*0.95
#     s2 = float(s[13]) * float(s[14])*0.90
#     s3 = float(s[15]) * float(s[16])*0.85
#     s4 = float(s[17]) * float(s[18])*0.80
#     s5 = float(s[19]) * float(s[20])*0.75
#     sum = s1 + s2 + s3 + s4 + s5
#     sell.append(sum)
  
    
# df = pd.DataFrame(data=data)
# for w in range(0,len(df)):
#     df[0][w]=date_normalizer(df[0][w])
# for c in range(0,len(df.columns)):
#     counter=0
#     for r in df[c]:
        
#         df[c][counter]=float(r)
        
#         counter+=1
#     df[c]=general_normalizer(df[c])
# #print(df)  

# genel=[]
# genel.append(buy)
# genel.append(sell)

"-----------------------------------------------------------------------------"  

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


# def testdata(Model,girdi):
#     #normalized_input=general_normalizer(girdi)
#     output=Model.predict(girdi)
#     for i in output:
#         if i>0.66:
#             print("SAT")
#         elif 0.66>=i>0.33:
#             print("Bekle")
#         else:
#             print("AL")
            


"-----------------------------------------------------------------------------"
data=getdata("www.txt")
df = pd.DataFrame(data=data)

df1=pd.to_numeric(df[1])
df11= pd.to_numeric(df[11])



price=df1
situation=[]
for i in range(0,len(price)):
    try:
        current_price=price[i]
        after_price=price[i+400]
        if current_price==after_price:
            situation.append(2) #Bekle
        elif current_price>after_price:
            situation.append(1) #SAT
        else:
            situation.append(0) #AL
    except:
        situation.append(2)

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
    

trained=df.iloc[:,0:19]
y_data=df[20]

# model = Sequential()
# model.add(Dense(input_dim=trained.shape[1],
#                 output_dim = 19,
#                 init =   'uniform',
#                 activation = 'tanh'))


# model.add(Dense(50, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(64, init='uniform'))
# model.add(Activation('relu'))
# model.add(Dense(10, init='uniform'))
# model.add(Activation('softmax'))

# y_train_ohe = to_categorical(y_data)

# sgd= SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# model.compile(loss = 'categorical_crossentropy',
#               optimizer = sgd)

# model.fit(trained,
#           y_train_ohe,
#           nb_epoch = 50,
#           batch_size = 500,
#           validation_split = 0.1,
#           verbose = 1)

# y_test_predictions = model.predict_classes(np.zeros((2,19)), verbose = 1)


model = Sequential()
model.add(Dense(2, input_dim=19, activation='relu'))
model.add(Dense(4, activation='relu'))
# model.add(BatchNormalization())
model.add(Dense(8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trained, y_data, epochs=200, batch_size=50)

_, accuracy = model.evaluate(trained, y_data)
print('Accuracy: %.2f' % (accuracy*100))
# print(model.summary())

results = testdata(model,np.zeros((2,19)))
print(results)



































# trained=df.iloc[:,0:19]
# y_data=df[20]

# model = Sequential()
# model.add(Dense(10, input_dim=19, activation='relu'))
# model.add(Dense(64, activation='relu'))
# # model.add(BatchNormalization())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='tanh'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(trained, y_data, epochs=20, batch_size=5)

# _, accuracy = model.evaluate(trained, y_data)
# print('Accuracy: %.2f' % (accuracy*100))
# # print(model.summary())

# results = testdata(model,np.zeros((2,19)))
# print(results)



        

