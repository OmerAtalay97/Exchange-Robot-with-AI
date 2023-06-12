import pandas as pd
import numpy as np

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

output_size = 2         # sonraki 4 degeri tahmin etmeye calis
epochs = 700            # islemi tekrar sayisi
features = 3

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
def get_data(y):
    # lstm datayi (rows, timesteps, features) formatinda ister
    # rows : ornek sayimiz
    # timesteps : zaman adimlari
    # features : ogrenme verimizin sutun sayisi
    # training verisi (rows, 1, features)
    # output verisi (rows, output_size)

    train_size = int(len(y)*0.7)                # Verinin %70 ini egitim icin kalaninin da test icin ayiracagiz, 180

    train = y[0:train_size]
    test = y[train_size:len(y)]
    # print("Shape : ", train.shape)

    ############## TRAIN DATA ####################
    train_x = []
    train_y = []
    for i in range(0, train_size - features - output_size):
        tmp_x = y[i:(i+features)]
        tmp_y = y[(i+features):(i+features+output_size)]
        train_x.append(np.reshape(tmp_x, (1, features)))
        train_y.append(tmp_y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    ########### TEST DATA ########################
    test_x = []
    test_y = []
    last = len(y) - output_size - features
    for i in range(train_size, last):
        tmp_x = y[i:(i+features)]
        tmp_y = y[(i + features):(i + features + output_size)]
        test_x.append(np.reshape(tmp_x, (1, features)))
        test_y.append(tmp_y)

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    ######## Tahmin edilecek data #######################
    data_x = []
    tmp_x = y[-features:len(y)]
    data_x.append(np.reshape(tmp_x, (1, features)))
    data_x = np.array(data_x)

    return train_x, train_y, test_x, test_y, data_x

data=getdata("xxx.txt")
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

raw_data = pd.read_csv('./datasets/ue128.csv', header=None, names=["i", "t", "y"])

t = np.array(raw_data.t.values)
y = np.array(raw_data.y.values)

min = y.min()
max = y.max()

y = np.interp(y, (min, max), (-1, +1))

x_train, y_train, x_test, y_test, data_x = get_data(y)

model = Sequential()
model.add(LSTM(5, input_shape=(1, features), return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(7))
model.add(Dense(output_size))

model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epochs, verbose=0)

score = model.evaluate(x_test, y_test)
print("%2s: %.2f%%" % (model.metrics_names[1], score[1]*100))
model.summary()

data_y = model.predict(data_x)

result = np.interp(data_y, (-1, +1), (min, max))

print("Gelecekteki Degerler (output_size) :", result)