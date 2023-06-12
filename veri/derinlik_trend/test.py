import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow import keras
from sklearn.model_selection import train_test_split



txt_files = glob.glob(r"C:\Users\OmerAtalay\Desktop\derinlik_trend\*.txt")



data = []
count = 0
for file in txt_files:
    f = open(file, "r")
    #append each line in the file to a list
    df = pd.read_csv(file, delimiter = ";")
    df = df.drop('//', 1)
    df = df.replace(np.nan,0)
    d = df.iloc[:,1:].values
    
   
    count += 1
    
    
    # if count ==  1:
    #     print(file)
    #     break
    print(file)
    if file == r"C:\Users\OmerAtalay\Desktop\derinlik_trend\DOHOL.txt": # TODO: drive'a tüm veriyi yükle ve tüm doholleri ver!!!! (mart1-2-3 dosyasından DOHOL ya da FDOHOL verilecek)
        data.append(d)
        print("t")
        
        
data = np.vstack(data)

new_data = []
for i in range(data.shape[0]):
    row = []
    # buy_prices = []
    # buy_lots = []

    # sell_prices = []
    # sell_lots = []

    #for k in range(5):
    #buy_prices.append(data[i][0])
    #buy_lots.append(data[i][k * 2 + 1])

    #sell_prices.append(data[i][k * 2 + 10])
    #sell_lots.append(data[i][k * 2 + 11])

    row.append(data[i][0]) 
    #row.append(np.min(sell_prices))

    #row.append(np.sum(buy_lots))
    #row.append(np.sum(sell_lots))

    ff = list(data[i]) + row

    new_data.append(ff)

new_data = np.asarray(new_data)


CNN = False
if CNN:
    col_maxes = np.max(new_data,axis=0)
    col_maxes.shape
    n_new_data = new_data / col_maxes
    new_data = n_new_data
    
    
buy_price = new_data[61, -1]

pump = np.where(new_data[62:, -1] >= (new_data[61, -1] + 0.02))
dump = np.where(new_data[62:, -1] <= (new_data[61, -1] - 0.02))


x_ = []
y_ = []
window_size = 60
b_price = 0

for i in range(new_data.shape[0] // window_size - 1):
    buy_price = new_data[(i+1)*window_size+1, -1]
    pump = np.where(new_data[(i+1)*window_size+2:, -1] >= (buy_price + 0.01))
    dump = np.where(new_data[(i+1)*window_size+2:, -1] <= (buy_price - 0.01))
    try:
        if pump[0][0] > dump[0][0]: # N adım sonraki 2 adımda alınan lot sayısı bir önceki adıma göre artmış mı?
            b_price = 1 # dump
            y_.append([b_price])
            x_.append(new_data[i*window_size:(i+1)*window_size, 2:-1]) # n satırı kes
        elif pump[0][0] < dump[0][0]:
            b_price = 0 # pump
            y_.append([b_price])
            x_.append(new_data[i*window_size:(i+1)*window_size, 2:-1]) # n satırı kes
    except:
        b_price = 2
        y_.append([b_price])
        x_.append(new_data[i*window_size:(i+1)*window_size, 2:-1]) # n satırı kes
    
    
X = np.asarray(x_)
Y = np.asarray(y_)

X = np.reshape(X, [-1, X.shape[1] * X.shape[2]]) # 1 boyut daha ekle


#X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2)
X_train, X_test, y_train, y_test = X[:694], X[694:], Y[:694], Y[694:]
pipe = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=1000))
pipe.fit(X_train, y_train.ravel())  # apply scaling on training data




print('Accuracy on the training subset: {:.3f}'.format(pipe.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(pipe.score(X_test, y_test)))

pred_train = pipe.predict(X)  # EĞİTİM VERİSİ










