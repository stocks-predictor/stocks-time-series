# -*- encoding: utf-8 -*-

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras import metrics
import math

PATH_IN= '/home/matheusgomes/TCC/stocks-time-serie/utils/dataSeriePosNeg.json'

df = pd.read_json(PATH_IN, orient='colums')

print(df.shape)

testLength = int(df.shape[0]*0.4)
trainLength = df.shape[0] - testLength

print('test length = ', testLength)
print('train length = ', trainLength)

timestamps = df.loc[0:, 'timestamp'].values

mpsTrain = df.loc[0:trainLength, 'midPriceStocks']  # já está normalizado
posRateTrain = df.loc[0:trainLength, 'pos(rate)']
mpsTest = df.loc[trainLength : trainLength + testLength, 'midPriceStocks'] # já está normalizado
posRateTest = df.loc[trainLength : trainLength + testLength, 'pos(rate)']

# .values para pegar o numpy array
posRateTrain = posRateTrain.values
mpsTrain = mpsTrain.values
posRateTest = posRateTest.values
mpsTest = mpsTest.values

look_back = 50

features_set_train = [] 
labels = []  
for i in range(look_back, trainLength):  
    features_set_train.append(mpsTrain[i-look_back:i])
    labels.append(mpsTrain[i])

features_set_train, labels = np.array(features_set_train), np.array(labels)

features_set_train = np.reshape(features_set_train, (features_set_train.shape[0], features_set_train.shape[1], 1))

print(features_set_train.shape)

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set_train.shape[1], 1)))  
model.add(Dropout(0.2))  

model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))

model.add(LSTM(units=50))  
model.add(Dropout(0.2))  

model.add(Dense(units = 1))  
model.summary()
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=[metrics.MAE, metrics.MSE])  

epochs = 1000
batch_size = 32

model.fit(features_set_train, labels, epochs = epochs, batch_size = batch_size)

#predição com o conjunto de treino
trainPredictions = model.predict(features_set_train, batch_size=batch_size)

feature1_testSet = [] 
labels_test = []
for i in range(look_back, testLength):  
    feature1_testSet.append(mpsTest[i-look_back:i])
    labels_test.append(mpsTest[i])

features_set_test, labels_test = np.array(feature1_testSet), np.array(labels_test)

features_set_test = np.reshape(features_set_test, (features_set_test.shape[0], features_set_test.shape[1], 1))

evalResults = model.evaluate(x=features_set_test, y=labels_test)

print('\n\n#test results')
print('#mse = %.4f   - mae = %.4f'%(evalResults[2], evalResults[1]))

#predição com o conjunto de teste
testPredictions = model.predict(features_set_test, batch_size=batch_size)

# shift train prediction for plotting (só para compensar o look_back)
emptyNan = np.empty_like(np.zeros(look_back))
emptyNan[:] = np.nan
trainPredictPlot = np.concatenate((np.reshape(emptyNan, (look_back, 1)), trainPredictions))

# shift test predictions for plotting
emptyNan = np.empty_like(np.zeros(trainLength + look_back))
emptyNan[:] = np.nan
testPredictPlot = np.concatenate((np.reshape(emptyNan, (trainLength + look_back, 1)), testPredictions))


def generatexTicks(interval, nlocs, labels):
    locs, nlabels = [], []
    for idx in range(0, nlocs, interval):
        locs.append(idx)
        ts = pd.to_datetime(str(labels[idx])) 
        nlabels.append(ts.strftime('%Y.%m.%d  %H:%Mh'))
    return locs, nlabels

locs, labels = generatexTicks(interval=10 , nlocs=len(posRateTrain) + len(posRateTest), labels=timestamps)

plt.figure(figsize=(10,6))  
plt.plot(np.concatenate((mpsTrain, mpsTest)), color='blue', label='average stock prices') 
plt.plot(trainPredictPlot , color='red', label='predicted stock prices') 
plt.plot(testPredictPlot , color='green', label='predicted stock prices test')
plt.title('Down jones index predictions (1 feature)')  
plt.xlabel('Date')  
plt.ylabel('Down jones index')
plt.xticks(locs, labels, rotation='90')
plt.grid(axis='x')  
plt.legend()  
plt.show()  
