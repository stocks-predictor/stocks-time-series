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
from keras.layers import Dense, Dropout, LSTM, Conv2D, Reshape
from keras import metrics
from sklearn.metrics import mean_squared_error
import math
import datetime

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

def moving_average(signal, period):
    buffer = [0] * period
    for i in range(period,len(signal)):
        buffer.append(signal[i-period:i].mean())
    return buffer

# .values para pegar o numpy array
posRateTrain = moving_average(posRateTrain.values, 10)
mpsTrain = mpsTrain.values
posRateTest = moving_average(posRateTest.values, 10)
mpsTest = mpsTest.values

#preparando conjunto de treino

look_back = 50

features1_set = [] 
features2_set = []  
labels = []  
for i in range(look_back, trainLength):  
    features1_set.append(mpsTrain[i-look_back:i])
    features2_set.append(posRateTrain[i-look_back:i])
    labels.append(mpsTrain[i])

features1_set, features2_set, labels = np.array(features1_set), np.array(features2_set), np.array(labels)
features_set_train = np.array([features1_set, features2_set])
features_set_train = np.reshape(features_set_train, (features_set_train.shape[1], features_set_train.shape[2], features_set_train.shape[0]))

print('features_set_train shape = ', features_set_train.shape)

model = Sequential()

model.add(Conv2D(1, kernel_size=(1, 2), activation='relu', input_shape=(features_set_train.shape[1], features_set_train.shape[2], 1)))
model.add(Reshape((features_set_train.shape[1], 1)))
print(model.summary())
model.add(LSTM(units=50, return_sequences=True))  
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

features_set_train = np.expand_dims(features_set_train, axis=3)
model.fit(features_set_train, labels, epochs = epochs, batch_size = batch_size)

#predição com o conjunto de treino
trainPredictions = model.predict(features_set_train, batch_size=batch_size)

#preparando conjunto de teste 

feature1_testSet = [] 
feature2_testSet = []  
labels_test = []
for i in range(look_back, testLength):  
    feature1_testSet.append(mpsTest[i-look_back:i])
    feature2_testSet.append(posRateTest[i-look_back:i])
    labels_test.append(mpsTest[i])

feature1_testSet, feature2_testSet, labels_test = np.array(feature1_testSet), np.array(feature2_testSet), np.array(labels_test)
features_set_test = np.array([feature1_testSet, feature2_testSet])
features_set_test = np.reshape(features_set_test, (features_set_test.shape[1], features_set_test.shape[2], features_set_test.shape[0]))

features_set_test = np.expand_dims(features_set_test, axis=3)
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
plt.plot(np.concatenate((posRateTrain, posRateTest)), color='burlywood', label='positive news rate')
plt.plot(trainPredictPlot , color='red', label='predicted stock prices train')  
plt.plot(testPredictPlot , color='green', label='predicted stock prices test')
plt.title('Down jones index predictions (2 features)')  
plt.xlabel('Date')  
plt.ylabel('Down jones index')  
plt.xticks(locs, labels, rotation='90')
plt.grid(axis='x')
plt.legend()  
plt.show()
#plt.savefig("time-serie-predictor-2features.eps", format='eps')
#plt.clf()  
