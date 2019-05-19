# -*- encoding: utf-8 -*-

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import argparse
import math
import datetime
from models.model2features import LstmTwoFeature
from models.model1features import LstmOneFeature
from keras import metrics
from keras.models import load_model

class PredictorTwoFeat():
    def __init__(self, look_back):
        self.model = None
        self.look_back = look_back

    def predict_model(self, batch_size = 32):
        self.predictions = self.model.predict([self.features1_set, self.features2_set], batch_size=batch_size)
        
        evalResults = self.model.evaluate(x=[self.features1_set, self.features2_set], y=self.labels)
        print('\n\n#test results')
        print('#mse = %.4f   - mae = %.4f'%(evalResults[2], evalResults[1]))

    def data_preparation(self, dataSeriePath):
        df = pd.read_json(dataSeriePath, orient='colums')

        self.length = int(df.shape[0])

        self.timestamps = df.loc[0:, 'timestamp'].values

        self.mps = df.loc[0:self.length, 'midPriceStocks']  # já está normalizado
        self.posRate = df.loc[0:self.length, 'pos(rate)']
        
        def moving_average(signal, period):
            buffer = [0] * period
            for i in range(period,len(signal)):
                buffer.append(signal[i-period:i].mean())
            return buffer

        # .values para pegar o numpy array
        self.posRate = self.posRate.values
        self.mps = self.mps.values
        
        self.features1_set = [] 
        self.features2_set = []
        self.labels = []

        for i in range(self.look_back, self.length):  
            self.features1_set.append(self.mps[i-self.look_back:i])
            self.features2_set.append(self.posRate[i-self.look_back:i])
            self.labels.append(self.mps[i])  

        self.features1_set, self.features2_set, self.labels = np.array(self.features1_set), np.array(self.features2_set), np.array(self.labels)
        self.features1_set =  np.expand_dims(self.features1_set, axis=3)
        self.features2_set =  np.expand_dims(self.features2_set, axis=3)

    def plot_serie(self):
        def generatexTicks(interval, nlocs, labels):
            locs, nlabels = [], []
            for idx in range(0, nlocs, interval):
                locs.append(idx)
                ts = pd.to_datetime(str(labels[idx])) 
                nlabels.append(ts.strftime('%Y.%m.%d  %H:%Mh'))
            return locs, nlabels

        locs, labels = generatexTicks(interval=10 , nlocs=len(self.posRate), labels=self.timestamps)

        plt.figure(figsize=(10,6))
        plt.plot(self.mps, color='blue', label='average stock prices') 
        plt.plot(self.posRate, color='burlywood', label='positive news rate')
        plt.plot(self.predictions , color='red', label='predicted stock prices')  
        plt.title('predictions (2 features)')  
        plt.xlabel('Date')  
        plt.ylabel('predictions')  
        plt.xticks(locs, labels, rotation='90')
        plt.grid(axis='x')
        plt.legend()  
        plt.show()
    
    def load_model(self, name = 'my_model_2features'):
        if(os.path.exists('%s.h5'%(name))):
            self.model = load_model('%s.h5'%(name))

class PredictorOneFeat():
    def __init__(self, look_back):
        self.model = None
        self.look_back = look_back

    def predict_model(self, batch_size = 32):
        self.predictions = self.model.predict(self.features1_set, batch_size=batch_size)
        
        evalResults = self.model.evaluate(x=self.features1_set, y=self.labels)
        print('\n\n#test results')
        print('#mse = %.4f   - mae = %.4f'%(evalResults[2], evalResults[1]))

    def data_preparation(self, dataSeriePath):
        df = pd.read_json(dataSeriePath, orient='colums')

        self.length = int(df.shape[0])

        self.timestamps = df.loc[0:, 'timestamp'].values

        self.mps = df.loc[0:self.length, 'midPriceStocks']  # já está normalizado
        
        def moving_average(signal, period):
            buffer = [0] * period
            for i in range(period,len(signal)):
                buffer.append(signal[i-period:i].mean())
            return buffer

        # .values para pegar o numpy array
        self.mps = self.mps.values
        
        self.features1_set = [] 
        self.labels = []

        for i in range(self.look_back, self.length):  
            self.features1_set.append(self.mps[i-self.look_back:i])
            self.labels.append(self.mps[i])  

        self.features1_set, self.labels = np.array(self.features1_set), np.array(self.labels)
        self.features1_set = np.expand_dims(self.features1_set, axis=3)

    def plot_serie(self):
        def generatexTicks(interval, nlocs, labels):
            locs, nlabels = [], []
            for idx in range(0, nlocs, interval):
                locs.append(idx)
                ts = pd.to_datetime(str(labels[idx])) 
                nlabels.append(ts.strftime('%Y.%m.%d  %H:%Mh'))
            return locs, nlabels

        locs, labels = generatexTicks(interval=10 , nlocs=len(self.mps), labels=self.timestamps)

        plt.figure(figsize=(10,6))
        plt.plot(self.mps, color='blue', label='average stock prices') 
        plt.plot(self.predictions , color='red', label='predicted stock prices')  
        plt.title('predictions (1 features)')  
        plt.xlabel('Date')  
        plt.ylabel('predictions')  
        plt.xticks(locs, labels, rotation='90')
        plt.grid(axis='x')
        plt.legend()  
        plt.show()
    
    def load_model(self, name = 'my_model_1feature'):
        if(os.path.exists('%s.h5'%(name))):
            self.model = load_model('%s.h5'%(name))

class TimeSeriePredictorTwoFeatures():
    def __init__(self, look_back):
        self.model = None
        self.look_back = look_back
    def train_model(self, epochs, batch_size = 32):
        print('feature1 shape = ', self.features1_train_set.shape)
        print('feature2 shape = ', self.features2_train_set.shape)
        print('labels shape = ', self.labels.shape)

        self.model = LstmTwoFeature(self.features1_train_set.shape[1], self.features2_train_set.shape[1]).get_model()
        self.model.summary()
        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=[metrics.MAE, metrics.MSE])
        self.model.fit([self.features1_train_set, self.features2_train_set], self.labels, epochs = epochs, batch_size = batch_size)
        
        self.trainPredictions = self.model.predict([self.features1_train_set, self.features2_train_set], batch_size=batch_size)

    def test_model(self, batch_size = 32):
        self.testPredictions = self.model.predict([self.features1_test_set, self.features2_test_set], batch_size=batch_size)
        
        evalResults = self.model.evaluate(x=[self.features1_test_set, self.features2_test_set], y=self.labels_test)
        print('\n\n#test results')
        print('#mse = %.4f   - mae = %.4f'%(evalResults[2], evalResults[1]))

    def data_preparation(self, dataSeriePath):
        df = pd.read_json(dataSeriePath, orient='colums')

        self.testLength = int(df.shape[0]*0.4)
        self.trainLength = df.shape[0] - self.testLength

        print('test length = ', self.testLength)
        print('train length = ', self.trainLength)

        self.timestamps = df.loc[0:, 'timestamp'].values

        self.mpsTrain = df.loc[0:self.trainLength, 'midPriceStocks']  # já está normalizado
        self.posRateTrain = df.loc[0:self.trainLength, 'pos(rate)']
        self.mpsTest = df.loc[self.trainLength : self.trainLength + self.testLength, 'midPriceStocks'] # já está normalizado
        self.posRateTest = df.loc[self.trainLength : self.trainLength + self.testLength, 'pos(rate)']

        def moving_average(signal, period):
            buffer = [0] * period
            for i in range(period,len(signal)):
                buffer.append(signal[i-period:i].mean())
            return buffer

        # .values para pegar o numpy array
        self.posRateTrain = self.posRateTrain.values
        self.mpsTrain = self.mpsTrain.values
        self.posRateTest = self.posRateTest.values
        self.mpsTest = self.mpsTest.values

        self.features1_train_set = [] 
        self.features2_train_set = []
        self.labels = []

        for i in range(self.look_back, self.trainLength):  
            self.features1_train_set.append(self.mpsTrain[i-self.look_back:i])
            self.features2_train_set.append(self.posRateTrain[i-self.look_back:i])
            self.labels.append(self.mpsTrain[i])  

        self.features1_train_set, self.features2_train_set, self.labels = np.array(self.features1_train_set), np.array(self.features2_train_set), np.array(self.labels)
        self.features1_train_set =  np.expand_dims(self.features1_train_set, axis=3)
        self.features2_train_set =  np.expand_dims(self.features2_train_set, axis=3)
        
        #Preparando conjunto de teste
        self.features1_test_set = []
        self.features2_test_set = []
        self.labels_test = []
        for i in range(self.look_back, self.testLength):  
            self.features1_test_set.append(self.mpsTest[i-self.look_back:i])
            self.features2_test_set.append(self.posRateTest[i-self.look_back:i])
            self.labels_test.append(self.mpsTest[i])

        self.features1_test_set, self.features2_test_set, self.labels_test = np.array(self.features1_test_set), np.array(self.features2_test_set), np.array(self.labels_test)
        
        self.features1_test_set =  np.expand_dims(self.features1_test_set, axis=3)
        self.features2_test_set =  np.expand_dims(self.features2_test_set, axis=3)

    def plot_serie(self):
        def generatexTicks(interval, nlocs, labels):
            locs, nlabels = [], []
            for idx in range(0, nlocs, interval):
                locs.append(idx)
                ts = pd.to_datetime(str(labels[idx])) 
                nlabels.append(ts.strftime('%Y.%m.%d  %H:%Mh'))
            return locs, nlabels

        # shift train prediction for plotting (só para compensar o look_back)
        emptyNan = np.empty_like(np.zeros(self.look_back))
        emptyNan[:] = np.nan
        trainPredictPlot = np.concatenate((np.reshape(emptyNan, (self.look_back, 1)), self.trainPredictions))

        # shift test predictions for plotting
        emptyNan = np.empty_like(np.zeros(self.trainLength + self.look_back))
        emptyNan[:] = np.nan
        testPredictPlot = np.concatenate((np.reshape(emptyNan, (self.trainLength + self.look_back, 1)), self.testPredictions))

        locs, labels = generatexTicks(interval=10 , nlocs=len(self.posRateTrain) + len(self.posRateTest), labels=self.timestamps)

        plt.figure(figsize=(10,6))
        plt.plot(np.concatenate((self.mpsTrain, self.mpsTest)), color='blue', label='average stock prices') 
        plt.plot(np.concatenate((self.posRateTrain, self.posRateTest)), color='burlywood', label='positive news rate')
        plt.plot(trainPredictPlot , color='red', label='predicted stock prices train')  
        plt.plot(testPredictPlot , color='green', label='predicted stock prices test')
        plt.title('Down jones index predictions (2 features)')  
        plt.xlabel('Date')  
        plt.ylabel('Down jones index')  
        plt.xticks(locs, labels, rotation='90')
        plt.grid(axis='x')
        plt.legend()  
        plt.show()
    
    def save(self, name = 'my_model_2features'):
        self.model.save('%s.h5'%(name))
    
    def load_model(self, name = 'my_model_2features'):
        if(os.path.exists('%s.h5'%(name))):
            self.model = load_model('%s.h5'%(name))

class TimeSeriePredictorOneFeature():
    def __init__(self, look_back):
        self.model = None
        self.look_back = look_back
    def train_model(self, epochs, batch_size = 32):
        print('feature1 shape = ', self.features1_train_set.shape)
        print('labels shape = ', self.labels.shape)

        self.model = LstmOneFeature((self.features1_train_set.shape[1], 1)).get_model()
        self.model.summary()
        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=[metrics.MAE, metrics.MSE])
        self.model.fit(self.features1_train_set, self.labels, epochs = epochs, batch_size = batch_size)
        
        self.trainPredictions = self.model.predict(self.features1_train_set, batch_size=batch_size)

    def test_model(self, batch_size = 32):
        self.testPredictions = self.model.predict(self.features1_test_set, batch_size=batch_size)
        
        evalResults = self.model.evaluate(x=self.features1_test_set, y=self.labels_test)
        print('\n\n#test results')
        print('#mse = %.4f   - mae = %.4f'%(evalResults[2], evalResults[1]))

    def data_preparation(self, dataSeriePath):
        df = pd.read_json(dataSeriePath, orient='colums')

        self.testLength = int(df.shape[0]*0.4)
        self.trainLength = df.shape[0] - self.testLength

        print('test length = ', self.testLength)
        print('train length = ', self.trainLength)

        self.timestamps = df.loc[0:, 'timestamp'].values

        self.mpsTrain = df.loc[0:self.trainLength, 'midPriceStocks']  # já está normalizado
        self.mpsTest = df.loc[self.trainLength : self.trainLength + self.testLength, 'midPriceStocks'] # já está normalizado

        def moving_average(signal, period):
            buffer = [0] * period
            for i in range(period,len(signal)):
                buffer.append(signal[i-period:i].mean())
            return buffer

        # .values para pegar o numpy array
        self.mpsTrain = self.mpsTrain.values
        self.mpsTest = self.mpsTest.values

        self.features1_train_set = [] 
        self.labels = []

        for i in range(self.look_back, self.trainLength):  
            self.features1_train_set.append(self.mpsTrain[i-self.look_back:i])
            self.labels.append(self.mpsTrain[i])  

        self.features1_train_set, self.labels = np.array(self.features1_train_set), np.array(self.labels)
        self.features1_train_set =  np.expand_dims(self.features1_train_set, axis=3)
        
        #Preparando conjunto de teste
        self.features1_test_set = []
        self.labels_test = []
        for i in range(self.look_back, self.testLength):  
            self.features1_test_set.append(self.mpsTest[i-self.look_back:i])
            self.labels_test.append(self.mpsTest[i])

        self.features1_test_set, self.labels_test = np.array(self.features1_test_set), np.array(self.labels_test)
        
        self.features1_test_set =  np.expand_dims(self.features1_test_set, axis=3)
    
    def plot_serie(self):
        def generatexTicks(interval, nlocs, labels):
            locs, nlabels = [], []
            for idx in range(0, nlocs, interval):
                locs.append(idx)
                ts = pd.to_datetime(str(labels[idx])) 
                nlabels.append(ts.strftime('%Y.%m.%d  %H:%Mh'))
            return locs, nlabels

        # shift train prediction for plotting (só para compensar o look_back)
        emptyNan = np.empty_like(np.zeros(self.look_back))
        emptyNan[:] = np.nan
        trainPredictPlot = np.concatenate((np.reshape(emptyNan, (self.look_back, 1)), self.trainPredictions))

        # shift test predictions for plotting
        emptyNan = np.empty_like(np.zeros(self.trainLength + self.look_back))
        emptyNan[:] = np.nan
        testPredictPlot = np.concatenate((np.reshape(emptyNan, (self.trainLength + self.look_back, 1)), self.testPredictions))

        locs, labels = generatexTicks(interval=10 , nlocs=len(self.mpsTrain) + len(self.mpsTest), labels=self.timestamps)

        plt.figure(figsize=(10,6))
        plt.plot(np.concatenate((self.mpsTrain, self.mpsTest)), color='blue', label='average stock prices') 
        plt.plot(trainPredictPlot , color='red', label='predicted stock prices train')  
        plt.plot(testPredictPlot , color='green', label='predicted stock prices test')
        plt.title('Down jones index predictions (1 features)')  
        plt.xlabel('Date')  
        plt.ylabel('Down jones index')  
        plt.xticks(locs, labels, rotation='90')
        plt.grid(axis='x')
        plt.legend()  
        plt.show()
    
    def save(self, name = 'my_model_1feature'):
        self.model.save('%s.h5'%(name))
    
    def load_model(self, name = 'my_model_1feature'):
        if(os.path.exists('%s.h5'%(name))):
            self.model = load_model('%s.h5'%(name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='time series stocks predictor')
    parser.add_argument('-i', '--input', type=str, default= 'utils/dataSeriePosNeg_60min.json', help='dataset path')
    parser.add_argument('-e', '--epochs', type=int, default= 5, help='number of epochs to fit the model')
    parser.add_argument('-lb', '--look_back', type=int, default= 50, help='how many previous time the lstm see')
    parser.add_argument('-t', '--type', type=str, default='twoFeatures', help='what predictor do you usage. (options: twoFeatures, oneFeature, predictorTwoFeat)')
    ####################################GLOBAL VARIABLES#############################################################
    #test_set_size= 50
    PATH_IN = parser.parse_args().input
    epochs = parser.parse_args().epochs
    look_back = parser.parse_args().look_back
    predictorType = parser.parse_args().type
    #################################################################################################################

    predictorsOptions = {'oneFeature': TimeSeriePredictorOneFeature, 'twoFeatures': TimeSeriePredictorTwoFeatures, 'predictorTwoFeat': PredictorTwoFeat, 'predictorOneFeat': PredictorOneFeat}

    # se quiser fazer predição com base em um modelo pré treinado
    if (predictorType == 'predictorTwoFeat' or predictorType == 'predictorOneFeat'):
        predictor = predictorsOptions[predictorType](look_back = look_back)
        predictor.load_model()
        predictor.data_preparation(PATH_IN)
        predictor.predict_model()
        predictor.plot_serie()
    else:
        predictor = predictorsOptions[predictorType](look_back=look_back)
        predictor.data_preparation(PATH_IN)
        predictor.train_model(epochs = epochs, batch_size = 32)
        predictor.save()
        predictor.test_model(batch_size= 32)
        predictor.plot_serie()
