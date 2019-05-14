from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input
import keras

class LstmTwoFeature():
    def __init__(self, featureA_shape, featureB_shape):
        self.featureA_shape = featureA_shape
        self.featureB_shape = featureB_shape

        featureAinput = Input(shape=(featureA_shape, 1))
        featureBinput = Input(shape=(featureB_shape, 1))
        
        # For featureA input
        x = LSTM(units=50, return_sequences=True)(featureAinput)
        #x = Dropout(0.2)(x)
        #x = LSTM(units=50, return_sequences=True)(x)
        #x = Dropout(0.2)(x)
        #x = LSTM(units=50, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(units=50)(x)
        lstmA = Dropout(0.2)(x)
        featureA_output = Dense(1, activation = 'relu')(lstmA)

        # For featureB input
        x = LSTM(units=50, return_sequences=True)(featureBinput)
        #x = Dropout(0.2)(x)
        #x = LSTM(units=50, return_sequences=True)(x)
        #x = Dropout(0.2)(x)
        #x = LSTM(units=50, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(units=50)(x)
        lstmB = Dropout(0.2)(x)
        featureB_output = Dense(1, activation = 'relu')(lstmB)

        AB_output = keras.layers.concatenate([featureA_output, featureB_output])

        modelOutput = Dense(1, activation = 'relu')(AB_output)

        self.model = Model(inputs=[featureAinput, featureBinput], output=modelOutput)
    def get_model(self):
        return self.model
        