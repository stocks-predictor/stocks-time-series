from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

class LstmOneFeature():
    def __init__(self, input_shape, outputLength):
        self.model = Sequential()

        self.model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))  
        self.model.add(Dropout(0.2))  

        self.model.add(LSTM(units=50, return_sequences=True))  
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=50, return_sequences=True))  
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=50))  
        self.model.add(Dropout(0.2))  

        self.model.add(Dense(units = outputLength))  
    
    def get_model(self):
        return self.model