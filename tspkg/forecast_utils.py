import pandas as pd
from tspkg.utils import * 
from sklearn.preprocessing import StandardScaler
import numpy as np

class ForecastSeries:
    def __init__(self, data: np.array, timesteps, prediction_length):
        
        self.data = data
        print(type(self.data))
        self.timesteps = timesteps
        self.prediction_length = prediction_length
        
        #oggetto StandardScaler di sklearn fittato sulla serie
        self.scaler_obj = self.scaler((self.data))
        #intera serie differenziata e scalata
        self.prepared_data = self.scaler_obj.transform(self.difference().reshape(-1,1))
        #valori per invertire la differenziazione dei valori predetti
        self.val_indif = data[-(2*prediction_length + 1)]
        self.test_indif = data[-(prediction_length + 1)]

        #train set convertito come un set di dati per supervised learning sul quale verrà fatto il fit
        self.X, self.y = split_sequence(self.prepared_data[:-(2*prediction_length)], self.timesteps, self.prediction_length)
        
        #set di dati per il test set, i dati per il fit (lunghi timesteps) sono elaborati mentre 
        #quelli di actual sono raw
        self.test_fit = self.prepared_data[(-self.prediction_length - self.timesteps):-prediction_length]
        self.test_actual = self.data[-prediction_length:]

        #set di dati per il set di validation, i dati per il fit (lunghi timesteps) sono elaborati mentre 
        #quelli di actual sono raw
        self.validation_fit = self.prepared_data[(- 2*self.prediction_length - self.timesteps):-2*prediction_length]
        #expressed with raw data
        self.validation_actual = self.data[-2*prediction_length:-prediction_length]

    #qui perdiamo il valore iniziale o tanti valori quanto è lungo intervaò
    def difference(self):
        interval = 1
        print(self.data.shape[0])
        diff = list()
        for i in range(interval, (self.data.shape[0])):
            value = self.data[i] - self.data[i - interval]
            diff.append(value)
        return np.array(diff)
    
    def scaler(self, dataset):
        obj = StandardScaler()
        obj = obj.fit(dataset.reshape(-1,1))
        return obj
    
    #intanto li implemento solo per interval = 1
    def inverse_validation_difference(self, data):
        new = list()
        for i in range(1,len(data)):
            new.append(data + self.val_indif + data[i-1])
        return new

    def inverse_test_difference(self, data):
        new = list()
        for i in range(1,len(data)):
            new.append(data + self.test_indif + data[i-1])
        return new
    


if __name__ == "__main__":
    arr = np.array([10, 20, 40, 50, 60, 80, 90, 110, 120, 130])
    obj = ForecastSeries(data = arr, timesteps = 2, prediction_length = 1)
    print(obj.prepared_data)
    print(obj.X)

    #print(obj.scaler_obj.var_)

    