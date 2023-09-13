import pandas as pd
from tspkg.utils import * 
from tspkg.metrics import *
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D, LSTM, RepeatVector, TimeDistributed
import optuna


#qui perdiamo il valore iniziale o tanti valori quanto è lungo l'intervallo
def difference(data: np.array) -> np.array:
    interval = 1
    diff = list()
    for i in range(interval, (data.shape[0])):
        value = data[i] - data[i - interval]
        diff.append(value)
    return np.array(diff)


def inverse_difference(starting_value, data):
    restored = list()
    for i in range(len(data)):
        if i == 0:
            restored.append(data[0] + starting_value)
        else:
            restored.append(data[i] + restored[-1])
    return restored


class NNSeries:
    def __init__(self, data: np.array, timesteps: int, prediction_length: int):
        
        self.data = data
        self.timesteps = timesteps
        self.prediction_length = prediction_length
        self.restoration_data = dict()
        
        #oggetto StandardScaler di sklearn fittato sulla serie
        self.scaler_obj = self.scaler((self.data))

        #intera serie differenziata e scalata
        self.prepared_data = self.scaler_obj.transform(difference(self.data).reshape(-1,1))

        #valori per invertire la differenziazione dei valori predetti, sia per il set di validation che test
        self.restoration_data['validation_indif'] = data[-(2*prediction_length + 1)]
        self.restoration_data['test_indif']= data[-(prediction_length + 1)]

        #Primo valore della serie
        self.restoration_data['first_value'] = self.data[0]

        #train set convertito come un set di dati per supervised learning sul quale verrà fatto il fit, viene usato per il tuning
        self.X, self.y = split_sequence(self.prepared_data[:-(2*prediction_length)], self.timesteps, self.prediction_length)
        
        #set di dati per il test set, i dati per il fit (lunghi timesteps) sono elaborati mentre 
        #quelli di actual sono raw
        self.test_fit = self.prepared_data[(-self.prediction_length - self.timesteps):-prediction_length]
        self.test_actual = self.data[-prediction_length:]
        #per calcolare il mase ho bisogno di un valore in più a sx di actual
        self.test_actual_mase = self.data[-(prediction_length+1):]
        

        #set di dati per il set di validation, i dati per il fit (lunghi timesteps) sono elaborati mentre 
        #quelli di actual sono raw
        self.validation_predict = self.prepared_data[(- 2*self.prediction_length - self.timesteps):-2*prediction_length]
        #expressed with raw data
        self.validation_actual = self.data[-2*prediction_length:-prediction_length]
        #per calcolare il mase ho bisogno di un valore in più a sx di actual
        self.validation_actual_mase = self.data[-(2*prediction_length+1):-prediction_length]
    
    def scaler(self, dataset):
        obj = StandardScaler()
        obj = obj.fit(dataset.reshape(-1,1))
        return obj
    
    #intanto li implemento solo per interval = 1
    def inverse_validation_difference(self, data):
        # new = list()
        # for i in range(1,len(data)):
        #     new.append(data + self.val_indif + data[i-1])
        # return new
        return inverse_difference(starting_value=self.restoration_data['validation_indif'], data = data)

    def inverse_test_difference(self, data):
        # new = list()
        # for i in range(1,len(data)):
        #     new.append(data + self.test_indif + data[i-1])
        # return new
        return inverse_difference(starting_value=self.restoration_data['test_indif'], data = data)

    def validation_prediction_conversion(self, data):
        diff_data = self.scaler_obj.inverse_transform(data.reshape(-1,1)).flatten()
        return self.inverse_validation_difference(diff_data)
    
    def test_prediction_conversion(self, data):
        diff_data = self.scaler_obj.inverse_transform(data.reshape(-1,1)).flatten()
        return self.inverse_test_difference(diff_data)



def create_univariate_cnn(trial, n_steps_in, n_prediction):
    """Crea un modello di rete neurale convoluzionale di keras con layer: Conv1D, MaxPooling1D, Flatten, Dense, Dense
    a partire dai parametri passati da optuna.
    """
    # num_cnn_blocks = trial.suggest_int('num_cnn_blocks', 2, 4)
    num_filters = trial.suggest_categorical('conv_filters', [32, 64, 128])
    k_size = trial.suggest_int('kernel_size', 2, 4)
    n_dense_nodes = trial.suggest_int('num_dense_nodes', 25, 100, 25)
    n_pooling = trial.suggest_categorical('num_pooling', [1, 2, 4])
    #il batch size va specificato nel fit, per ora non lo includo
    # batch_size = trial.suggest_categorical('batch_size', [32, 64, 96, 128])
    
    # creo il modello
    model = Sequential()
    model.add(Conv1D(filters= num_filters, kernel_size= k_size, activation='relu', input_shape=(n_steps_in, 1)))
    model.add(MaxPooling1D(pool_size= n_pooling))
    model.add(Flatten())
    model.add(Dense(n_dense_nodes, activation='relu'))
    model.add(Dense(n_prediction))
    model.compile(optimizer='adam', loss='mse')
    
    return model

def create_univariate_lstm(trial, n_steps_in, n_prediction):
    """Crea un modello di rete neurale lstm di keras con layer: ...a partire dai parametri passati da optuna."""
    nodes1 = trial.suggest_int('nodes1', 50, 200, 25)
    nodes2 = trial.suggest_int('nodes2', 50, 200, 25)
    model = Sequential()
    # Modello encoder-decoder:
    # model.add(LSTM(nodes1, activation='relu', input_shape=(n_steps_in, 1)))
    # model.add(RepeatVector(n_prediction))
    # model.add(LSTM(nodes2, activation='relu', return_sequences=True))
    # model.add(TimeDistributed(Dense(1)))
    # Vector output:
    model.add(LSTM(nodes1, activation='relu', return_sequences=True, input_shape=(n_steps_in, 1)))
    model.add(LSTM(nodes2, activation='relu'))
    model.add(Dense(n_prediction))
    model.compile(optimizer='adam', loss='mse')
    return model

class UnivariateLSTMObjective:
    """Classe che verrà utilizzata come funzione objective da inserire come argomento a optuna optimize.
    E' stata implementata come una classe per avere la possibilità di memorizzare all'interno dell'objective i dati per un migliore incapsulamento
    """
    def __init__(self, series):
        self.series = series
    
    # Verrà chiamata dallo study, crea il modello usando gli iperparametri in trial e ritorna il mase della predizione del validation
    def __call__(self, trial):
        model = create_univariate_lstm(trial, 
                                      n_steps_in= self.series.timesteps, 
                                      n_prediction = self.series.prediction_length)
        #dipendentemente dal tipo di modelli, y deve avere dimensione diversa, per il modello encoder/decoder
        #bisogna fare il reshape come per X
        model.fit(self.series.X.reshape((self.series.X.shape[0], self.series.X.shape[1], 1)),
                  self.series.y,
                  #self.series.y.reshape((self.series.y.shape[0], self.series.y.shape[1], 1)),
                epochs=10,
                verbose=0)
        prediction = model.predict(self.series.validation_predict.reshape(1, self.series.timesteps, 1), verbose=0)
        score = mase(self.series.validation_actual_mase, self.series.validation_prediction_conversion(prediction))
        print(f"------ Score lstm: {score}--------")
        print(f"----------Actual: {self.series.validation_actual}---------")
        print(f"----------Predicted: {self.series.validation_prediction_conversion(prediction)}------------")
        return score

class UnivariateCNNObjective:
    """Classe che verrà utilizzata come funzione objective da inserire come argomento a optuna optimize.
    E' stata implementata come una classe per avere la possibilità di memorizzare all'interno dell'objective i dati per un migliore incapsulamento
    """
    def __init__(self, series):
        self.series = series
    
    # Verrà chiamata dallo study, crea il modello usando gli iperparametri in trial e ritorna il mase della predizione del validation
    def __call__(self, trial):
        model = create_univariate_cnn(trial, 
                                      n_steps_in= self.series.timesteps, 
                                      n_prediction = self.series.prediction_length)
        #I dati X per il fit devono avere dimensione [samples, timesteps, features], y invece non ha bisogno di manipolazioni
        model.fit(self.series.X.reshape((self.series.X.shape[0], self.series.X.shape[1], 1)), 
                self.series.y,
                epochs=500,
                verbose=0)
        print(f"{self.series.X.shape}")
        prediction = model.predict(self.series.validation_predict.reshape(1, self.series.timesteps, 1), verbose=0)
        score = mase(self.series.validation_actual_mase, self.series.validation_prediction_conversion(prediction))
        print(f"------ Score CNN: {score}--------")
        print(f"----------Actual: {self.series.validation_actual}---------")
        print(f"----------Predicted: {self.series.validation_prediction_conversion(prediction)}------------")
        return score

def tests():
    arr = np.array([10, 20, 40, 50, 60, 80, 90, 110, 120, 130])
    obj = NNSeries(data = arr, timesteps = 2, prediction_length = 1)
    print(obj.prepared_data)
    # print(obj.X)
    print(f"Test fit :{obj.test_fit}")
    print(f"Test actual : {obj.test_actual}")
    print(f"Validation fit: {obj.validation_predict}")
    print(f"Test actual: {obj.validation_actual}")
    #print(obj.scaler_obj.var_)
    diff_data = (obj.scaler_obj.inverse_transform(obj.prepared_data))
    #verrà printato un vettore di vettori, risolvo con flatten
    print(f'Inverse scaling: {diff_data.flatten()}')
    # restored = inverse_difference(obj.restoration_data['first_value'], diff_data)
    # print(f'Restored data: {restored}')
    # #considero che -1.53999164 sia la previsione sia per il validation che per il test
    # #c'è sempre il problema che lo scaler accetta vettori di due dimensioni, si può risolvere aggiungendo un reshape(1)
    # test_pred = inverse_difference(obj.restoration_data['test_indif'], (obj.scaler_obj.inverse_transform(np.array(-1.53999164).reshape(-1,1))).reshape(1))
    # validation_pred = inverse_difference(obj.restoration_data['validation_indif'], (obj.scaler_obj.inverse_transform(np.array(-1.53999164).reshape(-1,1))).reshape(1))
    # print(f"Test prediction: {test_pred}", f"Val prediciton: {validation_pred}")

    #funziona tutto, l'unico problema è sulla dimensionalità dei valori nell'array

def test_series():
    with open(processed_dir / 'cluster1.pkl', 'rb') as f :
        dic = pickle.load(f)

    series = dic[list(dic.keys())[0]]
    print(series.tail(15))
    #print(np.array(series['Amount_sold']))
    test = NNSeries(np.array(series['Amount_sold']), timesteps = 730, prediction_length=7)
    print(f"Actual validation: {test.validation_actual}")
    print(f"Actual test: {test.test_actual}")

def test_cnn_tuning():
    with open(processed_dir / 'cluster1.pkl', 'rb') as f :
        dic = pickle.load(f)
    series = dic[list(dic.keys())[0]]
    series = NNSeries(np.array(series['Amount_sold']), timesteps= 730, prediction_length=7)
    cnn_objective = UnivariateCNNObjective(series)

    study = optuna.create_study(direction='minimize')
    study.optimize(cnn_objective, n_trials = 10)
    print(study.best_params)

def test_lstm_tuning():
    with open(processed_dir / 'cluster1.pkl', 'rb') as f :
        dic = pickle.load(f)
    series = dic[list(dic.keys())[0]]
    series = NNSeries(np.array(series['Amount_sold']), timesteps= 730, prediction_length=7)
    lstm_objective = UnivariateLSTMObjective(series)

    study = optuna.create_study(direction='minimize')
    study.optimize(lstm_objective, n_trials = 10)
    print(study.best_params)

def test_weekly_cnn_tuning():
    with open(processed_dir / 'cluster1.pkl', 'rb') as f :
        dic = pickle.load(f)
    conv_to_weekly(dic)
    series = dic[list(dic.keys())[0]]
    series = NNSeries(np.array(series['Amount_sold']), timesteps= 104, prediction_length=1)
    cnn_objective = UnivariateCNNObjective(series)

    study = optuna.create_study(direction='minimize')
    study.optimize(cnn_objective, n_trials = 20)
    print(f"Migliori parametri: {study.best_params}")

def test_weekly_lstm_tuning():
    with open(processed_dir / 'cluster1.pkl', 'rb') as f :
        dic = pickle.load(f)
    conv_to_weekly(dic)
    series = dic[list(dic.keys())[0]]
    series = NNSeries(np.array(series['Amount_sold']), timesteps= 104, prediction_length=1)
    lstm_objective = UnivariateLSTMObjective(series)

    study = optuna.create_study(direction='minimize')
    study.optimize(lstm_objective, n_trials = 10)
    print(study.best_params)


if __name__ == "__main__":
    #la rete lstm ha ancora problemi con i dati giornalieri, restituisce previsioni NaN
    #test_lstm_tuning()
    #test_weekly_cnn_tuning()
    test_cnn_tuning()
    #test_weekly_lstm_tuning()