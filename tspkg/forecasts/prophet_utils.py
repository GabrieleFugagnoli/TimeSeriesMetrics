import sys
#path per il desk a casa
sys.path.append(r"C:\Users\gabriele\Desktop\python_programs\TimeSeriesMetrics") 
from tspkg.paths import *
from tspkg.utils import *
import tspkg.metrics as metric
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import optuna
import pandas as pd

"""14/09: Avevo iniziato il processo di tuning del prophet ma ho scelto di sviluppare prima la
matrice delle metriche senza tunare i modelli al di fuori delle reti neurali
Il prophet durante la cross validation fa un fit separato ogni iterazione, non migliora i pesi.
Il cross validation torna utile per fare una previsione su dati presenti nella serie, il downside è
che bisogna fare prima il fit sulla serie e quindi il modello "vede" i dati da predirre.
Per fare delle prediction pure dovrò dividere la serie.
 """



class ProphetObjective:
    """Classe che verrà utilizzata come funzione objective da inserire come argomento a optuna optimize.
    E' stata implementata come una classe per avere la possibilità di memorizzare all'interno dell'objective
    i dati per un migliore incapsulamento.
    """
    def __init__(self, train_series: pd.DataFrame, n_prediction: int):
        #viene passata una serie di train sotto forma di datafarame con i nomi delle colonne originali
        #prophet_predict si occupa della preparazione del dataframe
        self.train = train_series.iloc[:-n_prediction]
        #per confrontare le previsioni ho bisogno di array
        self.validation_mase = np.array(train_series['Amount_sold'].iloc[-(n_prediction+1):])
        self.n_prediction = n_prediction
    
    def __call__(self, trial):
        params = dict()
        params['changepoint_prior_scale'] = trial.suggest_float('changepoint_prior_scale', low = 0.05, high = 0.5, step = 0.05)
        params['seasonality_prior_scale'] = trial.suggest_float('seasonality_prior_scale', 1, 10,step = 1)
        params['holidays_prior_scale'] = trial.suggest_float('holidays_prior_scale', 1, 10, step = 1)
        params['seasonality_mode'] = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
        prediction = prophet_predict(self.train, n_prediction = self.n_prediction, params = params)
        score = metric.mase(self.validation_mase, prediction)
        return score

# def objective_prophet(trial):
#     params = dict()

#     params['changepoint_prior_scale'] = trial.suggest_float('changepoint_prior_scale', 0.0001, 0.5, 0.05)
#     params['seasonality_prior_scale'] = trial.suggest_float('seasonality_prior_scale', 0.01, 10, 1)
#     params['holidays_prior_scale'] = trial.suggest_float('holidays_prior_scale', 0.01, 10, 1)
#     params['seasonality_mode'] = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
    

def prophet_tune(train_series, n_prediction):
    #usa la classe Prohet Objective e il metodo study
    #riceve in input i dati di train e ProphetObjective li trasformerà in train e validate
    objective = ProphetObjective(train_series, n_prediction)
    study = optuna.create_study(direction = 'minimize')
    study.optimize(objective, n_trials = 30)
    print(study.best_params)
    return study.best_params

def prophet_predict(series: pd.DataFrame, n_prediction: int, params: dict) -> np.array:
    """Fit su series e predizione con orizzonte n_prediction.
        Computa la predizione e la restituisce come array numpy.
      Series è una singola serie che contiene solo i dati già tagliati su cui fare il fit."""
    data = series.reset_index()
    data.columns = ['ds', 'y']
    m = Prophet(**params)
    print(f"-----Changepoint effettivo: {m.changepoint_prior_scale}---------")
    
    m.fit(data)
    future = m.make_future_dataframe(periods=n_prediction)
    
    forecast = m.predict(future)
    #seleziono solo i valori effettivamente predetti, prophet restituisce anche tutti i dati precedenti
    prediction = np.array(forecast['yhat'].iloc[-n_prediction:])
    return prediction


def prophet_compute_metrics(series_dic: dict(), n_prediction: int) -> pd.Series:
    """Computa le metriche di errore utilizzando la funzione prophet_predict, le restituisce sotto forma di pandas Series
    1. Split delle serie in train e test
    2. Predizioni
    3. Confronto delle predizioni con i test 
    
    train e test saranno dizionari di dataframe mentre pred sarà un dizionario di array""" 
    
    train = dict()
    pred = dict()
    test = dict()
   
    for item in series_dic:
        train[item] = series_dic[item].iloc[:-n_prediction]
        test[item] = series_dic[item].iloc[-n_prediction:]
        print(f"Lunghezza train set: {len(train[item])}")
        print(f"Lunghezza test set: {len(test[item])}")
    
    #per ogni serie ottengo la previsione
    for item in series_dic:
        pred[item] = prophet_predict(train[item], n_prediction)
    
    #l'unica metrica particolare è il mase che ha bisogno di una valore in più nell'array di actual
    
    #Creo prima un array e poi creo la serie metrics = pd.Series(name = "Prophet")
    mase = list() 
    mape = list() 
    wape = list() 
    rmse = list()
    

    for item in series_dic:
        mase.append(metric.mase(actual = np.array(series_dic[item]['Amount_sold'].iloc[-(n_prediction +1):]), predicted= pred[item]))
        mape.append(metric.mape(np.array(test[item]['Amount_sold']), pred[item]))
        wape.append(metric.wape(np.array(test[item]['Amount_sold']), pred[item]))
        rmse.append(metric.rmse(np.array(test[item]['Amount_sold']), pred[item]))
    
    series_metrics = list() 

    for i in range(mase):
        series_metrics.append(pd.Series([mase[i], mape[i], wape[i], rmse[i] ], 
                        index = ['mase', 'mape', 'wape', 'rmse'], 
                        name = "Prophet"))

    cluster_metrics = pd.Series([np.mean(mase), np.mean(mape), np.mean(wape), np.mean(rmse) ], 
                        index = ['mase', 'mape', 'wape', 'rmse'], 
                        name = "Prophet")
    
    return cluster_metrics, series_metrics

def prophet_tune_compute_metrics(series_dic: dict(), n_prediction: int) -> pd.Series:
    """Fa il tuning del modello serie per serie e computa le metriche di errore utilizzando la funzione prophet_predict, le restituisce sotto forma di pandas Series
    1. Split delle serie in train e test
    2. Predizioni
    3. Confronto delle predizioni con i test 
    
    train e test saranno dizionari di dataframe mentre pred sarà un dizionario di array
    :param dict series_dic: dizionario raw contenenti le serie""" 
    train = dict()
    pred = dict()
    test = dict()
   
    for item in series_dic:
        train[item] = series_dic[item].iloc[:-n_prediction]
        test[item] = series_dic[item].iloc[-n_prediction:]
        print(f"Lunghezza train set: {len(train[item])}")
        print(f"Lunghezza test set: {len(test[item])}")
    
    #per ogni serie ottengo la previsione
    for item in series_dic:
        parameters = prophet_tune(train[item], n_prediction)
        pred[item] = prophet_predict(train[item], n_prediction, parameters)
    
    #l'unica metrica particolare è il mase che ha bisogno di una valore in più nell'array di actual
    
    #Creo prima un array e poi creo la serie metrics = pd.Series(name = "Prophet")
    mase = list() 
    mape = list() 
    wape = list() 
    rmse = list()
    
    #Inserisco nelle liste i valori delle metriche, avro' 4 liste di lunghezza n_features
    for item in series_dic:
        mase.append(metric.mase(actual = np.array(series_dic[item]['Amount_sold'].iloc[-(n_prediction +1):]), predicted= pred[item]))
        mape.append(metric.mape(np.array(test[item]['Amount_sold']), pred[item]))
        wape.append(metric.wape(np.array(test[item]['Amount_sold']), pred[item]))
        rmse.append(metric.rmse(np.array(test[item]['Amount_sold']), pred[item]))
    
    series_metrics = list() 

    for i in range(len(mase)):
        series_metrics.append(pd.Series([mase[i], mape[i], wape[i], rmse[i] ], 
                        index = ['mase', 'mape', 'wape', 'rmse'], 
                        name = "Prophet"))

    cluster_metrics = pd.Series([np.mean(mase), np.mean(mape), np.mean(wape), np.mean(rmse) ], 
                        index = ['mase', 'mape', 'wape', 'rmse'], 
                        name = "Prophet")
    
    return cluster_metrics, series_metrics

if __name__ == "__main__":
    
    with open(processed_dir / 'cluster1.pkl', 'rb') as f :
        dic = pickle.load(f)
    series = copy.deepcopy(dic)


    #print(prophet_tuning(series))

    print(prophet_tune_compute_metrics(series, n_prediction=7))