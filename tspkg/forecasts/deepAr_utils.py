from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import OffsetSplitter
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation import make_evaluation_predictions
from tspkg.paths import *
from tspkg.utils import *    
import tspkg.metrics as metric
import optuna

class   DeepARObjective:
    def __init__(self, data: dict, n_prediction: int):
        self.data = data
        self.n_prediction = n_prediction
        self.train = self.make_train()
        self.validation_mase = self.make_actual_mase()
    #devo dividere un'altra volta il dataset per il validation
    
    def make_train(self): 
        train = dict()
        for item in self.data:
            train[item] = self.data[item].iloc[:-self.n_prediction]
        
        return train 


    def make_actual_mase(self):
        
        actual_plus_one = list()
        for item in self.data:
            actual_plus_one.append(np.array(self.data[item].iloc[-(self.n_prediction + 1):]))

        return actual_plus_one

    def __call__(self, trial):
        num_layers = trial.suggest_int("num_layers", low = 1, high = 5, step = 1)
        hidden_size = trial.suggest_int("hidden_size", low = 10, high = 50, step = 5)

        train_ds = PandasDataset(self.train, target = 'Amount_sold', freq='D')
        estimator = DeepAREstimator(num_layers= num_layers, hidden_size= hidden_size, freq='D', prediction_length=7, trainer_kwargs={'max_epochs':10})
        predictor = estimator.train(train_ds, num_workers=2)
        pred = list(predictor.predict(train_ds))
        
        pred_mean = list()
        for elem in pred:
            pred_mean.append(elem.samples.mean(axis = 0))

        mase = list()

        for i in range(len(pred_mean)):
            mase.append(metric.mase(self.validation_mase[i], pred_mean[i]))

        return np.mean(mase)

        
        

def deepAr_tune_compute_metrics(dic: dict(), n_prediction: int) -> pd.Series:
    """Partendo da un dizionario di serie e un orizzonte di previsione, restituisce le metriche di 
    errore sotto forma di pandas Series"""

    keys = list(dic.keys())
    train = dict()
    
    actual_dic = dict()
    for p in dic:
        train[p] = dic[p].iloc[:-n_prediction]
        actual_dic[p] = dic[p].iloc[-n_prediction:]

    actual_plus_one = list()
    for item in dic:
        actual_plus_one.append(np.array(dic[item].iloc[-(n_prediction + 1):]))
    
    #Posso inserire qui il tuning usando il solito oggetto e poi passare i due parametri a DeepAREstimator
    objective = DeepARObjective(train, n_prediction=7)
    study = optuna.create_study(direction = 'minimize')
    study.optimize(objective, n_trials = 10)

    train_ds = PandasDataset(train, target = 'Amount_sold', freq='D')

    estimator = DeepAREstimator(freq='D', prediction_length=7, **study.best_params, trainer_kwargs={'max_epochs':25})
    predictor = estimator.train(train_ds, num_workers=2)
    #lista di 11 elementi contenente ognuno 100 valori per ogni previsione
    pred = list(predictor.predict(train_ds))

    pred_mean = list()
    #ottengo la media delle previsioni arrivando a una lista di 11 elementi contenenti le previsioni lunghe n_pred
    for elem in pred:
        pred_mean.append(elem.samples.mean(axis = 0))
    
    actual = list()
    for item in dic:
        actual.append(np.array(actual_dic[item]['Amount_sold']))
    
    mase = list() 
    mape = list() 
    wape = list() 
    rmse = list()

    for i in range(len(pred_mean)):
        mase.append(metric.mase(actual_plus_one[i], pred_mean[i]))
        mape.append(metric.mape(actual[i], pred_mean[i]))
        wape.append(metric.wape(actual[i], pred_mean[i]))
        rmse.append(metric.rmse(actual[i], pred_mean[i]))

    series_metrics = list() 

    for i in range(len(mase)):
        series_metrics.append(pd.Series([mase[i], mape[i], wape[i], rmse[i] ], 
                        index = ['mase', 'mape', 'wape', 'rmse'], 
                        name = "DeepAr"))

    cluster_metrics = pd.Series([np.mean(mase), np.mean(mape), np.mean(wape), np.mean(rmse) ], 
                        index = ['mase', 'mape', 'wape', 'rmse'], 
                        name = "DeepAr")
    
    return cluster_metrics, series_metrics

    

if __name__ == "__main__":
    with open(processed_dir / 'cluster1.pkl', 'rb') as f :
        dic = pickle.load(f)

    print(deepAr_tune_compute_metrics(dic, n_prediction=7))
