from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import OffsetSplitter
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation import make_evaluation_predictions
from tspkg.paths import *
from tspkg.utils import *    
import tspkg.metrics as metric

def deepAr_compute_metrics(dic: dict(), n_prediction: int) -> pd.Series:
    """Partendo da un dizionario di serie e un orizzonte di previsione, restituisce le metriche di 
    errore sotto forma di pandas Series"""

    keys = list(dic.keys())
    train = dict()
    test = dict()
    for p in dic:
        train[p] = dic[p].iloc[:-n_prediction]
        test[p] = dic[p].iloc[-n_prediction:]

    actual_plus_one = list()
    for item in dic:
        actual_plus_one.append(np.array(dic[item].iloc[-(n_prediction + 1):]))
       
    train_ds = PandasDataset(train, target = 'Amount_sold', freq='D')

    estimator = DeepAREstimator(freq='D', prediction_length=7, trainer_kwargs={'max_epochs':1000})
    predictor = estimator.train(train_ds, num_workers=2)
    pred = list(predictor.predict(train_ds))

    pred_mean = list()

    for elem in pred:
        pred_mean.append(elem.samples.mean(axis = 0))
    
    actual = list()
    for item in dic:
        actual.append(np.array(test[item]['Amount_sold']))
    
   
    mase = list() 
    mape = list() 
    wape = list() 
    rmse = list()

    for i in range(len(pred_mean)):
        mase.append(metric.mase(actual_plus_one[i], pred_mean[i]))
        mape.append(metric.mape(np.array(test[keys[i]]['Amount_sold']), pred_mean[i]))
        wape.append(metric.wape(np.array(test[keys[i]]['Amount_sold']), pred_mean[i]))
        rmse.append(metric.rmse(np.array(test[keys[i]]['Amount_sold']), pred_mean[i]))

    metrics = pd.Series([np.mean(mase), np.mean(mape), np.mean(wape), np.mean(rmse) ], 
                        index = ['mase', 'mape', 'wape', 'rmse'], 
                        name = "DeepAr")
    
    return metrics

    

if __name__ == "__main__":
    with open(processed_dir / 'cluster1.pkl', 'rb') as f :
        dic = pickle.load(f)

    print(deepAr_compute_metrics(dic, n_prediction=7))
