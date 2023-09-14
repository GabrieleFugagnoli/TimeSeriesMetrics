from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tspkg.paths import *
from tspkg.utils import *

def ets_predict(series: pd.DataFrame, n_prediction: int) -> np.array:
    """Computa la predizione e la restituisce come array numpy.
      Series è una singola serie che contiene solo i dati già tagliati su cui fare il fit.
      Eventualmente si potrà aggiungere il processo di tuning qua dentro"""
   
    data = np.array(series['Amount_sold'])
    model = ETSModel(data)
    fit = model.fit(maxiter=1000)
    pred = fit.forecast(n_prediction)
    return pred

#def ets_compute_metrics(series_dic: dict(), n_prediction: int) -> pd.Series:


if __name__ == "__main__":
    with open(processed_dir / 'cluster1.pkl', 'rb') as f :
        dic = pickle.load(f)
    
    keys = list(dic.keys())
    serie = dic[keys[0]]

    ret = ets_predict(serie, 7)
    print(type(ret))
    print(ret)
