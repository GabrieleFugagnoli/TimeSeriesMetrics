import pandas as pd
import sys
sys.path.append(r"c:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject") 
from tspkg.paths import *
from tspkg.utils import *
from prophet import Prophet
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import copy

#ritorna un dizionario con i valori di trend e seasonality associati agli id del prodotto
def stl(dic, period = 7):
    # for p in dic:
    #   dic[p].reset_index(inplace = True)
    #   dic[p].columns = ['ds', 'y']

    # pred = dict()
    # m = Prophet()

    # for p in dic.keys():
      
    #   m.fit(dic[p])
    #   future = m.make_future_dataframe(periods=30)
    #   pred[p] = m.predict(future)

    # with open(processed_dir / 'pred.pkl', 'wb') as f:
    #   pickle.dump(pred, f)

    # return pred 
    decompose = seasonal_decompose(dic[55]['Amount_sold'],model='additive', period=period)
    decompose.plot()
    plt.show()

    df = copy.deepcopy(dic[55])
    
    df["Seasonal"] = decompose.seasonal
    df["Trend"] =  decompose.trend
    df["Resid"] = decompose.resid
    

    return df

    



if __name__ == "__main__":
    with open(processed_dir / 'dictionary.pkl', 'rb') as f:
        dic = pickle.load(f)
    
    decomposed = stl(dic)
    decomposed.plot()
    plt.show()
    #stl_par = stl(dic)
    
    # dic[55].reset_index(inplace = True)
    # dic[55].columns = ['ds', 'y']
    # print(dic[55].tail())

    # pred = dict()
    # m = Prophet()

      
    # m.fit(dic[55])
    # future = m.make_future_dataframe(periods=30)
    # pred[55] = (m.predict(future).iloc[:len(future) - 30]).loc[:,['trend', 'weekly', 'yearly']]
    # plt.plot(pred[55])
    # plt.show()
    

    #print(pred[55].columns.tolist())