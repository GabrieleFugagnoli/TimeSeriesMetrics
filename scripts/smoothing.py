import pandas as pd
import sys
sys.path.append(r"c:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject") 
from tspkg.paths import *
from tspkg.utils import *
import matplotlib.pyplot as plt
import copy

"""
Modificano il dizionario passato, lavorano in shallow copy
"""
def sma(dic: dict, window_size: int = 30, dropna: bool = False):
   
   for p in dic.keys():
       #print(f"SMA smoothing del prodotto {p}")
       dic[p].Amount_sold = dic[p].Amount_sold.rolling(window_size).mean()
       if dropna:
        dic[p] = dic[p].dropna()
    

def ewm(dic: dict, a: int):
   
   for p in dic.keys():
       dic[p].Amount_sold = dic[p].Amount_sold.ewm(alpha = a).mean().dropna()
       print(f"SMA smoothing del prodotto {p}")


       


if __name__ == "__main__":
    
    with open(processed_dir / 'dictionary.pkl', 'rb') as f:
        dic = pickle.load(f)
    
    old_dic = copy.deepcopy(dic)

    sma(dic, 30)
    #ewm(dic, 0.1)
    
    print(dic[55].head())

    
    
    
    
    #plt.style.use('fivethirtyeight')

    # plt.figure(1)
    # plt.title("Prodotto 55")
    # plt.figure(1)
    # plt.figure(figsize = (13,7))
    # plt.plot(old_dic[55],label='Vendite raw',linewidth=2)
    # plt.plot(dic[55],label='Vendite in SMA',linewidth=1.5)
    # plt.ylim([0, 1000])
    # plt.legend()
    # plt.show()

    # plt.figure(2)
    # plt.title("Prodotto 14405")
    # plt.figure(figsize = (13,7))
    # plt.plot(old_dic[14405],label='Vendite raw',linewidth=2)
    # plt.plot(dic[14405],label='Vendite in SMA',linewidth=1.5)
    # plt.ylim([0, 10])
    # plt.legend()
    # plt.show()

