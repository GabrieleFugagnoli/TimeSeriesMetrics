import pandas as pd
import sys
sys.path.append(r"c:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject") 
from tspkg.paths import *
from tspkg.utils import *
import matplotlib.pyplot as plt
import copy
    
    
def conv_to_weekly(dic):

    for p in dic.keys():
        dic[p]=dic[p].resample('w').sum()
    

    with open(processed_dir / 'weekly_dictionary.pkl', 'wb') as f:
        pickle.dump(dic, f)



if __name__ == "__main__":

    dic = dict()

    with open(dict_path, 'rb') as f:
        dic = pickle.load(f)

    old_dic = copy.deepcopy(dic)
    conv_to_weekly(dic)

    plt.figure(1)
    plt.title("Prodotto 91548")
    plt.figure(1)
    plt.figure(figsize = (13,7))
    plt.plot(old_dic[91548],label='Vendite raw',linewidth=2)
    plt.plot(dic[91548],label='Vendite settimanali',linewidth=2)
    plt.ylim([0, 100])
    plt.show()
"""
    plt.figure(2)
    plt.title("Prodotto 14405")
    plt.figure(figsize = (13,7))
    plt.plot(old_dic[14405],label='Vendite raw',linewidth=2)
    plt.plot(dic[14405],label='Vendite settimanali',linewidth=2)
    plt.ylim([0, 28])
    plt.show()
"""