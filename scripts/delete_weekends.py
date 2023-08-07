import pandas as pd
import numpy as np
from datetime import datetime

import sys
sys.path.append(r"c:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject") 
from tspkg.paths import *
from tspkg.utils import *

with open(processed_dir / 'dictionary.pkl', 'rb') as f:
   dic = pickle.load(f)

def delete_weekends(dic):
    for p in dic.keys():
        dropped = 0 
        iszero = 0
        j = 0
        size = dic[p].size
        while(j < dic[p].size):
            date = datetime.date(dic[p].index[j])
            weekno = date.weekday()
            if weekno > 4:
                if(dic[p].Amount_sold[j]== 0): iszero +=1
                else:
                    dic[p].Amount_sold[max(0,j-1)] += dic[p].Amount_sold[j]
                dic[p].drop(dic[p].index[j], inplace = True)
                dropped += 1
            else: j+=1

    with open(processed_dir / 'nowknd_dictionary.pkl', 'wb') as f:
        pickle.dump(dic, f)                  
    return dic


if __name__ == "__main__":

    with open(processed_dir / 'dictionary.pkl', 'rb') as f:
        dic = pickle.load(f)
    delete_weekends(dic)


"""#codice opzionale per leggere da file il nuovo dizionario senza dover far girare il codice generatore 
with open(processed_dir / 'nowknd_dictionary.pkl', 'rb') as f:
   nowknd_dictionary = pickle.load(f)


with open(processed_dir / 'nowknd_dictionary.pkl', 'wb') as f:
    pickle.dump(compact_dic, f)"""