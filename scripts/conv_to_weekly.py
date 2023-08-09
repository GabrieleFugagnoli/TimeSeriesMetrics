import pandas as pd
import sys
sys.path.append(r"c:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject") 
from tspkg.paths import *
from tspkg.utils import *
import matplotlib.pyplot as plt
import copy
    
    



if __name__ == "__main__":
    """Script that converts a dictionary stored in pkl format from daily data to weekly data"""

    with open(dict_path, 'rb') as f:
        dic = pickle.load(f)


    for p in dic.keys():
        dic[p]=dic[p].resample('w').sum()
    

    with open(processed_dir / 'weekly_dictionary.pkl', 'wb') as f:
        pickle.dump(dic, f)
