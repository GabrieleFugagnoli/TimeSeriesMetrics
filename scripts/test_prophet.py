import pandas as pd 
import numpy as np
import pickle
import matplotlib.pyplot as plt
#Importo le mie funzioni
import sys
sys.path.append(r"c:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject") 
from tspkg.paths import *
from tspkg.utils import *
from prophet import Prophet
from prophet.diagnostics import cross_validation



def apply_prophet(dic):
   for p in dic:
      dic[p].reset_index(inplace = True)
      dic[p].columns = ['ds', 'y']


   pred = dict()

   for p in dic.keys():
      m = Prophet()
      m.fit(dic[p])
      pred[p] = cross_validation(m, initial='540 days', period='30 days', horizon = '30 days')

   with open(processed_dir / 'pred.pkl', 'wb') as f:
      pickle.dump(pred, f)

   return pred


if __name__ == "__main__":
   with open(processed_dir / 'nowknd_dictionary.pkl', 'rb') as f:
      dic = pickle.load(f)

   pred = apply_prophet(dic)

   """fig1= m.plot(pred[55])
   plt.show()

   fig2= m.plot(pred[41094])
   plt.show()"""


