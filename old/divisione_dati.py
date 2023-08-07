import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('slim_timeseries.csv')

products = df.Product_ID.unique()

dataframes = dict()

for p in products:
    dataframes[p] = (df.loc[df.Product_ID == p, ['Date', 'Amount_sold']]).set_index("Date") 

print(dataframes[55])
print(type(dataframes[55]))

with open('dictionary.pkl', 'wb') as f:
    pickle.dump(dataframes, f)
        
#with open('saved_dictionary.pkl', 'rb') as f:
#   loaded_dict = pickle.load(f)