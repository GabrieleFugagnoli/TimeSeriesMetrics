import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('target_time_series_D.csv').iloc[:, 0:3]

df.columns = ["Product_ID", "Date", "Amount_sold"]


df.Date = pd.to_datetime(df.Date)
df = df.set_index("Date")

df.to_csv('slim_timeseries.csv')

