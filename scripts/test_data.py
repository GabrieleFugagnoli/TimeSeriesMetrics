import pandas as pd
import numpy as np
import sys
sys.path.append(r"c:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject") 
from tspkg.paths import *
from tspkg.utils import *


df = pd.read_csv(dict_path)
print(df.head())
