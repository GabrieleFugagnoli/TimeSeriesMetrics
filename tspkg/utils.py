#Importo le mie funzioni
import sys
sys.path.append(r"c:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject") 
from tspkg.paths import *
from tspkg.utils import *

import numpy as np
import pandas as pd
import pickle




def specific_data_processing(df) -> dict:
    """Partendo da un dataframe che contiene entry |ProductID - Data - Venduto - altre possibili colonne| ritorno un dataframe che 
    contiene solo |ProductID - Data - Venduto|, che ha come indice la data in formato pandas datetime.
    
    :param pd.DataFrame df: Dataframe da modificare

    :return: Il dataframe modificato
    
    :rtype: pd.DataFrame
    
    """

    df = df.iloc[:, 0:3]

    df.columns = ["Product_ID", "Date", "Amount_sold"]

    df.Date = pd.to_datetime(df.Date)
   
    df = df.set_index("Date")

    return df

def from_df_to_dict(df) -> dict:

    """Partendo da un dataframe che contiene entry |ProductID - Data - Venduto| ritono un dizionario che ha come chiavi 
    gli id dei prodotti e come valori associati i corrispondenti dataframe |Data - Amount_sold|

    :param pd.DataFrame df: Dataframe da modificare

    :return: Il dataframe modificato
    
    :rtype: pd.DataFrame

    """

    products = df.Product_ID.unique()

    dataframes = dict() 

    for p in products:
        dataframes[p] = (df.loc[df.Product_ID == p, ['Amount_sold']])

    return dataframes



def zeros_in_dic(dictionary: dict) -> dict: 

    """Ritorna un dizionario che contiene coppie [ID prodotto : Numero di zeri del prodotto] dei prodotti di dictionary
    che hanno almeno uno zero nella corrispondete serie storica.
    
    :param dict dictionary: The Dictionary that contains dataframes

    :return: A dictionary that contains the numbers of zeros inside the series of each key
    
    :rtype: dict
    
    """
    prods = dictionary.keys()

    zeros_dic = dict()

    for p in prods:
        array = (dictionary[p].loc[:,'Amount_sold']).to_numpy()
        zeros = np.count_nonzero(array == 0)
        if( zeros > 0):
            zeros_dic[p] = zeros

    return zeros_dic


def nan_counter(dic: dict) -> int:
    """Finds how many series inside the dictionary have at least one NaN value

    :param dict dic: the dictionary to analyze

    :return: The number of series with at least a NaN value
    
    :rtype: int

    """

    nan_polluted_series_counter = 0
    for p in dic:
        if dic[p].isnull().sum().sum() > 0:
            nan_polluted_series_counter+=1
    return nan_polluted_series_counter


def pearson(dic: dict) -> pd.DataFrame:
    """Calculates a correlation matrix of all the series inside a dictionary
    
    :param dict dic: The dictionary containing time series in dataframe format

    :return: A dataframe matrix that contains the correlation matrix

    :rtype: pd.DataFrame

    """

    df = pd.DataFrame()
    for p in dic:
        df = pd.concat([df, dic[p].rename(columns={'Amount_sold': p})], axis = 1)
    corr = df.corr(method = 'pearson')
    return corr
    

if __name__ == "__main__":

    with open(dict_path, 'rb') as f :
        dic = pickle.load(f)

    #nan_polluted_series_counter = nan_counter(dic)

    df = pearson(dic)

    print(df)

    

    

