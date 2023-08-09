#Importo le mie funzioni
import sys
sys.path.append(r"c:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject") 
from tspkg.paths import *
from tspkg.utils import *
import copy

import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.seasonal import seasonal_decompose




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


def continous_zeros(dic: dict, n_elements: int = 200) -> dict:
    """Calculates the list of continous zeros of the series inside a dictionary
        input: dizionario [chiave : dataframe] dove le serie storiche sono giornaliere
        output dizionario [chiave : lista] di zeri consecutivi negli ultimi n_elements giorni prima dell'inizio della prediction
     
      :param dict dic: Dictionary that contains dataframes

      :param int n_elements: The number of elements in the dataframe to use from the last value of the series

      :return: dictionary that contains the list of zeros for each element

      :rtype: dict
      """
    #non voglio modificare dic
    temp = pd.DataFrame()
    zeros_dic = dict()
    for p in dic.keys():
        streak_count = 0
        zeros_list = []
        temp = dic[p].reset_index()
        streak = False
        values = temp.loc[(len(temp)-n_elements):(len(temp)-1), ['Amount_sold']].to_numpy()
        for i in values:
            #print(i)
            if i == 0 or np.isnan(i):
                if streak:
                    #eravamo già in streak
                    streak_count += 1
                else:
                    #inizia la streak
                    streak = True
                    streak_count +=1
            else:
                if streak:
                    #finisce la streak
                    streak = False 
                    zeros_list.append(streak_count)
                    streak_count = 0

        if streak: zeros_list.append(streak_count)
        zeros_dic[p] = zeros_list

    return zeros_dic


def delete_gaps(dic_arg: dict, zeros_dic: dict, cutoff: int = 100) -> dict:
    """Deletes from the dictionary the dataframes with elements bigger than cutoff inside their respective zeros_dict

    :param dict dic_arg: The dictionary from which to delete the entries

    :param dict zeros_dic: A dictionary that pairs every key to the list of continous zero of the relative dataframe

    :param int cutoff: The cutoff number of continous zeros above which to delete the series

    :return: A dictionary without the deleted elements

    :rtype: dict """

    counter = 0
    dic = copy.deepcopy(dic_arg)
    for p in zeros_dic:
        if any(num > cutoff  for num in zeros_dic[p]):
            print(f"Cancello il prodotto n.{p}")
            print(f"Lista di zeri del prodotto n.{p}: {zeros_dic[p]}")
            counter +=1
            del dic[p]
    print(f"Prodotti eliminati: {counter}")
    return dic


    print(len(dic))


def conv_to_weekly(dic):

    for p in dic.keys():
        dic[p]=dic[p].resample('w').sum()
    

    with open(processed_dir / 'weekly_dictionary.pkl', 'wb') as f:
        pickle.dump(dic, f)


def stl(dic, period = 7):
    """Ritorna un dizionario con i valori di trend e seasonality associati agli id del prodotto, ancora da testare, non l'ho più usata"""
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
    # decompose.plot()
    # plt.show()

    df = copy.deepcopy(dic[55])
    
    df["Seasonal"] = decompose.seasonal
    df["Trend"] =  decompose.trend
    df["Resid"] = decompose.resid
    

    return df

#sma e ewm sono da rendere deep copy e da scrivere la documentazione
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



    
# if __name__ == "__main__":

#     with open(dict_path, 'rb') as f :
#         dic = pickle.load(f)

#     #nan_polluted_series_counter = nan_counter(dic)

#     df = pearson(dic)

#     print(df)

    

    

