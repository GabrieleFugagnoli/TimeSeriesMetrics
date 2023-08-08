#Per caricare il dizionario di dataframe:
import pickle

#Importo le mie funzioni
import sys
sys.path.append(r"c:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject") 
from tspkg.paths import *
from tspkg.utils import *

import numpy as np
import copy
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    

    with open(dict_path, 'rb') as f:
        dic = pickle.load(f)

    zeros_dic = continous_zeros(dic)

    for p in zeros_dic:
        print(f"Prodotto {p} : \n Numero di zeri: {sum(zeros_dic[p])} \n {zeros_dic[p]}\n ----------")
    
    
    print(len(dic))

    delete_gaps(dic, zeros_dic)

    with open(processed_dir / 'nogaps_dictionary.pkl', 'wb') as f:
        pickle.dump(dic, f)


    print(len(dic))

    

    


    
