import os
from pyts.approximation import SymbolicAggregateApproximation
import bz2
# import copy
# from sklearn.manifold import MDS
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import random
# from matplotlib.backends.backend_pdf import PdfPages
from tspkg.paths import *

def sax(dic: dict, bins: int = 26, strategy: str = 'quantile') -> list:
    """
    Starting form a dictionary of dataframes it creates a list of strings rapresenting the time series inside the dataframe.
    :param dict dic: A dictionary that containes id:dataframe entries
    :param int bins: The number of bins to use in the sax.
    :param str strategy: The type of sax strategy to follow, can be "uniform" or "quantile".
    :return: A list of strings rapresenting the time series inside the dataframe."""

    series_list = []
    series_sax_list = []

    #creo una lista di serie storiche (lista di)
    for p in dic:
        series_list.append(dic[p]['Amount_sold'].to_numpy())


    transformer = SymbolicAggregateApproximation(n_bins = bins, strategy = strategy)

    series_sax_list = (transformer.transform(series_list))

    return series_sax_list


def old_half_distance_matrix(series_sax_list: list) -> list:

    cbd_matrix = []
    series_sax_list_compressed = [bz2.compress(elem) for elem in series_sax_list]

    for x in range(len(series_sax_list)):
        len_1 = len(series_sax_list_compressed[x])
        cbd_elem = []
        print(f"Elaborazione cbd elemento {x}\n")
   
        for y in range(x, len(series_sax_list)):
        
            len_2 = len(series_sax_list_compressed[y])
            len_combined = len(bz2.compress(np.concatenate((series_sax_list[x], series_sax_list[y]), axis = None)))
            #Standard CBM distance
            cbd_elem.append(len_combined/(len_1 + len_2))
        
    
        cbd_matrix.append(cbd_elem)

    #specchio la matrice
    for i in range(len(cbd_matrix)):
        for x in range(i-1, -1, -1):
            cbd_matrix[i].insert(0, cbd_matrix[x][i])

    #mancano gli zeri nella diagonale
    
    return cbd_matrix

def distance_matrix(series_sax_list: list) -> list:
    cbd_matrix = []
    ssl_compressed_lenght = [compression_lenght(elem) for elem in series_sax_list]

    for x in range(len(series_sax_list)):
        len_1 = ssl_compressed_lenght[x]
        cbd_elem = []
        print(f"Elaborazione cbd elemento {x}\n")
   
        for y in range(len(series_sax_list)):
        
            len_2 = ssl_compressed_lenght[y]
            len_combined = compression_lenght(np.concatenate((series_sax_list[x], series_sax_list[y]), axis = None))
            #Standard CBM distance
            cbd_elem.append(len_combined/(len_1 + len_2))
        
        cbd_matrix.append(cbd_elem)
    
    return cbd_matrix

def agglomerative_clusters(dic: dict, distance_matrix: list, n_clusters: int = 4) -> list:

    agg = AgglomerativeClustering(n_clusters= n_clusters, metric = 'precomputed', linkage = 'average')
    labels = agg.fit_predict(distance_matrix)  # Returns class labels.

    cluster = [{} for _ in range(n_clusters)]

    i = 0
    for p in dic:
        cluster[labels[i]][p] = dic[p]
        i +=1
    
    return cluster
    
def visualize_clusters(cluster: list, file):
    mean_list = []
    random_key = random.choice(list(cluster[0].keys())) 
    ascissa = cluster[0][random_key].index

    n_clusters = len(cluster)
    fig, axs = plt.subplots(2, 3, figsize=(22,9))
    axs = axs.flatten()
    fig.suptitle(f"Divisione in {n_clusters} Clusters del sample")
    for i in range(len(axs)):
            if i < n_clusters:
                axs[i].set_title(f"Cluster {i+1}")
                plot_list = []
                for p in cluster[i]:
                    axs[i].plot(cluster[i][p], c="gray",alpha=0.3)
                    plot_list.append(cluster[i][p]['Amount_sold'].to_numpy())
                
                mean_list = np.mean(plot_list, axis=0)
                axs[i].plot(ascissa, mean_list, c="red",alpha=0.7)

    # fig.savefig(fname = path, format = 'pdf')
    file.savefig()
    plt.close()

def visualize_series(dic: dict, title: str, file):
    keys = list(dic.keys())
    fig, axs = plt.subplots(3, 5, figsize=(22,9))
    fig.suptitle(f"{title}")
    axs = axs.flatten()
    for i in range(len(axs)):
        if i < len(keys):
            axs[i].set_title(f"{i}")
            axs[i].plot(dic[keys[i]], linewidth=0.7)
    file.savefig()
    plt.close()
   

def visualize_matrix(file, matrix_arg):
    plt.matshow(matrix_arg)
    plt.suptitle("Matrice di similarità")
    file.savefig()
    plt.close()

def compression_lenght(string_arg: list) -> int:

    string = ''.join(str(elem) for elem in string_arg)

    ofile = bz2.BZ2File(processed_dir / "compressedStrings/compressed_file.txt", "wb")
    ofile.write(string.encode('ascii'))
    ofile.close()

    return os.path.getsize(processed_dir / "compressedStrings/compressed_file.txt")



