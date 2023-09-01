import pickle as pkl 
import sys
sys.path.append(r"c:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject") 
from tspkg.paths import *
from tspkg.utils import *
from tspkg.cbd_utils import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from matplotlib.backends.backend_pdf import PdfPages

def raw(dic_arg: dict, path, n_bins: int = 26, clusters: int = 4, matrix: bool = False):
   
   #dizionario per il clustering
   dic = copy.deepcopy(dic_arg)

   #dizionario per la visualizzazione
   dic_vis = copy.deepcopy(dic_arg)
   sma(dic_vis, dropna = True)
   scaler = StandardScaler()
   for p in dic_vis:
        dic_vis[p].loc[:, 'Amount_sold'] = scaler.fit_transform(dic_vis[p])

   #dic non viene modificato
   series_sax_list = sax(dic, bins = n_bins)

   cbd_matrix = distance_matrix(series_sax_list)

   clusters = agglomerative_clusters(dic_vis, cbd_matrix, n_clusters = clusters)

   visualize_series(dic_vis, file = path, title =f"Visualizzazione di 15 serie della selezione")

   if matrix:
       visualize_matrix(path, cbd_matrix)

   #stampo i clusters
   count= 1
   for elem in clusters:
      visualize_series(elem, file = path, title =f"Visualizzazione di 15 serie del cluster {count}")
      count+=1


   visualize_clusters(clusters, path)

def std(dic_arg: dict, path, n_bins: int = 26, clusters: int = 4, matrix: bool = False):
   
   #dizionario per il clustering
   dic = copy.deepcopy(dic_arg)

   scaler = StandardScaler()

   for p in dic:
        dic[p].loc[:, 'Amount_sold'] = scaler.fit_transform(dic[p])

   #dizionario per la visualizzazione: sia sma che std
   dic_vis = copy.deepcopy(dic_arg)
   sma(dic_vis, dropna = True)
   scaler = StandardScaler()
   for p in dic_vis:
        dic_vis[p].loc[:, 'Amount_sold'] = scaler.fit_transform(dic_vis[p])

   #dic non viene modificato
   series_sax_list = sax(dic, bins = n_bins)

   cbd_matrix = distance_matrix(series_sax_list)

   clusters = agglomerative_clusters(dic_vis, cbd_matrix, n_clusters = clusters)
   visualize_series(dic_vis, file = path, title =f"Visualizzazione di 15 serie della selezione")
   
   if matrix:
       visualize_matrix(path, cbd_matrix)

   #stampo i clusters
   count= 1
   for elem in clusters:
      visualize_series(elem, file = path, title =f"Visualizzazione di 15 serie del cluster {count}")
      count+=1


   visualize_clusters(clusters, path)

def sma_cbd(dic_arg: dict, path, n_bins: int = 26, clusters: int = 4, sma_window = 30, matrix: bool = False):
   
   #dizionario per il clustering
   dic = copy.deepcopy(dic_arg)

   sma(dic, sma_window, dropna = True)

   #dizionario per la visualizzazione: sia sma che std
   dic_vis = copy.deepcopy(dic_arg)
   sma(dic_vis, dropna = True)
   scaler = StandardScaler()
   for p in dic_vis:
        dic_vis[p].loc[:, 'Amount_sold'] = scaler.fit_transform(dic_vis[p])

   #dic non viene modificato
   series_sax_list = sax(dic, bins = n_bins)

   cbd_matrix = distance_matrix(series_sax_list)

   clusters = agglomerative_clusters(dic_vis, cbd_matrix, n_clusters = clusters)

   visualize_series(dic_vis, file = path, title = f"Visualizzazione di 15 serie della selezione")
   
   if matrix:
       visualize_matrix(path, cbd_matrix)

   #stampo i clusters
   count= 1
   for elem in clusters:
      visualize_series(elem, file = path, title =f"Visualizzazione di 15 serie del cluster {count}")
      count+=1


   visualize_clusters(clusters, path)


def sma_n_std(dic_arg: dict, path, n_bins: int = 26, clusters: int = 4, sma_window = 30, matrix: bool = False):
   
   #dizionario per il clustering
   dic = copy.deepcopy(dic_arg)

   sma(dic, sma_window, dropna = True)

   scaler = StandardScaler()

   for p in dic:
        dic[p].loc[:, 'Amount_sold'] = scaler.fit_transform(dic[p])


   #dic non viene modificato
   series_sax_list = sax(dic, bins = n_bins)

   cbd_matrix = distance_matrix(series_sax_list)

   clusters = agglomerative_clusters(dic, cbd_matrix, n_clusters = clusters)
   visualize_series(dic, file = path, title =f"Visualizzazione di 15 serie della selezione")

   if matrix:
       visualize_matrix(path, cbd_matrix)

   #stampo i clusters
   count= 1
   for elem in clusters:
      visualize_series(elem, file = path, title =f"Visualizzazione di 15 serie del cluster {count}")
      count+=1


   visualize_clusters(clusters, path)

def few_series(samples, clusters, bins, path):
   n_clusters = clusters

   selection = samples

   pdf = PdfPages(path / "raw.pdf")
   raw(selection, path = pdf, n_bins = bins, clusters = n_clusters, matrix = True)

   pdf.close()

   pdf = PdfPages(path / 'std.pdf')
   std(selection, path = pdf, n_bins = bins, clusters = n_clusters, matrix = True)
   pdf.close()

   pdf = PdfPages(path / 'sma.pdf')
   sma_cbd(selection, path = pdf, n_bins = bins, clusters = n_clusters, matrix = True)
   pdf.close()

   pdf = PdfPages(path / 'sma_n_std.pdf')
   sma_n_std(selection, path = pdf, n_bins = bins, clusters = n_clusters, matrix = True)
   pdf.close()

def more_series(samples, clusters, bins, path):
   
   n_clusters = clusters
   selection = samples

   pdf = PdfPages(path / 'sma.pdf')
   sma_cbd(selection, path = pdf, n_bins = bins, clusters = n_clusters, matrix = False)
   pdf.close()

    

if __name__ == "__main__":
   
   with open(processed_dir / 'final_filtered_bigger_dictionary.pkl', 'rb') as f:
        dic = pickle.load(f)

   # selection = sample(dic, 15)

   # few_series(selection, clusters = 3, bins = 26, path =  grafici / "26_bins")

   # few_series(selection, clusters = 3, bins = 20, path =  grafici / "20_bins")

   # few_series(selection, clusters = 3, bins = 10, path =  grafici / "10_bins")

   # few_series(selection, clusters = 3, bins = 4, path =  grafici / "4_bins")

   selection = sample(dic, 80)
   more_series(selection, clusters = 6, bins = 4, path =  grafici / "more_series")







