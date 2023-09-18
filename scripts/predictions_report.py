"""
Questo file conterrà script per calcolare le predizioni con i quattro modelli diversi e presentare i risultati in un pdf.

CNN: utlizzo i parametri trovati in precedenza a causa dei lunghi tempi di compilazione del tuning
Prophet: essendo veloce posso fare anche il tuning
DeepAR: Anche questo tuning è veloce
npts: non faccio tuning

Per ogni  modello otterrò due oggetti cluster_metrics e series_metrics.

Per le metriche di cluster mi basterà unire le serie in cluster_metrics in un unico dataframe.
Riguardo le metriche per serie dovrò iterare tra gli oggetti in series_metrics per ogni modello.

Il primo passo sarà un piccolo refactoring degli utils per la cnn per poter dare in input i parametri."""

from tspkg.forecasts.deepAr_utils import *
from tspkg.forecasts.nn_forecast_utils import *
from tspkg.forecasts.npts_utils import *
from tspkg.forecasts.prophet_utils import *
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def compute_n_save_metrics():
    with open(processed_dir / 'cluster1.pkl', 'rb') as f :
        dic = pickle.load(f)
    #?per sicurezza faccio una copia del dizionario ogni volta per evitare che modifiche che non ricordo falsino i risultati
    
    #uso parametri da tuning passati perchè il tuning per questa cnn ha tempi di compilazione molto lunghi
    params= {
        'conv_filters' : 32, 
        'kernel_size' : 4,
        'n_dense_nodes' : 100,
        'n_pooling' : 2,
    }

    #Vista la lunga compilazione, mi salvo questi file
    cnn_cluster_metrics, cnn_series_metrics, _ = cnn_compute_metrics(dic, params = params, n_timesteps = 730, n_prediction = 7)
    cnn_cluster_metrics.to_csv(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\cnn_cluster_metrics.csv')
    with open(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\cnn_series_metrics.pkl', 'wb') as f:
        pickle.dump(cnn_series_metrics, f)
    #cnn_series_metrics.to_csv(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\cnn_series_metrics.csv')
    
    prophet_cluster_metrics, prophet_series_metrics = prophet_tune_compute_metrics(dic, n_prediction=7)
    prophet_cluster_metrics.to_csv(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\prophet_cluster_metrics.csv')
    with open(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\prophet_series_metrics.pkl', 'wb') as f:
        pickle.dump(prophet_series_metrics, f)
    #prophet_series_metrics.to_csv(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\prophet_series_metrics.csv')
    
    deepAr_cluster_metrics, deepAr_series_metrics = deepAr_tune_compute_metrics(dic, n_prediction=7)
    deepAr_cluster_metrics.to_csv(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\deepAr_cluster_metrics.csv')
    with open(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\deepAr_series_metrics.pkl', 'wb') as f:
        pickle.dump(deepAr_series_metrics, f)
    #deepAr_series_metrics.to_csv(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\deepAr_series_metrics.csv')
    
    npts_cluster_metrics, npts_series_metrics = npts_compute_metrics(dic, n_prediction=7)
    npts_cluster_metrics.to_csv(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\npts_cluster_metrics.csv')
    with open(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\npts_series_metrics.pkl', 'wb') as f:
        pickle.dump(npts_series_metrics, f)
    #npts_series_metrics.to_csv(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\npts_series_metrics.csv')

def make_n_save_tables():
    cnn_cluster_metrics = pd.read_csv(r"C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\cnn_cluster_metrics.csv", index_col = 0)
    with open(r"C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\cnn_series_metrics.pkl", 'rb') as f:
        cnn_series_metrics = pickle.load(f)
    #cnn_series_metrics = pd.read_csv(r"C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\cnn_series_metrics.csv", header = None)
    prophet_cluster_metrics = pd.read_csv(r"C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\prophet_cluster_metrics.csv", index_col = 0)
    with open(r"C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\prophet_series_metrics.pkl", 'rb') as f:
        prophet_series_metrics = pickle.load(f)
    #prophet_series_metrics = pd.read_csv(r"C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\prophet_series_metrics.csv", header = None)
    deepAr_cluster_metrics = pd.read_csv(r"C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\deepAr_cluster_metrics.csv", index_col = 0)
    with open(r"C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\deepAr_series_metrics.pkl", 'rb') as f:
        deepAr_series_metrics = pickle.load(f)
    #deepAr_series_metrics = pd.read_csv(r"C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\deepAr_series_metrics.csv", header = None)
    npts_cluster_metrics = pd.read_csv(r"C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\npts_cluster_metrics.csv", index_col = 0)
    with open(r"C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\npts_series_metrics.pkl", 'rb') as f:
        npts_series_metrics = pickle.load(f)
    #npts_series_metrics = pd.read_csv(r"C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\npts_series_metrics.csv", header = None)

    cnn_cluster_metrics = cnn_cluster_metrics['CNN']
    prophet_cluster_metrics = prophet_cluster_metrics['Prophet']
    deepAr_cluster_metrics = deepAr_cluster_metrics['DeepAr']
    npts_cluster_metrics = npts_cluster_metrics['NPTS']


    cluster_table = pd.concat(objs = [cnn_cluster_metrics, prophet_cluster_metrics, deepAr_cluster_metrics, npts_cluster_metrics], axis = 1)

    series_tables = list()
    for i in range(len(cnn_series_metrics)):
        series_tables.append(pd.concat(objs = [cnn_series_metrics[i], prophet_series_metrics[i], deepAr_series_metrics[i], npts_series_metrics[i]], axis = 1))

    print(cluster_table)
    print(f"Lunghezza series table{len(series_tables)}")
    print(series_tables[0])

    #salvo tutto 
    cluster_table.to_csv(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\cluster_table.csv')

    with open(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\series_tables.pkl', 'wb') as f:
        pickle.dump(series_tables, f)

def make_pdf():

    cluster_table = pd.read_csv(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\cluster_table.csv', index_col = 0)
    print(cluster_table)

    with open(r"C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\metrics\series_tables.pkl", 'rb') as f:
        series_tables = pickle.load(f)
    
    with open(processed_dir / 'cluster1.pkl', 'rb') as f :
        dic = pickle.load(f)
    
    pdf = PdfPages(r"C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\predictions_report\report_forecasts.pdf")

    keys = list(dic.keys())
    fig, axs = plt.subplots(3, 4, figsize=(19,9))
    fig.suptitle(f"Serie nel Cluster")
    axs = axs.flatten()
    for i in range(len(axs)):
        if i < len(keys):
            axs[i].set_title(f"{i}")
            axs[i].plot(dic[keys[i]], linewidth=0.7)
    pdf.savefig()
    plt.close()

    fig, axs = plt.subplots(3, 4, figsize=(19,9))
    fig.suptitle(f"Metriche per ogni serie")
    axs = axs.flatten()
    for i in range(len(series_tables)):
        axs[i].axis('off')
        axs[i].set_title(f"{i}")
        #axs[i].axis('tight')
        table = pd.plotting.table(axs[i], series_tables[i].round(4), loc='center', cellLoc='center', rowLoc = 'center', colWidths=[.22, .22, .22, .22])
        table.scale(1, 2)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
    pdf.savefig()
    plt.close()

    

    fig, axs = plt.subplots(figsize=(19,9))
    fig.suptitle(f"Media delle metriche di tutte le serie")
    axs.axis('off')
    table = pd.plotting.table(axs, cluster_table.round(4), loc='center', cellLoc='center', rowLoc = 'center')
    table.scale(1, 2)
    pdf.savefig()



    pdf.close()


if __name__ == "__main__":
    #compute_n_save_metrics()
    #make_n_save_tables()
    make_pdf()