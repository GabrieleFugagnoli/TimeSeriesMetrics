import sys
sys.path.append(r"c:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject") 
from tspkg.paths import *
from tspkg.utils import *
from data_setup import data_setup
from smoothing import sma
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans


# total arguments
"""name = sys.argv[1]

dic = data_setup(name)"""



#Il file è stato salvato come dictionary.pkl nella directory processed
with open(processed_dir / 'nowknd_dictionary.pkl', 'rb') as f:
        dic = pickle.load(f)

index_list = [55, 60, 107, 160, 257, 307, 376, 93289, 93309, 95780, 99999]
test_data = dict()
for p in index_list:
      test_data[p] = dic[p]


object = StandardScaler()

for p in test_data:
    test_data[p]['Amount_sold'] = object.fit_transform(test_data[p])

sma(test_data)

series_list = []

for p in test_data:
    series_list.append(test_data[p]['Amount_sold'].to_numpy())



km = TimeSeriesKMeans(n_clusters=10, metric="dtw")

labels = km.fit_predict(series_list)
print(labels)




# plt.figure(1)
# plt.figure(figsize=(14, 5))
# plt.subplot(121)
# plt.plot(test_data[376])
# plt.subplot(122)
# plt.tight_layout()
# plt.plot(test_data[55])
# plt.show()

# correlation = pearson(test_data)
# print(correlation)



#A questo punto viene fatta l'analisi e la conseguente divisione in batches i quali subiranno diverse elaborazioni.
#I batches (ancora dizionari) vengono rielaborati e contemporaneamente salvati con pkl