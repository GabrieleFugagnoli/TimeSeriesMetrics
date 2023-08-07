import sys
sys.path.append(r"c:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject") 
from tspkg.paths import *
from tspkg.utils import *
import pickle



def data_setup(csv_name):

    df = pd.read_csv(raw_dir/csv_name, header = None)

    df = specific_data_processing(df)

    dataframes = dict()

    dataframes = from_df_to_dict(df)

    #print(len(dataframes))

    with open(processed_dir / 'dictionary.pkl', 'wb') as f:
        pickle.dump(dataframes, f)

    return dataframes


if __name__ == "__main__":
    data_setup("target_time_series_D.csv")