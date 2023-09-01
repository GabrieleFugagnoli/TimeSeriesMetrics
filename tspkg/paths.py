from pathlib import Path

raw_dir = Path(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\raw')

processed_dir = Path(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed')

grafici = Path(r'C:\Users\gabriele.fugagnoli\Desktop\gabrielefugagnoli\TimeSeriesMetricsProject\data\processed\grafici')

first_cv_path = raw_dir / 'target_time_series_D.csv'

dict_path = processed_dir / 'dictionary.pkl'

#print(first_cv_path)