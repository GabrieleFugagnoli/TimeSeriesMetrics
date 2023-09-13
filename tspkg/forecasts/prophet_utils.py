from tspkg.paths import *
from tspkg.utils import *
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import optuna

def objective_prophet(trial):
    params = dict()

    params['changepoint_prior_scale'] = trial.suggest_float('changepoint_prior_scale', 0.0001, 0.5, 0.05)
    params['seasonality_prior_scale'] = trial.suggest_float('seasonality_prior_scale', 0.01, 10, 1)
    params['holidays_prior_scale'] = trial.suggest_float('holidays_prior_scale', 0.01, 10, 1)
    params['seasonality_mode'] = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
    



def prophet_tuning(test_series):
    
    test_series.reset_index(inplace = True)
    test_series.columns = ['ds', 'y']

    
    m = Prophet().fit(series)
    cutoff = [test_series['ds'].iloc[-8]]   
    pred = cross_validation(m, initial='730 days', horizon = '7 days', cutoffs=cutoff)
    #volendo posso accedere alle metriche gi calcolate
    #error = performance_metrics(pred)['rmse'].iloc[-1]
    np.array(pred['yhat'])


if __name__ == "__main__":
    
    with open(processed_dir / 'cluster1.pkl', 'rb') as f :
        dic = pickle.load(f)
    series = dic[list(dic.keys())[0]]

    print(prophet_tuning(series))