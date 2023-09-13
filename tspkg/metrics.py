from tspkg.paths import *
from tspkg.utils import *
import numpy as np


def mase(actual, predicted) -> int:
    
    if(len(actual) != len(predicted) + 1):
         raise ValueError("L'array actual deve avere un valore in più di predicted per calcolare il MAE")
    
    differences = []
    for i in range(len(actual)-1):
        differences.append(np.abs(actual[i+1]-actual[i]))
        
    mae_error = np.mean(differences)

    values = []
    for i in range(len(predicted)):
        values.append(np.abs((actual[i+1] - predicted[i]) / mae_error))
    return np.mean(values)

    # values = []
    # for i in range(1, len(actual)):
    #     values.append(np.abs(actual[i] - predicted[i]) / (np.abs(actual[i] - actual[i - 1]) / len(actual) - 1))
    # return np.mean(values)


def mape(actual, predicted) -> int:
    return (np.abs((actual - predicted) / actual).sum() / len(actual)) * 100


def wape(actual, predicted) -> int:
    return np.abs(actual - predicted).sum() / np.abs(actual).sum()


def rmse(actual, predicted) -> int:
    sum = 0
    for i in range(len(actual)):
            sum += np.power(actual[i]-predicted[i], 2)
    return np.sqrt(sum / len(actual))     

#unit test per serie uguali   
np.testing.assert_equal(mase([1,1,1], [1,1]), np.nan)
assert mape(np.array([1,1]), np.array([1,1])) ==  0
assert wape(np.array([1,1]), np.array([1,1])) ==  0
assert rmse([1,1], [1,1]) ==  0

#unit test per serie diverse 
assert mase(np.array([1,1,2,3]), np.array([1,2,1])) == 1
assert mase(np.array([1,1,2,3]), np.array([2,2,1])) == 1.5
assert mase(np.array([1,3]), np.array([4])) == 0.5
assert mape(np.array([1,2,3]), np.array([1,2,1])) == 200/9
assert wape(np.array([1,2,3]), np.array([1,2,1])) == 1/3
assert rmse(np.array([1,2,3]), np.array([1,2,1])) == np.sqrt(4/3)

# if __name__ == "__main__":
#      print(mase(np.array([1,2,3]), np.array([1,2,1])))

# if __name__ == "__main__":
#      print(mase([1,0,1,0,1],[100,45,38,50]))