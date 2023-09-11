import pytest
from tspkg.forecast_utils import ForecastSeries
import numpy as np

#pytest.arr = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
#print(np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]).reshape(-1,1).shape)

class TestForecastSeries:

    def test_val_indif(self):
        arr = np.array([10, 20, 40, 50, 60, 80, 90, 110, 120, 130])
        print(arr.shape)
        obj = ForecastSeries(data = arr, timesteps = 2, prediction_length = 1)
        assert obj.val_indif == 80
        assert (obj.data == [10, 20, 40, 50, 60, 80, 90, 110, 120, 130]).all()
        assert obj.test_indif == 110
        assert len(obj.X) == 5
        assert len(obj.y) == 5
        