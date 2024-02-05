import pytest
import numpy as np
from topsearch.potentials.dataset_fitting import DatasetRegression, \
    DatasetInterpolation
from topsearch.data.model_data import ModelData

def test_function_regression():
    model_data = ModelData(training_file='training2.txt',
                           response_file='response2.txt')
    regression = DatasetRegression(model_data)
    position = np.array([7.9, 3.8, 6.4])
    f_val = regression.function(position)
    assert f_val == pytest.approx(2.54062841)
    position = np.array([5.9, 3.0, 5.1])
    f_val = regression.function(position)
    assert f_val == pytest.approx(1.96005548)
    position = np.array([6.15, 3.2, 5.25])
    f_val = regression.function(position)
    assert f_val == pytest.approx(2.01908417)
    score = regression.model.score(regression.model_data.training,
                                   regression.model_data.response)
    assert score == pytest.approx(0.4650225296543671)

def test_function_interpolation():
    model_data = ModelData(training_file='training.txt',
                           response_file='response.txt')
    interpolation = DatasetInterpolation(model_data)
    position = np.array([-0.5, 1.2])
    f_val = interpolation.function(position)
    assert f_val == pytest.approx(8.0)
    position = np.array([1.0, 1.4])
    f_val = interpolation.function(position)
    assert f_val == pytest.approx(3.1)
    position = np.array([1.75, 1.5])
    f_val = interpolation.function(position)
    assert f_val == pytest.approx(8.05163747)
