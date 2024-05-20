import pytest
import numpy as np
import os
from topsearch.potentials.dataset_fitting import DatasetRegression, \
    DatasetInterpolation
from topsearch.data.model_data import ModelData

current_dir = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))

def test_function_regression():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_fitting2.txt',
                           response_file=f'{current_dir}/test_data/response_fitting2.txt')
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

def test_get_model_error():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_fitting2.txt',
                           response_file=f'{current_dir}/test_data/response_fitting2.txt')
    regression = DatasetRegression(model_data)
    error = regression.get_model_error()
    assert error == pytest.approx(0.23901445200699975)

def test_regression_refit_model():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_fitting2.txt',
                           response_file=f'{current_dir}/test_data/response_fitting2.txt')
    regression = DatasetRegression(model_data)
    error = regression.get_model_error()
    assert error == pytest.approx(0.23901445200699975)
    model_data.standardise_response()
    regression.refit_model()
    error2 = regression.get_model_error()
    assert error2 == pytest.approx(0.110453626329787)

def test_function_interpolation():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_fitting1.txt',
                           response_file=f'{current_dir}/test_data/response_fitting1.txt')
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

def test_interpolation_refit_model():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_fitting1.txt',
                           response_file=f'{current_dir}/test_data/response_fitting1.txt')
    interpolation = DatasetInterpolation(model_data)
    position = np.array([-0.5, 1.2])
    f_val = interpolation.function(position)
    assert f_val == pytest.approx(8.0)
    model_data.standardise_response()
    interpolation.refit_model()
    position = np.array([-0.5, 1.2])
    f_val = interpolation.function(position)
    assert f_val == pytest.approx(0.8979663321173383)
