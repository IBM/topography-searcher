import pytest
import numpy as np
import os.path
import networkx as nx
from topsearch.potentials.gaussian_process import GaussianProcess
from topsearch.data.model_data import ModelData

current_dir = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))

def test_function_rbf():
    # Fix theta for testing
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_gp.txt',
                           response_file=f'{current_dir}/test_data/response_gp.txt')
    gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                         kernel_bounds=[(1e1, 1.00000001e1),
                                        (1e1, 1.00000001e1),
                                        (1e1, 1.00000001e1)],
                         standardise_response=False)
    assert np.all(gp.gpr.kernel_.bounds[0, :] == 
                  pytest.approx(np.array([2.30258509, 2.30258509])))
    assert np.all(gp.gpr.kernel_.bounds[1, :] == 
                  pytest.approx(np.array([2.30258509, 2.30258509])))
    assert np.all(gp.gpr.kernel_.bounds[2, :] == 
                  pytest.approx(np.array([2.30258509, 2.30258509])))
    gp_mean = gp.function(np.array([0.0, 0.0]))
    assert gp_mean == pytest.approx(12.302915954764156)
    gp_mean = gp.function(np.array([0.75, 1.1]))
    assert gp_mean == pytest.approx(12.295985976340937)
    
def test_function_and_std_rbf():
    # Fix theta for testing
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_gp.txt',
                           response_file=f'{current_dir}/test_data/response_gp.txt')
    gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                         kernel_bounds=[(1e1, 1.00000001e1),
                                        (1e1, 1.00000001e1),
                                        (1e1, 1.00000001e1)],
                         standardise_response=False)
    assert np.all(gp.gpr.kernel_.bounds[0, :] == 
                  pytest.approx(np.array([2.30258509, 2.30258509])))
    assert np.all(gp.gpr.kernel_.bounds[1, :] == 
                  pytest.approx(np.array([2.30258509, 2.30258509])))
    assert np.all(gp.gpr.kernel_.bounds[2, :] == 
                  pytest.approx(np.array([2.30258509, 2.30258509])))
    gp_mean, gp_std = gp.function_and_std(np.array([0.0, 0.0]))
    assert gp_mean == pytest.approx(12.302915954764156)
    assert gp_std == pytest.approx(3.21608076)
    gp_mean, gp_std = gp.function_and_std(np.array([0.75, 1.1]))
    assert gp_mean == pytest.approx(12.295985976340937)
    assert gp_std == pytest.approx(3.21782038)

def test_function_rbf_fit():
    # Fit the rbf kernel
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_gp.txt',
                           response_file=f'{current_dir}/test_data/response_gp.txt')
    gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                         kernel_bounds=[(1e-1, 1e2), (1e-1, 1e2),
                                        (1e-5, 1e-4)],
                         standardise_response=False)
    assert np.all(gp.gpr.kernel_.bounds[0, :] == 
                  pytest.approx(np.array([-2.30258509, 4.60517019])))
    assert np.all(gp.gpr.kernel_.bounds[1, :] == 
                  pytest.approx(np.array([ -2.30258509, 4.60517019])))
    assert np.all(gp.gpr.kernel_.bounds[2, :] == 
                  pytest.approx(np.array([-11.51292546, -9.210340371976182])))
    assert np.all(gp.gpr.kernel_.theta == 
                  pytest.approx(np.array([0.46979649, -1.78642308,
                                          -9.21034037])))
    assert gp.get_score() == pytest.approx(0.9999999859966073)
    gp_mean = gp.function(np.array([0.0, 0.0]))
    assert gp_mean == pytest.approx(1.9089953273359477)
    gp_mean = gp.function(np.array([0.75, 1.1]))
    assert gp_mean == pytest.approx(3.4859748905013)

def test_standardise_training():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_gp.txt',
                           response_file=f'{current_dir}/test_data/response_gp.txt')
    gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                         kernel_bounds=[(1e-1, 1e2), (1e-1, 1e2),
                                        (1e-5, 1e-4)],
                         standardise_training=True,
                         standardise_response=False)
    assert np.all(np.mean(gp.model_data.training, axis=0) == 
                  pytest.approx(np.array([0.0, 0.0])))
    assert np.all(np.std(gp.model_data.training, axis=0) == 
                  pytest.approx(np.array([1.0, 1.0])))
    assert np.all(gp.gpr.kernel_.theta == 
                  pytest.approx(np.array([-0.0782580832765277, -1.9290125537753147,
                                          -9.210340371976182])))
    assert gp.get_score() == pytest.approx(0.9999999859966073)

def test_standardise_response():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_gp.txt',
                           response_file=f'{current_dir}/test_data/response_gp.txt')
    gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                         kernel_bounds=[(1e-1, 1e2), (1e-1, 1e2),
                                        (1e-5, 1e-4)],
                         standardise_response=True)
    assert np.mean(gp.model_data.response) == pytest.approx(0.0)
    assert np.std(gp.model_data.response) == pytest.approx(1.0)
    assert np.all(gp.gpr.kernel_.theta == 
                  pytest.approx(np.array([-0.94938977, 0.27690632,
                                          -9.21034037])))
    gp_mean = gp.function(np.array([0.75, 1.1]))
    assert gp_mean == pytest.approx(-0.5768242790490604)

def test_standardise_both():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_gp.txt',
                           response_file=f'{current_dir}/test_data/response_gp.txt')
    gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                         kernel_bounds=[(1e-1, 1e2), (1e-1, 1e2),
                                        (1e-5, 1e-1)],
                         standardise_training=True,
                         standardise_response=True)
    assert np.all(np.mean(gp.model_data.training, axis=0) == 
                  pytest.approx(np.array([0.0, 0.0])))
    assert np.all(np.std(gp.model_data.training, axis=0) == 
                  pytest.approx(np.array([1.0, 1.0])))

def test_function_rbf_limit():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_gp.txt',
                           response_file=f'{current_dir}/test_data/response_gp.txt')
    mean = np.mean(model_data.response)
    gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                         kernel_bounds=[(1e-1, 1e2), (1e-1, 1e2),
                                        (1e-5, 1e-1)],
                         standardise_response=False,
                         limit_highest_data=True)
    assert np.max(gp.model_data.response) == np.abs(mean*5.0)

def test_function_matern():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_gp.txt',
                           response_file=f'{current_dir}/test_data/response_gp.txt')
    gp = GaussianProcess(model_data=model_data, kernel_choice='Matern',
                         kernel_bounds=[(1e-1, 1e2), (1e-1, 1e2),
                                        (1e-5, 1e-2)],
                         matern_nu=1.5)
    assert np.all(gp.gpr.kernel_.theta == pytest.approx([-0.72664135,
                                                         0.49700888,
                                                         -4.60517019]))

def test_write_fit():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_gp.txt',
                           response_file=f'{current_dir}/test_data/response_gp.txt')
    gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                         kernel_bounds=[(1e-1, 1e2), (1e-1, 1e2),
                                        (1e-5, 1e-1)])
    gp.write_fit()

def test_refit_model():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_gp.txt',
                           response_file=f'{current_dir}/test_data/response_gp.txt')
    gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                         kernel_bounds=[(1e-1, 1e2), (1e-1, 1e2),
                                        (1e-5, 1e-4)],
                         standardise_response=False)
    assert np.all(gp.gpr.kernel_.theta == 
                  pytest.approx(np.array([0.46979649, -1.78642308,
                                          -9.21034037])))
    model_data.standardise_response()
    gp.refit_model()
    assert np.all(gp.gpr.kernel_.theta == 
                  pytest.approx(np.array([-0.94938977, 0.27690632,
                                          -9.21034037])))
    
def test_update_bounds():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_gp.txt',
                           response_file=f'{current_dir}/test_data/response_gp.txt')
    gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                         kernel_bounds=[(1e1, 1.00000001e1),
                                        (1e1, 1.00000001e1),
                                        (1e-5, 1e-4)],
                         standardise_response=False)
    gp.update_bounds(0.99)
    gp.update_bounds(0.99)
    gp.update_bounds(0.99)
    gp.refit_model()
    assert np.all(gp.gpr.kernel_.theta == 
                  pytest.approx(np.array([0.46979649, -1.78642308,
                                          -9.21034037])))
