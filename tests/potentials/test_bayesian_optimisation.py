import pytest
import numpy as np
import os
from topsearch.potentials.gaussian_process import GaussianProcess
from topsearch.potentials.bayesian_optimisation import UpperConfidenceBound, \
    ExpectedImprovement
from topsearch.data.model_data import ModelData

current_dir = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))

def test_ei_function():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_bayesopt1.txt',
                           response_file=f'{current_dir}/test_data/response_bayesopt1.txt')
    gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                         kernel_bounds=[(1e-1, 1e2), (1e-1, 1e2),
                                        (1e-5, 1e-1)],
                         standardise_response=True)
    ei = ExpectedImprovement(gaussian_process=gp, zeta=0.1)
    improv = ei.function(np.array([0.0, 0.0]))
    assert improv == pytest.approx(0.0)
    improv = ei.function(np.array([0.75, 1.1]))
    assert improv == pytest.approx(0.0)

def test_ei_function2():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_bayesopt2.txt',
                           response_file=f'{current_dir}/test_data/response_bayesopt2.txt')
    gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                         kernel_bounds=[(1e-1, 1e2), (1e-5, 1e-1)])
    ei = ExpectedImprovement(gaussian_process=gp, zeta=0.3)
    improv = ei.function(np.array([0.5]))
    assert improv == pytest.approx(0.03386785)
    ei.zeta = 0.15
    improv = ei.function(np.array([0.5]))
    assert improv == pytest.approx(0.14304196)

def test_ucb_function():
    model_data = ModelData(training_file=f'{current_dir}/test_data/training_bayesopt1.txt',
                           response_file=f'{current_dir}/test_data/response_bayesopt1.txt')
    gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                         kernel_bounds=[(1e-1, 1e2), (1e-1, 1e2),
                                        (1e-5, 1e-1)])
    ucb = UpperConfidenceBound(gaussian_process=gp, zeta=0.1)
    improv = ucb.function(np.array([0.0, 0.0]))
    assert improv == pytest.approx(-0.66490006)
    improv = ucb.function(np.array([0.75, 1.1]))
    assert improv == pytest.approx(-0.61416774)
