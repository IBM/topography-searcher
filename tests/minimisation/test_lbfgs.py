import pytest
import numpy as np
from topsearch.data.coordinates import StandardCoordinates
from topsearch.minimisation.lbfgs import minimise
from topsearch.potentials.test_functions import Camelback, Schwefel

def test_lbfgs():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camelback = Camelback()
    min_coords, f_val, res_dict = \
        minimise(func_grad=camelback.function_gradient,
                 initial_position = np.array([0.1, -0.7]),
                 bounds=coords.bounds,
                 conv_crit=1e-6)
    assert np.all(min_coords == pytest.approx(np.array([0.0898420385,
                                                        -0.712656397])))
    assert f_val == pytest.approx(-1.031628422)
    assert np.max(camelback.gradient(min_coords)) < 1e-6

def test_lbfgs2():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camelback = Camelback()
    min_coords, f_val, res_dict = \
        minimise(func_grad=camelback.function_gradient,
                 initial_position = np.array([-0.1, 0.7]),
                 bounds=coords.bounds,
                 conv_crit=1e-6)
    assert np.all(min_coords == pytest.approx(np.array([-0.0898420385,
                                                         0.712656397])))
    assert f_val == pytest.approx(-1.031628422)
    assert np.max(camelback.gradient(min_coords)) < 1e-6

def test_lbfgs3():
    coords = StandardCoordinates(ndim=3, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0),
                                                 (-500.0, 500.0)])
    schwefel = Schwefel()
    min_coords, f_val, res_dict = \
        minimise(func_grad=schwefel.function_gradient,
                 initial_position = np.array([422.0, 422.0, 422.0]),
                 bounds=coords.bounds,
                 conv_crit=1e-6)
    assert np.all(min_coords == pytest.approx(np.array([420.96874612,
                                                        420.96874612,
                                                        420.96874612])))
    assert f_val < 1e-3
    assert np.max(schwefel.gradient(min_coords)) < 1e-6
