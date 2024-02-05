import pytest
import numpy as np
from topsearch.potentials.test_functions import Camelback, Schwefel, Quadratic

######### QUADRATIC FUNCTION #############

def test_quadratic_function():
    quadratic = Quadratic()
    position = np.array([0.0, 0.0])
    assert quadratic.function(position) == pytest.approx(0.0)

def test_quadratic_gradient():
    quadratic = Quadratic()
    position = np.array([0.0, 0.0])
    assert np.max(quadratic.gradient(position)) < 1e-4

def test_quadratic_function_gradient():
    quadratic = Quadratic()
    position = np.array([0.0, 0.0])
    f_val, grad = quadratic.function_gradient(position)
    assert f_val == pytest.approx(0.0)
    assert np.max(grad) < 1e-4

def test_quadratic_hessian():
    quadratic = Quadratic()
    position = np.array([0.0, 0.0])
    hess = quadratic.hessian(position)
    assert np.all(hess == pytest.approx(np.array([[2.0, 0.0], [0.0, 2.0]])))

def test_quadratic_function2():
    quadratic = Quadratic()
    position = np.array([1.0, 0.6])
    assert quadratic.function(position) == pytest.approx(1.36)

def test_quadratic_gradient2():
    quadratic = Quadratic()
    position = np.array([1.0, 0.6])
    grad = quadratic.gradient(position)
    assert np.all(grad == pytest.approx(np.array([2.0, 1.2])))

def test_quadratic_function_gradient2():
    quadratic = Quadratic()
    position = np.array([1.0, 0.6])
    f_val, grad = quadratic.function_gradient(position)
    assert f_val == pytest.approx(1.36)
    assert np.all(grad == pytest.approx(np.array([2.0, 1.2])))

def test_quadratic_hessian2():
    quadratic = Quadratic()
    position = np.array([1.0, 0.6])
    hess = quadratic.hessian(position)
    assert np.all(hess == \
                  pytest.approx(np.array([[2.0, 0.0], [0.0, 2.0]]), abs=1e-4))

######### CAMELBACK FUNCTION #############

def test_camel_function():
    camelback = Camelback()
    position = np.array([0.0898, -0.7126])
    assert camelback.function(position) == pytest.approx(-1.031628422)

def test_camel_gradient():
    camelback = Camelback()
    position = np.array([0.0898, -0.7126])
    assert np.max(camelback.gradient(position)) < 1e-3

def test_camel_function_gradient():
    camelback = Camelback()
    position = np.array([0.0898, -0.7126])
    f_val, grad = camelback.function_gradient(position)
    assert f_val == pytest.approx(-1.031628422)
    assert np.max(grad) < 1e-3

def test_camel_hessian():
    camelback = Camelback()
    position = np.array([0.0898, -0.7126])
    hess = camelback.hessian(position)
    assert np.all(hess == pytest.approx(
        np.array([[7.79743648, 1.0],[1.0, 16.37434048]]), abs=1e-4))

def test_camel_function2():
    camelback = Camelback()
    position = np.array([0.0, 0.0])
    assert camelback.function(position) == pytest.approx(0.0)

def test_camel_gradient2():
    camelback = Camelback()
    position = np.array([0.0, 0.0])
    grad = camelback.gradient(position)
    assert np.max(grad) < 1e-3

def test_camel_function_gradient2():
    camelback = Camelback()
    position = np.array([0.0, 0.0])
    f_val, grad = camelback.function_gradient(position)
    assert f_val == pytest.approx(0.0)
    assert np.max(grad) < 1e-3

def test_camel_hessian2():
    camelback = Camelback()
    position = np.array([0.0, 0.0])
    hess = camelback.hessian(position)
    assert np.all(hess == pytest.approx(
        np.array([[8.0, 1.0], [1.0, -8.0]]), abs=1e-4))

###### SCHWEFEL FUNCTION ############

def test_schwefel_function():
    schwefel = Schwefel()
    position = np.array([420.9687, 420.9687, 420.9687])
    f_val = schwefel.function(position)
    assert f_val < 1e-4

def test_schwefel_gradient():
    schwefel = Schwefel()
    position = np.array([420.9687, 420.9687, 420.9687])
    grad = schwefel.gradient(position)
    assert np.max(grad) < 1e-4

def test_schwefel_function_gradient():
    schwefel = Schwefel()
    position = np.array([420.9687, 420.9687, 420.9687])
    f_val, grad = schwefel.function_gradient(position)
    assert f_val < 1e-4
    assert np.max(grad) < 1e-4

def test_schwefel_hessian():
    schwefel = Schwefel()
    position = np.array([420.9687, 420.9687, 420.9687])
    hess = schwefel.hessian(position)
    assert np.all(hess == pytest.approx(
        np.array([[0.22737368, 0.05684342, 0.05684342],
                  [0.05684342, 0.22737368, 0.05684342],
                  [0.05684342, 0.05684342, 0.22737368]]), abs=1e-4))

def test_schwefel_function2():
    schwefel = Schwefel()
    position = np.array([20.0, 40.0, 60.0])
    f_val = schwefel.function(position)
    assert f_val == pytest.approx(1215.06960408814)

def test_schwefel_gradient2():
    schwefel = Schwefel()
    position = np.array([20.0, 40.0, 60.0])
    grad = schwefel.gradient(position)
    assert np.all(grad == pytest.approx(
        np.array([1.50334654, -3.20093022, -1.41169903]), abs=1e-4))

def test_schwefel_function_gradient2():
    schwefel = Schwefel()
    position = np.array([20.0, 40.0, 60.0])
    f_val, grad = schwefel.function_gradient(position)
    assert f_val == pytest.approx(1215.06960408814)
    assert np.all(grad == pytest.approx(
        np.array([1.50334654, -3.20093022, -1.41169903]), abs=1e-4))

def test_schwefel_hessian2():
    schwefel = Schwefel()
    position = np.array([20.0, 40.0, 60.0])
    hess = schwefel.hessian(position)
    assert np.all(hess == pytest.approx(
        np.array([[-0.22737368, -0.05684342,  0.05684342],
                  [-0.05684342, -0.17053026, -0.05684342],
                  [ 0.05684342, -0.05684342,  0.28421709]]), abs=1e-4))
