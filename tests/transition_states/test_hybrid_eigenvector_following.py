import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import math
import os
from topsearch.data.coordinates import StandardCoordinates, AtomicCoordinates
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.potentials.test_functions import Camelback, Schwefel, Potential
from topsearch.potentials.atomic import LennardJones
from topsearch.potentials.dataset_fitting import DatasetInterpolation
from topsearch.data.model_data import ModelData

current_dir = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))

# Smallest eigenvalue tests

def test_generate_random_vector():
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-2)
    vec = single_ended.generate_random_vector(2)
    assert vec.size == 2
    assert np.linalg.norm(vec) == pytest.approx(1.0)
    vec = single_ended.generate_random_vector(4)
    assert vec.size == 4
    assert np.linalg.norm(vec) == pytest.approx(1.0)

def test_rayleigh_ritz_function_gradient():
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-2)
    vec = np.array([-0.06213744, 0.9980676])
    f_val, grad = single_ended.rayleigh_ritz_function_gradient(vec, 0.0, 0.0)
    assert f_val == pytest.approx(-8.06224187173922, abs=1e-4)
    assert np.max(np.abs(grad)) < 1e-3

def test_rayleigh_ritz_function_gradient2():
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-2)
    vec = np.array([0.06213744, -0.9980676])
    f_val, grad = single_ended.rayleigh_ritz_function_gradient(vec, 0.0, 0.0)
    assert f_val == pytest.approx(-8.06224187173922, abs=1e-4)
    assert np.max(np.abs(grad)) < 1e-3

def test_rayleigh_ritz_function_gradient3():
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-2)
    vec = np.array([-0.9980676, -0.06213744])
    f_val, grad = single_ended.rayleigh_ritz_function_gradient(vec, 0.0, 0.0)
    assert f_val == pytest.approx(8.06224187173922, abs=1e-4)
    assert np.max(np.abs(grad)) < 1e-3

def test_rayleigh_ritz_function_gradient4():
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-2)
    vec = np.array([0.514496, 0.857493])
    f_val, grad = single_ended.rayleigh_ritz_function_gradient(vec, 0.0, 0.0)
    assert f_val == pytest.approx(-2.8823422966763426, abs=1e-4)
    assert np.max(np.abs(grad)) > 5.0
    
def test_check_valid_eigenvector():
    coords = StandardCoordinates(ndim=2, bounds=[(-100.0, 100.0), (-100.0, 100.0)])
    coords.position = np.array([0.0, 0.0])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-2)
    valid = single_ended.check_valid_eigenvector(np.array([0.0, 0.0]),
                                                 10.0, coords)
    assert valid == False
    valid = single_ended.check_valid_eigenvector(np.array([1.0, np.nan]),
                                                 10.0, coords)
    assert valid == False
    valid = single_ended.check_valid_eigenvector(np.array([1.0, 1.0]),
                                                 0.0, coords)
    assert valid == False
    coords.position = np.array([100.0, -100.0])
    valid = single_ended.check_valid_eigenvector(np.array([1.0, -1.0]),
                                                 1.0, coords)
    assert valid == False
    coords.position = np.array([100.0, 10.0])
    valid = single_ended.check_valid_eigenvector(np.array([1.0, -1.0]),
                                                 1.0, coords)
    assert valid == True
    coords.position = np.array([10.0, 10.0])
    valid = single_ended.check_valid_eigenvector(np.array([1.0, -1.0]),
                                                 1.0, coords)
    assert valid == True

def test_check_eigenvector_direction():
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-2)
    position = np.array([-1.16046538004201443, -0.8050858882331980881])
    eigenvector = np.array([-0.99947827, 0.03229852])
    eig1 = single_ended.check_eigenvector_direction(eigenvector, position)
    eigenvector = np.array([ 0.99947828, -0.03229824])
    eig2 = single_ended.check_eigenvector_direction(eigenvector, position)
    assert np.all(eig1 == pytest.approx([-0.99947827, 0.03229852]))
    assert np.all(eig2 == pytest.approx([-0.99947828, 0.03229824]))

def test_parallel_component():
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-2)
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    p_comp = single_ended.parallel_component(vec1, vec2)
    assert np.all(p_comp == pytest.approx(np.array([0.0, 0.0, 0.0])))
    vec1 = np.array([0.4, 0.3, 0.1])
    vec2 = np.array([0.8, 0.6, 0.2])
    p_comp = single_ended.parallel_component(vec1, vec2)
    assert np.all(p_comp == pytest.approx(np.array([0.4, 0.3, 0.1])))
    vec1 = np.array([0.4, 0.3])
    vec2 = np.array([0.8, 0.1])
    p_comp = single_ended.parallel_component(vec1, vec2)
    assert np.all(p_comp == pytest.approx(np.array([0.4307692307692308,
                                                    0.05384615384615385])))

def test_perpendicular_component():
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-2)
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    p_comp = single_ended.perpendicular_component(vec1, vec2)
    assert np.all(p_comp == pytest.approx(np.array([1.0, 0.0, 0.0])))
    vec1 = np.array([0.4, 0.3, 0.1])
    vec2 = np.array([0.8, 0.6, 0.2])
    p_comp = single_ended.perpendicular_component(vec1, vec2)
    assert np.all(p_comp == pytest.approx(np.array([0.0, 0.0, 0.0])))
    vec1 = np.array([0.4, 0.3])
    vec2 = np.array([0.8, 0.1])
    p_comp = single_ended.perpendicular_component(vec1, vec2)
    assert np.all(p_comp == pytest.approx(np.array([-0.0307692307692308,
                                                    0.2461538462])))

def test_update_eigenvector_bounds():
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-2)
    single_ended.eigenvector_bounds = [(-math.inf, math.inf)]*3
    lower_bounds = np.array([0, 0, 0])
    upper_bounds = np.array([0, 0, 0])
    single_ended.update_eigenvector_bounds(lower_bounds, upper_bounds)
    assert single_ended.eigenvector_bounds == \
        [(-math.inf, math.inf), (-math.inf, math.inf), (-math.inf, math.inf)]
    lower_bounds = np.array([1, 0, 0])
    upper_bounds = np.array([0, 1, 0])
    single_ended.update_eigenvector_bounds(lower_bounds, upper_bounds)
    assert single_ended.eigenvector_bounds == \
        [(0.0, math.inf), (-math.inf, 0.0), (-math.inf, math.inf)]

@pytest.mark.parametrize(
    "lower_bounds, upper_bounds, vector, expected_projection",
    [
        ([0, 0], [0, 1], [0.5, 1.5], [1.0, 0.0]),
        ([0, 0], [0, 1], [-0.3, 0.1], [-1.0, 0.0]),
        ([0, 1, 1], [0, 0, 0], [0.3, -0.1, -0.2], [1.0, 0.0, 0.0]),
        ([0, 1, 1], [0, 0, 0], [0.3, 0.1, 0.2], [0.80178373, 0.26726124, 0.53452248])
    ])
def test_project_onto_bounds(lower_bounds, upper_bounds, vector, expected_projection):
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-2)
    proj = single_ended.project_onto_bounds(np.array(vector), np.array(lower_bounds), np.array(upper_bounds))
    assert_array_almost_equal(proj, expected_projection)
    assert np.linalg.norm(proj) == pytest.approx(1.0)

def test_get_smallest_eigenvalue1():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    coords.position = np.array([0.0, 0.0])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-2)
    rand_vec = single_ended.generate_random_vector(2)
    lower_bounds = np.array([0, 0])
    upper_bounds = np.array([0, 0])
    single_ended.eigenvector_bounds = [(-math.inf, math.inf)]*2
    eigv, eig, steps = single_ended.get_smallest_eigenvector(rand_vec, coords,
                                          lower_bounds, upper_bounds)
    if eigv[0] < 0.0:
        eigv *= -1.0
    diff = np.abs(eigv - np.array([0.06213744, -0.9980676]))
    assert eig == pytest.approx(-8.0622418717394, abs=1e-4)
    assert np.all(diff < 1e-3)

def test_get_smallest_eigenvalue2():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    coords.position = np.array([-1.296046538004201443, -0.6050858882331980881])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-2)
    rand_vec = single_ended.generate_random_vector(2)
    lower_bounds = np.array([0, 0])
    upper_bounds = np.array([0, 0])
    single_ended.eigenvector_bounds = [(-math.inf, math.inf)]*2
    eigv, eig, steps = single_ended.get_smallest_eigenvector(rand_vec, coords,
                                          lower_bounds, upper_bounds)
    if eigv[0] > 0.0:
        eigv *= -1.0
    assert eig == pytest.approx(-6.177671098551178, abs=1e-4)
    diff = np.abs(eigv - np.array([-0.997991, 0.06335585]))
    assert np.all(diff < 1e-3)

def test_get_smallest_eigenvalue3():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    coords.position = np.array([1.0, 2.0])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-4)
    rand_vec = single_ended.generate_random_vector(2)
    lower_bounds = np.array([0, 0])
    upper_bounds = np.array([0, 1])
    single_ended.eigenvector_bounds = [(-math.inf, math.inf)]*2
    eigv, eig, steps = single_ended.get_smallest_eigenvector(rand_vec, coords,
                                          lower_bounds, upper_bounds)
    if eigv[0] < 0.0:
        eigv *= -1.0
    assert np.abs(eig + 7.205229981722495) < 1e-2
    diff = np.abs(eigv - np.array([0.99998632, -0.00522991]))
    assert np.all(diff < 1e-2)

def test_get_smallest_eigenvalue4():
    coords = StandardCoordinates(ndim=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
    model_data = ModelData(f'{current_dir}/test_data/training_ts.txt',
                           f'{current_dir}/test_data/response_ts.txt')
    model_data.feature_subset([0, 1])
    model_data.remove_duplicates()
    model_data.normalise_training()
    model_data.normalise_response()
    interpolation = DatasetInterpolation(model_data, 1e-5)
    single_ended = HybridEigenvectorFollowing(potential=interpolation,
                                 ts_conv_crit=5e-4,
                                 ts_steps=150,
                                 pushoff=5e-3,
                                 eigenvalue_conv_crit=1e-3)
    rand_vec = single_ended.generate_random_vector(2)
    lower_bounds = np.array([0, 0])
    upper_bounds = np.array([0, 0])
    coords.position = np.array([0.39560017, 0.11547777])
    eigv, eig, steps = single_ended.get_smallest_eigenvector(rand_vec, coords,
                                                      lower_bounds, upper_bounds)
    assert eig == pytest.approx(-6556.503538596112, abs=5e-1)
    assert np.all(eigv == pytest.approx(np.array([-0.93580141, 0.35252763]), abs=1e-1))

def test_analytical_step_size():
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-4)
    grad = np.array([1.0, 0.0])
    eigenvector = np.array([0.707, 0.707])
    eigenvalue = 0.0
    length = single_ended.analytic_step_size(grad, eigenvector, eigenvalue)
    assert length == 1e-7

def test_analytical_step_size2():
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-4,
                                              max_uphill_step_size=1.0)
    grad = np.array([1.0, 0.0])
    eigenvector = np.array([0.707, 0.707])
    eigenvalue = -2.0
    length = single_ended.analytic_step_size(grad, eigenvector, eigenvalue)
    assert length == pytest.approx(0.31779805424079116)

def test_analytical_step_size3():
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-4,
                                              max_uphill_step_size=0.1)
    grad = np.array([1.0, 0.0])
    eigenvector = np.array([0.707, 0.707])
    eigenvalue = -2.0
    length = single_ended.analytic_step_size(grad, eigenvector, eigenvalue)
    assert length == pytest.approx(0.1)

def test_take_uphill_step():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    coords.position = np.array([1.0, 1.0])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-4,
                                              positive_eigenvalue_step=0.2)
    single_ended.take_uphill_step(coords, np.array([1.0, 0.0]), 12.0)
    assert np.all(coords.position == np.array([1.2, 1.0]))

def test_take_uphill_step2():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    coords.position = np.array([1.0, 1.0])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-4,
                                              positive_eigenvalue_step=0.2,
                                              max_uphill_step_size=0.1)
    single_ended.take_uphill_step(coords, np.array([1.0, 0.0]), -1.0)
    assert np.all(coords.position == np.array([1.1, 1.0]))

def test_subspace_function_gradient():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    coords.position = np.array([0.05, 0.3])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-4)
    f_val, grad = single_ended.subspace_function_gradient(coords.position, 0.0, 1.0)
    assert f_val == pytest.approx(-0.30261311979166666)
    assert np.all(grad == pytest.approx(np.array([0.69895062, 0.0])))

def test_subspace_minimisation():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    coords.position = np.array([0.05, 0.3])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-4)
    eigenvector = np.array([0.0, 1.0])
    pos, energy, r_dict = single_ended.subspace_minimisation(coords, eigenvector)
    assert energy == pytest.approx(-0.3332291642152334)
    assert np.all(pos == pytest.approx(np.array([-0.03755558, 0.3])))

def test_subspace_minimisation2():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    coords.position = np.array([0.3, 0.3])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-4)
    eigenvector = np.array([0.0, 1.0])
    pos, energy, r_dict = single_ended.subspace_minimisation(coords, eigenvector)
    print("pos, energy = ", pos, energy)
    assert energy == pytest.approx(-0.14619315859200002)
    assert np.all(pos == pytest.approx(np.array([0.18, 0.3])))

def test_get_local_bounds():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    coords.position = np.array([0.05, 0.3])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-4)
    subspace_bounds = single_ended.get_local_bounds(coords)
    assert subspace_bounds[0] == pytest.approx((-0.07, 0.17))
    assert subspace_bounds[1] == pytest.approx((0.22, 0.38))

def test_get_local_bounds2():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    coords.position = np.array([0.05, -2.0])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-4)
    subspace_bounds = single_ended.get_local_bounds(coords)
    assert subspace_bounds[0] == pytest.approx((-0.07, 0.17))
    assert subspace_bounds[1] == pytest.approx((-2.0, -1.92))

def test_get_local_bounds3():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    coords.position = np.array([3.0, -2.0])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-4)
    subspace_bounds = single_ended.get_local_bounds(coords)
    assert subspace_bounds[0] == pytest.approx((2.88, 3.0))
    assert subspace_bounds[1] == pytest.approx((-2.0, -1.92))

def test_test_convergence():
    position = np.array([-1.296046538004201443, -0.6050858882331980881])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 5e-4, 50, 1e-2)
    conv = single_ended.test_convergence(position, np.array([0, 0]),
                                         np.array([0, 0]))
    assert conv == True

def test_test_convergence2():
    position = np.array([-1.43, 0.3050858882331980881])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 5e-4, 50, 1e-2)
    conv = single_ended.test_convergence(position, np.array([0, 0]),
                                         np.array([0, 0]))
    assert conv == False

def test_test_convergence3():
    position = np.array([-0.041, 0.3050858882331980881])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 5e-2, 50, 1e-2)
    conv = single_ended.test_convergence(position, np.array([0, 0]),
                                         np.array([0, 1]))
    assert conv == True

def test_test_convergence_with_some_dimensions_at_bounds(mocker):
    position = np.array([0.2, 1, 0.5, -1])
    fake_func = Potential()
    fake_func.gradient = mocker.MagicMock(return_value=np.array([2e-2, 0.5, 1e-2, -0.7]))
    single_ended = HybridEigenvectorFollowing(fake_func, 5e-2, 50, 1e-2)
    conv = single_ended.test_convergence(position, np.array([0, 0, 0, 1]),
                                         np.array([0, 1, 0, 0]))
    assert conv == True

def test_do_pushoff():
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 5e-2, 50, 1e-2)
    ts_position = np.array([0.0, 0.0])
    eigenvector = np.array([-0.06213744, 0.9980676])
    new_pos = single_ended.do_pushoff(ts_position, eigenvector, 1e-2, 5)
    assert np.all(new_pos == pytest.approx(np.array([-0.00310687, 0.04990338])))

def test_do_pushoff2():
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 5e-2, 50, 1e-2)
    ts_position = np.array([0.5, 1.0])
    eigenvector = np.array([-0.2, 0.6])
    new_pos = single_ended.do_pushoff(ts_position, eigenvector, 1e-1, 10)
    assert np.all(new_pos == pytest.approx(np.array([0.3, 1.6])))

def test_find_pushoff():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 5e-2, 50, 1e-2)
    coords.position = np.array([0.0, 0.0])
    eigenvector = np.array([-0.06213744, 0.9980676])
    pos_x, neg_x = single_ended.find_pushoff(coords, eigenvector)
    assert np.all(pos_x == pytest.approx(np.array([-6.213744e-05,
                                                   9.980676e-04])))
    assert np.all(neg_x == pytest.approx(np.array([ 6.213744e-05,
                                                   -9.980676e-04])))

def test_find_pushoff2():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 5e-2, 50, 1e-4)
    coords.position = np.array([0.0, -0.1])
    eigenvector = np.array([-0.06213744, 0.9980676])
    pos_x, neg_x = single_ended.find_pushoff(coords, eigenvector)
    assert np.all(pos_x == pytest.approx(np.array([-1.24274880e-05,
                                                   -9.98003865e-02])))
    assert np.all(neg_x == pytest.approx(np.array([ 6.21374400e-07,
                                                   -1.00009981e-01])))

def test_find_pushoff3():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 5e-2, 50, 1e-4)
    coords.position = np.array([0.0, 0.1])
    eigenvector = np.array([-0.06213744, 0.9980676])
    pos_x, neg_x = single_ended.find_pushoff(coords, eigenvector)
    assert np.all(pos_x == pytest.approx(np.array([-6.21374400e-07,
                                                   1.00009981e-01])))
    assert np.all(neg_x == pytest.approx(np.array([1.24274880e-05,
                                                   9.98003865e-02])))

def test_find_pushoff4():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 5e-2, 50, 1e-2)
    coords.position = np.array([-1.296046538004201443, -0.6050858882331980881])
    eigenvector = np.array([-0.997991, 0.06335585])
    pos_x, neg_x = single_ended.find_pushoff(coords, eigenvector)
    assert np.all(pos_x == pytest.approx(np.array([-1.29704453, -0.60502253])))
    assert np.all(neg_x == pytest.approx(np.array([-1.29504855, -0.60514924])))

def test_steepest_descent():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-1)
    coords.position = np.array([0.0, 0.0])
    eigenvector = np.array([-0.06213744, 0.9980676])
    plus_min, plus_e, minus_min, minus_e = single_ended.steepest_descent(coords,
                                                                         eigenvector)
    assert np.all(plus_min == pytest.approx(np.array([-0.08984219,
                                                      0.71265638]), abs=1e-4))
    assert np.all(minus_min == pytest.approx(np.array([ 0.08984219,
                                                       -0.71265638]), abs=1e-4))

def test_steepest_descent2():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 5e-1)
    coords.position = np.array([-1.296046538004201443, -0.6050858882331980881])
    eigenvector = np.array([-0.997991, 0.06335585])
    plus_min, plus_e, minus_min, minus_e = single_ended.steepest_descent(coords,
                                                                         eigenvector)                                                       
    assert np.all(plus_min == pytest.approx(np.array([-1.60710468,
                                                      -0.56865148]), abs=1e-4))
    assert np.all(minus_min == pytest.approx(np.array([0.0898424,
                                                       -0.71265596]), abs=1e-4))
    
def test_steepest_descent_paths():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 1e-1)
    coords.position = np.array([-1.296046538004201443, -0.6050858882331980881])
    position, energy, dict = single_ended.steepest_descent_paths(coords)
    assert np.all(position == pytest.approx(np.array([0.08984218,
                                                       -0.71265642]), abs=1e-4))
    assert energy == pytest.approx(-1.031628453489768)

def test_steepest_descent_paths2():
    coords = StandardCoordinates(ndim=2, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0)])
    schwefel = Schwefel()
    single_ended = HybridEigenvectorFollowing(schwefel, 1e-5, 50, 1e-1)
    coords.position = np.array([-500.0, -100.0])
    position, energy, dict = single_ended.steepest_descent_paths(coords)
    assert np.all(position == pytest.approx(np.array([-500.0, 
                                                      -124.82933592]), abs=1e-4))
    assert energy == pytest.approx(534.5004679547465)

def test_run():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-6, 50, 5e-1)
    coords.position = np.array([-1.16046538004201443, -0.8050858882331980881])
    coords, e_ts, plus_min, e_plus, minus_min, e_minus, eig = \
        single_ended.run(coords)
    assert np.all(coords == pytest.approx(np.array([-1.29607027,
                                                    -0.60508439])))
    assert e_ts == pytest.approx(2.229470818029859)
    if plus_min[0] < minus_min[0]:
        assert np.all(plus_min == pytest.approx(np.array([-1.60710468,
                                                          -0.56865148])))
        assert np.all(minus_min == pytest.approx(np.array([ 0.08984202,
                                                           -0.7126564 ])))
        assert e_plus == pytest.approx(2.10425)
        assert e_minus == pytest.approx(-1.0316284534898772)
    else:
        assert np.all(minus_min == pytest.approx(np.array([-1.60710468,
                                                          -0.56865148])))
        assert np.all(plus_min == pytest.approx(np.array([ 0.08984202,
                                                           -0.7126564 ])))
        assert e_minus == pytest.approx(2.10425)
        assert e_plus == pytest.approx(-1.0316284534898772)

def test_remove_zero_eigenvectors():
    position = np.genfromtxt(f'{current_dir}/test_data/lj13.xyz')
    lj = LennardJones()
    single_ended = HybridEigenvectorFollowing(lj, 1e-6, 50, 5e-1)
    vec = np.array([  2.94274405e-02,  2.30205592e-01,  3.49224659e-02,
                     -7.50724582e-02, -1.24368964e-01,  2.65231712e-01,
                     -1.40278400e-01,  3.27853226e-02, -2.51176931e-01,
                     -1.86634429e-01,  2.97426803e-04, -1.01521770e-01,
                     -3.59128516e-01, -1.42529543e-01,  1.38298343e-02,
                     -1.91543215e-02, -3.79265746e-01, -2.44532714e-01,
                      1.38475633e-01,  1.16391983e-01, -3.22238064e-01,
                     -3.12317475e-01,  9.80138033e-02, -7.95901436e-02,
                      8.43064907e-02,  3.63203220e-03,  1.03146789e-02,
                      6.47939981e-02, -4.29072832e-02,  7.19787712e-03,
                      9.34287777e-02, -6.26051408e-02, -1.07448714e-02,
                     -9.75565088e-02, -9.14992957e-02,  7.97870299e-03,
                     -1.79584905e-01, -1.59155289e-01,  1.39074326e-01])
    vec = single_ended.remove_zero_eigenvectors(vec, position.flatten())
    assert np.all(vec == pytest.approx(np.array([0.12545495, 0.28450384, 0.11350183, -0.01106076,
                                                 -0.06487376, 0.35934976, -0.04026297, 0.08331985,
                                                 -0.18301086, -0.1472667, 0.04797959, -0.03846154,
                                                 -0.23882716, -0.10639233, 0.04194187, 0.04384718,
                                                 -0.34916713, -0.2303408, 0.18201613, 0.16080047,
                                                 -0.26826612, -0.18016716, 0.13449425, -0.05106284,
                                                 0.0850619, 0.04496626, 0.05779555, 0.14874466,
                                                 -0.01233587, 0.02175931, 0.13344495, -0.03546468,
                                                 -0.00359458, -0.00218044, -0.06966015, -0.00171502,
                                                 -0.09880458, -0.11817034, 0.18210343]), abs=1e-4))

def test_rayleigh_ritz_function_gradient5():
    position = np.genfromtxt(f'{current_dir}/test_data/lj13.xyz')
    lj = LennardJones()
    single_ended = HybridEigenvectorFollowing(lj, 1e-6, 50, 5e-1)
    single_ended.remove_trans_rot = True
    vec = np.array([  2.94274405e-02,  2.30205592e-01,  3.49224659e-02,
                     -7.50724582e-02, -1.24368964e-01,  2.65231712e-01,
                     -1.40278400e-01,  3.27853226e-02, -2.51176931e-01,
                     -1.86634429e-01,  2.97426803e-04, -1.01521770e-01,
                     -3.59128516e-01, -1.42529543e-01,  1.38298343e-02,
                     -1.91543215e-02, -3.79265746e-01, -2.44532714e-01,
                      1.38475633e-01,  1.16391983e-01, -3.22238064e-01,
                     -3.12317475e-01,  9.80138033e-02, -7.95901436e-02,
                      8.43064907e-02,  3.63203220e-03,  1.03146789e-02,
                      6.47939981e-02, -4.29072832e-02,  7.19787712e-03,
                      9.34287777e-02, -6.26051408e-02, -1.07448714e-02,
                     -9.75565088e-02, -9.14992957e-02,  7.97870299e-03,
                     -1.79584905e-01, -1.59155289e-01,  1.39074326e-01])
    f_val, grad = single_ended.rayleigh_ritz_function_gradient(vec, *position.flatten())
    assert f_val == pytest.approx(125.33577098153548, abs=1e-4)
    assert np.all(grad == pytest.approx(np.array([-3.42179306, -14.61058916, 9.47070617, 28.24454099, -30.54033991,
                                                  -36.79959485, 12.17165542, 68.54885275, -31.09259249, 45.3887703,
                                                   15.32494429, -58.82903373, -4.72585149, 8.07193141, 13.7266261,
                                                   17.48514736, 81.02926141, 2.47881102, -27.54984715, -0.24138185,
                                                  -12.78632718, 20.39685638, -6.85447997, -40.90169008, -29.31240567,
                                                  -43.7539737, 44.47936809, 0.81148729, 3.28306977, -6.89091314,
                                                  -17.15607116, 11.28301764, -28.44961534, 15.71178257, -3.54161929,
                                                   -4.95709362, -58.0442718, -87.99869339, 150.55134906]), abs=1e-3))

def test_run2():
    position = np.genfromtxt(f'{current_dir}/test_data/lj13.xyz')
    atom_labels = ['C', 'C', 'C', 'C', 'C', 'C', 'C',
                   'C', 'C', 'C', 'C', 'C', 'C']
    coords = AtomicCoordinates(atom_labels, position.flatten())
    lj = LennardJones()
    single_ended = HybridEigenvectorFollowing(lj, 1e-4, 50, 5e-1,
                                              eigenvalue_conv_crit=1e-5)
    ts_position, e_ts, plus_min, e_plus, minus_min, e_minus, eig = \
        single_ended.run(coords)
    assert e_ts == pytest.approx(-40.4326640775, abs=1e-4)
    if e_plus < e_minus:
        assert e_plus == pytest.approx(-44.3268014195, abs=1e-4)
        assert e_minus == pytest.approx(-41.4719798478, abs=1e-4)
    else:
        assert e_minus == pytest.approx(-44.3268014195, abs=1e-4)
        assert e_plus == pytest.approx(-41.4719798478, abs=1e-4)
