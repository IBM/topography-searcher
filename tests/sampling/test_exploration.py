import pytest
import numpy as np
import ase.io
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.similarity.molecular_similarity import MolecularSimilarity
from topsearch.potentials.test_functions import Camelback
from topsearch.potentials.dataset_fitting import DatasetInterpolation
from topsearch.data.model_data import ModelData
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.data.coordinates import StandardCoordinates, MolecularCoordinates
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.global_optimisation.perturbations import StandardPerturbation
from topsearch.sampling.exploration import NetworkSampling
from topsearch.analysis.minima_properties import get_minima_energies
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand

def test_get_minima():
    step_taking = StandardPerturbation(max_displacement=0.7,
                                       proportional_distance=True)
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    camelback = Camelback()
    basin_hopping = BasinHopping(ktn=ktn, potential=camelback,
                                 similarity=similarity,
                                 step_taking=step_taking)
    sampler = NetworkSampling(ktn, None, basin_hopping, None, None, None)
    sampler.get_minima(coords, 500, 1e-5, 100.0, test_valid=False)
    energies = get_minima_energies(ktn)
    assert np.sort(energies)[0] == pytest.approx(-1.03162845)

def test_get_minima2():
    step_taking = StandardPerturbation(max_displacement=0.7,
                                       proportional_distance=True)
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    camelback = Camelback()
    basin_hopping = BasinHopping(ktn=ktn, potential=camelback,
                                 similarity=similarity,
                                 step_taking=step_taking)
    sampler = NetworkSampling(ktn, None, basin_hopping, None, None, None)
    sampler.get_minima(coords, 500, 1e-5, 100.0, test_valid=True)
    energies = get_minima_energies(ktn)
    assert np.all(np.sort(energies[:6]) == pytest.approx(np.array([-1.03162845,
                                                               -1.03162845,
                                                               -0.21546382,
                                                               -0.21546382,
                                                               2.10425031,
                                                               2.10425031])))

def test_check_pair():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.sampling')
    sampler = NetworkSampling(ktn, None, None, None, None, None)
    good, reps = sampler.check_pair(4, 4)
    assert good == False
    assert reps == 0
    good, reps = sampler.check_pair(3, 6)
    assert good == False
    assert reps == 0
    good, reps = sampler.check_pair(0, 2)
    assert good == True
    assert reps == 1
    good, reps = sampler.check_pair(0, 3)
    assert good == False
    assert reps == 4

def test_write_connection_attempt():
    ktn = KineticTransitionNetwork()
    sampler = NetworkSampling(ktn, None, None, None, None, None)
    sampler.write_connection_output(10.0, 6.0, 4.0)

def test_write_failure_condition():
    ktn = KineticTransitionNetwork()
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-6, 50, 5e-1)
    sampler = NetworkSampling(ktn, None, None, single_ended, None, None)
    single_ended.failure = 'SDpaths'
    sampler.write_failure_condition()
    single_ended.failure = 'eigenvector'
    sampler.write_failure_condition()
    single_ended.failure = 'eigenvalue'
    sampler.write_failure_condition()
    single_ended.failure = 'bounds'
    sampler.write_failure_condition()
    single_ended.failure = 'steps'
    sampler.write_failure_condition()
    single_ended.failure = 'pushoff'
    sampler.write_failure_condition()
    single_ended.failure = 'invalid_ts'
    sampler.write_failure_condition()

def test_select_minima():
    step_taking = StandardPerturbation(max_displacement=0.7,
                                       proportional_distance=True)
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.sampling')
    ktn.remove_ts(3, 4)
    camelback = Camelback()
    basin_hopping = BasinHopping(ktn=ktn, potential=camelback,
                                 similarity=similarity,
                                 step_taking=step_taking)
    sampler = NetworkSampling(ktn, coords, basin_hopping, None, None, similarity)
    pairs = sampler.select_minima(coords, 'ConnectUnconnected', 2)
    assert pairs == [[0, 7], [3, 4], [0, 6], [2, 3], [2, 6], [4, 7]]

def test_select_minima2():
    step_taking = StandardPerturbation(max_displacement=0.7,
                                       proportional_distance=True)
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.sampling')
    camelback = Camelback()
    basin_hopping = BasinHopping(ktn=ktn, potential=camelback,
                                 similarity=similarity,
                                 step_taking=step_taking)
    sampler = NetworkSampling(ktn, coords, basin_hopping, None, None, similarity)
    pairs = sampler.select_minima(coords, 'ClosestEnumeration', 2)
    assert pairs == [[0, 1], [1, 2], [5, 8], [3, 7], [0, 3],
                     [2, 3], [6, 7], [4, 5], [4, 8], [3, 6], [2, 8]]

def test_prepare_connection_attempt():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    similarity = StandardSimilarity(0.1, 0.1)
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.sampling')
    sampler = NetworkSampling(ktn, None, None, None, None, similarity)
    min1, min2, reps, perm = sampler.prepare_connection_attempt(coords, [0, 1])
    assert min1 == None
    assert min2 == None
    
def test_prepare_connection_attempt2():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    similarity = StandardSimilarity(0.1, 0.1)
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.sampling')
    sampler = NetworkSampling(ktn, None, None, None, None, similarity)
    min1, min2, reps, perm = sampler.prepare_connection_attempt(coords, [0, 8])
    assert np.all(min1 == pytest.approx(np.array([5.000000000000000000e+00,
                                                  -3.308186350707335688e+00,
                                                  4.244925221708777840e+00])))
    assert np.all(min2 == pytest.approx(np.array([-3.909913273993829375e+00,
                                                  4.027576960747261126e+00,
                                                  3.373227339563876548e+00])))

def test_prepare_connection_attempt3():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    init_position = position.copy()
    coords.position[0:3] = np.array([-1.2854, 0.2499, 0.0])
    coords.position[3:6] = np.array([0.0072, -0.5687, 0.0])
    coords.position[9:12] = np.array([-1.3175, 0.8784, 0.89])
    coords.position[15:18] = np.array([0.0392, -1.1972, 0.89])
    similarity = MolecularSimilarity(0.01, 0.05, weighted=False)
    ktn = KineticTransitionNetwork()
    ktn.add_minimum(init_position, 1.0)
    ktn.add_minimum(coords.position, 2.0)
    sampler = NetworkSampling(ktn, None, None, None, None, similarity)
    min1, min2, reps, perm = sampler.prepare_connection_attempt(coords, [0, 1])
    assert np.all(perm == np.array([1, 0, 2, 5, 4, 3, 6, 7, 8]))

def test_connection_attempt():
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.sampling2')
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 5e-1)
    double_ended = NudgedElasticBand(camel, 50.0, 4.0, 50, 1e-2)
    sampler = NetworkSampling(ktn, coords, None, single_ended,
                              double_ended, similarity)
    sp_info = sampler.connection_attempt([0, 1])
    assert np.all(sp_info[0][0] == pytest.approx(np.array([-1.10920535, 0.7682681])))
    assert sp_info[0][1] == pytest.approx(0.5437186009781857)
    if sp_info[0][3] < sp_info[0][5]:
        assert np.all(sp_info[0][2] == pytest.approx(np.array([-0.08984202, 0.7126564])))
        assert sp_info[0][3] == pytest.approx(-1.0316284534898772)
        assert np.all(sp_info[0][4] == pytest.approx(np.array([-1.70360671, 0.79608357])))
        assert sp_info[0][5] == pytest.approx(-0.21546382438371803)
    else:
        assert np.all(sp_info[0][4] == pytest.approx(np.array([-0.08984202, 0.7126564])))
        assert sp_info[0][5] == pytest.approx(-1.0316284534898772)
        assert np.all(sp_info[0][2] == pytest.approx(np.array([-1.70360671, 0.79608357])))
        assert sp_info[0][3] == pytest.approx(-0.21546382438371803)

def test_connection_attempt2():
    similarity = StandardSimilarity(0.05, 5e-2, True)
    coords = StandardCoordinates(ndim=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
    ktn = KineticTransitionNetwork()
    ktn.add_minimum(np.array([0.2442886393482816554, 0.2223237906150130339]), 0.37907)
    ktn.add_minimum(np.array([0.2664846253865226222, 0.2771414190591802718]), 0.49790)
    model_data = ModelData('test_data/training_sampling.txt',
                           'test_data/response_sampling.txt')
    model_data.feature_subset([0, 1])
    model_data.remove_duplicates()
    model_data.normalise_training()
    model_data.normalise_response()
    interpolation = DatasetInterpolation(model_data, 1e-4)
    single_ended = HybridEigenvectorFollowing(potential=interpolation,
                                 ts_conv_crit=5e-4,
                                 ts_steps=150,
                                 pushoff=5e-3,
                                 steepest_descent_conv_crit=1e-4,
                                 min_uphill_step_size=1e-8,
                                 eigenvalue_conv_crit=1e-6,
                                 positive_eigenvalue_step=1e-2)
    double_ended = NudgedElasticBand(potential=interpolation,
                        force_constant=5e3,
                        image_density=50.0,
                        max_images=30,
                        neb_conv_crit=1e-2)
    sampler = NetworkSampling(ktn, coords, None, single_ended,
                              double_ended, similarity)
    sp_info = sampler.connection_attempt([0, 1])
    assert np.all(sp_info[0][0] == pytest.approx(np.array([0.25610073, 0.24901663]), 1e-4))
    assert sp_info[0][1] == pytest.approx(0.6286582534803529)
    if sp_info[0][3] < sp_info[0][5]:
        assert np.all(sp_info[0][2] == pytest.approx(np.array([0.23169999, 0.22230662]), 1e-4))
        assert sp_info[0][3] == pytest.approx(0.3565880004171049)
        assert np.all(sp_info[0][4] == pytest.approx(np.array([0.26648462, 0.27714142]), 1e-4))
        assert sp_info[0][5] == pytest.approx(0.497898398711186)
    else:
        assert np.all(sp_info[0][4] == pytest.approx(np.array([0.23169999, 0.22230662]), 1e-4))
        assert sp_info[0][5] == pytest.approx(0.3565880004171049)
        assert np.all(sp_info[0][2] == pytest.approx(np.array([0.26648462, 0.27714142]), 1e-4))
        assert sp_info[0][3] == pytest.approx(0.497898398711186)

def test_connection_attempt3():
    similarity = StandardSimilarity(0.05, 5e-2, True)
    coords = StandardCoordinates(ndim=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
    ktn = KineticTransitionNetwork()
    ktn.add_minimum(np.array([0.2316999844237584150, 0.2223066239688433754]), 0.35659)
    ktn.add_minimum(np.array([0.1744027872860626494, 0.1636957501001815307]), -0.10940)
    model_data = ModelData('test_data/training_sampling.txt',
                           'test_data/response_sampling.txt')
    model_data.feature_subset([0, 1])
    model_data.remove_duplicates()
    model_data.normalise_training()
    model_data.normalise_response()
    interpolation = DatasetInterpolation(model_data, 1e-4)
    single_ended = HybridEigenvectorFollowing(potential=interpolation,
                                 ts_conv_crit=5e-4,
                                 ts_steps=150,
                                 pushoff=5e-3,
                                 steepest_descent_conv_crit=1e-4,
                                 min_uphill_step_size=1e-8,
                                 eigenvalue_conv_crit=1e-6,
                                 positive_eigenvalue_step=1e-2)
    double_ended = NudgedElasticBand(potential=interpolation,
                        force_constant=5e3,
                        image_density=50.0,
                        max_images=30,
                        neb_conv_crit=1e-2)
    sampler = NetworkSampling(ktn, coords, None, single_ended,
                              double_ended, similarity)
    sp_info = sampler.connection_attempt([0, 1])
    assert np.all(sp_info[0][0] == pytest.approx(np.array([0.21620406, 0.2035577]), 1e-4))
    assert sp_info[0][1] == pytest.approx(0.6876465405903218)
    if sp_info[0][3] < sp_info[0][5]:
        assert np.all(sp_info[0][2] == pytest.approx(np.array([0.17440265, 0.16369579]), 1e-4))
        assert sp_info[0][3] == pytest.approx(-0.10940039307661209)
        assert np.all(sp_info[0][4] == pytest.approx(np.array([0.23169998, 0.22230662]), 1e-4))
        assert sp_info[0][5] == pytest.approx(0.3565880004189239)
    else:
        assert np.all(sp_info[0][4] == pytest.approx(np.array([0.23169999, 0.22230662]), 1e-4))
        assert sp_info[0][5] == pytest.approx(0.3565880004171049)
        assert np.all(sp_info[0][2] == pytest.approx(np.array([0.17440265, 0.16369579]), 1e-4))
        assert sp_info[0][3] == pytest.approx(-0.10940039307661209)

def test_run_connection_attempts():
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.sampling2')
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 5e-1)
    double_ended = NudgedElasticBand(camel, 50.0, 4.0, 50, 1e-2)
    sampler = NetworkSampling(ktn, coords, None, single_ended,
                              double_ended, similarity)
    sampler.run_connection_attempts([[0, 1], [0, 5], [1, 2]])
    assert np.all(ktn.G[0][1]['coords'] == pytest.approx(np.array([-1.109205336676489795, 0.7682680961466419323])))
    assert ktn.G[0][1]['energy'] == pytest.approx(0.543718600978186)
    assert np.all(ktn.G[0][5]['coords'] == pytest.approx(np.array([-1.638068007045644592, -0.228673952850872414])))
    assert ktn.G[0][5]['energy'] == pytest.approx(2.229357197530713)
    assert np.all(np.abs(ktn.G[1][2]['coords']) < 1e-6)
    assert ktn.G[1][2]['energy'] == pytest.approx(0.0000)

def test_run_connection_attempts2():
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.sampling2')
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 5e-1)
    double_ended = NudgedElasticBand(camel, 50.0, 4.0, 50, 1e-2)
    sampler = NetworkSampling(ktn, coords, None, single_ended,
                              double_ended, similarity, multiprocessing_on=True,
                              n_processes=3)
    sampler.run_connection_attempts([[0, 1], [0, 5], [1, 2]])
    assert np.all(ktn.G[0][1]['coords'] == pytest.approx(np.array([-1.109205336676489795, 0.7682680961466419323])))
    assert ktn.G[0][1]['energy'] == pytest.approx(0.543718600978186)
    assert np.all(ktn.G[0][5]['coords'] == pytest.approx(np.array([-1.638068007045644592, -0.228673952850872414])))
    assert ktn.G[0][5]['energy'] == pytest.approx(2.229357197530713)
    assert np.all(np.abs(ktn.G[1][2]['coords']) < 1e-6)
    assert ktn.G[1][2]['energy'] == pytest.approx(0.0000)

def test_get_transition_states():
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.sampling2')
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 5e-1)
    double_ended = NudgedElasticBand(camel, 50.0, 4.0, 50, 1e-2)
    sampler = NetworkSampling(ktn, coords, None, single_ended,
                              double_ended, similarity)
    sampler.get_transition_states('ClosestEnumeration', 2)
    assert np.all(ktn.G[0][1]['coords'] == pytest.approx(np.array([-1.109205336676489795, 0.7682680961466419323])))
    assert ktn.G[0][1]['energy'] == pytest.approx(0.543718600978186)
    assert np.all(ktn.G[0][5]['coords'] == pytest.approx(np.array([-1.638068007045644592, -0.228673952850872414])))
    assert ktn.G[0][5]['energy'] == pytest.approx(2.229357197530713)
    assert np.all(np.abs(ktn.G[1][2]['coords']) < 1e-6)
    assert ktn.G[1][2]['energy'] == pytest.approx(0.0000)
    assert np.all(ktn.G[1][3]['coords'] == pytest.approx(np.array([1.296070266636779600, 0.6050843818689891629])))
    assert ktn.G[1][3]['energy'] == pytest.approx(2.22947081802986)
    assert np.all(ktn.G[2][4]['coords'] == pytest.approx(np.array([1.109205312695322965, -0.7682680828827476160])))
    assert ktn.G[2][4]['energy'] == pytest.approx(0.5437186009781854)
    assert np.all(ktn.G[2][5]['coords'] == pytest.approx(np.array([-1.2960702652540481, -0.6050844002904648])))
    assert ktn.G[2][5]['energy'] == pytest.approx(2.2294708180298604)
    assert np.all(ktn.G[3][4]['coords'] == pytest.approx(np.array([1.638067985280741601, 0.2286740494415425706])))
    assert ktn.G[3][4]['energy'] == pytest.approx(2.229357197530713)

def test_get_transition_states2():
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.sampling2')
    ktn.add_minimum(np.array([-3.0, 2.0]), 1.0)
    ktn.add_minimum(np.array([-3.0, 1.0]), 1.0)
    assert ktn.n_minima == 8
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 5e-1)
    double_ended = NudgedElasticBand(camel, 50.0, 4.0, 50, 1e-2)
    sampler = NetworkSampling(ktn, coords, None, single_ended,
                              double_ended, similarity)
    sampler.get_transition_states('ClosestEnumeration', 2, remove_bounds_minima=True)
    assert ktn.n_minima == 6

def test_reconverge_minima():
    similarity = StandardSimilarity(0.05, 0.01)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.sampling3')
    camel = Camelback()
    sampler = NetworkSampling(ktn, coords, None, None, None, similarity)
    sampler.reconverge_minima(camel, 1e-6)
    assert np.all(ktn.get_minimum_coords(0) == pytest.approx(np.array([-1.70360671, 0.79608357])))
    assert np.all(ktn.get_minimum_coords(1) == pytest.approx(np.array([-0.08984201, 0.71265641])))
    assert np.all(ktn.get_minimum_coords(2) == pytest.approx(np.array([ 0.089842, -0.71265639])))
    assert np.all(ktn.get_minimum_coords(3) == pytest.approx(np.array([1.60710475, 0.56865145])))
    assert np.all(ktn.get_minimum_coords(4) == pytest.approx(np.array([ 1.70360672, -0.79608357])))
    assert np.all(ktn.get_minimum_coords(5) == pytest.approx(np.array([-1.60710475, -0.56865145])))
    assert ktn.get_minimum_energy(0) == pytest.approx(-0.21546382438371758)

def test_reconverge_landscape():
    similarity = StandardSimilarity(0.05, 0.01)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.sampling4')
    camel = Camelback()
    single_ended = HybridEigenvectorFollowing(camel, 1e-5, 50, 5e-1)
    sampler = NetworkSampling(ktn, coords, None, single_ended, None, similarity)
    sampler.reconverge_landscape(camel, 1e-6)
    assert np.all(ktn.get_minimum_coords(0) == pytest.approx(np.array([-1.70360671, 0.79608357]), abs=1e-4))
    assert np.all(ktn.get_minimum_coords(1) == pytest.approx(np.array([-0.08984201, 0.71265641]), abs=1e-4))
    assert np.all(ktn.get_minimum_coords(2) == pytest.approx(np.array([ 0.089842, -0.71265639]), abs=1e-4))
    assert np.all(ktn.get_minimum_coords(3) == pytest.approx(np.array([1.60710475, 0.56865145]), abs=1e-4))
    assert np.all(ktn.get_minimum_coords(4) == pytest.approx(np.array([ 1.70360672, -0.79608357]), abs=1e-4))
    assert np.all(ktn.get_minimum_coords(5) == pytest.approx(np.array([-1.60710475, -0.56865145]), abs=1e-4))

    assert np.all(ktn.get_ts_coords(0, 5) == pytest.approx(np.array([-1.63806757, -0.22867399]), abs=1e-4))
    assert np.all(ktn.get_ts_coords(0, 1) == pytest.approx(np.array([-1.10920534, 0.76826805]), abs=1e-4))
    assert np.all(ktn.get_ts_coords(1, 2) == pytest.approx(np.array([6.15560809e-09, -9.98103796e-08]), abs=1e-4))
    assert np.all(ktn.get_ts_coords(1, 3) == pytest.approx(np.array([1.2960702, 0.60508434]), abs=1e-4))
    assert np.all(ktn.get_ts_coords(2, 4) == pytest.approx(np.array([ 1.10920525, -0.76826793]), abs=1e-4))
    assert np.all(ktn.get_ts_coords(2, 5) == pytest.approx(np.array([-1.29606974, -0.60508394]), abs=1e-4))
    assert np.all(ktn.get_ts_coords(3, 4) == pytest.approx(np.array([1.63806798, 0.22867407]), abs=1e-4))
