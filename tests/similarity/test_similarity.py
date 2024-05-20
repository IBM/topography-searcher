import pytest
import numpy as np
import os
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.data.coordinates import StandardCoordinates
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork

current_dir = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))

def test_closest_distance():

    coords = StandardCoordinates(ndim=2, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0)])
    similarity = StandardSimilarity(0.1, 0.1)
    coords.position = np.array([76.0, 31.0])
    coords_b = np.array([-22.0, 50.0])
    dist = similarity.closest_distance(coords, coords_b)
    assert dist == pytest.approx(99.82484660644363)
    coords_c = np.array([76.0, 31.0])
    dist = similarity.closest_distance(coords, coords_c)
    assert dist == pytest.approx(0.0)

def test_distance():
    coords = StandardCoordinates(ndim=2, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0)])
    similarity = StandardSimilarity(0.1, 0.1)
    coords.position = np.array([76.0, 31.0])
    coords_b = np.array([-22.0, 50.0])
    dist = similarity.distance(coords.position, coords_b)
    assert dist == pytest.approx(99.82484660644363)

def test_centre():
    coords = StandardCoordinates(ndim=2, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0)])
    similarity = StandardSimilarity(0.1, 0.1)
    coords.position = np.array([76.0, 31.0])
    centred = similarity.centre(coords)
    assert np.all(centred == np.array([76.0, 31.0]))

def test_optimal_alignment():
    coords = StandardCoordinates(ndim=2, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0)])
    similarity = StandardSimilarity(0.1, 0.1)
    coords.position = np.array([76.0, 31.0])
    coords_b = np.array([-22.0, 50.0])
    dist, coords1, coords2, perm = similarity.optimal_alignment(coords, coords_b)
    assert dist == pytest.approx(99.82484660644363)
    assert np.all(coords1 == np.array([76.0, 31.0]))
    assert np.all(coords2 == np.array([-22.0, 50.0]))

def test_test_same():
    coords = StandardCoordinates(ndim=2, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0)])
    similarity = StandardSimilarity(0.3, 0.1)

    coords.position = np.array([110.1, 230.0])
    energy_a = 0.0
    coords_b = np.array([110.05, 230.1])
    energy_b = 0.01
    coords_c = np.array([75.0, 221.0])
    energy_c = 0.0
    coords_d = np.array([110.1, 230.05])
    energy_d = 1.6

    a_b_same = similarity.test_same(coords, coords_b, energy_a, energy_b)
    a_c_same = similarity.test_same(coords, coords_c, energy_a, energy_c)
    a_d_same = similarity.test_same(coords, coords_d, energy_a, energy_d)

    assert a_b_same == True
    assert a_c_same == False
    assert a_d_same == False

def test_test_same_proportional():
    coords = StandardCoordinates(ndim=2, bounds=[(-500.0, 500.0),
                                                 (-5.0, 5.0)])
    similarity = StandardSimilarity(0.01, 0.1, True)

    coords.position = np.array([110.1, 2.30])
    energy_a = 0.0
    coords_b = np.array([110.05, 2.30])
    energy_b = 0.01
    coords_c = np.array([75.0, 2.1])
    energy_c = 0.0
    coords_d = np.array([110.1, 2.30])
    energy_d = 1.6
    coords_e = np.array([115.1, 2.31])
    energy_e = 0.0

    a_b_same = similarity.test_same(coords, coords_b, energy_a, energy_b)
    a_c_same = similarity.test_same(coords, coords_c, energy_a, energy_c)
    a_d_same = similarity.test_same(coords, coords_d, energy_a, energy_d)
    a_e_same = similarity.test_same(coords, coords_e, energy_a, energy_e)

    assert a_b_same == True
    assert a_c_same == False
    assert a_d_same == False
    assert a_e_same == True

def test_is_new_minimum():
    coords = StandardCoordinates(ndim=2, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0)])
    similarity = StandardSimilarity(0.1, 0.1)
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.similarity')
    coords.position = np.array([5.0, -3.3082, 4.245])
    min_energy = -2.2938
    new, match = similarity.is_new_minimum(ktn, coords, min_energy)
    assert new == False
    assert match == 0
    coords.position = np.array([5.0, -5.0, 5.0])
    new, match = similarity.is_new_minimum(ktn, coords, min_energy)    
    assert new == True
    assert match == None

def test_is_new_ts():
    coords = StandardCoordinates(ndim=2, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0)])
    similarity = StandardSimilarity(0.1, 0.1)
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.similarity')
    coords.position = np.array([-1.0277237, -4.53884, -2.0018477])
    ts_energy = -1.26930
    new, match1, match2 = similarity.is_new_ts(ktn, coords, ts_energy)
    assert new == False
    assert match1 == 3
    assert match2 == 6
    coords.position = np.array([5.0, -5.0, 5.0])
    new, match1, match2 = similarity.is_new_ts(ktn, coords, ts_energy)    
    assert new == True
    assert match1 == None

def test_test_new_minimum():
    coords = StandardCoordinates(ndim=2, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0)])
    similarity = StandardSimilarity(0.1, 0.1)
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.similarity')
    coords.position = np.array([5.0, -3.3082, 4.245])
    min_energy = -2.2938
    similarity.test_new_minimum(ktn, coords, min_energy)
    assert ktn.n_minima == 9
    coords.position = np.array([5.0, -5.0, 5.0])
    similarity.test_new_minimum(ktn, coords, min_energy)    
    assert ktn.n_minima == 10
    min_coords = ktn.get_minimum_coords(9)
    min_energy = ktn.get_minimum_energy(9)
    assert np.all(min_coords == np.array([5.0, -5.0, 5.0]))
    assert min_energy == pytest.approx(-2.2938)

def test_test_new_ts():
    coords = StandardCoordinates(ndim=2, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0)])
    similarity = StandardSimilarity(0.1, 0.1)
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.similarity')
    coords.position = np.array([-1.0277237, -4.53884, -2.0018477])
    ts_energy = -1.26930
    old_min1 = np.array([5.00, -3.308, 4.2449])
    old_min2 = np.array([-2.60664, -4.4795e-01, 3.79735])
    old_energy1 = -2.29238
    old_energy2 = -1.67184
    new_min1 = np.array([5.0, -5.0, 5.0])
    new_min2 = np.array([-5.0, 5.0, -5.0])
    new_energy1 = -1.0
    new_energy2 = -0.5
    # Repeated transition state
    similarity.test_new_ts(ktn, coords, ts_energy, old_min1, old_energy1,
                           old_min2, old_energy2)
    assert ktn.n_ts == 8
    # New transition state, known connected minima
    coords.position = np.array([5.0, 5.0, 5.0])
    similarity.test_new_ts(ktn, coords, ts_energy, old_min1, old_energy1,
                           old_min2, old_energy2)
    assert ktn.n_ts == 9
    assert ktn.n_minima == 9
    ts_coords = ktn.get_ts_coords(0, 2)
    assert np.all(ts_coords == np.array([5.0, 5.0, 5.0]))
    # New transition state, unknown connected minima
    coords.position = np.array([-5.0, -5.0, -5.0])
    similarity.test_new_ts(ktn, coords, ts_energy, new_min1, new_energy1,
                           new_min2, new_energy2)
    assert ktn.n_ts == 10
    assert ktn.n_minima == 11
    ts_coords = ktn.get_ts_coords(9, 10)
    assert np.all(ts_coords == np.array([-5.0, -5.0, -5.0]))
    min_coords1 = ktn.get_minimum_coords(9)
    min_coords2 = ktn.get_minimum_coords(10)
    assert np.all(min_coords1 == np.array([5.0, -5.0, 5.0]))
    assert np.all(min_coords2 == np.array([-5.0, 5.0, -5.0]))
