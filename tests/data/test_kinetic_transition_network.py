import pytest
import numpy as np
import networkx as nx
import os.path
from topsearch.potentials.test_functions import Schwefel
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.data.coordinates import StandardCoordinates
from topsearch.similarity.similarity import StandardSimilarity

def test_read_network():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.ktn')
    edges = []
    for i in ktn.G.edges:
        edges.append(i)
    assert ktn.n_minima == 9
    assert ktn.n_ts == 8
    assert edges == [(0, 1), (1, 2), (2, 8), (3, 6),
                     (3, 7), (3, 4), (4, 8), (4, 5)]

def test_dump_network():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.ktn')
    ktn.dump_network(text_string='.new')
    ktn_new = KineticTransitionNetwork()
    ktn_new.read_network(text_string='.new')
    edges = []
    for i in ktn.G.edges:
        edges.append(i)
    edges_new = []
    for i in ktn_new.G.edges:
        edges_new.append(i)
    assert ktn.n_minima == ktn_new.n_minima
    assert ktn.n_ts == ktn_new.n_ts
    assert edges == edges_new
    os.remove('min.coords.new')
    os.remove('min.data.new')
    os.remove('ts.data.new')
    os.remove('ts.coords.new')
    os.remove('pairlist.new')

def test_reset_network():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.ktn')
    assert ktn.n_minima == 9
    assert ktn.n_ts == 8
    ktn.reset_network()
    assert ktn.n_minima == 0
    assert ktn.n_ts == 0

def test_get_minimum_coords():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.ktn')
    coords1 = ktn.get_minimum_coords(1)
    coords4 = ktn.get_minimum_coords(4)
    assert np.all(coords1 == np.array([4.109132805750269846e+00,
                                       3.273730547854070583e+00,
                                       4.538571449978961780e+00]))
    assert np.all(coords4 == np.array([-1.363035511400476407e+00,
                                       4.412693961918607854e+00,
                                       -1.288026657948501130e+00]))

def test_get_minimum_energy():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.ktn')
    energy2 = ktn.get_minimum_energy(2)
    energy3 = ktn.get_minimum_energy(3)
    assert energy2 == pytest.approx(-1.67184)
    assert energy3 == pytest.approx(-1.44769)

def test_get_ts_coords():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.ktn')
    coords0_1 = ktn.get_ts_coords(0, 1)
    coords4_8 = ktn.get_ts_coords(4, 8)
    assert np.all(coords0_1 == np.array([5.099999999999999645e+00,
                                         -5.744108274046482582e-02,
                                         4.506128521495991635e+00]))
    assert np.all(coords4_8 == np.array([-2.223257713299624516e+00,
                                         4.620540748546810406e+00,
                                         1.065783454677023068e+00]))

def test_get_ts_energy():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.ktn')
    energy0_1 = ktn.get_ts_energy(0, 1)
    energy4_8 = ktn.get_ts_energy(4, 8)
    assert energy0_1 == pytest.approx(-1.26162)
    assert energy4_8 == pytest.approx(-0.79084)

def test_add_minimum():
    coords = StandardCoordinates(ndim=2, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0)])
    schwefel = Schwefel()
    ktn = KineticTransitionNetwork()

    # Add the global minimum to the network
    coords.position = np.array([420.9687, 420.9687])
    min_energy = schwefel.function(coords.position)
    ktn.add_minimum(coords.position, min_energy)

    # Get back the data from the network for that minimum
    network_coords = ktn.get_minimum_coords(0)
    network_energy = ktn.get_minimum_energy(0)

    # Check that input is the same as that accessed from network
    assert np.array_equal(network_coords, coords.position)
    assert network_energy == pytest.approx(min_energy)
    assert ktn.n_minima == 1

def test_add_ts():
    ktn = KineticTransitionNetwork()

    # Transition state data
    ts_coords = np.array([-124.830, 124.830])
    ts_energy = 837.9658

    # Add minima that are connected to transition state
    ktn.add_minimum(np.array([-124.830, 203.81]), 513.24641)
    ktn.add_minimum(np.array([-124.830, 65.55]), 651.45464)

    # Directly add transition state between minima
    ktn.add_ts(ts_coords, ts_energy, 1, 2)

    # Retrieve transition state information
    network_coords = ktn.get_ts_coords(1,2)
    network_energy = ktn.get_ts_energy(1,2)

    # Check that input is same as output from network
    assert np.array_equal(network_coords, ts_coords)
    assert network_energy == pytest.approx(ts_energy)
    assert ktn.n_ts == 1

def test_remove_minimum():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.ktn')
    ktn.remove_minimum(4)
    assert ktn.n_minima == 8
    assert ktn.get_minimum_energy(4) == pytest.approx(-1.71037)
    assert np.all(ktn.get_minimum_coords(4) == \
                  np.array([-3.547887856175243826e+00,
                            4.813255670440161893e+00,
                            -4.752587599226376192e+00]))
    ktn.remove_minimum(5)
    assert ktn.n_minima == 7
    assert ktn.get_minimum_energy(6) == pytest.approx(-1.27458)
    assert np.all(ktn.get_minimum_coords(6) == \
                  np.array([-3.909913273993829375e+00,
                            4.027576960747261126e+00,
                            3.373227339563876548e+00]))
    assert ktn.get_minimum_energy(4) == pytest.approx(-1.71037)
    assert np.all(ktn.get_minimum_coords(4) == \
                  np.array([-3.547887856175243826e+00,
                            4.813255670440161893e+00,
                            -4.752587599226376192e+00]))

def test_remove_minima():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.ktn')
    ktn.remove_minima(np.array([5, 2]))
    assert ktn.n_minima == 7
    assert ktn.get_minimum_energy(2) == pytest.approx(-1.44769)
    assert np.all(ktn.get_minimum_coords(2) == \
                  np.array([2.222731533857001951e-01,
                            -2.422236084835572179e+00,
                            -1.847285290881183872e+00]))
    assert ktn.get_minimum_energy(4) == pytest.approx(-1.28440)
    assert np.all(ktn.get_minimum_coords(4) == \
                  np.array([-1.396292692621145637e+00,
                            -5.000000000000000000e+00,
                            -2.061979532075426391e+00]))
    edges = []
    for i in ktn.G.edges:
        edges.append(i)
    assert edges == [(0, 1), (2, 4), (2, 5), (2, 3), (3, 6)]
    assert ktn.n_ts == 5

def test_remove_ts():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.ktn')
    ktn.remove_ts(3, 7)
    edges = []
    for i in ktn.G.edges:
        edges.append(i)
    assert ktn.n_ts == 7
    assert edges == [(0, 1), (1, 2), (2, 8), (3, 6),
                     (3, 4), (4, 8), (4, 5)]

def test_remove_tss():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.ktn')
    ktn.remove_tss([(3, 7), (1, 2)])
    edges = []
    for i in ktn.G.edges:
        edges.append(i)
    assert ktn.n_ts == 6
    assert edges == [(0, 1), (2, 8), (3, 6),
                     (3, 4), (4, 8), (4, 5)]

def test_dump_minima_csv():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.ktn')
    ktn.dump_minima_csv()
    assert os.path.exists('mol_data.csv') == True
    os.remove('mol_data.csv')

def test_combine_networks():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.ktn_combined')
    other_ktn = KineticTransitionNetwork()
    other_ktn.read_network(text_path='test_data/',
                           text_string='.ktn')
    similarity = StandardSimilarity(0.05, 0.1, proportional_distance=True)
    coords = StandardCoordinates(ndim=3, bounds=[(-5.0, 5.0), (-5.0, 5.0),
                                                 (-5.0, 5.0)])
    ktn.add_network(other_ktn, similarity, coords)
    assert len(ktn.G.nodes()) == 9
    assert len(ktn.G.edges()) == 8
