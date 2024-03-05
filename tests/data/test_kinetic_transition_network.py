import pytest
import numpy as np
from numpy.testing import assert_array_equal
import networkx as nx
import os
import os.path
from topsearch.potentials.test_functions import Schwefel
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.data.coordinates import StandardCoordinates
from topsearch.similarity.similarity import StandardSimilarity

current_dir = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))

def test_read_network():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn')
    edges = []
    for i in ktn.G.edges():
        edges.append(i)
    assert ktn.n_minima == 9
    assert ktn.n_ts == 8
    assert edges == [(0, 1), (1, 2), (2, 8), (3, 6),
                     (3, 7), (3, 4), (4, 8), (4, 5)]
    
    # check result for a kinetic transition network where nodes 0, 1 have two transition states
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn_multipleTS')
    edges = []
    for i in ktn.G.edges:
        edges.append(i)
    assert ktn.n_minima == 9
    assert ktn.n_ts == 9
    assert edges == [(0, 1, 0), (0, 1, 1), (1, 2, 0), (2, 8, 0), (3, 6, 0),
                     (3, 7, 0), (3, 4, 0), (4, 8, 0), (4, 5, 0)]
    
    # check that one can read a kinetic transition network with 1 minimum only
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn_single_min')
    edges = []
    for i in ktn.G.edges:
        edges.append(i)
    assert ktn.n_minima == 1
    assert ktn.n_ts == 0
    assert edges == []

def test_read_network_from_default_path():
    ktn = KineticTransitionNetwork(dump_path=f"{current_dir}/test_data/", dump_suffix=".ktn")
    ktn.read_network()
    edges = []
    for i in ktn.G.edges():
        edges.append(i)
    assert ktn.n_minima == 9
    assert ktn.n_ts == 8
    assert edges == [(0, 1), (1, 2), (2, 8), (3, 6),
                     (3, 7), (3, 4), (4, 8), (4, 5)]
    
def test_read_network_with_single_ts(tmp_path):
    dump_path = str(tmp_path) + "/"
    ktn = KineticTransitionNetwork(dump_path, ".single_ts")
    ktn.add_minimum(np.array([0,0]), 0)
    ktn.add_minimum(np.array([1,1]), 0)
    ktn.add_ts(np.array([0.5,0.5]),0,0,1)
    ktn.dump_network()

    ktn_new = KineticTransitionNetwork(dump_path, ".single_ts")
    ktn_new.read_network()

    assert_array_equal(ktn.get_ts_coords(0,1), np.array([0.5,0.5]))

def test_read_network_with_multiple_ts_per_node():
     # check result for a kinetic transition network where nodes 0, 1 have two transition states
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn_multipleTS')
    edges = []
    for i in ktn.G.edges:
        edges.append(i)
    assert ktn.n_minima == 9
    assert ktn.n_ts == 9
    assert edges == [(0, 1, 0), (0, 1, 1), (1, 2, 0), (2, 8, 0), (3, 6, 0),
                     (3, 7, 0), (3, 4, 0), (4, 8, 0), (4, 5, 0)]

def test_read_network_with_single_minimum(tmp_path):
    dump_path = str(tmp_path) + "/"
    ktn = KineticTransitionNetwork(dump_path, ".single_min")
    ktn.add_minimum(np.array([1,1,1]), 1.0)
    ktn.dump_network()

    ktn_new = KineticTransitionNetwork(dump_path, ".single_min")
    ktn_new.read_network()

    assert_array_equal(ktn_new.get_minimum_coords(0), np.array([1,1,1]))
    assert ktn_new.get_minimum_energy(0) == 1.0

def test_read_network_with_single_attempted_coord(tmp_path):
    dump_path = str(tmp_path) + "/"
    ktn = KineticTransitionNetwork(dump_path, ".single_attempted")
    ktn.add_minimum(np.array([1,1,1]), 1.0)
    ktn.add_attempted_position(np.array([1,1,1]))
    ktn.dump_network()

    ktn_new = KineticTransitionNetwork(dump_path, ".single_attempted")
    ktn_new.read_network()

    assert_array_equal(ktn_new.get_attempted_positions(), np.array([[1,1,1]]))

def test_dump_network_to_specified_path(tmp_path):
    dest_path = str(tmp_path) + "/"
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn')
    ktn.dump_network(text_string='.new', text_path=dest_path)
    ktn_new = KineticTransitionNetwork()
    ktn_new.read_network(text_string='.new', text_path=dest_path)
    edges = []
    for i in ktn.G.edges:
        edges.append(i)
    edges_new = []
    for i in ktn_new.G.edges:
        edges_new.append(i)
    assert ktn.n_minima == ktn_new.n_minima
    assert ktn.n_ts == ktn_new.n_ts
    assert edges == edges_new

def test_dump_network_to_default_path(tmp_path):
    dest_path = str(tmp_path) + "/"
    ktn = KineticTransitionNetwork(dump_path=dest_path, dump_suffix=".new")
    ktn.read_network(text_path=f"{current_dir}/test_data/",
                     text_string='.ktn')
    ktn.dump_network()
    ktn_new = KineticTransitionNetwork(dump_path=dest_path, dump_suffix=".new")
    ktn_new.read_network(text_string='.new', text_path=dest_path)
    edges = []
    for i in ktn.G.edges:
        edges.append(i)
    edges_new = []
    for i in ktn_new.G.edges:
        edges_new.append(i)
    assert ktn.n_minima == ktn_new.n_minima
    assert ktn.n_ts == ktn_new.n_ts
    assert edges == edges_new

def test_dump_network_appends_slash_if_needed(tmp_path):
    dest_path = str(tmp_path)
    ktn = KineticTransitionNetwork(dump_path=dest_path, dump_suffix=".new")
    ktn.read_network(text_path=f"{current_dir}/test_data/",
                     text_string='.ktn')
    ktn.dump_network()
    assert (tmp_path / 'min.data.new').exists()
    

def test_read_network_with_small_values(tmp_path):
    dump_path = str(tmp_path) + "/"
    ktn = KineticTransitionNetwork(dump_path, ".small_min")
    ktn.add_minimum(np.array([0,0]), -1.123456e-10)
    ktn.add_minimum(np.array([1,1]), 1.789101e-8)
    ktn.add_ts(np.array([0.5,0.5]),-1.3333333e-11,0,1)
    ktn.dump_network()

    ktn_new = KineticTransitionNetwork(dump_path, ".small_min")
    ktn_new.read_network()

    assert ktn.get_minimum_energy(0) == ktn_new.get_minimum_energy(0)
    assert ktn.get_minimum_energy(1) == ktn_new.get_minimum_energy(1)
    assert ktn.get_ts_energy(0,1) == ktn_new.get_ts_energy(0,1)


def test_reset_network():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn')
    assert ktn.n_minima == 9
    assert ktn.n_ts == 8
    ktn.reset_network()
    assert ktn.n_minima == 0
    assert ktn.n_ts == 0

def test_get_minimum_coords():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
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
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn')
    energy2 = ktn.get_minimum_energy(2)
    energy3 = ktn.get_minimum_energy(3)
    assert energy2 == pytest.approx(-1.67184)
    assert energy3 == pytest.approx(-1.44769)

def test_get_ts_coords():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
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
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn')
    energy0_1 = ktn.get_ts_energy(0, 1)
    energy4_8 = ktn.get_ts_energy(4, 8)
    assert energy0_1 == pytest.approx(-1.26162)
    assert energy4_8 == pytest.approx(-0.79084)
    # check result for a kinetic transition network where nodes 0, 1 have two transition states
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn_multipleTS')
    energy0_1_0 = ktn.get_ts_energy(0, 1)
    energy0_1_1 = ktn.get_ts_energy(0, 1, 1)
    energy4_8 = ktn.get_ts_energy(4, 8)
    assert energy0_1_0 == pytest.approx(-1.26162)
    assert energy0_1_1 == pytest.approx(-1.06162)
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
    ktn.add_ts(ts_coords, ts_energy, 0, 1)

    # Retrieve transition state information
    network_coords = ktn.get_ts_coords(0,1)
    network_energy = ktn.get_ts_energy(0,1)

    # Check that input is same as output from network
    assert np.array_equal(network_coords, ts_coords)
    assert network_energy == pytest.approx(ts_energy)
    assert ktn.n_ts == 1

    # Add a new transition state on top of the old one
    ts_coords_2 = np.array([-126.830, 126.830])
    ts_energy_2 = 840.9658
    
    # Directly add transition state between minima which are already connected!
    ktn.add_ts(ts_coords_2, ts_energy_2, 0, 1)

    # Retrieve transition state information
    network_coords_2 = ktn.get_ts_coords(0,1,1)
    network_energy_2 = ktn.get_ts_energy(0,1,1)

    # Check that input is same as output from network
    assert np.array_equal(network_coords, ts_coords)
    assert np.array_equal(network_coords_2, ts_coords_2)
    assert network_energy == pytest.approx(ts_energy)
    assert network_energy_2 == pytest.approx(ts_energy_2)
    assert ktn.n_ts == 2

def test_remove_minimum():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
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
    ktn.read_network(text_path=f'{current_dir}/test_data/',
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
    for i in ktn.G.edges():
        edges.append(i)
    assert edges == [(0, 1), (2, 4), (2, 5), (2, 3), (3, 6)]
    assert ktn.n_ts == 5

def test_remove_ts():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn')
    ktn.remove_ts(3, 7)
    edges = []
    for i in ktn.G.edges():
        edges.append(i)
    assert ktn.n_ts == 7
    assert edges == [(0, 1), (1, 2), (2, 8), (3, 6),
                     (3, 4), (4, 8), (4, 5)]
    
    # Can we remove the first transition state between two nodes, while keeping the second one?
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn_multipleTS')
    
    ktn.remove_ts(0, 1, 0)
    edges = []
    for i in ktn.G.edges:
        edges.append(i)
        
    assert ktn.n_ts == 8
    assert edges == [(0, 1, 1), (1, 2, 0), (2, 8, 0), (3, 6, 0),
                     (3, 7, 0), (3, 4, 0), (4, 8, 0), (4, 5, 0)]

def test_remove_all_ts_between_minima():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn_multipleTS')
    
    ktn.remove_all_ts(0, 1)
    edges = []
    for i in ktn.G.edges():
        edges.append(i)
        
    assert ktn.n_ts == 7
    assert edges == [(1, 2), (2, 8), (3, 6),
                     (3, 7), (3, 4), (4, 8), (4, 5)]

def test_remove_tss():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn')
    ktn.remove_tss([(3, 7), (1, 2)])
    edges = []
    for i in ktn.G.edges():
        edges.append(i)
    assert ktn.n_ts == 6
    assert edges == [(0, 1), (2, 8), (3, 6),
                     (3, 4), (4, 8), (4, 5)]

def test_remove_all_tss():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn_multipleTS')
    ktn.remove_all_tss([(0, 1), (3, 7), (1, 2)])
    edges = []
    for i in ktn.G.edges():
        edges.append(i)
    assert ktn.n_ts == 5
    assert edges == [(2, 8), (3, 6),
                     (3, 4), (4, 8), (4, 5)]
    
def test_dump_minima_csv():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn')
    ktn.dump_minima_csv()
    assert os.path.exists('mol_data.csv') == True
    os.remove('mol_data.csv')

def test_combine_networks():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn_combined')
    other_ktn = KineticTransitionNetwork()
    other_ktn.read_network(text_path=f'{current_dir}/test_data/',
                           text_string='.ktn')
    similarity = StandardSimilarity(0.05, 0.1, proportional_distance=True)
    coords = StandardCoordinates(ndim=3, bounds=[(-5.0, 5.0), (-5.0, 5.0),
                                                 (-5.0, 5.0)])
    ktn.add_network(other_ktn, similarity, coords)
    assert len(ktn.G.nodes()) == 9
    assert len(ktn.G.edges()) == 8

    # Can we combine networks that have more than 1 transition state
    # per pair of nodes? 
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn_combined')
    
    other_ktn = KineticTransitionNetwork()
    other_ktn.read_network(text_path=f'{current_dir}/test_data/',
                           text_string='.ktn_multipleTS')
    similarity = StandardSimilarity(0.05, 0.1, proportional_distance=True)
    coords = StandardCoordinates(ndim=3, bounds=[(-5.0, 5.0), (-5.0, 5.0),
                                                 (-5.0, 5.0)])
    ktn.add_network(other_ktn, similarity, coords)
    
    assert len(ktn.G.nodes()) == 9
    assert len(ktn.G.edges()) == 9

    # Can we combine networks that have more than 1 transition state
    # per pair of nodes? 
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.ktn_combined')
    
    other_ktn = KineticTransitionNetwork()
    other_ktn.read_network(text_path=f'{current_dir}/test_data/',
                           text_string='.ktn_multipleTS')
    similarity = StandardSimilarity(0.05, 0.1, proportional_distance=True)
    coords = StandardCoordinates(ndim=3, bounds=[(-5.0, 5.0), (-5.0, 5.0),
                                                 (-5.0, 5.0)])
    ktn.add_network(other_ktn, similarity, coords)
    
    assert len(ktn.G.nodes()) == 9
    assert len(ktn.G.edges()) == 9

def test_adds_attempted_initial_position_to_list():
    ktn = KineticTransitionNetwork()
    ktn.add_attempted_position(np.array([0.1, 0.2, 0.3]))
    ktn.add_attempted_position(np.array([0.4, 0.5, 0.6]))

    assert_array_equal(ktn.get_attempted_positions(), np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))