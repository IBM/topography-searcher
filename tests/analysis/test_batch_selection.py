import pytest
import numpy as np
from topsearch.analysis.batch_selection import get_excluded_minima, \
        monotonic_batch_selector, lowest_batch_selector, \
        barrier_batch_selector, sufficient_barrier, evaluate_batch, \
        generate_batch, select_batch, fill_batch, \
        topographical_batch_selector, get_batch_positions
from topsearch.data.coordinates import StandardCoordinates
from topsearch.potentials.test_functions import Camelback
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork

def test_get_excluded_minima():
    coords = StandardCoordinates(ndim=3, bounds=[(-5.0, 5.0),
                                                 (-5.0, 5.0),
                                                 (-5.0, 5.0)])
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.analysis')
    excluded = get_excluded_minima(ktn=ktn, energy_cutoff=1.0,
                                   penalise_edge=False,
                                   penalise_similarity=False)
    assert excluded == []
    excluded = get_excluded_minima(ktn=ktn, energy_cutoff=0.3,
                                   penalise_edge=False,
                                   penalise_similarity=False)
    assert excluded == [2, 3, 4, 5, 6, 7, 8]
    excluded = get_excluded_minima(ktn=ktn, penalise_edge=True,
                                   coords=coords)
    assert excluded == [0, 6, 7]
    excluded = get_excluded_minima(ktn, energy_cutoff=0.3,
                                   penalise_edge=True,
                                   coords=coords)
    assert excluded == [0, 2, 3, 4, 5, 6, 7, 8]
    known_points = np.array([[4.1091, 3.2737, 4.53857]])
    excluded = get_excluded_minima(ktn=ktn, penalise_similarity=True,
                                   proximity_measure=0.1,
                                   known_points=known_points)
    assert excluded == [1]

def test_select_batch():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.analysis')
    indices, points = select_batch(ktn, 3, 'Lowest', False)
    assert indices == [1, 0, 5]
    indices, points = select_batch(ktn, 3, 'Barrier', False, 0.5, [1])
    assert indices == [0, 5, 2]
    indices, points = select_batch(ktn, 5, 'Monotonic', True)
    assert indices == [1, 5, 3, 0, 2]

def test_generate_batch():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.analysis')
    indices = generate_batch(ktn, 'Lowest', [0, 2, 4], 1.0)
    assert indices == [1, 5, 3, 6, 8, 7]
    indices = generate_batch(ktn, 'Monotonic', [1], 1.0)
    assert indices == [0, 5, 2, 3]
    indices = generate_batch(ktn, 'Barrier', [1], 1.0)
    assert indices == [0]
    indices = generate_batch(ktn, 'Topographical', [1], 0.5)
    assert indices == [0, 5, 2, 3, 7]

def test_lowest_batch_selector():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.analysis')
    excluded_minima = []
    indices = lowest_batch_selector(ktn, excluded_minima)
    assert indices == [1, 0, 5, 2, 3, 4, 6, 8, 7]
    excluded_minima = np.array([0, 2, 4])
    indices = lowest_batch_selector(ktn, excluded_minima)
    assert indices == [1, 5, 3, 6, 8, 7]

def test_fill_batch():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.analysis')
    indices = fill_batch(ktn, [5, 4, 6], [0, 8])
    assert indices == [5, 4, 6, 1, 2, 3, 7]

def test_topographical_batch_selector():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.analysis')
    indices = topographical_batch_selector(ktn, [], 1.0)
    assert indices == [1, 5, 3]
    indices = topographical_batch_selector(ktn, [1], 1.0)
    assert indices == [0, 5, 2, 3]
    indices = topographical_batch_selector(ktn, [1], 0.5)
    assert indices == [0, 5, 2, 3, 7]

def test_barrier_batch_selector():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.analysis')
    indices = barrier_batch_selector(ktn, [], 1.0)
    assert indices == [1, 0]
    indices = barrier_batch_selector(ktn, [], 0.5)
    assert indices == [1, 0, 5, 2, 3, 7]
    indices = barrier_batch_selector(ktn, [1], 1.0)
    assert indices == [0]
    indices = barrier_batch_selector(ktn, [1], 0.5, [0])
    assert indices == [5, 2, 3, 7]

def test_sufficient_barrier():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.analysis')
    allowed = sufficient_barrier(ktn, 0, 1, -0.46971, 1.37082, 0.5)
    assert allowed == True
    allowed = sufficient_barrier(ktn, 0, 1, -0.46971, 1.37082, 1.04)
    assert allowed == False
    ktn.remove_ts(0, 1)
    allowed = sufficient_barrier(ktn, 0, 1, -0.46971, 1.37082, 1.04)
    assert allowed == False

def test_monotonic_batch_selector():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.analysis')
    indices = monotonic_batch_selector(ktn, excluded_minima=[])
    assert indices == [1, 5, 3]
    indices = monotonic_batch_selector(ktn, excluded_minima=[1])
    assert indices == [0, 5, 2, 3]

def test_evaluate_batch():
    camelback = Camelback()
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.analysis')
    indices = np.array([1, 3])
    points = np.array([[0.0898, -0.7126], [-0.0898, 0.7126]])
    response = evaluate_batch(camelback, indices, points)
    assert np.all(response == pytest.approx([-1.031628422, -1.031628422]))

def test_get_batch_positions():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path='test_data/',
                     text_string='.analysis')
    batch_points = get_batch_positions(ktn, [1, 4])
    assert np.all(batch_points == np.array([[4.109132805750269846e+00,
                                             3.273730547854070583e+00,
                                             4.538571449978961780e+00],
                                            [-1.363035511400476407e+00,
                                             4.412693961918607854e+00,
                                             -1.288026657948501130e+00]]))
