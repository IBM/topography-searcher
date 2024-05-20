import pytest
import numpy as np
import os
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.data.coordinates import StandardCoordinates
from topsearch.analysis.pair_selection import connect_unconnected, \
    connect_to_set, closest_enumeration, unique_pairs, read_pairs

current_dir = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))

def test_connect_unconnected():
    coords = StandardCoordinates(ndim=3, bounds=[(-5.0, 5.0),
                                                 (-5.0, 5.0),
                                                 (-5.0, 5.0)])
    ktn = KineticTransitionNetwork()
    similarity = StandardSimilarity(0.01, 0.01)
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    ktn.remove_ts(3, 4)
    total_pairs = connect_unconnected(ktn, similarity, coords, 2)
    assert total_pairs == [[0, 7], [3, 4], [0, 6], [2, 3], [2, 6], [4, 7]]

def test_connect_to_set():
    coords = StandardCoordinates(ndim=3, bounds=[(-5.0, 5.0),
                                                 (-5.0, 5.0),
                                                 (-5.0, 5.0)])
    ktn = KineticTransitionNetwork()
    similarity = StandardSimilarity(0.01, 0.01)
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    ktn.remove_ts(3, 4)
    total_pairs = connect_to_set(ktn, similarity, coords, 3, 2)
    assert total_pairs == [[2, 3], [3, 4]]

def test_connect_to_set2():
    coords = StandardCoordinates(ndim=3, bounds=[(-5.0, 5.0),
                                                 (-5.0, 5.0),
                                                 (-5.0, 5.0)])
    ktn = KineticTransitionNetwork()
    similarity = StandardSimilarity(0.01, 0.01)
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    total_pairs = connect_to_set(ktn, similarity, coords, 3, 2)
    assert total_pairs == []

def test_closest_enumeration():
    coords = StandardCoordinates(ndim=3, bounds=[(-5.0, 5.0),
                                                 (-5.0, 5.0),
                                                 (-5.0, 5.0)])
    ktn = KineticTransitionNetwork()
    similarity = StandardSimilarity(0.01, 0.01)
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    total_pairs = closest_enumeration(ktn, similarity, coords, 2)
    assert total_pairs == [[0, 1], [1, 2], [5, 8], [3, 7], [0, 3],
                           [2, 3], [6, 7], [4, 5], [4, 8], [3, 6], [2, 8]]

def test_read_pairs():
    pairs = read_pairs(text_path=f'{current_dir}/test_data/')
    assert pairs == [[0, 1], [1, 2], [1, 3]]

def test_unique_pairs():
    pairs = [[1, 0], [0, 2], [1, 0], [0, 1], [3, 4], [0, 0]]
    pairs2 = unique_pairs(pairs)
    assert pairs2 == [[0, 1], [0, 2], [3, 4]]
