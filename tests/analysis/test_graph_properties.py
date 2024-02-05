import pytest
import numpy as np
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.analysis.graph_properties import (unconnected_component,
                                                 are_nodes_connected,
                                                 all_minima_connected,
                                                 get_connections,
                                                 disconnected_height,
                                                 remove_edges_threshold)

def test_get_connections():
    ktn = KineticTransitionNetwork()
    ktn.read_network()
    transition_states1 = get_connections(ktn, 1)
    transition_states4 = get_connections(ktn, 4)
    transition_states7 = get_connections(ktn, 7)
    assert transition_states1.sort() == [0, 2].sort()
    assert transition_states4.sort() == [3, 5, 8].sort()
    assert transition_states7.sort() == [3].sort()

def test_unconnected_component():
    ktn = KineticTransitionNetwork()
    ktn.read_network()
    unconnected = unconnected_component(ktn)
    assert unconnected == set()
    ktn.remove_ts(0, 1)
    unconnected = unconnected_component(ktn)
    assert unconnected == set({0})
    ktn.remove_ts(4, 8)
    unconnected = unconnected_component(ktn)
    assert unconnected == set({0, 3, 4, 5, 6, 7})
    ktn.remove_ts(1, 2)
    unconnected = unconnected_component(ktn)
    assert unconnected == set({0, 2, 3, 4, 5, 6, 7, 8})

def test_check_if_nodes_connected():
    ktn = KineticTransitionNetwork()
    ktn.read_network()
    connected = are_nodes_connected(ktn, 0, 1)
    assert connected == True
    ktn.remove_ts(0, 1)
    connected = are_nodes_connected(ktn, 0, 1)
    assert connected == False

def test_all_minima_connected():
    ktn = KineticTransitionNetwork()
    ktn.read_network()
    all_connected = all_minima_connected(ktn)
    assert all_connected == True
    ktn.remove_ts(0, 1)
    all_connected = all_minima_connected(ktn)
    assert all_connected == False

def test_remove_edges_threshold():
    ktn = KineticTransitionNetwork()
    ktn.read_network()
    H = ktn.G.copy()
    cut_graph = remove_edges_threshold(H, -1.0)
    edges = []
    for i in cut_graph.edges:
        edges.append(i)
    assert edges == [(0, 1), (2, 8), (3, 6), (4, 5)]

def test_disconnected_height():
    ktn = KineticTransitionNetwork()
    ktn.read_network()
    height = disconnected_height(ktn, 0, 1, -0.46971, 2.11236)
    assert abs(height+1.26162) < 1e-2
    ktn.remove_ts(0, 1)
    height = disconnected_height(ktn, 0, 1, -0.46971, 2.11236)
    assert height == 1e10
    ktn.add_ts(np.array([1.0, 1.0, 1.0]), 0.0, 3, 2)
    height = disconnected_height(ktn, 1, 3, -0.46971, 2.11236)
    assert abs(height+0.68248) < 1e-2
