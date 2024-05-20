import pytest
import numpy as np
import os.path
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.plotting.network import plot_network, barrier_reweighting

current_dir = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))

def test_barrier_reweighting():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.network')
    ktn.G.edges[0,1]['energy'] = -3.0
    weighted_graph = barrier_reweighting(ktn)
    edges = []
    for i in ktn.G.edges:
        edges.append(i)
    weighted_edges = []
    for i in weighted_graph.edges:
        weighted_edges.append(i)
    # Make one edge lower in energy than minima
    assert edges == weighted_edges
    assert ktn.n_minima == weighted_graph.number_of_nodes()
    assert ktn.n_ts == weighted_graph.number_of_edges()

def test_plot_network():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.network')
    plot_network(ktn)
    assert os.path.exists('Network.png') == True
    os.remove('Network.png')
