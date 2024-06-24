import pytest
import numpy as np
import os.path
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.plotting.disconnectivity import plot_disconnectivity_graph, \
        cut_line_collection, get_line_collection, get_connectivity_graph
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

current_dir = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))

def test_plot_disconnectivity_graph():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.sp') 
    mpl.rcParams.update(mpl.rcParamsDefault)
    plot_disconnectivity_graph(ktn, 50)
    assert os.path.exists('DisconnectivityGraph.png') == True
    os.remove('DisconnectivityGraph.png')

def test_get_connectivity_graph():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.sp')
    c_graph = get_connectivity_graph(ktn, 2.5, -1.5, 25)
    assert len(c_graph.nodes()) == 110
    assert len(c_graph.edges()) == 109

def test_cut_line_collection():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.sp')
    c_graph = get_connectivity_graph(ktn, 1.5, -1.2, 10)
    lines = get_line_collection(c_graph, 1.5, -1.2, 10)
    lines = cut_line_collection(ktn, c_graph, lines, 1.5, -1.2, 10)
    assert len(lines) == 15

def test_plot_disconnectivity_graph2():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.disco')
    plot_disconnectivity_graph(ktn, 100, '2')
    assert os.path.exists('DisconnectivityGraph2.png') == True
    os.remove('DisconnectivityGraph2.png')

def test_cut_line_collection2():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.disco')
    c_graph = get_connectivity_graph(ktn, 1191.293301, -113.456472, 100)
    lines = get_line_collection(c_graph, 1191.293301, -113.456472, 100)
    lines = cut_line_collection(ktn, c_graph, lines, 1191.293301, -113.456472, 100)
    assert len(lines) == 204
