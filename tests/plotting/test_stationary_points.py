import pytest
import numpy as np
import os.path
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.potentials.test_functions import Camelback
from topsearch.plotting.stationary_points import plot_stationary_points, \
        self_connected
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

current_dir = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))

def test_plot_stationary_points():
    camelback = Camelback()
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.sp')
    mpl.rcParams.update(mpl.rcParamsDefault)
    plot_stationary_points(camelback, ktn, bounds=[(-3.0, 3.0), (-2.0, 2.0)],
                           label='1')
    assert os.path.exists('StationaryPoints1.png') == True
    os.remove('StationaryPoints1.png')
    # With self connections
    ktn.add_ts(np.array([0.1, 0.1]), -1.0, 0, 0)
    plot_stationary_points(camelback, ktn, bounds=[(-3.0, 3.0), (-2.0, 2.0)],
                           label='2')
    assert os.path.exists('StationaryPoints2.png') == True
    os.remove('StationaryPoints2.png')
    # With labels
    plot_stationary_points(camelback, ktn, bounds=[(-3.0, 3.0), (-2.0, 2.0)],
                           label='3', label_min=True)
    assert os.path.exists('StationaryPoints3.png') == True
    os.remove('StationaryPoints3.png')  

def test_self_connected():
    camelback = Camelback()
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.sp')
    assert self_connected(ktn) == 0
    ktn.add_ts(np.array([0.1, 0.1]), -1.0, 0, 0)
    assert self_connected(ktn) == 1
    ktn.add_ts(np.array([0.2, 0.2]), -1.0, 2, 2)
    assert self_connected(ktn) == 2
