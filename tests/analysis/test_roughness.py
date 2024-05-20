import pytest
import numpy as np
import os
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.analysis.roughness import get_population, roughness_metric

current_dir = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))

def test_get_population():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    population0_1 = get_population(ktn, min_node=0, ts_node=1, lengthscale=1.0)
    population1_0 = get_population(ktn, min_node=1, ts_node=0, lengthscale=1.0)
    population4_8 = get_population(ktn, min_node=4, ts_node=8, lengthscale=1.0)
    assert population0_1 == pytest.approx(0.00487914157909077)
    assert population1_0 == pytest.approx(0.002382065319350818)
    assert population4_8 == pytest.approx(0.0423493466740897)
    population = get_population(ktn, min_node=0, ts_node=1, lengthscale=2.0)
    assert population == pytest.approx(0.06985085238628638)

def test_roughness_metric_barrier():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    ktn.remove_minima(np.array([2,3,4,5,6,7,8]))
    roughness_default = roughness_metric(ktn, lengthscale=1.0)
    roughness_lengthscale = roughness_metric(ktn)
    assert roughness_default == pytest.approx(0.004087311062500195)
    assert roughness_lengthscale == pytest.approx(0.0001785639186812142)
    ktn.G.edges[0, 1]['energy'] = -3.0
    roughness = roughness_metric(ktn)
    assert roughness == 0.0

def test_roughness_metric():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    roughness = roughness_metric(ktn)
    assert roughness == pytest.approx(0.004325301709779069)

def test_roughness_metric_empty():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    ktn.reset_network()
    roughness = roughness_metric(ktn)
    assert roughness == 0.0

def test_roughness_metric_one_minimum():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f'{current_dir}/test_data/',
                     text_string='.analysis')
    ktn.reset_network()
    ktn.add_minimum(np.array([1.0, 1.0, 1.0]), 0.0)
    roughness = roughness_metric(ktn)
    assert roughness == 0.0
