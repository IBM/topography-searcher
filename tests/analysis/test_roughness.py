import pytest
from assertpy import assert_that
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import os

from topsearch.analysis.graph_properties import get_connections
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.analysis.roughness import get_population, roughness_metric, roughness_contributors

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
    ktn.G.edges[0, 1, 0]['energy'] = -3.0
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


def test_roughness_contributors_empty():
    ktn = KineticTransitionNetwork()
    contributors = roughness_contributors(ktn)
    assert_that(contributors).is_empty()


def test_roughness_contributors_two_minima_coords():
    ktn = KineticTransitionNetwork()
    ktn.add_minimum(np.array([1.0, 1.0]), 0.2)
    ktn.add_minimum(np.array([2.0, 2.0]), 0.3)
    ktn.add_ts(np.array([1.5,1.5]),0.5, 0, 1)

    contributors = roughness_contributors(ktn, 0.8)
    
    assert_that(contributors).is_length(2)

    assert_array_equal(contributors[0].minimum,np.array([1.0, 1.0]))
    assert_array_equal(contributors[1].minimum,np.array([2.0, 2.0]))

def test_roughness_contributors_two_minima_ts():
    ktn = KineticTransitionNetwork()
    ktn.add_minimum(np.array([1.0, 1.0]), 0.2)
    ktn.add_minimum(np.array([2.0, 2.0]), 0.3)
    ktn.add_ts(np.array([1.5,1.5]),0.5, 0, 1)

    contributors = roughness_contributors(ktn, 0.8)
    
    assert_that(contributors[0].ts).is_length(2)
    assert_that(contributors[0].frustration).is_close_to(0.203,0.001)


def test_roughness_contributors_two_minima_frustration():
    ktn = KineticTransitionNetwork()
    ktn.add_minimum(np.array([1.0, 1.0]), 0.2)
    ktn.add_minimum(np.array([2.0, 2.0]), 0.3)
    ktn.add_ts(np.array([1.5,1.5]),0.5, 0, 1)

    contributors = roughness_contributors(ktn, 0.8)
    
    assert_that(contributors[0].frustration).is_close_to(0.203,0.001)

def test_sorted_contributors():
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path=f"{current_dir}/test_data/",
                     text_string='.analysis')
    
    ranked_contributors = roughness_contributors(ktn)

    total_connections = sum([len(get_connections(ktn, minimum)) for minimum in range(ktn.n_minima)])

    assert_that(ranked_contributors).is_length(total_connections)
    for c in ranked_contributors:
        print(f"{c.minimum} : {c.frustration}")
    assert_allclose(ranked_contributors[0].ts, [-1.027724, -4.538841, -2.001848], rtol=1e-5)
    assert_allclose(ranked_contributors[0].minimum, [-1.396293, -5.      , -2.06198 ], rtol=1e-5)

def test_records_features():
    ktn = KineticTransitionNetwork()
    ktn.add_minimum(np.array([1.0, 1.0]), 0.2)
    ktn.add_minimum(np.array([2.0, 2.0]), 0.3)
    ktn.add_ts(np.array([1.5,1.5]),0.5, 0, 1)

    contributors = roughness_contributors(ktn, 0.8, [0,1])

    assert_that(contributors[0].features).is_equal_to([0,1])