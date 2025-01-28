from assertpy import assert_that
import numpy as np

from topsearch.analysis.roughness import RoughnessContribution
from topsearch.potentials.local_roughness import LocalRoughness

def test_single_roughness_contribution_value_at_mean_is_minus_one():

    contrib = RoughnessContribution(np.array([0,0]), np.array([1,1]), 1, [])

    potential = LocalRoughness([contrib], distance_to_ts=0.5, 
                                 parallel_scaling=0.5, orthogonal_scaling=0.5, 
                                 prefactor_scaling=1)
    
    assert_that(potential.function(np.array([0.5, 0.5]))).is_equal_to(-1)

def test_single_roughness_contribution_distance_to_ts_moves_mean():

    contrib = RoughnessContribution(np.array([0,0]), np.array([1,1]), 1, [])

    potential = LocalRoughness([contrib], distance_to_ts=0.75, 
                                 parallel_scaling=0.5, orthogonal_scaling=0.5, 
                                 prefactor_scaling=1)
    
    assert_that(potential.function(np.array([0.75, 0.75]))).is_equal_to(-1)



