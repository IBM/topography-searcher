## Generate complete landscape for the two-dimensional Schwefel function
## on both a single processor, and with 6 processors to highlight the
## acceleration possible with multiprocessing 

# IMPORTS

import numpy as np
from timeit import default_timer as timer
from topsearch.data.coordinates import StandardCoordinates
from topsearch.potentials.test_functions import Schwefel
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.global_optimisation.perturbations import StandardPerturbation
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.sampling.exploration import NetworkSampling
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand

# INITIALISATION

# Specify the coordinates for optimisation. We will optimise in a space
# of two dimensions, and select the standard bounds used for Schwefel
coords = StandardCoordinates(ndim=2, bounds=[(-500.0, 500.0),
                                             (-500.0, 500.0)])
# Specify the test function we will generate the landscape of
schwefel = Schwefel()
# Similarity object, decides if two points are the same or different
# Same if distance between points is less than distance_criterion
# and the difference in function value is less than energy_criterion
# Distance is a proportion of the range, in this case 0.05*1000.0
comparer = StandardSimilarity(distance_criterion=0.01,
                              energy_criterion=1e-2,
                              proportional_distance=True)
# Kinetic transition network to store stationary points
ktn = KineticTransitionNetwork()
# Perturbation scheme for proposing new positions in configuration space
# Standard perturbation just applies random perturbations
step_taking = StandardPerturbation(max_displacement=1.0,
                                   proportional_distance=True)
# Global optimisation class using basin-hopping
optimiser = BasinHopping(ktn=ktn, potential=schwefel, similarity=comparer,
                         step_taking=step_taking)
# Single ended transition state search object that locates transition
# states from a single starting position, using hybrid eigenvector-following
hef = HybridEigenvectorFollowing(schwefel, 1e-4, 75, 1.0)
# Double ended transition state search that locates approximate minimum energy
# pathways between two points using the nudged elastic band algorithm
neb = NudgedElasticBand(schwefel, 10.0, 50, 10, 1e-2)
# Serial sampling of configuration space using the global_optimisation
# and transition state search methods
explorer_serial = NetworkSampling(ktn=ktn, coords=coords,
                                  global_optimiser=optimiser,
                                  single_ended_search=hef,
                                  double_ended_search=neb,
                                  similarity=comparer)
# Parallel equivalent for searching configuration space
explorer_parallel = NetworkSampling(ktn=ktn, coords=coords,
                                    global_optimiser=optimiser,
                                    single_ended_search=hef,
                                    double_ended_search=neb,
                                    similarity=comparer,
                                    multiprocessing_on=True,
                                    n_processes=6)

# BEGIN CALCULATIONS

# Serial version first
serial_start_time = timer()
# Perform global optimisation of Schwefel function and locate
# other local minima
explorer_serial.get_minima(coords, 1000, 1e-5, 100.0, test_valid=True)
# Then get the transition states between them
explorer_serial.get_transition_states('ClosestEnumeration', 8,
                                      remove_bounds_minima=True)
serial_end_time = timer()

# Parallel version second
ktn.reset_network()
parallel_start_time = timer()
explorer_parallel.get_minima(coords, 1000, 1e-5, 100.0, test_valid=True)
explorer_parallel.get_transition_states('ClosestEnumeration', 8,
                                        remove_bounds_minima=True)
parallel_end_time = timer()

# Calculate the length of each landscape construction
serial_length = serial_end_time - serial_start_time
parallel_length = parallel_end_time - parallel_start_time

print("Serial runtime = ", serial_length)
print("Parallel runtime = ", parallel_length)
