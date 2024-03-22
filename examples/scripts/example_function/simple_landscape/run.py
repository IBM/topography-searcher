## Generate complete landscape for the two-dimensional Schwefel function
## locating both all the local minima and the transition states between
## them.

# IMPORTS

import numpy as np
from topsearch.data.coordinates import StandardCoordinates
from topsearch.potentials.test_functions import Schwefel
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.global_optimisation.perturbations import StandardPerturbation
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.sampling.exploration import NetworkSampling
from topsearch.plotting.disconnectivity import plot_disconnectivity_graph
from topsearch.plotting.stationary_points import plot_stationary_points
from topsearch.plotting.network import plot_network
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand

# INITIALISATION

# Specify the coordinates for optimisation. We will optimise in a space
# of two dimensions, and select the standard bounds used for Schwefel
coords = StandardCoordinates(ndim=2, bounds=[(-500.0, 500.0),
                                             (-500.0, 500.0)])
# Specify the Schwefel function that we will compute the landscape for
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
# Global optimisation class that uses basin-hopping to locate
# local minima and the global minimum
optimiser = BasinHopping(ktn=ktn,
                         potential=schwefel,
                         similarity=comparer,
                         step_taking=step_taking)
# Single ended transition state search object that locates transition
# states from a single starting position, using hybrid eigenvector-following
hef = HybridEigenvectorFollowing(potential=schwefel,
                                 ts_conv_crit=1e-4,
                                 ts_steps=75,
                                 pushoff=1.0)
# Double ended transition state search that locates approximate minimum energy
# pathways between two points using the nudged elastic band algorithm
neb = NudgedElasticBand(potential=schwefel,
                        force_constant=10.0,
                        image_density=10.0,
                        max_images=50,
                        neb_conv_crit=1e-2)
# Class for sampling configuration space by driving the global_optimisation
# and transition state search methods
explorer = NetworkSampling(ktn=ktn,
                           coords=coords,
                           global_optimiser=optimiser,
                           single_ended_search=hef,
                           double_ended_search=neb,
                           similarity=comparer)

# BEGIN CALCULATIONS

# Perform global optimisation to locate all local minima of the function
explorer.get_minima(coords=coords,
                    n_steps=1000,
                    conv_crit=1e-5,
                    temperature=100.0,
                    test_valid=True)
# Then get the transition states between them by attempting to connect
# each minimum with its eight nearest neighbours
explorer.get_transition_states(method='ClosestEnumeration',
                               cycles=8,
                               remove_bounds_minima=True)
# Write the network to files to store
ktn.dump_network()
# Plot the stationary points in various ways to visualise the surface
plot_disconnectivity_graph(ktn=ktn,
                           levels=100)
plot_network(ktn=ktn)
plot_stationary_points(potential=schwefel,
                       ktn=ktn,
                       bounds=coords.bounds,
                       contour_levels=100,
                       fineness=150)
