## Generate complete landscape for the two-dimensional Schwefel function
## starting from a partial landscape consisting of a subset of the total
## minima and transition states

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
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand

# INITIALISATION

# Specify the coordinates which we optimise in two dimensions,
# giving them appropriate bounds for the function
coords = StandardCoordinates(ndim=2, bounds=[(-500.0, 500.0),
                                             (-500.0, 500.0)])
# Specify the Schwefel function that we compute the landscape of
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
# Global optimisation class using basin-hopping to find local minima
optimiser = BasinHopping(ktn=ktn, potential=schwefel, similarity=comparer,
                         step_taking=step_taking)
# Single ended transition state search object that locates transition
# states from a single starting position, using hybrid eigenvector-following
hef = HybridEigenvectorFollowing(schwefel, 1e-4, 75, 1.0)
# Double ended transition state search that locates approximate minimum energy
# pathways between two points using the nudged elastic band algorithm
neb = NudgedElasticBand(schwefel, 10.0, 50, 10, 1e-2)
# Class for sampling configuration space by driving the global_optimisation
# and transition state search methods
explorer = NetworkSampling(ktn=ktn, coords=coords,
                           global_optimiser=optimiser,
                           single_ended_search=hef,
                           double_ended_search=neb,
                           similarity=comparer,
                           multiprocessing_on=True,
                           n_processes=4)

# BEGIN CALCULATIONS

# Read the partially-complete landscape in from file
ktn.read_network(text_string='.original')
# Plot this original landscape
plot_disconnectivity_graph(ktn, 100, label='Original')
plot_stationary_points(schwefel, ktn, bounds=coords.bounds, contour_levels=100,
                       fineness=150, label='Original')

# Then continue the sampling of the landscape to locate further intermediate
# transition states between the known minima
explorer.get_transition_states('ClosestEnumeration', 20)
# Write all the stationary points back to file
ktn.dump_network()
# Plot the complete landscape to compare with the original network read in
plot_disconnectivity_graph(ktn, 100, label='Final')
plot_stationary_points(schwefel, ktn, bounds=coords.bounds, contour_levels=100,
                       fineness=150, label='Final')
