## Global optimisation of the Schwefel function

# IMPORTS

import numpy as np
from topsearch.data.coordinates import StandardCoordinates
from topsearch.potentials.test_functions import Schwefel
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.global_optimisation.perturbations import StandardPerturbation
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.sampling.exploration import NetworkSampling
from topsearch.analysis.minima_properties import get_minima_energies

# INITIALISATION

# Specify the coordinates for optimisation. We will optimise in a space
# of three dimensions, and select the standard bounds used for Schwefel
coords = StandardCoordinates(ndim=3, bounds=[(-500.0, 500.0),
                                             (-500.0, 500.0),
                                             (-500.0, 500.0)])
# Specify the simple test function we will perform global optimisation on
schwefel = Schwefel()
# Similarity object, decides if two points are the same or different
# Same if distance between points is less than distance_criterion
# and the difference in function value is less than energy_criterion
# Distance is a proportion of the range, in this case 0.05*1000.0
comparer = StandardSimilarity(distance_criterion=0.05,
                              energy_criterion=1e-2,
                              proportional_distance=True)
# Kinetic transition network object to store stationary points
ktn = KineticTransitionNetwork()
# Perturbation scheme for proposing new positions in configuration space
# Standard perturbation just applies random perturbations
step_taking = StandardPerturbation(max_displacement=1.0,
                                   proportional_distance=True)
# Global optimisation class using basin-hopping
optimiser = BasinHopping(ktn=ktn,
                         potential=schwefel,
                         similarity=comparer,
                         step_taking=step_taking)
# Object that deals with sampling of configuration space,
# given the object for global optimisation
explorer = NetworkSampling(ktn=ktn, coords=coords,
                           global_optimiser=optimiser,
                           single_ended_search=None,
                           double_ended_search=None,
                           similarity=comparer)

## BEGIN CALCULATIONS

# Perform global optimisation of 3D Schwefel function
explorer.get_minima(coords=coords,
                    n_steps=500,
                    conv_crit=1e-5,
                    temperature=100.0,
                    test_valid=True)
# Dump the minima we found to files min.data and min.coords
ktn.dump_network()
# Give the energy and position of the global minimum
energies = get_minima_energies(ktn)
print("Global minimum energy = ", np.min(energies))
print("Coordinates = ", ktn.get_minimum_coords(np.argmin(energies)))
