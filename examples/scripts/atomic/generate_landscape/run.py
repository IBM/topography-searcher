## Generate the potential energy landscape for the LJ7 cluster

# IMPORTS

import numpy as np
from topsearch.data.coordinates import AtomicCoordinates
from topsearch.potentials.atomic import LennardJones
from topsearch.similarity.molecular_similarity import MolecularSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.global_optimisation.perturbations import AtomicPerturbation
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.sampling.exploration import NetworkSampling
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand

# INITIALISATION

# Specify the coordinates for optimisation. We will optimise in a space
# of two dimensions, and select the standard bounds used for Schwefel
atom_labels = ['C','C','C','C','C','C','C']
position = np.array([[0.0, 0.0, 0.0],
                     [3.0, 0.0, 0.0],
                     [3.0, 3.0, 0.0],
                     [0.0, 3.0, 0.0],
                     [0.0, 0.0, 3.0],
                     [0.0, 3.0, 3.0],
                     [3.0, 3.0, 3.0]])
coords = AtomicCoordinates(atom_labels=atom_labels,
                           position=position.flatten())
# Initialise the simple Lennard Jones potential for describing
# gas phase atomic species
lj = LennardJones()
# Similarity object, decides if two points are the same or different
# Same if distance between points is less than distance_criterion
# and the difference in function value is less than energy_criterion
comparer = MolecularSimilarity(distance_criterion=0.1,
                               energy_criterion=5e-2,
                               weighted=False)
# Kinetic transition network to store stationary points
ktn = KineticTransitionNetwork()
# Perturbation scheme for proposing new positions in configuration space
# Standard perturbation just applies random perturbations
step_taking = AtomicPerturbation(max_displacement=0.2,
                                 max_atoms=1)
# Global optimisation class that uses basin-hopping to locate
# local minima and the global minimum
optimiser = BasinHopping(ktn=ktn, potential=lj, similarity=comparer,
                         step_taking=step_taking)
# Single ended transition state search object that locates transition
# states from a single starting position, using hybrid eigenvector-following
hef = HybridEigenvectorFollowing(potential=lj,
                                 conv_crit=5e-5,
                                 ts_steps=50,
                                 pushoff=1e-1,
                                 max_uphill_step_size=0.3,
                                 positive_eigenvalue_step=0.3)
# Double ended transition state search that locates approximate minimum energy
# pathways between two points using the nudged elastic band algorithm
neb = NudgedElasticBand(potential=lj,
                        force_constant=50.0,
                        image_density=10.0,
                        max_images=20,
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
explorer.get_minima(coords, 2, 1e-5, 1.0, test_valid=False)
for i in range(ktn.n_minima):
    coords.position = ktn.get_minimum_coords(i)
    coords.write_xyz('%i' %i)
# Then get the transition states between them by attempting to connect
# each minimum with its eight nearest neighbours
#explorer.get_transition_states('ClosestEnumeration', 4,
#                               remove_bounds_minima=False)
# Write the network to files to store
#ktn.dump_network()
