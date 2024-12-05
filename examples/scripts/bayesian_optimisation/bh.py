## Generate complete landscape for the two-dimensional Schwefel function

# IMPORTS

import numpy as np
from topsearch.data.coordinates import AtomicCoordinates, HybridCoordinates
from topsearch.potentials.atomic import BinaryGupta
from topsearch.similarity.molecular_similarity import MolecularSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.global_optimisation.perturbations import AtomicPerturbation
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.sampling.exploration import NetworkSampling
from topsearch.plotting.disconnectivity import plot_disconnectivity_graph
from topsearch.minimisation.lbfgs import minimise
from scipy.optimize import fmin_l_bfgs_b

# INITIALISATION

# Specify the coordinates for optimisation. We will optimise in a space
# of three dimensions, and select the standard bounds used for Schwefel
atom_labels = ['Au', 'Ag', 'Au', 'Ag', 'Ag']
gupta = BinaryGupta(species=atom_labels)
# Make some Cartesian coordinates for visualisation of predictions
a_position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                      -0.1977281310, 0.4447221826, -0.6224697723,
                      -0.1822009635, 0.5970484122, 0.4844363476,
                       0.1822015272, -0.5970484858, -0.4844360463,
                       1.3, 1.3, 1.3])
coords = AtomicCoordinates(atom_labels=atom_labels,
                           position=a_position)
hy_coords = HybridCoordinates(atom_labels=atom_labels,
                              position=np.array([1.0, -1.0, 0.5, 0.1822015272, -0.5970484858, -0.4844360463, -0.1822009635, 0.5970484122, 0.4844363476]),
                              bounds=[(0.0, 6.0)]*(3*coords.n_atoms-6))
# Similarity object
comparer = MolecularSimilarity(distance_criterion=0.1,
                               energy_criterion=5e-2)
# Kinetic transition network to store stationary points
ktn = KineticTransitionNetwork()
# Perturbation scheme for proposing new positions in configuration space
# Standard perturbation just applies random perturbations
step_taking = AtomicPerturbation(max_displacement=2.0,
                                 max_atoms=3)
# Global optimisation class using basin-hopping
optimiser = BasinHopping(ktn=ktn, potential=gupta, similarity=comparer,
                         step_taking=step_taking)
# Sampling of congifuration space
explorer = NetworkSampling(ktn=ktn, coords=coords,
                           global_optimiser=optimiser,
                           single_ended_search=None,
                           double_ended_search=None,
                           similarity=comparer,
                           multiprocessing_on=False,
                           n_processes=1)

######## BEGIN CALCULATIONS ######

training = np.genfromtxt('training.txt')

for i in range(10):

    # Perform global optimisation of acquisition function
    coords.position = hy_coords.hybrid_to_cartesian(training[i])
    explorer.get_minima(coords, 100, 1e-3, 1.0, test_valid=True)
