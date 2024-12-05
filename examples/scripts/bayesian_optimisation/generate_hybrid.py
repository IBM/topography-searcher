## Generate complete landscape for the two-dimensional Schwefel function

# IMPORTS

import numpy as np
import matplotlib.pyplot as plt
from topsearch.data.coordinates import HybridCoordinates, AtomicCoordinates
from topsearch.potentials.gaussian_process import GaussianProcess
from topsearch.potentials.bayesian_optimisation import UpperConfidenceBound, ExpectedImprovement
from topsearch.potentials.atomic import BinaryGupta
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.similarity.molecular_similarity import MolecularSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.data.model_data import ModelData
from topsearch.global_optimisation.perturbations import StandardPerturbation
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.sampling.exploration import NetworkSampling
from topsearch.plotting.disconnectivity import plot_disconnectivity_graph
from topsearch.plotting.stationary_points import plot_stationary_points
from topsearch.plotting.network import plot_network
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand
from topsearch.analysis.batch_selection import get_excluded_minima, select_batch, \
        evaluate_batch
from smt.sampling_methods import LHS

# INITIALISATION

# Specify the coordinates for optimisation. We will optimise in a space
# of three dimensions, and select the standard bounds used for Schwefel
position = np.array([1.0, -1.0, 0.5, 0.1822015272, -0.5970484858, -0.4844360463,
                     -0.1822009635, 0.5970484122, 0.4844363476])
atom_labels = ['Au','Ag','Au', 'Ag', 'Ag']
bounds = [(0.0, 6.0), (0.0, 6.0), (0.0, 6.0),
          (0.0, 6.0), (0.0, 6.0), (0.0, 6.0),
          (0.0, 6.0), (0.0, 6.0), (0.0, 6.0)]
coords = HybridCoordinates(atom_labels, position, bounds=bounds)
gupta = BinaryGupta(species=atom_labels)
a_position = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0],
                       [3.0, 3.0, 0.0], [3.0, 0.0, 3.0]])
a_coords = AtomicCoordinates(atom_labels, a_position, 1.5)
a_coords_nolabel = AtomicCoordinates(['Si','C','Au','Ag','Ag'], a_position, 1.5)
comparer = MolecularSimilarity(distance_criterion=1e-6,
                               energy_criterion=1e-6,
                               allow_inversion=True)

######## BEGIN CALCULATIONS ######

def test_clashes(position, n_atoms):
    for i in range(n_atoms-1):
        for j in range(i+1, n_atoms):
            atom_i = position[i*3:(i*3)+3]
            atom_j = position[j*3:(j*3)+3]
            dist = np.linalg.norm(atom_i - atom_j)
            if dist < 1.8:
                return False
    return True

# Check if any atoms are dissociated from the rest
def test_empty(position, n_atoms):
    for i in range(n_atoms):
        dist_vector = np.zeros(n_atoms, dtype=float)
        for j in range(n_atoms):
            if i == j:
                dist_vector[j] = 1e3
            else:
                atom_i = position[i*3:(i*3)+3]
                atom_j = position[j*3:(j*3)+3]
                dist_vector[j] = np.linalg.norm(atom_i - atom_j)
        if np.all(dist_vector > 4.0):
            return False
    return True

def test_z_higher(position, n_atoms):
    z1 = position[2]
    if position[5] < z1:
        return False
    else:
        return True

#dat = np.random.rand(140, 9)
xlimits = np.array([[1.0, 6.0], [1.0, 6.0], [0.0, 6.0], [0.0, 6.0],
                    [0.0, 6.0], [0.0, 6.0], [0.0, 6.0], [0.0, 6.0],
                    [0.0, 6.0]])
sampling = LHS(xlimits=xlimits)
dat = sampling(5000)

#dat = (dat*8.0)-4.0
response = []
banned = []
for count, i in enumerate(dat):
    cart_i = coords.hybrid_to_cartesian(i)
    clash = test_clashes(cart_i, 5)
    if not clash:
        banned.append(count)
    dissociated = test_empty(cart_i, 5)
    if not dissociated:
        banned.append(count)
    z_higher = test_z_higher(cart_i, 5)
    if not z_higher:
        banned.append(count)
dat = np.delete(dat, np.asarray(banned), axis=0)

