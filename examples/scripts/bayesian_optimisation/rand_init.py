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
from smt.sampling_methods import LHS

# INITIALISATION

target = -13.47066
n_at = 5

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
step_taking = AtomicPerturbation(max_displacement=1.5,
                                 max_atoms=4)
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


xlimits = np.array([[1.0, 6.0], [1.0, 6.0], [0.0, 6.0], [0.0, 6.0],
                    [0.0, 6.0], [0.0, 6.0], [0.0, 6.0], [0.0, 6.0],
                    [0.0, 6.0]])
sampling = LHS(xlimits=xlimits)

calls = []
for c in range(10):
    dat = sampling(10000)

    response = []
    banned = []
    for count, i in enumerate(dat):
        cart_i = hy_coords.hybrid_to_cartesian(i)
        clash = test_clashes(cart_i, n_at)
        if not clash:
            banned.append(count)
        dissociated = test_empty(cart_i, n_at)
        if not dissociated:
            banned.append(count)
        z_higher = test_z_higher(cart_i, n_at)
        if not z_higher:
            banned.append(count)
    dat = np.delete(dat, np.asarray(banned), axis=0)

    total_calls = 0
    for i in dat:
        x_position = hy_coords.hybrid_to_cartesian(i)
        x, f, d = fmin_l_bfgs_b(func=gupta.function,
                                x0=x_position,
                                fprime=None,
                                approx_grad=True,
                                factr=1e-30,
                                pgtol=1e-3)
        print("f_val, f_calls = ", f, d['funcalls'])
        total_calls += d['funcalls']
        if f < target:
            calls.append(total_calls)
            break

mean = np.mean(np.asarray(calls))
std =  np.std(np.asarray(calls))
with open('random_initialisation.txt', 'w') as out_file:
    out_file.write('%6.1f %6.1f' %(mean, std))
