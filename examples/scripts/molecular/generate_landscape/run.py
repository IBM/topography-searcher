## Generate the potential energy landscape of a small molecule starting from
## an xyz file

# IMPORTS

import numpy as np
import ase.io
from topsearch.data.coordinates import MolecularCoordinates
from topsearch.potentials.dft import DensityFunctionalTheory
from topsearch.potentials.force_fields import MMFF94
from topsearch.similarity.molecular_similarity import MolecularSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.global_optimisation.perturbations import MolecularPerturbation
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.sampling.exploration import NetworkSampling
from topsearch.plotting.disconnectivity import plot_disconnectivity_graph
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand

# INITIALISATION

# Specify the molecule and its initial coordinates from the xyz file
# Here the molecule is ethanol, but any other molecule could be used
atoms = ase.io.read('ethanol.xyz')
species = atoms.get_chemical_symbols()
position = atoms.get_positions().flatten()
coords = MolecularCoordinates(species, position)
# Initialise the force field potential we use when DFT fails
# due to atom overlap. Empirical force fields continue to work
# and remove the clashes before we continue with DFT
ff = MMFF94('ethanol.xyz')
# Initialise the DFT potential that we will compute the landscape of
# Set all the Psi4 options. These are expensive but high quality parameters
options = {"method": 'WB97X',
           "basis": '6-31G*',
           "threads": 3,
           "E_CONVERGENCE": 1e-9,
           "DFT_SPHERICAL_POINTS": 770,
           "D_CONVERGENCE": 1e-9,
           "wcombine": False,
           "scf_type": "direct",
           "DFT_RADIAL_POINTS":100}
dft = DensityFunctionalTheory(atom_labels=species,
                              options=options,
                              force_field=ff)
# Similarity object, decides if two conformations are the same or different
# Same if distance between points is less than distance_criterion
# and the difference in energy is less than energy_criterion
comparer = MolecularSimilarity(distance_criterion=0.3,
                               energy_criterion=1e-3,
                               weighted=False)
# Kinetic transition network to store stationary points
ktn = KineticTransitionNetwork()
# Perturbation scheme for proposing new positions in configuration space
# Molecular perturbations rotate about a random subset of the flexible
# dihedrals up to a maximum angle max_displacement
step_taking = MolecularPerturbation(max_displacement=180.0,
                                    max_bonds=2)
# Global optimisation class that uses basin-hopping to locate
# local minima and the global minimum
optimiser = BasinHopping(ktn=ktn, potential=dft, similarity=comparer,
                         step_taking=step_taking)
# Single ended transition state search object that locates transition
# states from a single starting position, using hybrid eigenvector-following
hef = HybridEigenvectorFollowing(potential=dft,
                                 ts_conv_crit=1e-2,
                                 ts_steps=50,
                                 pushoff=0.8,
                                 max_uphill_step_size=0.3,
                                 positive_eigenvalue_step=0.1,
                                 steepest_descent_conv_crit=1e-2,
                                 eigenvalue_conv_crit=1e-3)
# Double ended transition state search that locates approximate minimum energy
# pathways between two points using the nudged elastic band algorithm
neb = NudgedElasticBand(potential=dft,
                        force_constant=50.0,
                        image_density=5.0,
                        max_images=10,
                        neb_conv_crit=1.0)
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
# The evaluation of the energy is expensive even for this small molecule
# so we limit the number of steps. This is still more than enough for ethanol
# but more may be necessary for larger molecules
explorer.get_minima(coords=coords,
                    n_steps=15,
                    conv_crit=1e-2,
                    temperature=100.0,
                    test_valid=True)

# Find the transition states between the three minima
explorer.get_transition_states(method='ClosestEnumeration',
                               cycles=2,
                               remove_bounds_minima=False)
# Write the network to files to store
ktn.dump_network()

# Once basin-hopping has finished we store the minima
# Drop the coordinates of each local minimum for visualisation in xyz file
for i in range(ktn.n_minima):
    coords.position = ktn.get_minimum_coords(i)
    coords.write_xyz('min%i' %i)
for u, v in ktn.G.edges():
    coords.position = ktn.get_ts_coords(u, v)
    coords.write_xyz('ts%i_%i' %(u, v))

# Plot a disconnectivity graph
plot_disconnectivity_graph(ktn=ktn,
                           levels=100)