## Generate the potential energy landscape of a small molecule starting from
## an xyz file

# IMPORTS

import numpy as np
import ase.io
from topsearch.data.coordinates import MolecularCoordinates
from sys import argv
# from topsearch.potentials.dft import DensityFunctionalTheory
from topsearch.potentials.force_fields import MMFF94
from topsearch.similarity.molecular_similarity import MolecularSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.global_optimisation.perturbations import MolecularPerturbation
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.sampling.exploration import NetworkSampling
from topsearch.plotting.disconnectivity import plot_disconnectivity_graph
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand
from topsearch.potentials.ml_potentials import MachineLearningPotential

# INITIALISATION

# Specify the molecule and its initial coordinates from the xyz file
# Here the molecule is ethanol, but any other molecule could be used
atfile = f'../../molecules_relax/{argv[1]}_relax.xyz'
atoms = ase.io.read(atfile)
species = atoms.get_chemical_symbols()
position = atoms.get_positions().flatten()
coords = MolecularCoordinates(species, position)

# mlp = MachineLearningPotential(species, 'aimnet2', 
#                                '/u/jdm/models/aimnet2_wb97m_0.jpt', # aimnet2_wb97m_0.jpt
#                                device='cuda') # should use the better, more expensive model for production run
mace_file = '/dccstor/chemistry_ai/mlp_landscapes/models/nequip/malonaldehyde/nequip_malonaldehyde.pth'
mlp = MachineLearningPotential(species, 'nequip', mace_file, 'cuda')
# Initialise the force field potential we use when DFT fails
# due to atom overlap. Empirical force fields continue to work
# and remove the clashes before we continue with DFT
ff = MMFF94(atfile)
# Initialise the DFT potential that we will compute the landscape of
# Set all the Psi4 options. These are expensive but high quality parameters
# Similarity object, decides if two conformations are the same or different
# Same if distance between points is less than distance_criterion
# and the difference in energy is less than energy_criterion
comparer = MolecularSimilarity(distance_criterion=1.0,
                               energy_criterion=5e-3,
                               weighted=False)
# comparer = MolecularSimilarity(distance_criterion=0.3,
#                                energy_criterion=1e-3,
#                                weighted=False)
# Kinetic transition network to store stationary points
ktn = KineticTransitionNetwork()
# Perturbation scheme for proposing new positions in configuration space
# Molecular perturbations rotate about a random subset of the flexible
# dihedrals up to a maximum angle max_displacement
step_taking = MolecularPerturbation(max_displacement=180.0,
                                    max_bonds=2)
# Global optimisation class that uses basin-hopping to locate
# local minima and the global minimum
optimiser = BasinHopping(ktn=ktn, potential=mlp, similarity=comparer,
                         step_taking=step_taking, ignore_relreduc=False, opt_method='ase')
# Single ended transition state search object that locates transition
# states from a single starting position, using hybrid eigenvector-following
hef = HybridEigenvectorFollowing(potential=mlp,
                                 ts_conv_crit=1e-2,
                                 ts_steps=100,
                                 pushoff=0.8,
                                 max_uphill_step_size=0.3,
                                 positive_eigenvalue_step=0.1,
                                 steepest_descent_conv_crit=1e-3,
                                 eigenvalue_conv_crit=5e-2)
# Double ended transition state search that locates approximate minimum energy
# pathways between two points using the nudged elastic band algorithm
neb = NudgedElasticBand(potential=mlp,
                        force_constant=50.0,
                        image_density=15.0,
                        max_images=20,
                        neb_conv_crit=1e-2, output_level=2)
# Class for sampling configuration space by driving the global_optimisation
# and transition state search methods
explorer = NetworkSampling(ktn=ktn,
                           coords=coords,
                           global_optimiser=optimiser,
                           single_ended_search=hef,
                           double_ended_search=neb,
                           similarity=comparer)

# BEGIN CALCULATIONS

explorer.get_minima(coords=coords,
                    n_steps=50,
                    conv_crit=1e-3,
                    temperature=100.0,
                    test_valid=True)
ktn.dump_network('.first')
explorer.get_transition_states(method='ClosestEnumeration',
                               cycles=2,
                               remove_bounds_minima=False)
ktn.dump_network()