## Generate the potential energy landscape 

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
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand

# INITIALISATION

# Specify the coordinates for optimisation. We will optimise in a space
# of two dimensions, and select the standard bounds used for Schwefel
atoms = ase.io.read('ethanol.xyz')
species = atoms.get_chemical_symbols()
position = atoms.get_positions().flatten()
coords = MolecularCoordinates(species, position)
# Initialise the force field potential we use when DFT fails
# due to atom overlap. Empirical force fields continue to work
# and remove the clashes before we continue with DFT
ff = MMFF94('ethanol.xyz')
# Initialise the DFT potential that we will compute the landscape of
# Set all the Psi4 options
options = {"method": 'pbe',
           "basis": 'def2-svp',
           "threads": 8,
           "E_CONVERGENCE": 1e-9,
           "DFT_SPHERICAL_POINTS": 770,
           "D_CONVERGENCE": 1e-9,
           "df_basis_scf": "def2-svp-RI",
           "wcombine": False,
           "scf_type": "direct",
           "DFT_RADIAL_POINTS":100}
dft = DensityFunctionalTheory(atom_labels=species,
                              options=options,
                              force_field=ff)
# Similarity object, decides if two points are the same or different
# Same if distance between points is less than distance_criterion
# and the difference in function value is less than energy_criterion
# Distance is a proportion of the range, in this case 0.05*1000.0
comparer = MolecularSimilarity(distance_criterion=0.3,
                               energy_criterion=0.05,
                               weighted=False)
# Kinetic transition network to store stationary points
ktn = KineticTransitionNetwork()
# Perturbation scheme for proposing new positions in configuration space
# Standard perturbation just applies random perturbations
step_taking = MolecularPerturbation(max_displacement=180.0,
                                    max_bonds=2)
# Global optimisation class that uses basin-hopping to locate
# local minima and the global minimum
optimiser = BasinHopping(ktn=ktn, potential=dft, similarity=comparer,
                         step_taking=step_taking)
# Single ended transition state search object that locates transition
# states from a single starting position, using hybrid eigenvector-following
hef = HybridEigenvectorFollowing(potential=dft,
                                 conv_crit=5e-5,
                                 ts_steps=50,
                                 pushoff=1e-1,
                                 max_uphill_step_size=0.3,
                                 positive_eigenvalue_step=0.3)
# Double ended transition state search that locates approximate minimum energy
# pathways between two points using the nudged elastic band algorithm
neb = NudgedElasticBand(potential=dft,
                        force_constant=250.0,
                        image_density=5.0,
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

# Read in the original network computed at lower level of theory
ktn.read_network('initial')
# Reconverge every stationary point in the network within the
# new dft potential
explorer.reconverge_landscape(potential=dft,
                              conv_crit=1e-5)
# Write the updated network to file
ktn.dump_network()
