import numpy as np
import ase.io
from topsearch.data.coordinates import MolecularCoordinates
from sys import argv
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
atfile = f'../../../molecules_relax/{argv[1]}_relax.xyz'
atoms = ase.io.read(atfile)
species = atoms.get_chemical_symbols()
position = atoms.get_positions().flatten()
coords = MolecularCoordinates(species, position)

if len(argv) < 3:
    argv.append('aimnet2')

mlp = MachineLearningPotential(species, 'aimnet2', argv[2], "cuda")
with open("run_info.txt", "w") as f:
    f.write(f"Used AIMNET model {argv[2]}\n")
ff = MMFF94(atfile)
comparer = MolecularSimilarity(distance_criterion=1.5, # standard settings used for all but ethanol, which uses 0.5, 1e-3
                               energy_criterion=5e-3,
                               weighted=False,
                               allow_inversion=False)
ktn = KineticTransitionNetwork()
step_taking = MolecularPerturbation(max_displacement=180.0,
                                    max_bonds=2)
optimiser = BasinHopping(ktn=ktn, potential=mlp, similarity=comparer,
                         step_taking=step_taking, opt_method='ase') # use ASE to do the minimisation: finds more transition states
hef = HybridEigenvectorFollowing(potential=mlp,
                                 ts_conv_crit=1e-2,
                                 ts_steps=100,
                                 pushoff=0.8,
                                 max_uphill_step_size=0.3,
                                 positive_eigenvalue_step=0.1,
                                 steepest_descent_conv_crit=1e-3,
                                 eigenvalue_conv_crit=5e-2,
                                 output_level=1)
neb = NudgedElasticBand(potential=mlp,
                        force_constant=50.0,
                        image_density=15.0,
                        max_images=20,
                        neb_conv_crit=1e-2,
                        output_level=1)
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
explorer.get_transition_states(method='ClosestEnumeration',
                               cycles=2,
                               remove_bounds_minima=False)
ktn.dump_network()