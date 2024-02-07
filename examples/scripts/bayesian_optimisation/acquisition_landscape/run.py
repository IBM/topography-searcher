## Compute the solution landscape of a three-dimensional upper
## confidence bound acquisition function

# IMPORTS

import numpy as np
from topsearch.data.coordinates import StandardCoordinates
from topsearch.potentials.gaussian_process import GaussianProcess
from topsearch.potentials.bayesian_optimisation import UpperConfidenceBound
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.data.model_data import ModelData
from topsearch.global_optimisation.perturbations import StandardPerturbation
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.sampling.exploration import NetworkSampling
from topsearch.plotting.disconnectivity import plot_disconnectivity_graph
from topsearch.plotting.network import plot_network
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand

# INITIALISATION

# Specify the coordinates that move around the three-dimensional
# feature space. With bounds specified to match the dataset
coords = StandardCoordinates(ndim=3, bounds=[(-5.0, 5.0),
                                             (-5.0, 5.0),
                                             (-5.0, 5.0)])
# Get the training data for building the Gaussian process
model_data = ModelData(training_file='training.txt',
                       response_file='response.txt')
# Initialise the Gaussian process class
gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                     kernel_bounds=[(1.0, 10.0), (1.0, 10.0),
                                    (1.0, 10.0), (1e-7, 0.1)])
# Initialise the acquisition function using the gaussian process
ucb = UpperConfidenceBound(gaussian_process=gp, zeta=0.2)
# Similarity object, decides if two points are the same or different
# Same if distance between points is less than distance_criterion
# and the difference in function value is less than energy_criterion
# Distance is a proportion of the range, in this case 0.03*10.0
comparer = StandardSimilarity(distance_criterion=0.03,
                              energy_criterion=1e-2,
                              proportional_distance=True)
# Kinetic transition network to store stationary points
ktn = KineticTransitionNetwork()
# Perturbation scheme for proposing new positions in feature space
# Standard perturbation just applies random perturbations
step_taking = StandardPerturbation(max_displacement=1.0,
                                   proportional_distance=True)
# Global optimisation class using basin-hopping
optimiser = BasinHopping(ktn=ktn, potential=ucb, similarity=comparer,
                         step_taking=step_taking)
# Single ended transition state search object that locates transition
# states from a single starting position, using hybrid eigenvector-following
hef = HybridEigenvectorFollowing(potential=ucb,
                                 conv_crit=1e-5,
                                 ts_steps=75,
                                 pushoff=3e-1)
# Double ended transition state search that locates approximate minimum energy
# pathways between two points using the nudged elastic band algorithm
neb = NudgedElasticBand(potential=ucb,
                        force_constant=25.0,
                        image_density=20.0,
                        max_images=20,
                        neb_conv_crit=1e-2)
# Object that deals with sampling of configuration space,
# given the object for global optimisation and transition state searches
explorer = NetworkSampling(ktn=ktn, coords=coords,
                           global_optimiser=optimiser,
                           single_ended_search=hef,
                           double_ended_search=neb,
                           similarity=comparer,
                           multiprocessing_on=True,
                           n_processes=4)

# BEGIN CALCULATIONS

# Perform global optimisation on the acquisition function to locate the 
# global minimum and other local minima. Each is a locally-optimal solution
# in sampling the acquisition surface
explorer.get_minima(coords, 1000, 1e-6, 100.0, test_valid=True)
# Compute the transition states that lie between the set of known minima
# These transition states encode the intermediate behaviour of the
# surface between solutions
explorer.get_transition_states('ClosestEnumeration', 4,
                               remove_bounds_minima=True)
# Dump the stationary point network we found to file
ktn.dump_network()
# Plot the landscape of solutions in various ways. Can't directly
# visualise due to dimensionality
plot_disconnectivity_graph(ktn, 100)
plot_network(ktn)
