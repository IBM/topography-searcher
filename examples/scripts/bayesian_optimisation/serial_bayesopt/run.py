## Perform serial Bayesian optimisation on the six-hump Camel function
## We compute, and visualise, the solution landscape at each iteration

# IMPORTS

import numpy as np
from topsearch.data.coordinates import StandardCoordinates
from topsearch.potentials.gaussian_process import GaussianProcess
from topsearch.potentials.bayesian_optimisation import UpperConfidenceBound
from topsearch.potentials.test_functions import Camelback
from topsearch.similarity.similarity import StandardSimilarity
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

# INITIALISATION

# Specify the coordinates that move around the space. Here, these
# match the dimensionality and bounds of the true function
# the six-hump Camel function
coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0),
                                             (-2.0, 2.0)])
# The true function we aim to optimise, six-hump Camelback function
camel = Camelback()
# Get the initial training data for BayesOpt
model_data = ModelData(training_file='training.txt',
                       response_file='response.txt')
# Initialise the Gaussian process class
gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                     kernel_bounds=[(1e-2, 30.0), (1e-2, 30.0), (1e-7, 0.1)])
# Initialise the acquisition function using the gaussian process
ucb = UpperConfidenceBound(gaussian_process=gp, zeta=0.1)
# Similarity object, decides if two points are the same or different
# Same if distance between points is less than distance_criterion
# and the difference in function value is less than energy_criterion
# Distance is a proportion of the range, in this case 0.03*1.0
comparer = StandardSimilarity(distance_criterion=0.03,
                              energy_criterion=1e-2,
                              proportional_distance=True)
# Kinetic transition network to store stationary points
ktn = KineticTransitionNetwork()
# Perturbation scheme for proposing new positions in configuration space
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
                        image_density=10.0,
                        max_images=30,
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

# In serial Bayesian optimisation we aim to locate the global minimum of a
# function (for which we don't know the gradient) in a small number of
# evaluations. Starting from an initial dataset we perform epochs in which
# we propose the optimal next point for function evaluation by selecting
# the global minimum of an acqusition function surface 

# Run 15 epochs of BayesOpt
for epoch in range(1, 5):

    # Write the epoch to file
    with open('logfile', 'a') as outfile:
        outfile.write('######## Epoch %i #########\n\n' %epoch)

    print("len(data) = ", len(gp.model_data.response))
    print("len(resp) = ", len(gp.model_data.training))
    print("resp = ", gp.model_data.response)
    # Refit the Gaussian process as it will gain additional data at each epoch
    gp.initialise_gaussian_process(100)
    print("hyps = ", gp.gpr.kernel_.theta)

    # Empty the landscape for the new acquisition function
    ktn.reset_network()

    # Perform global optimisation of acquisition function
    explorer.get_minima(coords, 250, 1e-6, 100.0, test_valid=True)
    # Then get the transition states between minima on acquistion surface
    explorer.get_transition_states('ClosestEnumeration', 4,
                                   remove_bounds_minima=True)

    # Dump the stationary points we found to files for storage
    ktn.dump_network(str(epoch))
    # Plot the stationary points in various ways
    plot_disconnectivity_graph(ktn, 100, label=f'{epoch}')
    plot_network(ktn, label=f'{epoch}')
    plot_stationary_points(ucb, ktn, bounds=coords.bounds, contour_levels=100,
                           fineness=150, label=f'{epoch}')

    # Generate the batch of minima from topographical information
    # First decide on minima to exclude - here minima at edges of space
    excluded_minima = get_excluded_minima(ktn, penalise_edge=True, coords=coords)
    print("excluded_minima = ", excluded_minima)
    # Then select the single lowest minimum as serial BayesOpt
    indices, points = select_batch(ktn, 1, 'Lowest', False,
                                   excluded_minima=excluded_minima)
    print("points = ", points)

    # Evaluate the batch, add these new points and write lowest to file
    f_vals = evaluate_batch(camel, indices, points)
    print("f_vals = ", f_vals)
    gp.add_data(points, f_vals)
    with open('logfile', 'a') as outfile:
        outfile.write("Lowest point = %8.5f\n" %np.min(model_data.response))

# Write the final dataset generated during Bayesian optimisation
model_data.write_data('finaltraining.txt', 'finalresponse.txt')
