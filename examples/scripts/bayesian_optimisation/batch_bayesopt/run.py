## Generate complete landscape for the two-dimensional Schwefel function

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

# Specify the coordinates for optimisation. We will optimise in a space
# of three dimensions, and select the standard bounds used for Schwefel
coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0),
                                             (-2.0, 2.0)])
# The true function we aim to optimise
camel = Camelback()
# Get the training data for building the Gaussian process
model_data = ModelData(training_file='training.txt',
                       response_file='response.txt')
# Get the Gaussian process class
gp = GaussianProcess(model_data=model_data, kernel_choice='RBF',
                     kernel_bounds=[(1e-2, 30.0), (1e-2, 30.0), (1e-7, 0.1)])
# Initialise the acquisition function using the gaussian process
ucb = UpperConfidenceBound(gaussian_process=gp, zeta=0.1)
# Similarity object
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
# Single ended transition state search
hef = HybridEigenvectorFollowing(ucb, 1e-5, 75, 3e-1)
# Double ended transition state search
neb = NudgedElasticBand(ucb, 10.0, 50, 10, 1e-2)
# Sampling of congifuration space
explorer = NetworkSampling(ktn=ktn, coords=coords,
                           global_optimiser=optimiser,
                           single_ended_search=hef,
                           double_ended_search=neb,
                           similarity=comparer,
                           multiprocessing_on=True,
                           n_processes=4)

######## BEGIN CALCULATIONS ######

# Loop over epochs
for epoch in range(1, 15):
   
    # Write the epoch to file
    with open('logfile', 'a') as outfile:
        outfile.write('######## Epoch %i #########\n\n' %epoch)

    print("len(data) = ", len(gp.model_data.response))
    print("len(resp) = ", len(gp.model_data.training))
    print("resp = ", gp.model_data.response)
    # Refit the Gaussian process as it will gain additional data at each epoch
    gp.initialise_gaussian_process(100)
    
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
    indices, points = select_batch(ktn, 3, 'Topographical', False,
                                   excluded_minima=excluded_minima)
    print("points = ", points)

    # Evaluate the batch, add these new points and write lowest to file
    f_vals = evaluate_batch(camel, indices, points)
    print("f_vals = ", f_vals)
    gp.add_data(points, f_vals)
    with open('logfile', 'a') as outfile:
        outfile.write("Lowest point = %8.5f\n" %np.min(model_data.response))

# Write the final dataset generated during optimisation
model_data.write_data('finaltraining.txt', 'finalresponse.txt')
