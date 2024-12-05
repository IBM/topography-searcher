## Generate complete landscape for the two-dimensional Schwefel function

# IMPORTS

import numpy as np
from topsearch.data.coordinates import HybridCoordinates, AtomicCoordinates
from topsearch.potentials.gaussian_process import GaussianProcess
from topsearch.potentials.bayesian_optimisation import UpperConfidenceBound, ExpectedImprovement
from topsearch.potentials.atomic import BinaryGupta
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
from topsearch.analysis.minima_properties import get_bounds_minima
from topsearch.analysis.graph_properties import unconnected_component

# INITIALISATION

# Specify the coordinates for optimisation. We will optimise in a space
# of three dimensions, and select the standard bounds used for Schwefel
position = np.array([1.0, -1.0, 0.5, 0.1822015272, -0.5970484858, -0.4844360463,
                     -0.1822009635, 0.5970484122, 0.4844363476])
atom_labels = ['Au','Ag','Au', 'Ag', 'Ag']
bounds = [(1.8, 6.0), (1.8, 6.0), (0.0, 6.0), (0.0, 6.0), (0.0, 6.0),
          (0.0, 6.0), (0.0, 6.0), (0.0, 6.0), (0.0, 6.0)]
coords = HybridCoordinates(atom_labels, position, bounds=bounds)
gupta = BinaryGupta(species=atom_labels)
# Make some Cartesian coordinates for visualisation of predictions
a_position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                      -0.1977281310, 0.4447221826, -0.6224697723,
                      -0.1822009635, 0.5970484122, 0.4844363476,
                       0.1822015272, -0.5970484858, -0.4844360463,
                       1.3, 1.3, 1.3])
a_coords = AtomicCoordinates(atom_labels=atom_labels,
                             position = a_position)
# Get the training data for building the Gaussian process
model_data = ModelData(training_file='training.txt',
                       response_file='response.txt')
#upper_limit = -1.0*np.min(model_data.response)
upper_limit = 0.0
min_conf = np.argmin(model_data.response)
ref_coords = model_data.training[min_conf]
# Get the Gaussian process class
gp = GaussianProcess(model_data=model_data, kernel_choice='PermInvariant',
#                     kernel_bounds=[(1e-1, 1e1), (1e-1, 1e1), (1e-1, 1e1),
#                                    (1e-1, 1e1), (1e-1, 1e1), (1e-1, 1e1),
#                                    (1e-1, 1e1), (1e-1, 1e1), (1e-1, 1e1),
#                                    (1e-5, 1e-3)],
                     kernel_bounds=[(1e-1, 1e1), (1e-5, 1e-1)],
                     standardise_response=False,
                     coords=coords,
                     reference=ref_coords)
# Initialise the acquisition function using the gaussian process
ucb = UpperConfidenceBound(gaussian_process=gp, zeta=0.3)
# Similarity object
comparer = StandardSimilarity(distance_criterion=0.2,
                              energy_criterion=5e-2,
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
hef = HybridEigenvectorFollowing(potential=ucb,
                                 ts_conv_crit=1e-3,
                                 ts_steps=150,
                                 pushoff=3e-1,
                                 max_uphill_step_size=3e-1,
                                 min_uphill_step_size=1e-9,
                                 steepest_descent_conv_crit=1e-4,
                                 eigenvalue_conv_crit=1e-6,
                                 positive_eigenvalue_step=3e-1)
# Double ended transition state search
neb = NudgedElasticBand(potential=ucb,
                        force_constant=10.0,
                        image_density=25.0,
                        max_images=75,
                        neb_conv_crit=1e-2)
# Sampling of congifuration space
explorer = NetworkSampling(ktn=ktn, coords=coords,
                           global_optimiser=optimiser,
                           single_ended_search=hef,
                           double_ended_search=neb,
                           similarity=comparer,
                           multiprocessing_on=True,
                           n_processes=8)

######## BEGIN CALCULATIONS ######

# In batch Bayesian optimisation we aim to locate the global minimum of a
# function (for which we don't know the gradient) in a small number of
# evaluations. Starting from an initial dataset we perform epochs in which
# we propose an optimal set of points for function evaluation based on the
# current data. We select these points by those low in acquisition whilst
# also selecting those minima that are diverse

# Loop over epochs
for epoch in range(1, 5):

    print("EPOCH ", epoch)
    # Write the epoch to file
    with open('logfile', 'a') as outfile:
        outfile.write('######## Epoch %i #########\n\n' %epoch)

    # Refit the Gaussian process as it will gain additional data at each epoch
    init_response = gp.model_data.response.copy()
    model_data.limit_response_maximum(upper_limit)
    gp.model_data.standardise_response()
    gp.refit_model(150)
    print("gp fit = ", np.exp(gp.gpr.kernel_.theta))
    
    # Clear the network again after visualisation
    ktn.reset_network()

    # Perform global optimisation of acquisition function
    explorer.get_minima(coords, 5000, 1e-4, 100.0, test_valid=True)
    print("n_minima after basin_hopping = ", ktn.n_minima)
    ktn.dump_network(text_string='.minima')

    # Remove minima before transition state searches
    removals = []
    for i in range(ktn.n_minima):
       c = ktn.get_minimum_coords(i)
       mean, std = gp.function_and_std(c)
       if mean > -0.2 or std > 0.99:
           removals.append(i)
    ktn.remove_minima(np.asarray(removals))
    print("minima after removals = ", ktn.n_minima)
    # Remove the bounds minima
    bounds_minima = get_bounds_minima(ktn, coords)
    ktn.remove_minima(bounds_minima)
    print("After removing bounds: ", ktn.n_minima)

    # Then get the transition states between minima on acquistion surface
    explorer.get_transition_states('ClosestEnumeration', 6,
                                   remove_bounds_minima=False)
    # Remove those high energy that are not connected to the global minimum
    removals = []
    for i in range(ktn.n_minima):
       c = ktn.get_minimum_coords(i)
       mean, std = gp.function_and_std(c)
       if mean > -5e-2 and std > 0.99:
           removals.append(i)
    unconnected = unconnected_component(ktn)
    un_removals = set.intersection(set(removals), unconnected)
    print("Round 1 removals: ", un_removals)
    ktn.remove_minima(np.asarray(list(un_removals)))
    print("Round 1: ", ktn.n_minima)

    # Repeat transition state searches for more attempts
    explorer.get_transition_states('ClosestEnumeration', 10,
                                   remove_bounds_minima=False)
    removals = []
    for i in range(ktn.n_minima):
       c = ktn.get_minimum_coords(i)
       mean, std = gp.function_and_std(c)
       if mean > -5e-2 and std > 0.99:
           removals.append(i)
    unconnected = unconnected_component(ktn)
    un_removals = set.intersection(set(removals), unconnected)
    print("Round 2 removals: ", un_removals)
    ktn.remove_minima(np.asarray(list(un_removals)))
    print("Round 2: ", ktn.n_minima)

    # Final round to TS searches
    explorer.get_transition_states('ConnectUnconnected', 14,
                                   remove_bounds_minima=False)
    removals = []
    for i in range(ktn.n_minima):
       c = ktn.get_minimum_coords(i)
       mean, std = gp.function_and_std(c)
       if mean > -5e-2 and std > 0.99:
           removals.append(i)
    unconnected = unconnected_component(ktn)
    un_removals = set.intersection(set(removals), unconnected)
    print("Round 3 removals: ", un_removals)
    ktn.remove_minima(np.asarray(list(un_removals)))
    print("Round 3: ", ktn.n_minima)

    # Try and search through the network for a batch
    indices = []
    if ktn.n_minima > 0:
        # Dump the stationary points we found to files for storage
        ktn.dump_network(f'.{str(epoch)}')
        print("Writing network")

        # Generate the batch of minima from topographical information
        # First decide on minima to exclude - here minima at edges of space
        excluded_minima = get_excluded_minima(ktn, penalise_edge=True, coords=coords,
                                              penalise_similarity=False, proximity_measure=0.05,
                                              known_points=model_data.training)
        # Then select the single lowest minimum as serial BayesOpt
        indices, points = select_batch(ktn, 5, 'Monotonic', False,
                                       excluded_minima=excluded_minima)

        print("indices = ", indices)

    # No minima we want to increase variance and take lowest
    if len(indices) == 0 or ktn.n_minima == 0:
        ktn.reset_network()
        ucb.zeta *= 50.0
        explorer.get_minima(coords, 1500, 1e-5, 100.0, test_valid=True)
        # Generate the batch of minima from topographical information
        # First decide on minima to exclude - here minima at edges of space
        excluded_minima = get_excluded_minima(ktn, penalise_edge=True, coords=coords, energy_cutoff=100.0,
                                              penalise_similarity=True, proximity_measure=0.01,
                                              known_points=model_data.training)
        # Then select the single lowest minimum as serial BayesOpt
        indices, points = select_batch(ktn, 5, 'Lowest', False,
                                       excluded_minima=excluded_minima)
        ucb.zeta /= 50.0
        print("Done the extra loop for high variance")
        print("zeta = ", ucb.zeta)

    f_vals = []
    for c, i in enumerate(points):
        f_vals.append(gupta.function(coords.hybrid_to_cartesian(i)))
        a_coords.position = coords.hybrid_to_cartesian(i)
        a_coords.write_xyz('%i_%i' %(epoch, c))
    f_vals = np.asarray(f_vals)
    print("f_vals = ", f_vals)
    print("points = ", points)
    
    # Evaluate the batch, add these new points and write lowest to file
    gp.model_data.response = init_response
    gp.model_data.append_data(points, f_vals)
    with open('logfile', 'a') as outfile:
        outfile.write("Lowest point = %8.5f\n" %gp.lowest_point())
    
    plot_disconnectivity_graph(ktn, 100, label='%i' %epoch)

# Write the final dataset generated during optimisation
model_data.write_data('finaltraining.txt', 'finalresponse.txt')
