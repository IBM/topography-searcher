## Compute the roughness of a given dataset using the energy landscape
## framework to compute the frustration metric

# IMPORTS

import numpy as np
from topsearch.data.coordinates import StandardCoordinates
from topsearch.potentials.dataset_fitting import DatasetInterpolation
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
from topsearch.analysis.roughness import roughness_metric

# Specify the coordinates we move around the space, 2-dimensional
# here as we apply the method to feature pairs that are normalised
# hence the bounds specified between 0 and 1
coords = StandardCoordinates(ndim=2, bounds=[(0.0, 1.0),
                                             (0.0, 1.0)])
# Get the training data for which we will estimate the roughness
model_data = ModelData(training_file='training.txt',
                       response_file='response.txt')
# Interpolation function class. This class takes in the training data
# fits a radial basis function interpolation with the specified smoothness
# and can then be queried to give function and gradient at any point
interpolation = DatasetInterpolation(model_data=model_data,
                                     smoothness=1e-4)
# Similarity object, decides if two points are the same or different
# Same if distance between points is less than distance_criterion
# and the difference in function value is less than energy_criterion
# Distance is a proportion of the range, in this case 0.05*1.0
comparer = StandardSimilarity(distance_criterion=0.02,
                              energy_criterion=5e-2,
                              proportional_distance=True)
# Kinetic transition network to store stationary points
ktn = KineticTransitionNetwork()
# Perturbation scheme for proposing new positions in configuration space
# Standard perturbation just applies random perturbations
step_taking = StandardPerturbation(max_displacement=1.0,
                                   proportional_distance=True)
# Global optimisation class using basin-hopping
optimiser = BasinHopping(ktn=ktn, potential=interpolation, similarity=comparer,
                         step_taking=step_taking)
# Single ended transition state search object that locates transition
# states from a single starting position, using hybrid eigenvector-following
hef = HybridEigenvectorFollowing(potential=interpolation,
                                 ts_conv_crit=5e-4,
                                 ts_steps=75,
                                 pushoff=5e-3,
                                 steepest_descent_conv_crit=1e-4,
                                 max_uphill_step_size=1e1,
                                 min_uphill_step_size=1e-8,
                                 eigenvalue_conv_crit=1e-3,
                                 positive_eigenvalue_step=1e-2)
# Double ended transition state search that locates approximate minimum energy
# pathways between two points using the nudged elastic band algorithm
neb = NudgedElasticBand(potential=interpolation,
                        force_constant=5e2,
                        image_density=50.0,
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
                           n_processes=8)

# BEGIN CALCULATIONS

# We take each subset of the data corresponding to only two features.
# Compute the landscape of the interpolated surface corresponding to the
# dataset, and compute the roughness from this.
# We then average the roughness over all feature pairs to get the overall value

# Generate all possible feature pairs
pairs = []
for i in range(0, model_data.n_dims-1):
    for j in range(i+1, model_data.n_dims):
        pairs.append([i, j])
frustration_values = []

#model_data.invert_response()

# Loop over all feature pairs to get the landscape and roughness
for i in pairs:

    print("Pair = ", i)

    # Get the corresponding data and normalise it
    model_data.read_data('training.txt', 'response.txt')
    model_data.feature_subset(i)
    model_data.remove_duplicates()
    model_data.normalise_training()
    model_data.normalise_response()
    model_data.convex_hull()
    interpolation.refit_model()

    # Empty the network so we can start from scratch for each feature pair
    ktn.reset_network()

    # Do short global optimisation runs from the position of each data point
    # There can be very small wells corresponding to a dataset so choosing
    # start points like this can still catch these
    for d_p in model_data.training:
        coords.position = d_p
        # Perform global optimisation of interpolated function
        explorer.get_minima(coords, 5, 1e-4, 100.0, test_valid=True)

    # Remove any minima that lie outside the convex hull of the original data
    # Outside of this region the interpolating function is untrustworthy due
    # to lack of data
    outside_hull = []
    for j in range(ktn.n_minima):
        point = ktn.get_minimum_coords(j)
        if not model_data.point_in_hull(point):
            outside_hull.append(j)
    ktn.remove_minima(np.asarray(outside_hull))

    # Get the transition states between the remaining minima to produce
    # the complete landscape for this dataset interpolation
    explorer.get_transition_states('ClosestEnumeration', 12,
                                   remove_bounds_minima=False)

    # Compute the roughness of the dataset using the frustration metric
    frustration = roughness_metric(ktn, lengthscale=0.8)
    frustration_values.append(frustration)

    # Plot the relevant surface descriptions for the feature pair
    plot_stationary_points(interpolation, ktn, bounds=coords.bounds, contour_levels=125,
                           fineness=175, label='%i_%i' %(i[0], i[1]))

    # Store the landscape to file for future reference
    ktn.dump_network('%i_%i' %(i[0], i[1]))

np.savetxt('frustration.txt', frustration_values)
