## Compute the roughness of a chemical dataset using the frustration metric
## and relate the roughness associated with specific features to model error

# IMPORTS

import numpy as np
import matplotlib.pyplot as plt
from topsearch.data.coordinates import StandardCoordinates
from topsearch.potentials.dataset_fitting import DatasetInterpolation, DatasetRegression
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.data.model_data import ModelData
from topsearch.global_optimisation.perturbations import StandardPerturbation
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.sampling.exploration import NetworkSampling
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand
from topsearch.analysis.roughness import roughness_metric

# Specify the coordinates we move around the space, 2-dimensional
# here as we apply the method to feature pairs that are normalised
# hence the bounds specified between 0 and 1
coords = StandardCoordinates(ndim=2, bounds=[(0.0, 1.0),
                                             (0.0, 1.0)])
# Get the training data for building the interpolation model
model_data = ModelData(training_file='training.txt',
                       response_file='response.txt')
original_dimensionality = model_data.n_dims
# Interpolation function class. This class takes in the training data
# fits a radial basis function interpolation with the specified smoothness
# and can then be queried to give function and gradient at any point
interpolation = DatasetInterpolation(model_data=model_data, smoothness=1e-5)
# Regression function class. Class fits a multi-layer perceptron to the
# model data and the regression model prediction can be made at any point
regression = DatasetRegression(model_data=model_data)
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
optimiser = BasinHopping(ktn=ktn,
                         potential=interpolation,
                         similarity=comparer,
                         step_taking=step_taking)
# Single ended transition state search object that locates transition
# states from a single starting position, using hybrid eigenvector-following
hef = HybridEigenvectorFollowing(potential=interpolation,
                                 conv_crit=1e-4,
                                 ts_steps=75,
                                 pushoff=1e-1)
# Double ended transition state search that locates approximate minimum energy
# pathways between two points using the nudged elastic band algorithm
neb = NudgedElasticBand(potential=interpolation,
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

# Here, we take a chemical dataset and compute the landscape
# for each possible pair of features. At each feature pair we compute
# the roughness and take an average roughness for the feature set
# without each feature in turn. We fit an MLP to the dataset without a single
# feature and show the correlation of roughness and model error. Therefore,
# we can highlight features that hinder model performance


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
        explorer.get_minima(coords, 50, 1e-5, 100.0, test_valid=True)

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
    explorer.get_transition_states('ClosestEnumeration', 8,
                                   remove_bounds_minima=True)

    # Compute the roughness of the dataset using the frustration metric
    frustration = roughness_metric(ktn, lengthscale=0.8)
    frustration_values.append(frustration)

# We have all the frustration values so now to analyse the roughness
# relative to each feature

np.savetxt('frustration.txt', frustration_values)
frustration_values = np.asarray(frustration_values)

# Get the frustration with each feature excluded in turn
feature_frustration = []
for i in range(original_dimensionality):
    # Get all the feature pairs that don't include current
    idxs = []
    for j in range(len(pairs)):
        if i not in pairs[j]:
            idxs.append(j)
    # Compute the average frustration
    frustration_subset = frustration_values[idxs]
    feature_frustration.append(np.average(frustration_subset))

# Compute the model accuracy for each of the same conditions
# with a single feature missing
model_error = []
for i in range(original_dimensionality):
    model_data.read_data('training.txt', 'response.txt')
    subset = np.arange(model_data.n_dims)
    subset = np.delete(subset, i)
    model_data.feature_subset(subset)
    model_data.remove_duplicates()
    model_data.normalise_training()
    model_data.normalise_response()
    regression.refit_model()
    model_error.append(regression.get_model_error())

# Plot the correlation between the roughness and model error
# as different features are removed. The results highlight that
# feature n significantly degrades performance, which is
# highlighted by the roughness without any model fitting
plt.scatter(feature_frustration, model_error)
plt.xlabel('Frustration')
plt.ylabel('Model error')
plt.tight_layout()
plt.savefig('FeatureContribution.png')
