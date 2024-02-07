## Compute the roughness of multiple datasets using the frustration metric
## and correlate the topographical roughness with the model error for these
## datasets

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

# Specify the coordinates for optimisation. The space is of two dimensions
# here as we select feature pairs to define the surface, and the
# bounds fit data normalisation
coords = StandardCoordinates(ndim=2, bounds=[(0.0, 1.0),
                                             (0.0, 1.0)])
# Read in the dataset for which we compute the roughness
model_data = ModelData(training_file='training_ari.txt',
                       response_file='response_ari.txt')
original_dimensionality = model_data.n_dims
# Interpolation function class. This class takes in the training data
# fits a radial basis function interpolation with the specified smoothness
# and can then be queried to give function and gradient at any point
interpolation = DatasetInterpolation(model_data=model_data, smoothness=1e-5)
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
                                 conv_crit=1e-5,
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
explorer = NetworkSampling(ktn=ktn,
                           coords=coords,
                           global_optimiser=optimiser,
                           single_ended_search=hef,
                           double_ended_search=neb,
                           similarity=comparer,
                           multiprocessing_on=True,
                           n_processes=8)

# BEGIN CALCULATIONS

# Compute the roughness and model error (from MLP) for each selected dataset.
# The roughness is computed using the frustration metric, which determines the
# topographical roughness of the solution landscape using energy landscape theory.
# Finally, compare the frustration metric and model error to show the strong
# correlation between these two properties.

# Loop over multiple datasets getting frustration for each
datasets = ['caco2', 'hep', 'ppbr', 'vdss']

# Initialise the lists to store overall data
dataset_frustration = []
model_error = []

# Loop over all the datasets we want to test
for i in datasets:

    print("Dataset: ",  i)

    # Generate all possible feature pairs
    pairs = []
    for j in range(0, original_dimensionality-1):
        for k in range(j+1, original_dimensionality):
            pairs.append([j, k])

    # Compute the regression model error for each of the datasets
    model_data.read_data(f'training_{i}.txt', f'response_{i}.txt')
    model_data.normalise_training()
    model_data.normalise_response()
    regression = DatasetRegression(model_data=model_data)
    model_error.append(regression.get_model_error())

    frustration_values = []

    # Loop over all feature pairs for the current dataset
    for j in pairs:

        print("Pair = ", j)

        # Get the subset of data and normalise it
        model_data.read_data(f'training_{i}.txt', f'response_{i}.txt')
        model_data.feature_subset(j)
        model_data.remove_duplicates()
        model_data.normalise_training()
        model_data.normalise_response()
        model_data.convex_hull()
        interpolation.refit_model()

        # Clear the network for each separate feature pair
        ktn.reset_network()

        # Do short global optimisation runs from the position of each data point
        # There can be very small wells corresponding to a dataset so choosing
        # start points like this can still catch these
        for d_p in model_data.training:
            coords.position = d_p
            # Perform global optimisation of acquisition function
            explorer.get_minima(coords, 50, 1e-5, 100.0, test_valid=True)

        # Remove any minima that lie outside the convex hull of the original data
        # Outside of this region the interpolating function is untrustworthy due
        # to lack of data
        outside_hull = []
        for k in range(ktn.n_minima):
            point = ktn.get_minimum_coords(k)
            if not model_data.point_in_hull(point):
                outside_hull.append(k)
        ktn.remove_minima(np.asarray(outside_hull))

        # Get the transition states between the remaining minima to produce
        # the complete landscape for this dataset interpolation
        explorer.get_transition_states('ClosestEnumeration', 8,
                                       remove_bounds_minima=True)

        # Compute the roughness of the dataset using the frustration metric
        frustration = roughness_metric(ktn, lengthscale=0.8)
        frustration_values.append(frustration)

    # Add the overall frustration value to the list over datasets
    dataset_frustration.append(np.average(frustration_values))

# Plot the model error and roughness calculations we have computed
# to show the strong correlation between the different values
plt.scatter(dataset_frustration, model_error)
plt.xlabel('Frustration Metric')
plt.ylabel('Model error')
plt.savefig('ModelErrorVsFrustration', dpi=300)
