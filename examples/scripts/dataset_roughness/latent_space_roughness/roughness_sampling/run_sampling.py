# %% [markdown]
# # Import necessary packages

# %%
import os
import time
import numpy as np
import logging
from topsearch.data.coordinates import StandardCoordinates
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.global_optimisation.perturbations import StandardPerturbation
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.sampling.exploration import NetworkSampling
from topsearch.potentials.local_roughness import LocalRoughness
from topsearch.analysis.roughness import roughness_contributors
from topsearch.analysis.minima_properties import get_ordered_minima
from topsearch.utils.logging import configure_logging


# %%
# Initialise logging

def main():
    # %% [markdown]
    # # Read in the existing kinetic transition network

    # %%
    # Kinetic transition network to store stationary points
    ktn = KineticTransitionNetwork()
    ktn.read_network(text_path="../landscape_exploration/expected_output", text_string="_all_dimensions")


    # %% [markdown]
    # # Set the parameters for the multivariate Gaussians

    # %%
    distance_to_ts = 0.75
    parallel_scaling = 0.99
    orthogonal_scaling = 0.25
    prefactor_scaling = 10
    lengthscale = 80
    n_dim = 128
    experiment_timestamp = time.time()
    logger = logging.getLogger()

    logger.info("==========================================================")

    logger.info(f"distance_to_ts: {distance_to_ts}")
    logger.info(f"parallel_scaling: {parallel_scaling}")
    logger.info(f"orthogonal_scaling: {orthogonal_scaling}")
    logger.info(f"prefactor_scaling: {prefactor_scaling}")
    logger.info(f"lengthscale: {lengthscale}")
    logger.info(f"Output file timestamp: {experiment_timestamp}")

    # %% [markdown]
    # # Get all the Gaussians for each minima-transition state

    # %%
    # Compute the roughness of the dataset using the frustration metric
    contributions = roughness_contributors(ktn, lengthscale)

    # %% [markdown]
    # # Specify the local roughness function
     # Initialise a second ktn to store the points with highest local roughness
 
    ktn2 = KineticTransitionNetwork(dump_suffix=".sampling-checkpoint")
    if os.path.isfile("./min.data.sampling-checkpoint"):
        logger.info("Resuming previous sampling run")
        ktn2.read_network()

    logger.info("Computing gaussians for local roughness")
    l_roughness = LocalRoughness(contributions, parallel_scaling=parallel_scaling, orthogonal_scaling=orthogonal_scaling, prefactor_scaling=prefactor_scaling, distance_to_ts=distance_to_ts)

    # %% [markdown]
    # # Initialise basin-hopping

    # %%
   

    # Initialise the coordinates for the space of initial roughness
    coords = StandardCoordinates(ndim=n_dim, bounds=[(0.0, 1.0)]*n_dim)
    # Comparison object
    comparer = StandardSimilarity(distance_criterion=0.02,
                                energy_criterion=5e-2,
                                proportional_distance=True)
    # Perturbation scheme for proposing new positions in configuration space
    step_taking = StandardPerturbation(max_displacement=1.0,
                                    proportional_distance=True)

    # Global optimisation class using basin-hopping
    optimiser = BasinHopping(ktn=ktn2, potential=l_roughness, similarity=comparer,
                            step_taking=step_taking)
    # Sampling object
    explorer = NetworkSampling(ktn=ktn2, coords=coords,
                            global_optimiser=optimiser,
                            single_ended_search=None,
                            double_ended_search=None,
                            similarity=comparer, n_processes=8)

    # %% [markdown]
    # # Perform basin-hopping

    # %%
    logger.info("Basin-hopping starting from means of Gaussians")

    # %%
    explorer.get_minima(coords=coords,
                        n_steps=5, # order of magnitude less as we just want some sensible minima, don't need to globally optimise for now
                        conv_crit=0.5, # make much higher as the range is now large enough that this is relatively tight convergence
                        temperature=100.0,
                        test_valid=False, # turn off as the surface is smooth and it's difficult to generate invalid minima
                        initial_positions=l_roughness.means
                        ) 

    # %%
    ktn2.dump_network(f".roughness")

if __name__ == '__main__':
    main()
