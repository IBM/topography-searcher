#!/usr/bin/env python3.12

## Compute the roughness of a given dataset using the energy landscape
## framework to compute the frustration metric

# IMPORTS
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging
import math
import os
import pathlib
import time
from typing import Optional
import numpy as np
import sys
from multiprocessing import cpu_count
from topsearch.data.coordinates import StandardCoordinates
from topsearch.potentials.dataset_fitting import DatasetInterpolation, DatasetRegression
from topsearch.potentials.potential import Potential
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.data.model_data import ModelData
from topsearch.global_optimisation.perturbations import StandardPerturbation
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.sampling.exploration import NetworkSampling
from topsearch.transition_states.hybrid_eigenvector_following import HybridEigenvectorFollowing
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand
from topsearch.analysis.roughness import RoughnessContribution, roughness_contributors, roughness_metric
from topsearch.analysis.minima_properties import validate_minima
from topsearch.utils.logging import configure_logging
from tqdm.auto import tqdm
from multiprocessing import current_process, set_start_method

configure_logging()

def main(training_data_path: str,
         response_data_path: str,
         total_processes: int = cpu_count(), 
         log_level="WARNING",
         minimse=True,
         ts_search=True,
         data_dir=os.getcwd()
         ):

    set_start_method("spawn")

    logger = logging.getLogger()

    logger.info(f"Running with {total_processes} processes")
    logger.info(f"Training data: {training_data_path} Response data: {response_data_path}")

    start_time = time.time()

    # Get the training data for which we will estimate the roughness
    model_data = ModelData(training_data_path, response_data_path)
    model_data.normalise_training()
    model_data.normalise_response()

    # Specify the coordinates we move around the space, 
    # We either operate in all dimensions of the embedding space
    # Features are normalised hence the bounds specified between 0 and 1
    bounds = [(0.0, 1.0) for _ in range(model_data.n_dims)]
    coords = StandardCoordinates(ndim=model_data.n_dims, bounds=bounds)
 
    # Interpolation function class. This class takes in the training data
    # fits a radial basis function interpolation with the specified smoothness
    # and can then be queried to give function and gradient at any point
    interpolation = DatasetInterpolation(model_data=model_data,
                                        smoothness=1e-4)
    logger.info("Using DatasetInterpolation, smoothness = %e", interpolation.smoothness)

    # Similarity object, decides if two points are the same or different
    # Same if distance between points is less than distance_criterion
    # and the difference i  n function value is less than energy_criterion
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
                                    ts_conv_crit=5e-2,
                                    ts_steps=300,
                                    pushoff=5e-3, # doesn't make a difference until we find a transition state
                                    steepest_descent_conv_crit=1e-4,
                                    max_uphill_step_size=0.3,
                                    min_uphill_step_size=1e-7,
                                    eigenvalue_conv_crit=1e-2,
                                    positive_eigenvalue_step=0.3)
    # Double ended transition state search that locates approximate minimum energy
    # pathways between two points using the nudged elastic band algorithm
    neb = NudgedElasticBand(potential=interpolation,
                            force_constant=5e2,
                            image_density=10.0,
                            max_images=30,
                            neb_conv_crit=1e-1)
    # Object that deals with sampling of configuration space,
    # given the object for global optimisation and transition state searches
    explorer = NetworkSampling(ktn=ktn, coords=coords,
                            global_optimiser=optimiser,
                            single_ended_search=hef,
                            double_ended_search=neb,
                            similarity=comparer,
                            multiprocessing_on=True,
                            n_processes=total_processes,
                            log_dir=data_dir,
                            max_connection_attempts_per_pair=1)

    # BEGIN CALCULATIONS

    contributions: list[RoughnessContribution] = []
    dataset_frustration: float = 0 
    with open("frustration.txt", 'a') as frustration_log:
        frustration, contributions = compute_roughness(model_data = model_data, ktn = ktn, interpolation = interpolation, explorer = explorer, coords = coords, log_level=log_level, minimise=minimse, ts_search=ts_search, data_dir=data_dir)
        dataset_frustration = frustration
        frustration_log.write(f"All dimensions: {dataset_frustration}")

    total_time = time.time() - start_time
    
    logger.info("Top 10 contributors to roughness:")
    contributions.sort(key=lambda contributor: contributor.frustration, reverse=True)

    for contribution in contributions[:10]:
        logger.info(f"Minimum: {contribution.minimum} Transition state: {contribution.ts} Frustration: {contribution.frustration}" )
    
    logger.info(f"Average frustration: {dataset_frustration}")
    logger.info(f"Total time to compute: {total_time:.3f} seconds") 

def compute_roughness(model_data: ModelData, ktn: KineticTransitionNetwork, interpolation: DatasetInterpolation, explorer: NetworkSampling, coords: StandardCoordinates, log_level: str, minimise: bool = True, ts_search: bool = True, data_dir: str = os.getcwd()) -> tuple[float, list]:
    logger = logging.getLogger()
        
    ktn.dump_path = pathlib.Path(data_dir)
    ktn.dump_suffix = '_all_dimensions'
    min_path = ktn.dump_path / f"min.data{ktn.dump_suffix}"
    
    if min_path.exists():
        logger.info(f"Loading KTN from {ktn.dump_path}")
        ktn.read_network()
        
    if minimise:
        # Do short global optimisation runs from the position of each data point
        # There can be very small wells corresponding to a dataset so choosing
        # start points like this can still catch these
       
        # Perform global optimisation of interpolated function
        explorer.get_minima(initial_positions=model_data.training, coords=coords, n_steps=5, conv_crit=1e-4, temperature=100.0, test_valid=True)

        logger.info(f"Found {ktn.n_minima} minima")
        ktn.dump_network()

        validate_minima(ktn, model_data, coords, interpolation)

    frustration = 0
    contributors = []

    if ts_search: 
        # Get the transition states between the remaining minima to produce
        # the complete landscape for this dataset interpolation
        explorer.get_transition_states('ClosestEnumeration', 12,
                                    remove_bounds_minima=False)

        # Compute the roughness of the dataset using the frustration metric
        frustration = roughness_metric(ktn, lengthscale=0.8)
        contributors = roughness_contributors(ktn, lengthscale=0.8)

    # Store the landscape to file for future reference
    ktn.dump_network()

    return frustration, contributors

if __name__ == '__main__':    
    main(training_data_path="../data_generation/expected_output/selfies-ted-mini-training.txt", response_data_path="../data_generation/expected_output/selfies-ted-mini-response.txt", log_level="INFO", minimse=True, ts_search=True, data_dir="./expected_output")