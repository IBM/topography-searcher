""" Module contains the functions for analysing the properties of individual
    minima. All methods loop through and find minima that meet, or fail,
    specified criteria """

import logging

import numpy as np
from nptyping import NDArray
from numpy.testing import assert_allclose, assert_array_less
from scipy.optimize import fmin_l_bfgs_b

from topsearch.data.coordinates import StandardCoordinates
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.data.model_data import ModelData
from topsearch.potentials.potential import Potential
from topsearch.similarity.similarity import StandardSimilarity


def get_invalid_minima(ktn: KineticTransitionNetwork, potential: Potential, coords: StandardCoordinates) -> list[int]:
    """ Find any minima in the network G that do not meet the
        gradient or eigenspectrum criteria """
    invalid_minima = []
    # Check if each minimum passes the test for gradient and eigenvalues
    for i in range(ktn.n_minima):
        min_position = ktn.G.nodes[i]['coords']
        coords.position = min_position
        # Check if valid minimum by eigenvalues
        if not potential.check_valid_minimum(coords):
            invalid_minima.append(i)
    return invalid_minima


def get_bounds_minima(ktn: KineticTransitionNetwork, coords: StandardCoordinates) -> list:
    """ Find any minima that are at the bounds in any dimension """
    bounds_minima = []
    #  Loop over all current minima
    for i in range(ktn.n_minima):
        min_position = ktn.get_minimum_coords(i)
        coords.position = min_position
        if coords.at_bounds():
            bounds_minima.append(i)
    return bounds_minima


def get_all_bounds_minima(ktn: KineticTransitionNetwork, coords: StandardCoordinates) -> list:
    """ Find any minima that are at the bounds in all dimensions """
    bounds_minima = []
    #  Loop over all current minima
    for i in range(ktn.n_minima):
        min_position = ktn.get_minimum_coords(i)
        coords.position = min_position
        if coords.all_bounds():
            bounds_minima.append(i)
    return bounds_minima


def get_similar_minima(ktn: KineticTransitionNetwork,
                       proximity_measure: float,
                       comparison_points: NDArray) -> list:
    """ Locate any minima within proximity_measure of the comparison_points """
    similar_minima = []
    for i in range(ktn.n_minima):
        coords = ktn.get_minimum_coords(i)
        too_close = False
        for j in comparison_points:
            dist = np.linalg.norm(coords-j)
            if dist < proximity_measure:
                too_close = True
        # This minimum is too close to current data and is banned
        if too_close:
            similar_minima.append(i)
    return similar_minima


def get_minima_above_cutoff(ktn: KineticTransitionNetwork,
                            cutoff: float) -> list:
    """ Find all minima with an energy above cutoff """

    energies = get_minima_energies(ktn)
    high_energy_minima = energies > cutoff
    return np.where(high_energy_minima)[0].tolist()


def get_minima_energies(ktn: KineticTransitionNetwork) -> NDArray:
    """ Return the energies of all the minima """
    energies = np.zeros((ktn.n_minima), dtype=float)
    for i in range(ktn.n_minima):
        energies[i] = ktn.get_minimum_energy(i)
    return energies


def get_ordered_minima(ktn: KineticTransitionNetwork) -> NDArray:
    """ Return the ordered list of indices with increasing energy """
    energies = get_minima_energies(ktn)
    return np.argsort(energies)


def get_distance_matrix(ktn: KineticTransitionNetwork, similarity: StandardSimilarity, coords: StandardCoordinates) -> NDArray:
    """ Compute a distance matrix for all minima in the network """
    dist_matrix = np.zeros((ktn.n_minima, ktn.n_minima), dtype=float)
    for i in range(ktn.n_minima-1):
        coords.position = ktn.get_minimum_coords(i)
        for j in range(i+1, ktn.n_minima):
            coords2 = ktn.get_minimum_coords(j)
            dist_matrix[i, j] = similarity.closest_distance(coords, coords2)
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix


def get_distance_from_minimum(ktn: KineticTransitionNetwork, similarity: StandardSimilarity, coords: StandardCoordinates,
                              node: int) -> NDArray:
    """ Compute the distance of all nodes from specified minimum node1 """
    dist_vector = np.zeros((ktn.n_minima), dtype=float)
    coords.position = ktn.get_minimum_coords(node)
    for i in range(ktn.n_minima):
        coords2 = ktn.get_minimum_coords(i)
        dist_vector[i] = similarity.closest_distance(coords, coords2)
    return dist_vector

def validate_minima(ktn: KineticTransitionNetwork, model_data: ModelData, coords: StandardCoordinates, interpolation: Potential) -> None:
    logger = logging.getLogger("minima_validation")

    for i in range(ktn.n_minima):
        min_coords = ktn.get_minimum_coords(i)
        min_energy = ktn.get_minimum_energy(i)
        x,f,d = fmin_l_bfgs_b(func=interpolation.function_gradient,
                                x0=min_coords,
                                factr=1e-30,
                                bounds=coords.bounds,
                                pgtol=1e-3)
        logger.debug("Evaluating minimum index %i", i)
        logger.debug("From basin hopping: f = %e, x = %s", min_energy, min_coords)
        logger.debug("From lbfgs: f = %e, x = %s, d = %s", f, x, d)
        
        assert_allclose(min_coords, x, atol=1e-3, err_msg="Minimum coords from basin hopping do not match values from l-bfgs-b", verbose=True)
        assert_allclose(min_energy, f, atol=1e-3, err_msg="Minimum energy from basin hopping does not match value from l-bfgs-b", verbose=True)
        grad = interpolation.gradient(min_coords)
        f_plus = interpolation.function(np.clip(x + 1e-3, 0, 1))
        f_minus = interpolation.function(np.clip(x - 1e-3, 0, 1))
        logger.debug("f_plus: %f, f_minux: %f", f_plus, f_minus)

        for x_j, l_bound, u_bound, grad_j in zip (min_coords, coords.lower_bounds, coords.upper_bounds, grad):
            if x_j != l_bound and x_j != u_bound:
                assert_allclose(grad_j, 0, atol=1e-3, err_msg=f"Non-zero gradient {grad} at {min_coords}")
                assert_array_less(f, f_plus, err_msg="f_plus is less than f", verbose=True)
                assert_array_less(f, f_minus, err_msg="f_minus is less than f", verbose=True)