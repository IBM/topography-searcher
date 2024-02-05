""" Module contains the functions for analysing the properties of individual
    minima. All methods loop through and find minima that meet, or fail,
    specified criteria """

import numpy as np
from nptyping import NDArray


def get_invalid_minima(ktn: type, potential: type, coords: type) -> list:
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


def get_bounds_minima(ktn: type, coords: type) -> list:
    """ Find any minima that are at the bounds in any dimension """
    bounds_minima = []
    #  Loop over all current minima
    for i in range(ktn.n_minima):
        min_position = ktn.get_minimum_coords(i)
        coords.position = min_position
        if coords.at_bounds():
            bounds_minima.append(i)
    return bounds_minima


def get_all_bounds_minima(ktn: type, coords: type) -> list:
    """ Find any minima that are at the bounds in all dimensions """
    bounds_minima = []
    #  Loop over all current minima
    for i in range(ktn.n_minima):
        min_position = ktn.get_minimum_coords(i)
        coords.position = min_position
        if coords.all_bounds():
            bounds_minima.append(i)
    return bounds_minima


def get_similar_minima(ktn: type,
                       proximity_measure: float,
                       comparison_points: NDArray) -> list:
    """ Locate any minima with proximity_measure of the comparison_points """
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


def get_minima_above_cutoff(ktn: type,
                            cutoff: float) -> list:
    """ Find all minima with an energy above cutoff """

    energies = get_minima_energies(ktn)
    high_energy_minima = energies > cutoff
    return np.where(high_energy_minima)[0].tolist()


def get_minima_energies(ktn: type) -> NDArray:
    """ Return the energies of all the minima """
    energies = np.zeros((ktn.n_minima), dtype=float)
    for i in range(ktn.n_minima):
        energies[i] = ktn.get_minimum_energy(i)
    return energies


def get_ordered_minima(ktn: type) -> NDArray:
    """ Return the ordered list of indices with increasing energy """
    energies = get_minima_energies(ktn)
    return np.argsort(energies)


def get_distance_matrix(ktn: type, similarity: type, coords: type) -> NDArray:
    """ Compute a distance matrix for all minima in the network """
    dist_matrix = np.zeros((ktn.n_minima, ktn.n_minima), dtype=float)
    for i in range(ktn.n_minima-1):
        coords.position = ktn.get_minimum_coords(i)
        for j in range(i+1, ktn.n_minima):
            coords2 = ktn.get_minimum_coords(j)
            dist_matrix[i, j] = similarity.closest_distance(coords, coords2)
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix


def get_distance_from_minimum(ktn: type, similarity: type, coords: type,
                              node: int) -> NDArray:
    """ Compute the distance of all nodes from specified minimum node1 """
    dist_vector = np.zeros((ktn.n_minima), dtype=float)
    coords.position = ktn.get_minimum_coords(node)
    for i in range(ktn.n_minima):
        coords2 = ktn.get_minimum_coords(i)
        dist_vector[i] = similarity.closest_distance(coords, coords2)
    return dist_vector
