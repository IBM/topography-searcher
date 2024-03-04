""" Module containing batch selection methods for Bayesian optimisation.
    The methods take in a network encoding of the surface topography,
    and analyse it to select a diverse set of minima. """

import numpy as np
from nptyping import NDArray
from .graph_properties import disconnected_height
from .minima_properties import get_ordered_minima, get_minima_energies, \
                               get_bounds_minima, get_similar_minima, \
                               get_minima_above_cutoff


def get_excluded_minima(ktn: type,
                        energy_cutoff: float = 1.0,
                        penalise_edge: bool = False,
                        coords: type = None,
                        penalise_similarity: bool = False,
                        proximity_measure: float = 0.2,
                        known_points: NDArray = None) -> list:
    """ Apply specified criteria to decide if the minima should
        be allowed in the BayesOpt batch. Returns the list of those
        that should be excluded """

    energies = get_minima_energies(ktn)
    # Get the cutoffs for allowed minima
    abs_energy_cutoff = np.min(energies) + \
        energy_cutoff*(np.max(energies) - np.min(energies))
    excluded_minima = get_minima_above_cutoff(ktn, abs_energy_cutoff)
    if penalise_edge:
        edge_minima = get_bounds_minima(ktn, coords)
        excluded_minima += edge_minima
    if penalise_similarity:
        similar_minima = get_similar_minima(ktn, proximity_measure,
                                            known_points)
        excluded_minima += similar_minima
    # Retain unique elements
    return list(set(excluded_minima))


def select_batch(ktn: type, batch_size: int, batch_selection_method: str,
                 fixed_batch_size: bool, barrier_cutoff: float = 1.0,
                 excluded_minima: list = None) -> tuple[list, NDArray]:

    """
    Method to select the batch of points from a given network
    Returns the indices of the minima and their coordinates
    Type of batch selection depends on batch_selection_method
    Lowest - Select the batch_size lowest minima in value
    Monotonic - Select monotonic sequence basins of the network
    Barrier - Select minima that are separated by sufficiently large
                barriers counting from lowest in acquisition
    Topographical - First perform Monotonic, then fill batch using Barrier
    """

    if excluded_minima is None:
        excluded_minima = []
    # Get the value of acquisition function for each minimum
    energies = get_minima_energies(ktn)
    # Get the cutoffs for allowed minima
    abs_barrier_cutoff = barrier_cutoff * \
        (np.max(energies) - np.min(energies))
    # Write the batch scheme we are using
    with open('logfile', 'a', encoding="utf-8") as outfile:
        outfile.write("------- BATCH SELECTION ---------\n")
        outfile.write(f"{batch_selection_method}: {batch_size} minima\n")
    # Perform the batch selection scheme
    batch_indices = generate_batch(ktn, batch_selection_method,
                                   excluded_minima,
                                   abs_barrier_cutoff)
    # If we want a fixed batch size fill with lowest minima not in batch
    if fixed_batch_size and (len(batch_indices) < batch_size):
        batch_indices = fill_batch(ktn, batch_indices, excluded_minima)
    # Limit the batch to batch_size if more have been returned
    if np.size(batch_indices) > batch_size:
        batch_indices = batch_indices[:batch_size]
    batch_points = get_batch_positions(ktn, batch_indices)
    return batch_indices, batch_points


def generate_batch(ktn: type, batch_selection_method: str,
                   excluded_minima: list,
                   absolute_barrier_cutoff: float) -> NDArray:
    """ Generate the batch given the specified method """
    if batch_selection_method == "Lowest":
        batch_indices = lowest_batch_selector(ktn, excluded_minima)
    elif batch_selection_method == "Monotonic":
        batch_indices = monotonic_batch_selector(ktn, excluded_minima)
    elif batch_selection_method == "Barrier":
        batch_indices = barrier_batch_selector(ktn, excluded_minima,
                                               absolute_barrier_cutoff)
    elif batch_selection_method == "Topographical":
        batch_indices = topographical_batch_selector(ktn, excluded_minima,
                                                     absolute_barrier_cutoff)
    return batch_indices


def lowest_batch_selector(ktn: type, excluded_minima: list) -> list:
    """ Simply select the lowest minima of the acquisition function
        Returns list of indices the selected minima correspond to """
    # Starting from the lowest indices progressively add if not banned
    ordered_indices = get_ordered_minima(ktn).tolist()
    batch_indices = [i for i in ordered_indices if i not in excluded_minima]
    return batch_indices


def fill_batch(ktn: type, batch_indices: list,
               excluded_minima: list) -> NDArray:
    """ Return the batch after adding the remaining
        lowest minima in order """
    # Use lowest_batch_selector to fill in with lowest minima
    lowest_indices = lowest_batch_selector(ktn, excluded_minima+batch_indices)
    # And combine the two batches
    batch_indices += lowest_indices
    return batch_indices


def topographical_batch_selector(ktn: type,
                                 excluded_minima: list,
                                 absolute_barrier_cutoff: float) -> NDArray:
    """ Select a batch of minima based on the surface topography
        First finds monotonic sequence basins, and if not batch_size
        of them then use barrier_batch_selector to generate remaining
        Returns the indices of the selected nodes, and their
        corresponding coordinates """

    # Get the monotonic batch
    monotonic_indices = monotonic_batch_selector(ktn, excluded_minima)
    with open('logfile', 'a', encoding="utf-8") as outfile:
        outfile.write(f"Got {np.size(monotonic_indices)} "
                      f"minima from monotonic sequence basins\n")
    # Get the barrier batch
    barrier_indices = barrier_batch_selector(ktn,
                                             excluded_minima,
                                             absolute_barrier_cutoff,
                                             monotonic_indices)
    barrier_indices = \
        [i for i in barrier_indices if i not in monotonic_indices]
    return monotonic_indices + barrier_indices


def barrier_batch_selector(ktn: type,
                           excluded_minima: list,
                           absolute_barrier_cutoff: float,
                           current_batch_indices: list = None):
    """ Select a batch of points that are all separated by barrier heights
        greater than absolute_barrier_cutoff. Returns the indices """

    if current_batch_indices is None:
        current_batch_indices = []
    batch_indices = []
    # Find maximum ts energy for barrier calc
    ts_energies = []
    for u, v in ktn.G.edges():
        ts_energies.append(ktn.get_ts_energy(u, v))
    if not ts_energies:
        max_ts_energy = 1e5
    else:
        max_ts_energy = np.max(ts_energies)
    # Find energy range for barrier calc
    energies = get_minima_energies(ktn)
    e_range = np.max(energies) - np.min(energies)
    # Loop over all minima in network
    ordered_minima_indices = get_ordered_minima(ktn)
    for i in ordered_minima_indices:
        # Minimum is already in batch or banned so ignore
        if (i in current_batch_indices) or (i in excluded_minima):
            continue
        # Loop over current batch and check if allowed by barrier heights
        allowed = True
        for j in current_batch_indices:
            accept = sufficient_barrier(ktn, i, j, max_ts_energy, e_range,
                                        absolute_barrier_cutoff)
            # If conditions fail compared to any of current batch - reject
            if not accept:
                allowed = False
        # If the minimum has passed all checks then add to batch
        if allowed:
            # Add to batches
            batch_indices.append(i)
            current_batch_indices.append(i)
    return batch_indices


def sufficient_barrier(ktn: type, node_i: int, node_j: int,
                       max_ts_energy: float, e_range: float,
                       absolute_barrier_cutoff: float) -> bool:
    """ Check if minima pass criteria to be added to the batch
        Returns logical specified if pass or fail """
    height = disconnected_height(ktn, node_i, node_j, max_ts_energy, e_range)
    if height > 1e9:
        return False
    # Barriers, calculated in either way, from both minima
    barrier1 = height - ktn.get_minimum_energy(node_i)
    barrier2 = height - ktn.get_minimum_energy(node_j)
    # If barrier less than minimum allowed then reject
    if min(barrier1, barrier2) < absolute_barrier_cutoff:
        return False
    return True


def monotonic_batch_selector(ktn: type, excluded_minima: list) -> list:
    """ Find all monotonic sequence basins when excluding excluded_minima,
        and return their indices in increasing function value """

    batch_indices = []
    # Loop from lowest to highest energy
    ordered_indices = get_ordered_minima(ktn)
    for i in ordered_indices:
        monotonic = True
        energy_i = ktn.get_minimum_energy(i)
        edges_i = ktn.G.edges(i)
        # If no edges then ignore the minimum
        if (not edges_i) or (i in excluded_minima):
            continue
        # Loop over all edges getting the two connected minima
        for j in edges_i:
            if j[0] == j[1]:
                continue
            if j[1] == i:
                j.reverse()
            min1, min2 = j
            if min2 in excluded_minima:
                continue
            # Find other connected minimum and if lower not monotonic
            if ktn.get_minimum_energy(min2) <= energy_i:
                monotonic = False
        # If not connected to any lower minima then add to MSB
        if monotonic:
            batch_indices.append(i)
    return batch_indices


def get_batch_positions(ktn: type, batch_indices: list) -> NDArray:
    """ Get the corresponding batch position from the list of indices """
    batch_points = np.zeros((len(batch_indices),
                             ktn.get_minimum_coords(0).size), dtype=float)
    for idx, i in enumerate(batch_indices, 0):
        batch_points[idx, :] = ktn.get_minimum_coords(i)
    return batch_points


def evaluate_batch(true_potential: type, batch_indices: list,
                   batch_points: NDArray) -> NDArray:
    """ Evaluate the true potential at each point in the given batch
        Return the training and corresponding response values """
    new_response = np.apply_along_axis(
        true_potential.function, 1, batch_points)
    # Write the batch to file
    for i in range(len(batch_indices)):
        with open('logfile', 'a', encoding="utf-8") as outfile:
            outfile.write(f"{batch_indices[i]} Coords: ")
            for j in range(np.size(batch_points[i])):
                outfile.write(f"{batch_points[i][j]} ")
            outfile.write(f" Function: {new_response[i]}\n")
    return new_response
