""" Functions to compute the roughness of a given network.
    https://doi.org/10.26434/chemrxiv-2023-0zx26 """

import numpy as np
from .graph_properties import get_connections


def roughness_metric(ktn: type, lengthscale: float = 0.8) -> float:
    """ Compute the roughness metric for the current kinetic
        transition network """
    lengthscale = lengthscale**2
    # Roughness is zero for one minimum or empty network
    if ktn.n_minima in (0, 1):
        return 0.0
    frustration = 0.0
    # Loop over all minima in the set
    for i in range(ktn.n_minima):
        # Get the minimum energy
        min_energy = ktn.get_minimum_energy(i)
        # Find the directly connected transition states
        connections = get_connections(ktn, i)
        # Compute the barrier to each of these transition states
        for ts in connections:
            ts_energy = ktn.get_ts_energy(i, ts)
            barrier = ts_energy - min_energy
            if ts_energy < min_energy:
                barrier = 0.0
            population = get_population(ktn, i, ts, lengthscale)
            frustration += (population*barrier)
    frustration /= ktn.n_minima
    return frustration


def get_population(ktn: type, min_node: int, ts_node: int,
                   lengthscale: float) -> float:
    """ Compute the population of a given minimum in the network.
        Population is the value of the RBF kernel at the separation
        between minimum and transition state """
    min_coords = ktn.get_minimum_coords(min_node)
    ts_coords = ktn.get_ts_coords(min_node, ts_node)
    dist = np.linalg.norm(min_coords - ts_coords)
    population = np.exp(-1.0*((dist**2)/(2.0*lengthscale)))
    return population
