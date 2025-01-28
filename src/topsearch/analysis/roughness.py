""" Functions to compute the roughness of a given network, as described in
    https://doi.org/10.1039/D3ME00189J """

from typing import List
import numpy as np

from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from .graph_properties import get_connections


class RoughnessContribution:
    minimum: np.ndarray
    ts: np.ndarray
    features: List[int]

    def __init__(self, minimum: np.ndarray, ts: np.ndarray, frustration: float, features: List[int]) -> None:
        self.minimum = minimum
        self.ts = ts
        self.frustration = frustration
        self.features = features

def roughness_metric(ktn: KineticTransitionNetwork, lengthscale: float = 0.8) -> float:
    """ Compute the roughness metric for the current kinetic
    transition network """
    # Roughness is zero for one minimum or empty network
    if ktn.n_minima in (0, 1):
        return 0.0
    contributors = roughness_contributors(ktn, lengthscale)
    total_frustration = sum([contribution.frustration for contribution in contributors])
    return total_frustration / ktn.n_minima

def roughness_contributors(ktn: KineticTransitionNetwork, lengthscale: float = 0.8, features: List[int] = []) -> list[RoughnessContribution]:
    """ Compute the roughness metric for the current kinetic
        transition network, and return all the contributing transition states with their frustration values 
        
        Returns: a list of frustration contributions
        """
    lengthscale = lengthscale**2

    contributors = []
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
            contribution = RoughnessContribution(ktn.get_minimum_coords(i), ktn.get_ts_coords(i, ts), population*barrier, features)
            contributors.append(contribution)

    contributors.sort(key=lambda contributor: contributor.frustration, reverse=True)

    return contributors    

def get_population(ktn: KineticTransitionNetwork, min_node: int, ts_node: int,
                   lengthscale: float) -> float:
    """ Compute the population of a given minimum in the network.
        Population is the value of the RBF kernel at the separation
        between minimum and transition state """
    min_coords = ktn.get_minimum_coords(min_node)
    ts_coords = ktn.get_ts_coords(min_node, ts_node)
    dist = np.linalg.norm(min_coords - ts_coords)
    population = np.exp(-1.0*((dist**2)/(2.0*lengthscale)))
    return population
