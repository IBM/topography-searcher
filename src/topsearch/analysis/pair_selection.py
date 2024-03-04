""" Routines that act on a kinetic transition network to select pairs of minima
   for sampling by different criteria """

import numpy as np
import networkx as nx
from .graph_properties import unconnected_component
from .minima_properties import get_distance_matrix, get_distance_from_minimum


def connect_unconnected(ktn: type, similarity: type,
                        coords: type, neighbours: int) -> list:
    """
    Find all minima not connected to global minimum set and their nearest
    cycles minima. Run connection attempts for each minima pair
    """
    # Check for emptiness
    if ktn.n_minima == 0:
        return []
    # Get the set of minima not connected to the global minimum
    unconnected_set = unconnected_component(ktn)
    total_pairs = []
    if len(unconnected_set) > 0:
        for i in unconnected_set:
            pairs = connect_to_set(ktn, similarity, coords, i, neighbours)
            for j in pairs:
                total_pairs.append(j)
    # Can generate a lot of repeats so remove any repeated pairs
    total_pairs = unique_pairs(total_pairs)
    return total_pairs


def connect_to_set(ktn: type, similarity: type, coords: type,
                   node1: int, cycles: int) -> list:
    """
    Finds all minima connected to node1 and finds the pairs closest
    in distance where one is connected and one is not. Returns the
    set of minima pairs in a list for use in connect_unconnected
    """

    # Set of nodes connected to node1
    s_set = nx.node_connected_component(ktn.G, node1)
    # Find list of nodes not connected to node 1
    f_set = set(range(ktn.n_minima)) - set(s_set)
    if f_set == set():
        with open('logfile', 'a', encoding="utf-8") as outfile:
            outfile.write("No unconnected minima\n")
        return []
    # Get ordered distances to this node
    dist_vector = get_distance_from_minimum(ktn, similarity, coords, node1)
    nearest = np.argsort(dist_vector).tolist()[1:]
    # Remove any minima in same set
    pairs = [i for i in nearest if i in f_set]
    total_pairs = []
    for i in pairs[:cycles]:
        total_pairs.append([node1, i])
    return unique_pairs(total_pairs)


def closest_enumeration(ktn: type, similarity: type,
                        coords: type, neighbours: int) -> list:
    """
    Selector that attempts to connect all minima in the fewest number of
    attempts by connecting each minimum to its N nearest neighbours
    Generates a list of pairs and runs connection attempts for all
    """
    pairs = []
    dist_matrix = get_distance_matrix(ktn, similarity, coords)
    for i in range(ktn.n_minima):
        nearest = np.argsort(dist_matrix[i, :]).tolist()[1:neighbours+1]
        for j in nearest:
            pairs.append([i, j])
    return unique_pairs(pairs)


def read_pairs(text_path: str = ''):
    """ Pair selection by reading the information from file pairs.txt """
    pairs = np.genfromtxt(f'{text_path}pairs.txt', dtype=int)
    return unique_pairs(pairs.tolist())


def unique_pairs(initial_pairs: list) -> list:
    """ Remove any repeated pairs from a given list """
    # Sort the pairs as [0, 1] and [1, 0] are equivalent
    final_pairs = [tuple(sorted(i)) for i in initial_pairs if i != [0, 0]]
    return [list(i) for i in set(final_pairs)]
