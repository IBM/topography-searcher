""" Methods for analysing the network properties of the kinetic transition
    network graphs """

import networkx as nx
import numpy as np
from .minima_properties import get_minima_energies


def unconnected_component(ktn: type) -> set:
    """ Check which minima are not connected to the global minimum
        Return the set of minima that are unconnected """

    # Get global minimum node
    energies = get_minima_energies(ktn)
    min_node = np.argmin(energies)
    # Use networkx to find those minima not connected to it
    connected_set = nx.node_connected_component(ktn.G, min_node)
    unconnected_set = set(range(ktn.n_minima)) - set(connected_set)
    return unconnected_set


def are_nodes_connected(ktn: type, node_i: int, node_j: int) -> bool:
    """ Check if two minima are connected in a given graph """
    return bool(node_j in nx.node_connected_component(ktn.G, node_i))


def all_minima_connected(ktn: type) -> bool:
    """ Wrapper for checking if the graph is fully connected """
    return nx.is_connected(ktn.G)


def get_connections(ktn: type, min_node: int) -> list:
    """ Find the transition states directly connected to a given node """
    transition_states = []
    for i in ktn.G.edges(min_node):
        if i[0] == min_node:
            transition_states.append(i[1])
        else:
            transition_states.append(i[0])
    return transition_states


def disconnected_height(ktn: type, node_i: int, node_j: int,
                        max_ts_energy: float, e_range: float) -> float:
    """ Return the height at which two nodes become disconnected """
    # Create a graph copy
    H = ktn.G.copy()
    # Check if the minima are initially connected
    if not bool(node_j in nx.node_connected_component(H, node_i)):
        return 1e10
    # Loop downwards in e_range/intervals steps removing edges
    # until minima are disconnected
    intervals = 510
    initial_energy = max_ts_energy+(10*(e_range/intervals))
    for k in range(intervals+20):
        energy = initial_energy-(k*(e_range/intervals))
        H = remove_edges_threshold(H, energy)
        if not bool(node_j in nx.node_connected_component(H, node_i)):
            return energy
    return 1e10


def remove_edges_threshold(H: nx.Graph, energy1: float) -> nx.Graph:
    """ Remove any transition states that have an energy above energy1 """
    for u, v in H.edges():
        ts_energy = H[u][v]['energy']
        if ts_energy > energy1:
            H.remove_edge(u, v)
    return H
