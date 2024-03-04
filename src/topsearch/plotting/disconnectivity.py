""" Module that contains the functions to plot disconnectivity graphs from
    a network of minima and transition states. """

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from matplotlib import collections as mc
import pylab as pl
from ..analysis.graph_properties import remove_edges_threshold
from ..analysis.minima_properties import get_minima_energies

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams.update({'font.size': 18})


def get_connectivity_graph(ktn: type, start: float, finish: float,
                           levels: int) -> nx.Graph:
    """ Generate the connectivity graph for splitting the space. This
        is a network containing the subsets of minima that are connected
        as we progressively remove edges with a lower value """
    # Initialise the network
    connectivity_graph = nx.Graph()
    # Copy the original landscape network to operate on
    H = ktn.G.copy()
    node_count = 0
    spacing = (start - finish)/levels
    # Do first level separately
    H = remove_edges_threshold(H, start)
    # Find the separate subsets after removing edges
    for j in nx.connected_components(H):
        connectivity_graph.add_node(node_count, level=0, members=j)
        node_count += 1
    # Then loop over the rest of the levels
    for i in range(1, levels+1):
        # Decrease the cutoff value
        cutoff = start - (i*spacing)
        H = remove_edges_threshold(H, cutoff)
        # Find the connected subsets after edge removal
        for j in nx.connected_components(H):
            # Add each separate subset with their corresponding level
            connectivity_graph.add_node(node_count, level=i, members=j)
            # Locate which subset they came from in previous level
            parent = find_parent(connectivity_graph, list(j)[0], i-1)
            # Connect each to their parent node
            connectivity_graph.add_edge(parent, node_count)
            node_count += 1
    return connectivity_graph


def find_parent(H: nx.Graph, member: set, level: int) -> int:
    """ Find the parent node for the subset containing the member minimum """
    level_nodes = [x for x, y in H.nodes(data=True) if y['level'] == level]
    # Loop through all nodes at the level to find which contains member
    for i in level_nodes:
        members = H.nodes[i]['members']
        if member in members:
            return i
    return None


def get_line_collection(conn_graph: nx.Graph, start: float,
                        finish: float, levels: int) -> list:
    """ Produce the line collection from the connectivity graph.
        Specifies the width in x of each subset and uses this to
        place lines connected to their parents in the x axis """
    # Compute the x-range of the first level separately
    level_nodes = \
        [x for x, y in conn_graph.nodes(data=True) if y['level'] == 0]
    min_x = 1.0
    for j in level_nodes:
        max_x = min_x + len(conn_graph.nodes[j]['members'])
        conn_graph.nodes[j]['width'] = [min_x, max_x]
        min_x = max_x
    # Initialise
    lines = []
    spacing = (start - finish)/levels
    # Loop over all subsequent levels
    for i in range(1, levels+1):
        # Get nodes from previous and current level
        prev_level_nodes = \
            [x for x, y in conn_graph.nodes(data=True) if y['level'] == i-1]
        level_nodes = \
            [x for x, y in conn_graph.nodes(data=True) if y['level'] == i]
        # Start from the level above and compute what it split into
        for j in prev_level_nodes:
            width = conn_graph.nodes[j]['width']
            x1 = ((width[1] - width[0]) * 0.5) + width[0]
            y1 = start - (i-1)*spacing
            # Find the connections in the layer below
            neighbours = []
            for k in conn_graph.edges(j):
                if k[0] == j:
                    neighbours.append(k[1])
                else:
                    neighbours.append(k[0])
            neighbours = [n for n in neighbours if n in level_nodes]
            min_x = conn_graph.nodes[j]['width'][0]
            for k in neighbours:
                max_x = min_x + len(conn_graph.nodes[k]['members'])
                conn_graph.nodes[k]['width'] = [min_x, max_x]
                min_x = max_x
                width_n = conn_graph.nodes[k]['width']
                x2 = ((width_n[1] - width_n[0]) * 0.5) + width_n[0]
                y2 = start - (i*spacing)
                lines.append([(x1, y1), (x2, y2)])
    return lines


def cut_line_collection(ktn: type, conn_graph: nx.Graph, lines: list,
                        start: float, finish: float, levels: int) -> list:
    """ Take the lines that currently run to the lowest known energy and
        shorten each node to its corresponding energy when there is only
        one member in the subset """
    step = (start - finish)/levels
    # Final set of single occupancy nodes
    final_level =  \
        [x for x, y in conn_graph.nodes(data=True) if y['level'] == levels]
    # Initialise the store of which nodes to remove
    remove_lines = []
    # Go over each node
    for i in final_level:
        # Work out its x position
        x_width = conn_graph.nodes[i]['width']
        x_value = 0.5*(x_width[1] - x_width[0]) + x_width[0]
        # And where the line should stop in y
        member = list(conn_graph.nodes[i]['members'])[0]
        energy = ktn.G.nodes[member]['energy']
        # Calculate the highest y before it merges with other nodes
        # and add a replacement line starting from the right value
        max_y = -1e30
        y_values = []
        # Get all the lines y-values corresponding to the given x_value
        for count, j in enumerate(lines, 0):
            if j[0][0] == x_value and j[1][0] == x_value:
                y_values.append(j[1][1])
                remove_lines.append(count)
        # Account for lines merging to same x at lower y
        diffs = np.abs(np.diff(np.asarray(y_values)))
        idxs = np.where(diffs > 1.5*step)[0]
        # If splits in the y values, then make separate lines
        if np.any(idxs):
            min_y = energy
            for i in np.flip(idxs):
                max_y = y_values[i+1]
                lines.append([(x_value, min_y), (x_value, max_y+step)])
                min_y = y_values[i]
            lines.append([(x_value, min_y), (x_value, y_values[0]+step)])
        # Just a single line so connect both ends directly
        else:
            max_y = y_values[0]
            lines.append([(x_value, energy), (x_value, max_y+step)])
    # Remove the unneccesary lines
    for i in sorted(remove_lines, reverse=True):
        del lines[i]
    return lines


def plot_disconnectivity_graph(ktn: type, levels: int,
                               label: str = '') -> None:
    """ Compute the disconnectivity graph, plot and save to file """
    # Find the highest energy transition state as upper limit
    ts_energies = []
    for u, v in ktn.G.edges():
        ts_energies.append(ktn.get_ts_energy(u, v))
    if ts_energies:
        highest = np.max(ts_energies)
    else:
        highest = np.max(get_minima_energies(ktn))
    # Find the lowest minimum as lower limit
    lowest = np.min(get_minima_energies(ktn))
    # Add a little extra in y to space out plot
    start = highest + 0.05*(highest-lowest)
    finish = lowest - 0.1*(highest-lowest)
    # Get the connectivity graph for the subsets as they split
    c_graph = get_connectivity_graph(ktn, start, finish, levels)
    # Get the corresponding lines from this connectivity graph
    lines = get_line_collection(c_graph, start, finish, levels)
    # Edit these lines to have the correct length
    lines = cut_line_collection(ktn, c_graph, lines, start, finish, levels)
    fig, axes = pl.subplots()
    axes.get_xaxis().set_visible(False)
    axes.add_collection(mc.LineCollection(lines, linewidths=1))
    axes.autoscale()
    axes.margins(0.1)
    plt.savefig(f"DisconnectivityGraph{label}.png", dpi=300)
    plt.cla()
    plt.clf()
    plt.close()
