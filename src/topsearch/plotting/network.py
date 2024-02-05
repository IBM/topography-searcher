""" Functions for plotting the kinetic transition network as a weighted
    graph. Nodes are minima (coloured by their function value), edges
    are present when transition states connect minima. The separation
    is inversely proportional to barrier height """

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import numpy as np
import networkx as nx

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams.update({'font.size': 18})


def plot_network(ktn: type, label: str = '',
                 colour_scheme: str = 'cool') -> None:
    """ Plot the network using a weighted spring layout with larger
        barriers giving a larger separation between nodes """

    g_weighted = barrier_reweighting(ktn)
    pos = nx.spring_layout(g_weighted)
    colours = np.empty((0))
    for i in ktn.G.nodes:
        colours = np.append(colours, ktn.G.nodes[i]['energy'])
    network_contours = nx.draw_networkx_nodes(
        ktn.G, pos, node_color=colours,
        cmap=plt.get_cmap(colour_scheme))
    nx.draw_networkx_edges(ktn.G, pos)
    plt.colorbar(network_contours)
    plt.tight_layout()
    plt.savefig(f"Network{label}.png", dpi=300)
    plt.cla()
    plt.clf()
    plt.close()


def barrier_reweighting(ktn: type) -> nx.Graph:
    """ Calculate appropriate weighting for spring constants in the
        graph layout from barriers to ensure inversely proportional
        to barrier height. Returns reweighted network """

    g_weighted = nx.create_empty_copy(ktn.G, with_data=True)
    for node1, node2 in ktn.G.edges:
        energy1 = ktn.get_minimum_energy(node1)
        energy2 = ktn.get_minimum_energy(node2)
        energy_ts = ktn.get_ts_energy(node1, node2)
        min_barrier = float(min((energy_ts-energy1), (energy_ts-energy2)))
        if min_barrier < 0.0:
            min_barrier = 1e-5
        # Take inverse so minima separated by large barriers are far apart
        min_barrier = 1.0/min_barrier
        g_weighted.add_edge(node1, node2, weight=min_barrier)
    return g_weighted
