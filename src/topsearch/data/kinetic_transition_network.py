""" Module which stores all the KineticTransitionNetwork class with
    methods for storing and extracting stationary point information
    encoded as a network graph """

import networkx as nx
import numpy as np
from nptyping import NDArray


class KineticTransitionNetwork:

    """
    Description
    -------------
    A class to keep track of the network of minima and transition states
    Stationary points are stored in a networkx graph. All addition, removal
    and accessing of properties is performed through this class. Furthermore,
    provides functionality for computing different network properties

    Attributes
    -----------

    G : networkx graph
        The network object that contains all the minima (nodes) and transition
        states (edges between directly connected minima)
    n_minima : int
        Number of minima in the network
    n_ts : int
        Number of transition states in the network
    pairlist : numpy array
        Stores the pairs of minima between which connections have already been
        attempted. Useful to avoid repetition of calculations
    similarity : class instance
        Evalutes the similarity between any two given configurations
    """

    def __init__(self) -> None:
        self.G = nx.Graph()
        self.n_minima = 0
        self.n_ts = 0
        self.pairlist = np.empty((0, 2), dtype=int)
        # Make the logfile empty before writing
        with open('logfile', 'w', encoding="utf-8") as outfile:
            outfile.write(' ')

    def get_minimum_coords(self, minimum: int) -> None:
        """ Returns the coordinates of a given node minimum """
        return self.G.nodes[minimum]['coords']

    def get_minimum_energy(self, minimum: int) -> None:
        """ Returns the energy of a given node minimum """
        return self.G.nodes[minimum]['energy']

    def get_ts_coords(self, min_plus: int, min_minus: int) -> None:
        """ Returns coordinates of ts edge between min_plus and min_minus """
        return self.G[min_plus][min_minus]['coords']

    def get_ts_energy(self, min_plus: int, min_minus: int) -> None:
        """ Returns energy of ts edge between min_plus and min_minus """
        return self.G[min_plus][min_minus]['energy']

    def add_minimum(self, min_coords: NDArray, energy: float) -> None:
        """ Add a node to the network with data for the minimum """
        self.G.add_node(self.n_minima, coords=np.array(
            min_coords, copy=True), energy=energy)
        self.n_minima += 1

    def add_ts(self, ts_coords: NDArray, energy: float,
               min_plus: NDArray, min_minus: NDArray) -> None:
        """ Add an edge to the network with the transition state data """
        self.G.add_edge(min_plus, min_minus,
                        coords=np.array(ts_coords, copy=True),
                        energy=energy)
        self.n_ts += 1

    def remove_minimum(self, minimum: int) -> None:
        """ Remove a node from the network correpsonding to index minimum """
        new_order = np.arange(self.n_minima)
        new_order[minimum] = self.n_minima+1
        for i in range(minimum, self.n_minima):
            new_order[i] -= 1
        mapping = dict(zip(np.arange(self.n_minima), new_order))
        self.G = nx.relabel_nodes(self.G, mapping, copy=True)
        # Will remove any edges connected to this node so reduce n_ts
        self.n_ts -= len(self.G.edges(self.n_minima))
        self.G.remove_node(self.n_minima)
        self.n_minima -= 1

    def remove_minima(self, minima: NDArray) -> None:
        """ Remove the nodes with the given indices in removed_minima """
        for c, i in enumerate(np.sort(minima), 0):
            self.remove_minimum(i-c)

    def remove_ts(self, minimum1: int, minimum2: int) -> None:
        """ Remove the transition state connected the two passed minima """
        self.G.remove_edge(minimum1, minimum2)
        self.n_ts -= 1

    def remove_tss(self, minima: list) -> None:
        """ Remove an array of transition states in one go """
        for i in minima:
            self.remove_ts(i[0], i[1])

    #  INPUT/OUTPUT FUNCTIONS

    def reset_network(self) -> None:
        """ Empty the network """
        self.G = nx.Graph()
        self.n_minima = 0
        self.n_ts = 0
        self.pairlist = np.empty((0, 2), dtype=int)

    def dump_network(self, text_string: str = '') -> None:
        """
        Write network to text files:
        *.data store the index and energy
        ts.data also includes the connectivity
        min/ts.coords store the coordinates of each stationary point
        """
        # Get dimensionality of minima
        ndim = self.get_minimum_coords(0).shape[0]
        # Get minima data out of network
        minima_data = np.empty((0, 2), dtype=object)
        minima_coords = np.empty((0, ndim), dtype=object)
        for i in range(self.n_minima):
            minima_data = np.append(
                minima_data, [[i, self.G.nodes[i]['energy']]], axis=0)
            minima_coords = np.append(
                minima_coords, [self.G.nodes[i]['coords']], axis=0)
        # Get transition state data out of the network
        ts_data = np.empty((0, 3), dtype=object)
        ts_coords = np.empty((0, ndim), dtype=object)
        for node1, node2 in self.G.edges():
            ts_data = np.append(
                ts_data,
                [[node1, node2, self.G[node1][node2]['energy']]], axis=0)
            ts_coords = np.append(
                ts_coords, [self.G[node1][node2]['coords']], axis=0)
        # Write stationary point data and pairlist
        np.savetxt(f"ts.data{text_string}",
                   ts_data, fmt='%i %i %8.5f')
        np.savetxt(f"ts.coords{text_string}", ts_coords)
        np.savetxt(f"min.data{text_string}",
                   minima_data, fmt='%i %8.5f')
        np.savetxt(f"min.coords{text_string}", minima_coords)
        np.savetxt(f"pairlist{text_string}", self.pairlist, fmt='%i')

    def read_network(self, text_path: str = '', text_string: str = '') -> None:
        """ Returns G network from files that resulted from dump_network """

        # Get the data back from files
        minima_data = np.loadtxt(f"{text_path}min.data{text_string}")
        minima_coords = np.loadtxt(f"{text_path}min.coords{text_string}")
        ts_data = np.loadtxt(f"{text_path}ts.data{text_string}")
        ts_coords = np.loadtxt(f"{text_path}ts.coords{text_string}")
        self.pairlist = np.loadtxt(f"{text_path}pairlist{text_string}",
                                   ndmin=2, dtype=int)

        # Now build a graph from the data
        self.n_minima = np.size(minima_data, 0)
        for i in range(np.size(minima_data, 0)):
            self.G.add_node(int(minima_data[i, 0]),
                            energy=minima_data[i, 1],
                            coords=minima_coords[i, :])
        self.n_ts = np.size(ts_data, 0)
        for i in range(np.size(ts_data, 0)):
            self.G.add_edge(int(ts_data[i, 0]), int(ts_data[i, 1]),
                            energy=ts_data[i, 2], coords=ts_coords[i, :])

    def dump_minima_csv(self, text_string: str = '') -> None:
        """ Method to dump all the minima into a csv format """
        with open(f'mol_data{text_string}.csv',
                  'w', encoding="utf-8") as csv_file:
            csv_file.write('coords, energy\n')
            for i in range(self.n_minima):
                coords = self.get_minimum_coords(i)
                energy = self.get_minimum_energy(i)
                csv_file.write(f'{coords}, ')
                csv_file.write(f'{energy}\n')

    def add_network(self, other_ktn: type, similarity: type,
                    coords: type) -> None:
        """ Method to combine a second network with the current one.
            Compares all stationary points in other_ktn and adds any
            non-repeats to the current network """

        # Loop over all minima and test if new or not
        for i in range(other_ktn.n_minima):
            coords.position = other_ktn.get_minimum_coords(i)
            energy = other_ktn.get_minimum_energy(i)
            similarity.test_new_minimum(self, coords, energy)
        # Loop over all TSs and test if new or not
        for u, v in other_ktn.G.edges():
            coords.position = other_ktn.get_ts_coords(u, v)
            ts_energy = other_ktn.get_ts_energy(u, v)
            min1_coords = other_ktn.get_minimum_coords(u)
            min1_energy = other_ktn.get_minimum_energy(u)
            min2_coords = other_ktn.get_minimum_coords(v)
            min2_energy = other_ktn.get_minimum_energy(v)
            similarity.test_new_ts(self, coords, ts_energy,
                                   min1_coords, min1_energy,
                                   min2_coords, min2_energy)
        # Combine the pairlist files
        for i in other_ktn.pairlist:
            self.pairlist = np.append(
                self.pairlist, np.array([np.sort(i)]), axis=0)
