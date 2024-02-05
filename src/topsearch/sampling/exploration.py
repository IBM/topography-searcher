""" exploration module contains the algorithms for deciding on sampling
    of the function space to map the topography, and running different
    exploration algorithms """

from timeit import default_timer as timer
import multiprocessing
from copy import deepcopy
import numpy as np
from nptyping import NDArray
from ..analysis.pair_selection import connect_unconnected, \
    closest_enumeration, read_pairs
from ..analysis.minima_properties import get_invalid_minima, \
    get_bounds_minima, get_all_bounds_minima
from ..minimisation import lbfgs


class NetworkSampling:

    """
    Description
    -----------

    A class that can sample configuration space given a network of minima
    and transition states. Connections are attempted between minima to find
    intervening transition states. Sampling can be applied repeatedly to
    connect all minima by a series of transition states.

    Attributes
    -----------

    ktn : class instance
        Contains the network of all known minima and transition states, along
        with their properties and operations on the network. Decide where to
        sample next from network properties and add new stationary points
    coords : class instance
        The type of coordinates that are being using exploring space
    global_optimiser : class instance
        Global optimisation algorithm that locates low-valued minima of the
        given potential
    single_ended_search : class instance
        Single ended transition state search algorithms to locate transition
        states from a single given point
    double_ended_search : class instance
        Double ended transition state search algorithms to produce minimum
        energy paths between two minima that locate approximate transition
        states
    similarity : class instance
        The object that evaluates the similarity of two points in space
    multiprocessing_on : bool
        Specifies if exploration should be run in parallel
    n_processes : int
        The number of cores used if multiprocessing_on is True
    """

    def __init__(self,
                 ktn: type,
                 coords: type,
                 global_optimiser: type,
                 single_ended_search: type,
                 double_ended_search: type,
                 similarity: type,
                 multiprocessing_on: bool = False,
                 n_processes: int = None) -> None:
        self.ktn = ktn
        self.coords = coords
        self.global_optimiser = global_optimiser
        self.single_ended_search = single_ended_search
        self.double_ended_search = double_ended_search
        self.similarity = similarity
        self.multiprocessing_on = multiprocessing_on
        self.n_processes = n_processes

    # OVERALL LANDSCAPE EXPLORATION

    def get_minima(self, coords: type, n_steps: int, conv_crit: float,
                   temperature: float, test_valid: bool = True) -> None:
        """ Perform global optimisation to locate low-valued minima """
        self.global_optimiser.run(coords=coords, n_steps=n_steps,
                                  conv_crit=conv_crit, temperature=temperature)
        # After finishing basin-hopping remove any minima that are not allowed
        if test_valid:
            invalid_min = get_invalid_minima(self.ktn,
                                             self.global_optimiser.potential,
                                             coords)
            self.ktn.remove_minima(invalid_min)

    def get_transition_states(self, method: str, cycles: int,
                              remove_bounds_minima: bool = False,
                              all_bounds: bool = False) -> None:
        """ Default algorithm for generating a landscape
            Combines different sampling methods in sequence to produce
            a fully connected network.
            Updates the ktn.G network with any new transition states """

        # Remove any edge cases that are high in energy and not connected
        if remove_bounds_minima:
            if all_bounds:
                bounds_minima = get_all_bounds_minima(self.ktn, self.coords)
            else:
                bounds_minima = get_bounds_minima(self.ktn, self.coords)
            self.ktn.remove_minima(bounds_minima)
        # Run a set of initial connections for all minima
        pairs = self.select_minima(self.coords, method, cycles)
        self.run_connection_attempts(pairs)
        # Remove any additional bounds minima found during sampling
        if remove_bounds_minima:
            if all_bounds:
                bounds_minima = get_all_bounds_minima(self.ktn, self.coords)
            else:
                bounds_minima = get_bounds_minima(self.ktn, self.coords)
            self.ktn.remove_minima(bounds_minima)

    # CONNECTING MINIMA FUNCTIONS

    def run_connection_attempts(self, total_pairs: list) -> None:
        """
        Given the pairs of minima that have been selected for connection
        run the connection attempts in parallel or serial and add any
        new transition states to the network
        """

        if self.multiprocessing_on:
            # Set off connection attempts from the list total_pairs
            with multiprocessing.get_context('fork').Pool(
                    processes=self.n_processes) as pool:
                stationary_point_information = pool.map(
                    self.connection_attempt, total_pairs)
                # For each located transition state, attempt to add to network
                for i in stationary_point_information:
                    if i is not None:
                        for j in i:
                            self.coords.position = j[0]
                            self.similarity.test_new_ts(self.ktn,
                                                        self.coords, j[1],
                                                        j[2], j[3],
                                                        j[4], j[5])
        else:
            # Run connection attempts for each pair
            for i in total_pairs:
                stationary_point_information = self.connection_attempt(i)
                # For each transition state attempt to add to network
                if stationary_point_information is not None:
                    for j in stationary_point_information:
                        self.coords.position = j[0]
                        self.similarity.test_new_ts(self.ktn,
                                                    self.coords, j[1],
                                                    j[2], j[3],
                                                    j[4], j[5])
        # Write node pairs into pairlist to keep track of previous attempts
        if self.ktn.pairlist.size == 0:
            self.ktn.pairlist = np.empty((0, 2), dtype=int)
        for i in total_pairs:
            self.ktn.pairlist = np.append(
                self.ktn.pairlist, np.array([np.sort(i)]), axis=0)

    def connection_attempt(self, pair: list) -> list:
        """ Connection attempt between a pair of selected minima that are
            specified nodes of the network graph. Returns a list of all
            transition states and their connected minima
            with each transition state as a sublist """

        connection_start_time = timer()
        local_coords = deepcopy(self.coords)
        # Get the coordinates, check them and align them
        min1, min2, repeats, permutation = \
            self.prepare_connection_attempt(local_coords, pair)
        local_coords.position = min1
        if min1 is None:
            return []

        # Do a double ended transition state search
        neb_start_time = timer()
        candidates, positions = \
            self.double_ended_search.run(local_coords, min2, repeats,
                                         permutation)
        neb_end_time = timer()

        with open('logfile', 'a', encoding="utf-8") as outfile:
            outfile.write(f"{np.size(candidates)} candidates from NEB after "
                          f"Force constant: "
                          f"{self.double_ended_search.force_constant}\n")

        # Try and converge each candidate to a true transition state
        hef_start_time = timer()
        stationary_point_information = []
        for ts_search_number, ts_candidate in enumerate(positions, start=1):
            local_coords.position = ts_candidate
            ts_coords, e_ts, min_plus, e_plus, min_minus, e_minus, neg_eig = \
                self.single_ended_search.run(local_coords)
            # Check search was successful
            if ts_coords is not None:
                with open('logfile', 'a', encoding="utf-8") as outfile:
                    outfile.write(f"{ts_search_number}: ")
                stationary_point_information.append(
                        [ts_coords, e_ts, min_plus, e_plus, min_minus,
                         e_minus, neg_eig])
            else:
                self.write_failure_condition()

        #  Work out the timings for each part of the connection
        end_time = timer()
        self.write_connection_output(end_time-connection_start_time,
                                     neb_end_time - neb_start_time,
                                     end_time - hef_start_time)
        return stationary_point_information

    def write_connection_output(self, connection_length: float,
                                neb_length: float, hef_length: float) -> None:
        """ Write the timings for the connection attempt """

        with open('logfile', 'a', encoding="utf-8") as outfile:
            outfile.write(f"Completed connection in "
                          f"{connection_length} s:"
                          f"NEB ({neb_length}), HEF ({hef_length})\n")
            outfile.write('-------------------------\n')

    def prepare_connection_attempt(self, coords: type, pair: list) -> NDArray:
        """ Prepare the two minima for connection in a double ended
            transition state search. Find the coordinates of each
            minimum. Check that the pair is valid and how many times
            it has been repeated. Align the two conformations. """

        node1 = pair[0]
        node2 = pair[1]
        with open('logfile', 'a', encoding="utf-8") as outfile:
            outfile.write(f"Connection attempt between minima pair: "
                          f"{node1} {node2}\n\n")
        # Check if we should connect this pair
        allowed, repeats = self.check_pair(node1, node2)
        if not allowed:
            return None, None, None, None
        # Calculate the positions of nodes in network
        min1 = self.ktn.get_minimum_coords(node1)
        coords.position = min1
        min2 = self.ktn.get_minimum_coords(node2)
        # Align the second coordinates to the first to make
        # double ended search more efficient
        dist, min1, min2, permutation = \
            self.similarity.optimal_alignment(coords, min2)
        return min1, min2, repeats, permutation

    def check_pair(self, node1: int, node2: int) -> (bool, int):
        """ Determines if this pair should be tried again
            Could have already been attempted too many times or
            already have a transition state. Return logical
            that determines if pair should be attemped """

        # Calculate the number of times this pair has been tried
        repeats = 0
        for i in self.ktn.pairlist:
            if np.array_equal(i, np.sort([node1, node2])):
                repeats += 1
        # Report if retrying a connection
        if repeats in (1, 2):
            with open('logfile', 'a', encoding="utf-8") as outfile:
                outfile.write("Connection already attempted. Trying again\n")
        #  Only try three times before giving up
        elif repeats > 2:
            with open('logfile', 'a', encoding="utf-8") as outfile:
                outfile.write("Connection attempted too many times.")
            return False, repeats
        # Don't repeat connections that are already directly connected
        if self.ktn.G.has_edge(node1, node2):
            with open('logfile', 'a', encoding="utf-8") as outfile:
                outfile.write("Nodes already connected by transition state\n")
            return False, repeats
        # Avoid connections between minimum and itself
        if node1 == node2:
            return False, repeats
        return True, repeats

    def write_failure_condition(self) -> None:
        """ Writes reason for transition state location failure to file """
        with open('logfile', 'a', encoding="utf-8") as outfile:
            outfile.write("TS search did not converge. ")
            if self.single_ended_search.failure == 'SDpaths':
                outfile.write("Steepest descent paths did not converge\n")
            elif self.single_ended_search.failure == 'eigenvector':
                outfile.write("Eigenvector is zero\n")
            elif self.single_ended_search.failure == 'eigenvalue':
                outfile.write("Eigenvalue is zero\n")
            elif self.single_ended_search.failure == 'bounds':
                outfile.write("Transition state outside bounds\n")
            elif self.single_ended_search.failure == 'steps':
                outfile.write("Too many steps\n")
            elif self.single_ended_search.failure == 'pushoff':
                outfile.write("Can't find a pushoff\n")
            elif self.single_ended_search.failure == 'invalid_ts':
                outfile.write("Invalid transition state\n")

    def select_minima(self, coords: type, option: str,
                      neighbours: int) -> list:
        """
        Make decisions about which pairs of nodes we should be connecting
        Pick one of four schemes designed to connect with certain properties
        'ClosestEnumeration' - for each minimum generate the neighbours pairs
                               that are closest in Euclidean distance
        'ConnectUnconnected' - select pairs of minima that are closest in
                               Euclidean distance with the constraint that
                               one minimum must be connected to the
                               global minimum, and one must not
        'ReadPairs' - take the list of connections from the file 'pairs.txt'
        """

        #  Write which method we are using to logfile
        with open('logfile', 'a', encoding="utf-8") as outfile:
            outfile.write("------- SAMPLING THE NETWORK ---------\n")
            outfile.write(f"Exploring the landscape for {self.ktn.n_minima}"
                          f" after removing minima at bounds\n")
            if option == "ClosestEnumeration":
                outfile.write("Scheme: ClosestEnumeration\n")
            elif option == "ConnectUnconnected":
                outfile.write("Scheme: ConnectUnconnected\n")
            elif option == "ReadPairs":
                outfile.write("Scheme: ReadPairs\n")
            outfile.write("-----------------\n")
        #  Run the pair selection method
        if option == "ClosestEnumeration":
            pairs = closest_enumeration(self.ktn, self.similarity,
                                        coords, neighbours)
        elif option == "ConnectUnconnected":
            pairs = connect_unconnected(self.ktn, self.similarity,
                                        coords, neighbours)
        elif option == "ReadPairs":
            pairs = read_pairs()
        return pairs

    def reconverge_minima(self, potential: type, reconv_crit: float) -> None:
        """ Reconverge the minima we have found in ktn, allows high
            accuracy without expending large cost at every BH step """

        start_time = timer()
        # Get the coordinates of all minima to reconverge
        coordinates = []
        for i in range(self.ktn.n_minima):
            coordinates.append(self.ktn.get_minimum_coords(i))
        # Reoptimise all the minima and accumulate in list
        minima_information = []
        for minimum in coordinates:
            reconv_coords, reconv_energy, results_dict = \
                lbfgs.minimise(func_grad=potential.function_gradient,
                               initial_position=minimum,
                               bounds=self.coords.bounds,
                               conv_crit=reconv_crit)
            minima_information.append([reconv_coords, reconv_energy])
        # Empty the current network
        self.ktn.reset_network()
        # Get the unique minima from the reconverged
        for i in range(len(minima_information)):
            self.coords.position = minima_information[i][0]
            energy = minima_information[i][1]
            self.similarity.test_new_minimum(self.ktn, self.coords, energy)
        end_time = timer()
        with open('logfile', 'a', encoding="utf-8") as outfile:
            outfile.write(f"{self.ktn.n_minima} distinct minima after "
                          f"reconvergence\nReconvergence completed in "
                          f"{end_time - start_time}\n\n")

    def reconverge_landscape(self, potential: type,
                             reconv_crit: float) -> None:
        """ Reconverge all stationary points in a landscape. Useful
            for finding the corresponding points in a different potential """

        start_time = timer()
        # Get the coordinates of all stationary points to reconverge
        min_coordinates = []
        ts_coordinates = []
        for i in range(self.ktn.n_minima):
            min_coordinates.append(self.ktn.get_minimum_coords(i))
        for u, v in self.ktn.G.edges():
            ts_coordinates.append(self.ktn.get_ts_coords(u, v))
        # Reoptimise all the minima
        minima_information = []
        for minimum in min_coordinates:
            reconv_coords, reconv_energy, results_dict = \
                lbfgs.minimise(func_grad=potential.function_gradient,
                               initial_position=minimum,
                               bounds=self.coords.bounds,
                               conv_crit=reconv_crit)
            minima_information.append([reconv_coords, reconv_energy])
        # Then the transition states
        ts_information = []
        for ts in ts_coordinates:
            self.coords.position = ts
            ts_information.append(self.single_ended_search.run(self.coords))
        # Reset the network
        self.ktn.reset_network()
        # Test for uniqueness and add each to the kinetic transition network
        for i in range(len(minima_information)):
            self.coords.position = minima_information[i][0]
            energy = minima_information[i][1]
            self.similarity.test_new_minimum(self.ktn, self.coords, energy)
        for i in ts_information:
            self.coords.position = i[0]
            self.similarity.test_new_ts(self.ktn, self.coords, i[1], i[2],
                                        i[3], i[4], i[5])
        end_time = timer()
        with open('logfile', 'a', encoding="utf-8") as outfile:
            outfile.write(f"{self.ktn.n_minima} distinct minima and "
                          f"{self.ktn.n_ts} transition states after "
                          f"reconvergence\nReconvergence completed in "
                          f"{end_time - start_time}\n\n")
