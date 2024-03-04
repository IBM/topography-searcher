""" Similarity module evaluates the similarity of two different points
    in a space. These could be atomic configurations or simply points
    in a Euclidean space """

from copy import deepcopy
import numpy as np
from nptyping import NDArray


class StandardSimilarity:

    """
    Description
    ------------

    Base class for assessing the similarity between two configurations
    The base class deals with non-atomic systems, which are considerably
    simpler than atomic systems due to the absence of permutation and rotation

    Attributes
    -----------

    distance_criterion : float
        The distance under which two minima are considered the same, for
        proportional_distance set to True this is not an absolute value,
        but the proportion of the total function range
    energy_criterion : float
        The value that the difference in function values must be below
        to be considered the same
    proportional_distance : logical
        Set if the distance_criterion is an absolute value or a proportion
        of the function bounds range
    """

    def __init__(self,
                 distance_criterion: float,
                 energy_criterion: float,
                 proportional_distance: bool = False):
        self.distance_criterion = distance_criterion
        self.energy_criterion = energy_criterion
        self.proportional_distance = proportional_distance

    def distance(self, coords1: type, coords2: NDArray) -> float:
        """ Returns Euclidean distance between two configurations """
        return np.linalg.norm(coords1 - coords2)

    def centre(self, coords1: type) -> NDArray:
        """ Returns coords again, included for ease with other classes """
        return coords1.position

    def closest_distance(self, coords1: type, coords2: NDArray) -> float:
        """ Returns the closest distance between coords1 and coords2
            Just the distance for non-atomic systems """
        return self.distance(coords1.position, coords2)

    def optimal_alignment(self, coords1: type,
                          coords2: NDArray) -> (float, NDArray, NDArray):
        """
        Placeholder method to allow the same routines to be used
        in atomic classes, does nothing in this case
        """
        dist = self.closest_distance(coords1, coords2)
        return dist, coords1.position, coords2, None

    def test_same(self, coords1: type, coords2: NDArray,
                  energy1: float, energy2: float) -> bool:
        """ Returns logical that specifies if two points are the same """
        # Compute the energy difference
        energy_difference = np.abs(energy1 - energy2)
        # For proportional distance we scale by each bound range
        if self.proportional_distance:
            # Find the allowed differences given the bounds
            allowed_differences = (coords1.upper_bounds -
                                   coords1.lower_bounds) * \
                                    self.distance_criterion
            # Check if within specified ellipsoid
            sum_distances = np.sum(np.divide(np.subtract(
                coords1.position, coords2), allowed_differences)**2)
            # If within ellipsoid and energy range then accept
            return bool((sum_distances <= 1.0) and
                        (energy_difference < self.energy_criterion))
        distance = self.distance(coords1.position, coords2)
        return bool((distance < self.distance_criterion) and
                    (energy_difference < self.energy_criterion))

    def is_new_minimum(self, ktn: type, min_coords: type,
                       min_energy: float) -> tuple[bool, NDArray]:
        """ Compare to all existing minima and add if different to all
        minima currently in network return True """

        # Loop over all other minima and if same as any then do not add
        for i in range(ktn.n_minima):
            if self.test_same(min_coords,
                              ktn.get_minimum_coords(i),
                              min_energy,
                              ktn.get_minimum_energy(i)):
                return False, i
        return True, None

    def is_new_ts(self, ktn: type, ts_coords: type,
                  ts_energy: float) -> tuple[bool, NDArray]:
        """ Compare transition state to all other currently in the network G
            and return whether it is the same as any of them """

        # Loop over all transition states
        for node1, node2 in ktn.G.edges():
            # Check if each transition state is a match
            if self.test_same(ts_coords,
                              ktn.get_ts_coords(node1, node2),
                              ts_energy,
                              ktn.get_ts_energy(node1, node2)):
                with open('logfile', 'a', encoding="utf-8") as outfile:
                    outfile.write("Repeated transition state connecting "
                                  f"{node1} and {node2}\n")
                return False, node1, node2
        return True, None, None

    def test_new_minimum(self, ktn: type, min_coords: type,
                         min_energy: float) -> None:
        """ Evaluate if a minimum is different to all current minima,
            check the minimum passes the requires test, and if both are
            True then add to network """

        if not self.is_new_minimum(ktn, min_coords, min_energy)[0]:
            return
        ktn.add_minimum(min_coords.position, min_energy)

    def test_new_ts(self, ktn: type,
                    ts_coords: type, ts_energy: float,
                    min_plus_coords: NDArray, e_plus: float,
                    min_minus_coords: NDArray, e_minus: float) -> None:
        """ Evaluate if the transition state is not a repeat, and that
            all stationary points are valid, if they are then add
            transition state to network with indices of connected minima """

        # Check if the transition state is new or repeated
        if not self.is_new_ts(ktn, ts_coords, ts_energy)[0]:
            with open('logfile', 'a', encoding="utf-8") as outfile:
                outfile.write("Repeated transition state\n")
            return

        # Set up classes for the minima to compare
        min_plus = deepcopy(ts_coords)
        min_minus = deepcopy(ts_coords)
        min_plus.position = min_plus_coords
        min_minus.position = min_minus_coords
        # New transition state and all valid stationary points
        # Find the indices of the connected minima
        index_plus = self.is_new_minimum(ktn, min_plus, e_plus)[1]
        index_minus = self.is_new_minimum(ktn, min_minus, e_minus)[1]

        if index_plus is None:
            ktn.add_minimum(min_plus.position, e_plus)
            index_plus = ktn.n_minima-1
        if index_minus is None:
            ktn.add_minimum(min_minus.position, e_minus)
            index_minus = ktn.n_minima-1

        # Add the transition state to the network
        with open('logfile', 'a', encoding="utf-8") as outfile:
            outfile.write(f"New transition state connecting minima"
                          f" {index_plus} and {index_minus}\n")

        ktn.add_ts(ts_coords.position, ts_energy, index_plus, index_minus)
