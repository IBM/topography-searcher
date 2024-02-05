""" Module that contains the classes for evaluating similarity of
    atomic or molecular conformations """

import numpy as np
from nptyping import NDArray
import scipy.optimize
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation as rotations
from .similarity import StandardSimilarity


class MolecularSimilarity(StandardSimilarity):

    """
    Description
    ------------
    Separate class to deal with the extra complexity of atomic systems.
    They require translation, rotation, centering and permutation

    Attributes
    -----------
    distance_criterion : float
        The distance under which two minima are considered the same, for
        proportional_distance set to True this is not an absolute value,
        but the proportion of the total function range
    energy_criterion : float
        The value that the difference in function values must be below
        to be considered the same
    weighted : bool
        Flag that specifies whether we include the relative atomic weights
        in alignment
    """

    def __init__(self, distance_criterion: float, energy_criterion: float,
                 weighted: bool = False):
        self.distance_criterion = distance_criterion
        self.energy_criterion = energy_criterion
        self.weighted = weighted

    def permutational_alignment(self, coords1: type, coords2: NDArray) -> tuple:
        """ Hungarian algorithm adapted to solve the linear assignment problem
            for each atom of each distinct element. Return the optimised coords2
            with best permutation """

        # Initial permutation vector
        perm_vec = np.zeros((coords1.n_atoms), dtype=int)
        # Calculate the number of different elements
        elements = list(set(coords1.atom_labels))
        # Loop over each element to perform alignment w.r.t each in turn
        for i in range(len(elements)):
            curr_element = elements[i]
            element_atoms = [i for i, x in enumerate(coords1.atom_labels)
                             if x == curr_element]
            if len(element_atoms) > 1:
                # Get coordinates of just atoms of this element
                coords1_element = np.take(coords1.position.reshape(-1,3),
                                          element_atoms, axis=0)
                coords2_element = np.take(coords2.reshape(-1,3),
                                          element_atoms, axis=0)
                # Calculate distance matrix for these subset of atoms
                dist_matrix = distance_matrix(coords1_element,
                                              coords2_element)
                # Optimal permutational alignment
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_matrix**2)
                for j in range(len(element_atoms)):
                    perm_vec[element_atoms[j]] = element_atoms[col_ind[j]]
                # Apply the permutation to subset of atoms
                permuted_coords = coords2.copy()
                for j in range(len(element_atoms)):
                    permuted_coords[element_atoms[j]*3:(element_atoms[j]*3)+3] = \
                        coords2[element_atoms[col_ind[j]]*3:(element_atoms[col_ind[j]]*3)+3]
                # Update the original coordinates for subsequent elements
                coords2 = permuted_coords.copy()
        return permuted_coords, perm_vec

    def rotational_alignment(self, coords1: type, coords2: NDArray) -> tuple:
        """ Find the rotation that minimises the distance between
            two sets of vectors using the Kabsch algorithm and apply it """
        if self.weighted:
            best_rotation, dist = \
                rotations.align_vectors(coords1.position.reshape(-1,3),
                                        coords2.reshape(-1,3),
                                        weights=coords1.atom_weights)
        else:
            best_rotation, dist = \
                rotations.align_vectors(coords1.position.reshape(-1,3),
                                        coords2.reshape(-1,3))
        coords2 = best_rotation.apply(coords2.reshape(-1,3))
        return dist, coords2.flatten()

    def random_rotation(self, position: NDArray) -> NDArray:
        """ Apply a uniformly distributed random rotation matrix """
        coords_rotated = \
            rotations.random(1).apply(position.reshape(-1,3))
        return coords_rotated.flatten()

    def centre(self, position: NDArray, weights: NDArray) -> NDArray:
        """ Returns coords after centre of mass has been moved to origin """
        if self.weighted:
            centre_of_mass = \
                np.array([np.average(position[0::3], weights=weights),
                          np.average(position[1::3], weights=weights),
                          np.average(position[2::3], weights=weights)])
        else:
            centre_of_mass = np.array([np.average(position[0::3]),
                                       np.average(position[1::3]),
                                       np.average(position[2::3])])
        position_centered = position.copy()
        for i in range(weights.size):
            position_centered[i*3:(i*3)+3] -= centre_of_mass
        return position_centered

    def align(self, coords1: type, coords2: NDArray) -> tuple:
        """ Perform rotational and permutational alignment of coords1 and
            coords2. Returns the optimised distance and corresponding coords2 """
        coords2_permuted, permutation = self.permutational_alignment(coords1,
                                                                     coords2)
        optimised_distance, coords2_optimised = \
            self.rotational_alignment(coords1, coords2_permuted)
        return optimised_distance, coords2_optimised, permutation

    def test_exact_same(self, coords1: type, coords2: NDArray) -> tuple:
        """ Routine to quickly test if two conformations are identical, 
            if not then we apply the normal alignment routines """

        # Find the furthest atom from the centre of mass
        # And any others within 0.05 to allow for small fluctuations
        # in compared structures changing it
        origin = np.zeros(3, dtype=float)
        distances1 = np.zeros(coords1.n_atoms, dtype=float)
        distances2 = np.zeros(coords1.n_atoms, dtype=float)
        for i in range(coords1.n_atoms):
            distances1[i] = self.distance(origin, coords1.position[(3*i):(3*i)+3])
            distances2[i] = self.distance(origin, coords2[(3*i):(3*i)+3])
        max_d1 = np.max(distances1)
        max_d2 = np.max(distances2)
        indices1 = [list(l) for l in np.where(distances1 > (max_d1-0.05))][0]
        indices2 = [list(l) for l in np.where(distances2 > (max_d2-0.05))][0]
        all_pairs = []
        for i in indices1:
            for j in indices2:
                all_pairs.append([i,j])
        # Pick the furthest atom from the origin in the direction
        # perpendicular to furthest atom for all atoms within the range
        for i in all_pairs:
            atom1 = i[0]
            atom2 = i[1]
            furthest1_coords = coords1.position[3*atom1:(3*atom1)+3]
            furthest2_coords = coords2[3*atom2:(3*atom2)+3]
            distances1 = np.zeros(coords1.n_atoms, dtype=float)
            distances2 = np.zeros(coords1.n_atoms, dtype=float)
            for j in range(coords1.n_atoms):
                # Get the perpendicular distance for atom1 in coords1
                perp_coords1 = coords1.position[3*j:(3*j)+3]
                cross_p = np.cross(-1.0*furthest1_coords,
                                   perp_coords1-furthest1_coords)
                perp_vec1 = np.linalg.norm(cross_p) / \
                    np.linalg.norm(perp_coords1-furthest1_coords)
                vec_dist = np.linalg.norm(origin-perp_vec1)
                if np.isnan(vec_dist):
                    distances1[j] = 0.0
                else:
                    distances1[j] = vec_dist
                # Get the perpendicular distance for atom2 in coords2
                perp_coords2 = coords2[3*j:(3*j)+3]
                cross_p = np.cross(-1.0*furthest2_coords,
                                   perp_coords2-furthest2_coords)
                perp_vec2 = np.linalg.norm(cross_p) / \
                    np.linalg.norm(perp_coords2-furthest2_coords)
                vec_dist = np.linalg.norm(origin-perp_vec2)
                if np.isnan(vec_dist):
                    distances2[j] = 0.0
                else:
                    distances2[j] = vec_dist
            # Find the furthest perpendicular atoms and any within 0.05 of this
            max_d1 = np.max(distances1)
            max_d2 = np.max(distances2)
            indices1 = [list(l) for l in np.where(distances1 > (max_d1-0.03))][0]
            indices2 = [list(l) for l in np.where(distances2 > (max_d2-0.03))][0]
            perp_pairs = []
            for j in indices1:
                for k in indices2:
                    perp_pairs.append([j,k])
            # For each pair of furthest atoms attempt to align with each pair
            # of perpendicular atoms within the distance range
            for j in perp_pairs:
                # Find rotation that transforms the two atoms
                subset1 = np.concatenate((coords1.position[3*atom1:(3*atom1)+3],
                                          coords1.position[3*j[0]:(3*j[0])+3]))
                subset2 = np.concatenate((coords2[3*atom2:(3*atom2)+3],
                                          coords2[3*j[1]:(3*j[1])+3]))
                subset1 = subset1.reshape((2, 3))
                subset2 = subset2.reshape((2, 3))
                best_rotation, dist = \
                    rotations.align_vectors(np.asarray(subset1),
                                            np.asarray(subset2))
                # Apply the transformation to the whole molecule
                coords2_rotated = best_rotation.apply(coords2.reshape(-1,3))
                # Find the optimal permutation at this rotation
                dist, coords2_aligned, perm_vec = \
                    self.align(coords1, coords2_rotated.flatten())
                # If close enough to be considered same then exit 
                if (dist / coords1.n_atoms) < self.distance_criterion:
                    return dist, coords2_aligned, perm_vec
        return dist, coords2_aligned, perm_vec

    def optimal_alignment(self, coords1: type, coords2: NDArray) -> tuple:
        """ Try to find the optimal alignment between coords1 and coords2
            Take multiple attempts from different starting orientations """
        coords1.position = self.centre(coords1.position, weights=coords1.atom_weights)
        coords2 = self.centre(coords2, weights=coords1.atom_weights)
        # See if the two structures are identical first
        dist, coords2_aligned, perm_vec = self.test_exact_same(coords1, coords2)
        if (dist / coords1.n_atoms) < self.distance_criterion:
            return dist, coords1, coords2_aligned, perm_vec
        # Not identical so try to find best permutational and rotational alignment
        best_dist = dist
        best_coords2 = coords2_aligned
        best_perm = perm_vec
        for i in range(100):
            coords2_rotated = self.random_rotation(coords2)
            dist, coords_opt, perm_vec = self.align(coords1, coords2_rotated)
            if (dist / coords1.n_atoms) < self.distance_criterion:
                return dist, coords1, coords_opt, perm_vec
            if dist < best_dist:
                best_dist = dist
                best_coords2 = coords_opt
                best_perm = perm_vec
        return best_dist, coords1, best_coords2, best_perm

    def test_same(self, coords1: type, coords2: NDArray,
                  energy1: float, energy2: float) -> bool:
        """ Test if two structures are the same to within a distance and energy
            criterion after finding the closest alignment. Returns logical """
        distance, coords1, coords2_optimised, perm_vec = \
            self.optimal_alignment(coords1, coords2)
        energy_difference = np.abs(energy1-energy2)
        return bool(((distance / coords1.n_atoms) < self.distance_criterion) and
                    (energy_difference < self.energy_criterion))

    def closest_distance(self, coords1: type, coords2: NDArray) -> float:
        """ Align two structures and return the optimised distance """
        return self.optimal_alignment(coords1, coords2)[0]
