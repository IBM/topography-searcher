""" Module that contains the class for evaluating similarity of
    atomic or molecular conformations. Alignment of atomistic systems
    requires consideration of translational, rotatational and permutational
    invariance, which makes it much more challenging than the Euclidean
    distance between configurations """

import numpy as np
from nptyping import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation as rotations
from .similarity import StandardSimilarity


class MolecularSimilarity(StandardSimilarity):

    """
    Description
    ------------
    Class to perform alignment of molecular and atomic coordinate
    objects, and highlight repeated minima and transition states.
    Alignment minimises the Euclidean distance between a pair of
    conformations accounting for translational, rotational and 
    permutational invariance.

    Attributes
    -----------
    distance_criterion : float
        The distance under which two minima are considered the same
    energy_criterion : float
        The value that the difference in function values must be below
        to be considered the same
    weighted : bool
        Flag that specifies whether we include the relative atomic weights
        in alignment
    allow_inversion : bool
        Flag that specifies if we consider conformations related by the
        inversion operation as the same
    """

    def __init__(self, distance_criterion: float, energy_criterion: float,
                 weighted: bool = False, allow_inversion: bool = False):
        self.distance_criterion = distance_criterion
        self.energy_criterion = energy_criterion
        self.weighted = weighted
        self.allow_inversion = allow_inversion

    def permutational_alignment(self, coords1: type,
                                coords2: NDArray) -> tuple:
        """ Hungarian algorithm adapted to solve the linear assignment problem
            for each atom of each distinct element. Finds the optimal
            permutation of all atoms respecting the atomic species """
        # Initialise permutation vector to track atom swaps
        permutation = np.zeros((coords1.n_atoms), dtype=int)
        # Copy second coordinates to leave unchanged
        permuted_coords = coords2.copy()
        perm_group1, perm_group2 = self.get_permutable_groups(coords1, coords2)
        # Loop over each perm group to perform alignment w.r.t each in turn
        for i in range(len(perm_group1)):
            # Get atoms of the given element
            perm_atoms1 = perm_group1[i]
            perm_atoms2 = perm_group2[i]
            if len(perm_atoms1) > 1:
                # Get coordinates of just atoms of this element
                coords1_element = np.take(coords1.position.reshape(-1, 3),
                                          perm_atoms1, axis=0)
                coords2_element = np.take(coords2.reshape(-1, 3),
                                          perm_atoms2, axis=0)
                # Calculate distance matrix for these subset of atoms
                dist_matrix = distance_matrix(coords1_element,
                                              coords2_element)
                if dist_matrix.shape[0] != dist_matrix.shape[1]:
                    print("Warning: distance matrix wrong shape")
                    break
                # Optimal permutational alignment
                col_ind = linear_sum_assignment(dist_matrix**2)[1]
                # Update coordinates and permutation vector
                for idx, atom in enumerate(perm_atoms1):
                    permutation[atom] = perm_atoms1[col_ind[idx]]
                    permuted_coords[atom*3:(atom*3)+3] = \
                        coords2[perm_atoms2[col_ind[idx]]*3:
                                (perm_atoms2[col_ind[idx]]*3)+3]
            # Only one atom so no need for permutational alignment
            else:
                permutation[perm_atoms1[0]] = perm_atoms1[0]
        return permuted_coords, permutation

    def get_permutable_groups(self, coords1: type, coords2: NDArray) -> list:
        """ Determine the subsets of atoms that are allowed to be permuted
            when minimising the distance between conformations. Must be
            of the same element and have the same bonds """
        # Initialise the sets of permutable atoms from coords1 and 2
        permutable_groups1 = []
        permutable_groups2 = []
        # Get the connected species for each atom as we
        # only want to allow permutation of atoms with same bonding
        original_coords = coords1.position.copy()
        bond_labels1 = coords1.get_connected_atoms()
        coords1.position = coords2
        bond_labels2 = coords1.get_connected_atoms()
        coords1.position = original_coords
        # Find the set of different elements in the molecule
        elements = list(set(coords1.atom_labels))
        # Loop over each species separately
        for element in elements:
            # Get each atom of a given element
            element_atoms = \
                [i for i, x in enumerate(coords1.atom_labels) if x == element]
            # Initialise the arrays to store the connected atoms of each in set
            elements_bonds1 = []
            elements_bonds2 = []
            # Loop over all atoms of element, adding their connections
            for i in element_atoms:
                elements_bonds1.append(bond_labels1[i])
                elements_bonds2.append(bond_labels2[i])
            # Find the unique set of environments for these atoms
            unique_envs1 = \
                list(set(tuple(sorted(row)) for row in elements_bonds1))
            # Loop over each unique environment finding the atoms with it
            for i in unique_envs1:
                perm_atoms1 = []
                perm_atoms2 = []
                # Loop over all the atoms of this element checking which
                # match the current environment
                for j in element_atoms:
                    if bond_labels1[j] == list(i):
                        perm_atoms1.append(j)
                    if bond_labels2[j] == list(i):
                        perm_atoms2.append(j)
                permutable_groups1.append(perm_atoms1)
                permutable_groups2.append(perm_atoms2)
        return permutable_groups1, permutable_groups2

    def rotational_alignment(self, coords1: type, coords2: NDArray) -> tuple:
        """ Find the rotation that minimises the distance between
            two sets of vectors using the Kabsch algorithm and apply it """
        if self.weighted:
            best_rotation, dist = \
                rotations.align_vectors(coords1.position.reshape(-1, 3),
                                        coords2.reshape(-1, 3),
                                        weights=coords1.atom_weights)
        else:
            best_rotation, dist = \
                rotations.align_vectors(coords1.position.reshape(-1, 3),
                                        coords2.reshape(-1, 3))
        coords2 = best_rotation.apply(coords2.reshape(-1, 3))
        return dist, coords2.flatten()

    def random_rotation(self, position: NDArray) -> NDArray:
        """ Apply a uniformly distributed random rotation matrix """
        coords_rotated = \
            rotations.random(1).apply(position.reshape(-1, 3))
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
        """ Perform permutational and rotational alignment of coords2
            relative to coords1 to find best alignment """
        coords2_permuted, permutation = \
            self.permutational_alignment(coords1, coords2)
        optimised_distance, coords2_optimised = \
            self.rotational_alignment(coords1, coords2_permuted)
        return optimised_distance, coords2_optimised, permutation

    def test_exact_same(self, coords1: type, coords2: NDArray) -> tuple:
        """ Routine to test if two conformations are identical by aligning
            furthest atom from centre in each, along with furthest in
            direction perpendicular to that """
        # Find furthest atoms from the origin
        indices1 = self.get_furthest_from_centre(coords1.position)
        indices2 = self.get_furthest_from_centre(coords2)
        # Generate all pairs of indices between conformations
        all_pairs = self.generate_pairs(indices1, indices2)
        # Store for best atom pair if we don't find exact match
        best_dist = 1e30
        best_coords2 = coords2
        best_perm = np.zeros(coords1.n_atoms, dtype=int)
        # Test all pairs of distant atoms
        for i in all_pairs:
            # Find furthest atoms from the origin in the direction
            # perpendicular to furthest atom for all atoms within the range
            indices1 = self.get_furthest_perpendicular(coords1.position, i[0])
            indices2 = self.get_furthest_perpendicular(coords2, i[1])
            perp_pairs = self.generate_pairs(indices1, indices2)
            # Test all pairs of perpendicular atoms
            for j in perp_pairs:
                # Align each pair of furthest atoms in both conformations
                subset1 = np.concatenate((coords1.position[3*i[0]:(3*i[0])+3],
                                          coords1.position[3*j[0]:(3*j[0])+3]))
                subset2 = np.concatenate((coords2[3*i[1]:(3*i[1])+3],
                                          coords2[3*j[1]:(3*j[1])+3]))
                best_rotation, dist = \
                    rotations.align_vectors(np.asarray(subset1.reshape(2, 3)),
                                            np.asarray(subset2.reshape(2, 3)))
                # Apply transformation to the whole molecule and align
                coords2_rotated = best_rotation.apply(coords2.reshape(-1, 3))
                dist, coords2_aligned, permutation = \
                    self.align(coords1, coords2_rotated.flatten())
                # If close enough to be considered same then exit
                if dist < self.distance_criterion:
                    return dist, coords2_aligned, permutation
                # If an improvement on previous alignment then update
                if dist < best_dist:
                    best_dist = dist
                    best_coords2 = coords2_aligned
                    best_perm = permutation
        return best_dist, best_coords2, best_perm

    def get_furthest_from_centre(self, position: NDArray) -> list:
        """ Find the atom furthest from the origin in position,
            and any other within 0.05 of this """
        origin = np.zeros(3, dtype=float)
        distances = np.zeros(int(position.size/3), dtype=float)
        for i in range(int(position.size/3)):
            distances[i] = self.distance(origin, position[(3*i):(3*i)+3])
        max_d = np.max(distances)
        # All atoms within 0.05 of furthest distance from origin
        return [list(j) for j in np.where(distances > (max_d-0.05))][0]

    def get_furthest_perpendicular(self, position: NDArray,
                                   ref_atom: int) -> list:
        """ Find the atoms furthest from the origin in position,
            perpendicular to the vector to position[ref_atom] """
        ref_coords = position[3*ref_atom:(3*ref_atom)+3]
        distances = np.zeros(int(position.size/3), dtype=float)
        for i in range(int(position.size/3)):
            # Get the perpendicular distance for atom1 in coords1
            atom_coords = position[3*i:(3*i)+3]
            cross_p = np.cross(-1.0*ref_coords,
                               atom_coords-ref_coords)
            if np.linalg.norm(cross_p) == 0.0:
                continue
            perp_vec = np.linalg.norm(cross_p) / \
                np.linalg.norm(atom_coords-ref_coords)
            vec_dist = np.linalg.norm(perp_vec)
            if not np.isnan(vec_dist):
                distances[i] = vec_dist
        # Find the furthest perpendicular atoms and any within 0.05 of this
        max_d = np.max(distances)
        return [list(j) for j in np.where(distances > (max_d-0.03))][0]

    def generate_pairs(self, indices1: list, indices2: list) -> list:
        """ Generate a list of all possible pairs from two input lists """
        pairs = []
        for i in indices1:
            for j in indices2:
                pairs.append([i, j])
        return pairs

    def optimal_alignment(self, coords1: type, coords2: NDArray) -> tuple:
        """ Try to find the optimal alignment between coords1 and coords2.
            Initially test if the structures are the same, and if not iterate
            alignment from different starting orientations """
        coords1.position = self.centre(coords1.position,
                                       weights=coords1.atom_weights)
        # Try without inversion first
        coords2 = self.centre(coords2, weights=coords1.atom_weights)
        # See if the two structures are identical first
        dist, coords2_aligned, permutation = \
            self.test_exact_same(coords1, coords2)
        # Return already if coords sufficiently close to be considered the same
        if dist < self.distance_criterion:
            return dist, coords1, coords2_aligned, permutation
        # Not identical so try to find best permutation and rotation
        best_dist = dist
        best_coords2 = coords2_aligned
        best_perm = permutation
        # Iterate 50 times from different overall rotations
        for i in range(150):
            coords2_rotated = self.random_rotation(coords2)
            dist, coords_opt, permutation = \
                self.align(coords1, coords2_rotated)
            # If sufficiently close to be a match leave the loop early
            if dist < self.distance_criterion:
                return dist, coords1, coords_opt, permutation
            # If an improvement on previous closest alignment then update
            if dist < best_dist:
                best_dist = dist
                best_coords2 = coords_opt
                best_perm = permutation
        if self.allow_inversion:
            # Try with inversion
            coords2 = self.invert(coords2)
            coords2 = self.centre(coords2, weights=coords1.atom_weights)
            # See if the two structures are identical first
            dist, coords2_aligned, permutation = \
                self.test_exact_same(coords1, coords2)
            # Return already if coords sufficiently close to be the same
            if dist < self.distance_criterion:
                return dist, coords1, coords2_aligned, permutation
            # Not identical so try to find best permutation and rotation
            if dist < best_dist:
                best_dist = dist
                best_coords2 = coords2_aligned
                best_perm = permutation
            # Iterate 50 times from different overall rotations
            for i in range(150):
                coords2_rotated = self.random_rotation(coords2)
                dist, coords_opt, permutation = \
                    self.align(coords1, coords2_rotated)
                # If sufficiently close to be a match leave the loop early
                if dist < self.distance_criterion:
                    return dist, coords1, coords_opt, permutation
                # If an improvement on previous closest alignment then update
                if dist < best_dist:
                    best_dist = dist
                    best_coords2 = coords_opt
                    best_perm = permutation
        return best_dist, coords1.position, best_coords2, best_perm

    def test_same(self, coords1: type, coords2: NDArray,
                  energy1: float, energy2: float) -> bool:
        """ Test if two structures are the same to within a distance and energy
            criterion after finding the closest alignment """
        within_distance = \
            self.closest_distance(coords1, coords2) < self.distance_criterion
        within_energy = np.abs(energy1-energy2) < self.energy_criterion
        return bool(within_distance and within_energy)

    def closest_distance(self, coords1: type, coords2: NDArray) -> float:
        """ Align two structures and return the optimised distance """
        return self.optimal_alignment(coords1, coords2)[0]

    def invert(self, position: NDArray) -> NDArray:
        """ Invert a conformation  """
        return -1.0*position
