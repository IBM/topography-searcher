""" Module containing the StandardCoordinates class
    used to store and change positions """

import numpy as np
import networkx as nx
from nptyping import NDArray
from ase.geometry.analysis import Analysis
from ase import neighborlist
from ase import Atoms
from scipy.spatial.transform import Rotation as rotations
from rdkit.Chem import AllChem


class StandardCoordinates:
    """
    Description
    ---------------

    Class to store the coordinate position and all of the operations
    applied to the positions

    Attributes
    ---------------
    bounds : list of tuples [(b1_min, b1_max), (b2_min, b2_max)...]
        The allowed ranges of the coordinates in each dimension
    ndim : int
        The dimensionality of the coordinates
    position : numpy array
        Array of size ndim containing the current position
    """

    def __init__(self, ndim: int, bounds: list = None) -> None:
        self.ndim = ndim
        self.bounds = bounds
        self.lower_bounds = np.asarray([i[0] for i in self.bounds])
        self.upper_bounds = np.asarray([i[1] for i in self.bounds])
        self.position = self.generate_random_point()

    def generate_random_point(self) -> NDArray:
        """ Return an random point within the bound ranges """
        return np.random.uniform(low=self.lower_bounds,
                                 high=self.upper_bounds)

    def check_bounds(self) -> NDArray:
        """ Check if provided coords are at the bounds and return boolean
            array denoting which dimensions are at the bounds """
        return np.invert((self.position > self.lower_bounds) &
                         (self.position < self.upper_bounds))

    def at_bounds(self) -> bool:
        """ Check if the position is at the bounds in any dimension """
        return np.any(self.check_bounds())

    def all_bounds(self) -> bool:
        """ Check if the position is at the bounds in all dimensions """
        return np.all(self.check_bounds())

    def active_bounds(self) -> (NDArray, NDArray):
        """ Checks whether the position is at the upper or lower bounds """
        below_bounds = self.position <= self.lower_bounds
        above_bounds = self.position >= self.upper_bounds
        return below_bounds, above_bounds

    def move_to_bounds(self):
        """ Move the coordinates to within the bounds if currently outside """
        self.position = np.clip(self.position,
                                self.lower_bounds,
                                self.upper_bounds)


class AtomicCoordinates(StandardCoordinates):
    """
    Description
    ---------------

    Class to store the coordinates of an atomic system in Euclidean space.
    Methods to modify atomic positions and write out configurations

    Attributes
    ---------------
    bounds : list of tuples [(b1_min, b1_max), (b2_min, b2_max)...]
        The allowed ranges of the coordinates in each dimension
    ndim : int
        The dimensionality of the coordinates
    position : numpy array
        Array of size ndim containing the current position
    atom_labels : list
        The species in the system, matching the order of position
    """

    def __init__(self, atom_labels: list, position: NDArray,
                 bond_cutoff: float = 1.5) -> None:
        self.atom_labels = atom_labels
        self.position = position
        self.ndim = position.size
        self.n_atoms = int(self.ndim / 3)
        self.bounds = [(-100.0, 100.0)] * self.ndim
        self.lower_bounds = np.asarray([i[0] for i in self.bounds])
        self.upper_bounds = np.asarray([i[1] for i in self.bounds])
        # Create the atoms object for the molecule
        self.atoms = Atoms(''.join(self.atom_labels),
                           positions=self.position.reshape(-1, 3))
        self.atom_weights = self.atoms.get_atomic_numbers()
        self.bond_cutoff = bond_cutoff

    def get_atom(self, atom_ind: int) -> NDArray:
        """ Get the Cartesian position of the atom given index atom_ind """
        return self.position[atom_ind*3:(atom_ind*3)+3]

    def write_xyz(self, label: str = '') -> None:
        """ Dump an xyz file for visualisation """
        with open(f'coords{label}.xyz', 'w', encoding="utf-8") as xyz_file:
            xyz_file.write(f'{self.n_atoms}\n')
            xyz_file.write('molecule\n')
            for i in range(self.n_atoms):
                atom_pos = self.get_atom(i)
                xyz_file.write(f'{self.atom_labels[i]} ')
                xyz_file.write(f'{atom_pos[0]} {atom_pos[1]} {atom_pos[2]}\n')

    def write_extended_xyz(self, energy: float, grad: NDArray,
                           label: str = '') -> None:
        """ Dump an extended xyz file for visualisation """
        with open(f'coords{label}.xyz', 'w', encoding="utf-8") as xyz_file:
            xyz_file.write(f'{self.n_atoms}\n')
            xyz_file.write(f'{energy}\n')
            for i in range(self.n_atoms):
                atom_pos = self.get_atom(i)
                xyz_file.write(f'{self.atom_labels[i]} ')
                xyz_file.write(f'{atom_pos[0]} {atom_pos[1]} {atom_pos[2]} ')
                xyz_file.write(f'{grad[i*3]} {grad[(i*3)+1]} {grad[(i*3)+2]}')
                xyz_file.write('\n')

    def same_bonds(self) -> bool:
        """ Check if we still have a connected network of atoms """
        # Construct adjacency matrix
        adj_matrix = np.zeros((self.n_atoms, self.n_atoms), dtype=int)
        for i in range(self.n_atoms-1):
            for j in range(i+1, self.n_atoms):
                dist = np.linalg.norm(self.get_atom(i)-self.get_atom(j))
                if dist < self.bond_cutoff:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
        # Turns adjacency matrix into graph
        adj_graph = nx.from_numpy_array(adj_matrix)
        # Test if all nodes are connected
        return nx.is_connected(adj_graph)

    def remove_atom_clashes(self):
        """ Routine to remove clashes between atoms that result in
            very large gradient and explosion of the cluster.
            Unused default parameter is for matching with other classes """
        centre_of_mass = np.array([np.average(self.position[0::3]),
                                   np.average(self.position[1::3]),
                                   np.average(self.position[2::3])])
        displacements = 0.25*self.bond_cutoff * \
            (self.position.reshape(-1, 3) - centre_of_mass).flatten()
        for i in range(25):
            if self.check_atom_clashes():
                self.position = self.position + displacements
            else:
                break

    def check_atom_clashes(self) -> bool:
        """ Determine if there are atom clashes within the configuration """
        for i in range(self.n_atoms-1):
            for j in range(i+1, self.n_atoms):
                atom1 = self.get_atom(i)
                atom2 = self.get_atom(j)
                # Compare distance to bond_cutoff
                bond_length = np.linalg.norm(atom1 - atom2)
                allowed_bond_length = 0.5*self.bond_cutoff
                if bond_length < allowed_bond_length:
                    return True
        return False


class MolecularCoordinates(AtomicCoordinates):
    """
    Description
    ---------------

    Class to store the coordinates of an molecular system.
    Methods to modify atomic positions and write out configurations.
    Inherits from AtomicCoordinates, but contains additional functionality
    to test rotatable dihedrals and bonding structure.


    Attributes
    ---------------
    bounds : list of tuples [(b1_min, b1_max), (b2_min, b2_max)...]
        The allowed ranges of the coordinates in each dimension
    ndim : int
        The dimensionality of the coordinates
    position : numpy array
        Array of size ndim containing the current position
    atom_labels : list
        The species in the system, matching the order of position
    """

    def __init__(self, atom_labels: list, position: NDArray) -> None:
        super().__init__(atom_labels, position)
        # Calculate reference properties to retain
        self.natural_cutoffs = neighborlist.natural_cutoffs(self.atoms)
        self.reference_bonds = self.get_bonds()
        self.get_rotatable_dihedrals()
        self.analysis = None

    def get_bonds(self) -> nx.Graph:
        """ Find the current bonding framework for self.position """
        nl = neighborlist.NeighborList(self.natural_cutoffs,
                                       self_interaction=False)
        self.atoms.set_positions(self.position.reshape(self.n_atoms, 3))
        nl.update(self.atoms)
        analysis = Analysis(self.position, nl=nl)
        # Find the bonds
        ref_bonds = analysis.unique_bonds[0]
        current_bonds = []
        for b in range(self.n_atoms):
            for b2 in ref_bonds[b]:
                current_bonds.append([b, b2])
        edgelist = [tuple(l) for l in current_bonds]
        return nx.Graph(edgelist)

    def same_bonds(self) -> bool:
        """ Compares the bonding framework of the current coordinates
            with the initial coordinates (self.reference_bonds) """
        current_bonds = self.get_bonds()
        if len(current_bonds.edges()) != len(self.reference_bonds.edges()):
            return False
        # Same length so compute the types of bonds
        current_bond_labels = []
        for u, v in current_bonds.edges():
            current_bond_labels.append(sorted([self.atom_labels[u],
                                               self.atom_labels[v]]))
        ref_bond_labels = []
        for u, v in self.reference_bonds.edges():
            ref_bond_labels.append(sorted([self.atom_labels[u],
                                           self.atom_labels[v]]))
        # Compare the sorted lists of bonds to test similarity
        if sorted(current_bond_labels) == sorted(ref_bond_labels):
            return True
        return False

    def get_planar_rings(self) -> list:
        """ Find the atoms that belong to a planar ring in the molecule """
        # Find closed loops in the system, which are rings
        loop_nodes = nx.cycle_basis(self.reference_bonds)
        ring_atoms = []
        # Check if each loop is planar
        for i in loop_nodes:
            loop_coords = []
            for j in i:
                loop_coords.append(self.get_atom(j))
            # Get the coordinates of the loop atoms
            loop_coords = np.asarray(loop_coords)
            # Pick three points to define a plane
            a, b, c = loop_coords[0:3]
            normal_vec = np.cross((b-a), (c-a))
            planar = True
            for j in range(len(loop_coords)-3):
                d = loop_coords[j+3]
                if np.abs(np.dot((a-d), normal_vec)) > 1e-1:
                    planar = False
            if planar:
                for j in i:
                    ring_atoms.append(j)
        return ring_atoms

    def get_rotatable_dihedrals(self) -> list:
        """ Find all the dihedral angles we can rotate about,
            along with the atoms that should be rotated for each """
        nl = neighborlist.NeighborList(self.natural_cutoffs,
                                       self_interaction=False)
        self.atoms.set_positions(self.position.reshape(self.n_atoms, 3))
        nl.update(self.atoms)
        analysis = Analysis(self.position, nl=nl)
        # Get all dihedrals
        ase_rotatable_dihedrals = analysis.unique_dihedrals[0]
        self.rotatable_dihedrals = []
        for i in range(self.n_atoms):
            if ase_rotatable_dihedrals[i]:
                for j in ase_rotatable_dihedrals[i]:
                    k = [i]+list(j)
                    self.rotatable_dihedrals.append(k)
        # Get the atoms that should not be allowed to move
        rigid_atoms = self.get_planar_rings()
        # Get the repeated dihedrals
        repeat_dihedrals = self.get_repeat_dihedrals(self.rotatable_dihedrals)
        # If both central atoms are rigid or repeated then will remove
        removals = []
        for dih in self.rotatable_dihedrals:
            if dih[1] in rigid_atoms and dih[2] in rigid_atoms:
                removals.append(dih)
            if dih in repeat_dihedrals:
                removals.append(dih)
        # Remove the repeats and fixed dihedrals
        set_dih = set(tuple(x) for x in self.rotatable_dihedrals)
        set_removals = set(tuple(x) for x in removals)
        self.rotatable_dihedrals = list(set_dih - set_removals)
        # Find rotatable atoms for each
        self.rotatable_atoms = []
        if self.rotatable_dihedrals:
            for i in self.rotatable_dihedrals:
                moved_atoms = self.get_movable_atoms([i[1], i[2]], 'dihedral',
                                                     self.reference_bonds)
                self.rotatable_atoms.append(moved_atoms)

    def get_repeat_dihedrals(self, dihedrals: list) -> list:
        """ Only retain one dihedral when multiple share the same
            central bond. Gives a list of repeats to remove """
        removals = []
        for i in range(1, len(dihedrals)):
            same_central_bond = False
            for j in range(i):
                pair1 = [dihedrals[i][1], dihedrals[i][2]]
                pair2 = [dihedrals[j][1], dihedrals[j][2]]
                if sorted(pair2) == sorted(pair1):
                    same_central_bond = True
            if same_central_bond:
                removals.append(dihedrals[i])
        return removals

    def get_movable_atoms(self, bond: list, bond_type: str,
                          bond_network: nx.Graph) -> list:
        """ For a given dihedral return the atoms that should be rotated """
        ring_atoms = nx.cycle_basis(bond_network)
        ring_atoms = [x for xs in ring_atoms for x in xs]
        if bond_type == 'dihedral' or bond_type == 'length':
            # Both central atoms are in the loop so treat differently
            if bond[0] in ring_atoms and bond[1] in ring_atoms:
                neighbours = bond_network.neighbors(bond[1])
                if bond_type == 'length':
                    moved_atoms = [bond[1]]
                else:
                    moved_atoms = []
                # Add each neighbour that's not in loop
                for j in neighbours:
                    if j not in ring_atoms:
                        moved_atoms.append(j)
                        # And all its connections too
                        l_bond_network = bond_network.copy()
                        l_bond_network.remove_edge(bond[1], j)
                        for k in nx.connected_components(l_bond_network):
                            if j in k:
                                for l in k:
                                    if l != j:
                                        moved_atoms.append(l)
            else:
                l_bond_network = bond_network.copy()
                l_bond_network.remove_edge(bond[0], bond[1])
                for k in nx.connected_components(l_bond_network):
                    if bond[1] in k:
                        moved_atoms = k
        elif bond_type == 'angle':
            # All in ring so can't do usual
            if bond[0] in ring_atoms and bond[1] in ring_atoms and \
                    bond[2] in ring_atoms:
                # Make atom two and it's neighbours not in the ring
                neighbours = bond_network.neighbors(bond[2])
                moved_atoms = []
                moved_atoms.append(bond[2])
                # Add each neighbour that's not in loop
                for j in neighbours:
                    if j not in ring_atoms:
                        moved_atoms.append(j)
                        # And all its connections too
                        l_bond_network = bond_network.copy()
                        l_bond_network.remove_edge(bond[2], j)
                        for k in nx.connected_components(l_bond_network):
                            if j in k:
                                for l in k:
                                    if l != j:
                                        moved_atoms.append(l)
            elif bond[1] in ring_atoms and bond[2] in ring_atoms:
                l_bond_network = bond_network.copy()
                l_bond_network.remove_edge(bond[1], bond[0])
                for k in nx.connected_components(l_bond_network):
                    if bond[0] in k:
                        moved_atoms = k
            else:
                l_bond_network = bond_network.copy()
                l_bond_network.remove_edge(bond[1], bond[2])
                for k in nx.connected_components(l_bond_network):
                    if bond[2] in k:
                        moved_atoms = k
        return list(moved_atoms)

    def rotate_dihedral(self, bond: list, angle: float,
                        moved_atoms: list) -> None:
        """ Perform rotation about central bond to change dihedral angle
            defined between four consecutive atoms """
        atom1 = self.get_atom(bond[0]).copy()
        self.position = self.position.reshape(-1, 3)
        self.position -= atom1
        atom2 = self.position[bond[1]].copy().reshape(-1, 3)
        # Rotate whole molecule so bond_vector lies on x
        unit_vector = np.array([[1, 0, 0]])
        best_rotation, dist = \
            rotations.align_vectors(unit_vector,
                                    atom2 / np.linalg.norm(atom2))
        self.position = best_rotation.apply(self.position)
        undo_rotation = np.transpose(best_rotation.as_matrix())
        # Perform rotation of selected atoms about x axis by angle
        rot_matrix = self.get_rotation_matrix(angle*(np.pi/180.0))
        for i in moved_atoms:
            self.position[i] = np.matmul(rot_matrix, self.position[i])
        # Undo rotation onto x
        rotation = rotations.from_matrix(undo_rotation)
        self.position = rotation.apply(self.position)
        # Undo translation to origin
        self.position += atom1
        self.position = self.position.reshape([-1])

    def get_rotation_matrix(self, angle: float) -> NDArray:
        """ Generate rotation matrix for rotation about x axis """
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, np.cos(angle), -1.0*np.sin(angle)],
                         [0.0, np.sin(angle), np.cos(angle)]])

    def rotate_angle(self, angle_atoms: list, angle: float,
                     moved_atoms: list) -> None:
        """ Rotate one of the bonds in order to change the angle
            between the two bonds defined by atom0-atom1 and
            atom1-atom2 """
        atom1 = self.get_atom(angle_atoms[0]).copy()
        atom2 = self.get_atom(angle_atoms[1]).copy()
        atom3 = self.get_atom(angle_atoms[2]).copy()
        # Get the angle vectors
        bond_vector1 = (atom2 - atom1).reshape(-1, 3)
        bond_vector2 = (atom3 - atom2).reshape(-1, 3)
        # Get normal vector
        cross_prod = np.cross(bond_vector1, bond_vector2)
        normal_vec = cross_prod / np.linalg.norm(cross_prod)
        # Set norm of normal vector to angle displacement
        normal_vec *= angle*(np.pi/180.0)
        rot_vec = rotations.from_rotvec(normal_vec)
        self.position = self.position.reshape(-1, 3) - atom2
        for j in moved_atoms:
            self.position[j] = rot_vec.apply(self.position[j])
        self.position = self.position + atom2
        self.position = self.position.flatten()

    def change_bond_length(self, bond: list, length: float,
                           moved_atoms: list) -> None:
        """ Modify the selected bond length by distance length
            along the current bond vector """
        atom1 = self.get_atom(bond[0])
        atom2 = self.get_atom(bond[1])
        # Get bond vector
        bond_vector = atom2 - atom1
        # Move all of those atoms by specified amount
        self.position = self.position.reshape(-1, 3)
        for j in moved_atoms:
            self.position[j] += length*bond_vector
        self.position = self.position.reshape(-1)

    def change_bond_lengths(self, bonds: list, step: list,
                            bond_network: nx.Graph) -> None:
        """ Given a list of bonds change their bond lengths by step """
        for count, i in enumerate(bonds, 0):
            moved_atoms = self.get_movable_atoms(i, 'length', bond_network)
            self.change_bond_length(i, step[count], moved_atoms)

    def change_bond_angles(self, angles: list, step: list,
                           bond_network: nx.Graph) -> None:
        """ Given a list of bond angles, update them by all by step """
        for count, i in enumerate(angles, 0):
            moved_atoms = self.get_movable_atoms(i, 'angle', bond_network)
            self.rotate_angle(i, step[count], moved_atoms)

    def change_dihedral_angles(self, bond_dihedrals: list, step: list,
                               bond_network: nx.Graph) -> None:
        """ Update a list of dihedrals by angles specified in step """
        for count, i in enumerate(bond_dihedrals, 0):
            moved_atoms = self.get_movable_atoms([i[1], i[2]], 'dihedral',
                                                 bond_network)
            self.rotate_dihedral([i[1], i[2]], step[count], moved_atoms)

    def get_bond_angle_info(self) -> tuple:
        """ Given the current Cartesian coordinates of the atoms in
            self.position compute all dihedrals, angles and bond lengths
            needed to define the molecular configuration """
        nl = neighborlist.NeighborList(self.natural_cutoffs,
                                       self_interaction=False)
        self.atoms.set_positions(self.position.reshape(self.n_atoms, 3))
        nl.update(self.atoms)
        self.analysis = Analysis(self.position, nl=nl)
        # Already have dihedrals in self.rotatable_dihedrals, find their angles
        dihedral_angles = []
        for i in self.rotatable_dihedrals:
            angle = self.atoms.get_dihedral(i[0], i[1], i[2], i[3])
            dihedral_angles.append(angle)
        # Next find all the bond angles
        uniq_angles = self.analysis.unique_angles[0]
        angles = []
        for i in range(self.n_atoms):
            if uniq_angles[i]:
                for j in uniq_angles[i]:
                    k = [i]+list(j)
                    angles.append(k)
        # Only retain n-1 bond angles for each central atom as the final
        # is defined by the rest
        angles = self.remove_repeat_angles(angles)
        # Then compute their values
        bond_angles = []
        for i in angles:
            angle = self.atoms.get_angle(i[0], i[1], i[2])
            bond_angles.append(angle)
        # Get all the bonds
        uniq_bonds = self.analysis.unique_bonds[0]
        bonds = []
        for i in range(self.n_atoms):
            if uniq_bonds[i]:
                for j in uniq_bonds[i]:
                    bonds.append([i, j])
        # And compute their lengths
        bond_lengths = []
        for i in bonds:
            coords1 = self.get_atom(i[0])
            coords2 = self.get_atom(i[1])
            bond_lengths.append(np.linalg.norm(coords1 - coords2))
        # Make dictionary to pair the data
        return bonds, bond_lengths, angles, bond_angles, \
            self.rotatable_dihedrals, dihedral_angles

    def remove_repeat_angles(self, angles: list) -> list:
        """ To change between conformations we can remove one angle
            centered on each atom as it is specified by the rest """
        central_atoms = set([i[1] for i in angles])
        unique_angles = []
        for i in central_atoms:
            specific_angles = []
            for j in angles:
                if j[1] == i:
                    specific_angles.append(j)
            if len(specific_angles) == 1:
                unique_angles.append(specific_angles)
            else:
                unique_angles.append(specific_angles[:-1])
        return [x for xs in unique_angles for x in xs]

    def get_specific_bond_angle_info(self, bonds: list, angles: list,
                                     dihedrals: list,
                                     permutation: NDArray) -> tuple:
        """ Find the bond lengths, bond angles and bond dihedrals for those
            specified in the input lists, accounting for the permutation of
            atoms according to permutation """

        # Set the atom positions to extract angles and dihedrals
        nl = neighborlist.NeighborList(self.natural_cutoffs,
                                       self_interaction=False)
        self.atoms.set_positions(self.position.reshape(self.n_atoms, 3))
        nl.update(self.atoms)
        self.analysis = Analysis(self.position, nl=nl)
        # Get the corresponding bond angles from bonds list
        bond_lengths = []
        for i in bonds:
            coords1 = self.get_atom(permutation[i[0]])
            coords2 = self.get_atom(permutation[i[1]])
            bond_lengths.append(np.linalg.norm(coords1-coords2))
        # Then angles from list angles
        bond_angles = []
        for i in angles:
            angle = self.atoms.get_angle(permutation[i[0]],
                                         permutation[i[1]],
                                         permutation[i[2]])
            bond_angles.append(angle)
        # And dihedrals from list dihedrals
        dihedral_angles = []
        for i in dihedrals:
            angle = self.atoms.get_dihedral(permutation[i[0]],
                                            permutation[i[1]],
                                            permutation[i[2]],
                                            permutation[i[3]])
            dihedral_angles.append(angle)
        return bond_lengths, bond_angles, dihedral_angles

    def remove_atom_clashes(self, force_field: type) -> None:
        """ Check for clashes between atoms. Any atom pairs closer than
            0.8*natural_cutoffs can cause a failure for electronic structure
            calculations so locally minimise to remove clashes with stable
            empirical force field """

        clash = False
        # Loop over all atom pairs in the molecule
        for i in range(self.n_atoms-1):
            for j in range(i+1, self.n_atoms):
                atom1 = self.get_atom(i)
                atom2 = self.get_atom(j)
                # Compare the bond length to the average for these atom types
                bond_length = np.linalg.norm(atom1 - atom2)
                allowed_bond_length = 0.9*(self.natural_cutoffs[i] +
                                           self.natural_cutoffs[j])
                if bond_length < allowed_bond_length:
                    clash = True
        # Loosely converge with force field that does not fail at high overlap
        if clash:
            force_field.set_xyz(self.position)
            AllChem.MMFFOptimizeMolecule(force_field.molecule)
            min_position = force_field.molecule.GetConformer().GetPositions()
            self.position = min_position.flatten()
