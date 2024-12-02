""" Module that contains the classes for evaluating similarity of
    atomic or molecular conformations based on the rotational distance
    that is required to move between them """

import numpy as np
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from .similarity import StandardSimilarity
from ase.geometry.analysis import Analysis
from ase import neighborlist
from nptyping import NDArray


class DihedralSimilarity(StandardSimilarity):

    """
    Description
    ------------
    Class to compute the dihedral similarity between two molecular conformations
    assuming only the same bonding framework, but not ordering. Finds the optimal
    permutational alignment of both structures considering only interconversion
    by dihedral rotation. Useful for understanding physical interconvertibility
    and preparing minimal action paths, explained in .

    Attributes
    -----------
    distance_criterion : float
        The distance under which two minima are considered the same, for
        proportional_distance set to True this is not an absolute value,
        but the proportion of the total function range
    energy_criterion : float
        The value that the difference in function values must be below
        to be considered the same
    n_paths : int
        The number of shortest paths to be returned, each with a separate
        permutation. Only 1 is allowed for use within landscape searching,
        more are allowed as a standalone search
    """

    def __init__(self, distance_criterion: float=10.0,
                 energy_criterion: float=1e-3):
        self.distance_criterion = distance_criterion
        self.energy_criterion = energy_criterion

    ### ROUTINES FOR LANDSCAPE SEARCHING COMPATABILITY

    def optimal_alignment(self, coords1: type, position2: NDArray) -> list:
        """ Find the n shortest paths between coords1 and coords2 assuming
            only dihedral rotation is possible """
        # Enforce a standard clockwise orientation for all permutable atoms
        bond_network1 = self.prepare_network(coords1)
        # Make the corresponding second coordinates object
        coords2 = deepcopy(coords1)
        coords2.position = position2
        bond_network2 = self.prepare_network(coords2)
        # Match the network nodes accounting for 3D rotations
        mappings = self.match_graphs(bond_network1, bond_network2, coords1.n_atoms)
        # Store the best mappings from each different base_mapping
        best_mappings = []
        best_dists = []
        for base_mapping in mappings:
            mapped_network2 = nx.relabel_nodes(bond_network2, base_mapping)
            # Remove the placeholder atoms
            network2 = self.remove_placeholders(mapped_network2)
            # After matching take the version without placholder nodes for augmenting
            # Get the permutable bonds and add edges attributes
            permutable_bonds = self.augment_network_edges(coords1, network2)
            # Add the info to the placeholder graph again
            nx.set_edge_attributes(mapped_network2, network2.edges)
            # Get rotatable bonds in the molecule, ordered into containing sets
            ordered_sets = self.order_bonds(permutable_bonds, mapped_network2)
            # Get all possible mappings of the permutations
            mappings = self.get_mappings(ordered_sets, mapped_network2)
            # Get the mapping that gives the smallest dihedral distance
            optimal_mappings, optimal_dists = \
                self.get_optimal_similarity(coords1, coords2, base_mapping,
                                            mappings, mapped_network2,
                                            ordered_sets, 1)
            # Undo the initial mapping from 2 -> 1
            for c, i in enumerate(optimal_mappings):
                best_mapping = self.undo_mapping(i, base_mapping,
                                                 coords1.n_atoms)
                best_mappings.append(best_mapping)
                best_dists.append(optimal_dists[c])
        # Take the optimal n_path examples across all base_mappings
        best_idxs = np.argsort(best_dists)[:1]
        final_mappings = []
        final_dists = []
        for i in best_idxs:
            final_mappings.append(best_mappings[i])
            final_dists.append(best_dists[i])
        return final_dists[0], coords1.position, position2, final_mappings[0]

    def test_same(self, coords1: type, coords2: NDArray,
                  energy1: float, energy2: float) -> bool:
        """ Test if two structures are the same to within a distance and energy
            criterion after finding optimal dihedral permutational alignment """
        within_distance = \
            self.closest_distance(coords1, coords2) < self.distance_criterion
        within_energy = np.abs(energy1-energy2) < self.energy_criterion
        return bool(within_distance and within_energy)

    def closest_distance(self, coords1: type, coords2: NDArray) -> float:
        """ Permute the two structures and return the optimised distance """
        return self.optimal_alignment(coords1, coords2)[0]

    ### PATH FINDING ROUTINE

    def find_best_alignments(self, coords1: type, coords2: type,
                             n_paths: int=1) -> list:
        """ Find the n shortest paths between coords1 and coords2 assuming
            only dihedral rotation is possible """
        # Enforce a standard clockwise orientation for all permutable atoms
        bond_network1 = self.prepare_network(coords1)
        bond_network2 = self.prepare_network(coords2)
        # Match the network nodes accounting for 3D rotations
        mappings = self.match_graphs(bond_network1, bond_network2, coords1.n_atoms)
        # Store the best mappings from each different base_mapping
        best_mappings = []
        best_dists = []
        for base_mapping in mappings:
            mapped_network2 = nx.relabel_nodes(bond_network2, base_mapping)
            # Remove the placeholder atoms
            network2 = self.remove_placeholders(mapped_network2)
            # After matching take the version without placholder nodes for augmenting
            # Get the permutable bonds and add edges attributes
            permutable_bonds = self.augment_network_edges(coords1, network2)
            # Add the info to the placeholder graph again
            nx.set_edge_attributes(mapped_network2, network2.edges)
            # Get rotatable bonds in the molecule, ordered into containing sets
            ordered_sets = self.order_bonds(permutable_bonds, mapped_network2)
            # Get all possible mappings of the permutations
            mappings = self.get_mappings(ordered_sets, mapped_network2)
            # Get the mapping that gives the smallest dihedral distance
            optimal_mappings, optimal_dists = \
                self.get_optimal_similarity(coords1, coords2, base_mapping,
                                            mappings, mapped_network2,
                                            ordered_sets, n_paths)
            # Undo the initial mapping from 2 -> 1
            for c, i in enumerate(optimal_mappings):
                best_mapping = self.undo_mapping(i, base_mapping,
                                                 coords1.n_atoms)
                best_mappings.append(best_mapping)
                best_dists.append(optimal_dists[c])
        # Take the optimal n_path examples across all base_mappings
        best_idxs = np.argsort(best_dists)[:n_paths]
        final_mappings = []
        final_dists = []
        for i in best_idxs:
            final_mappings.append(best_mappings[i])
            final_dists.append(best_dists[i])
        return final_mappings, final_dists

    ### PREPARE THE NETWORKS

    def prepare_network(self, coords: type) -> nx.Graph:
        """ Prepare the two networks in a consistent order and align them
            so they are correctly aligned up to permutations of exchangeable
            atoms that can be switched by dihedral rotation only """
        # Get the initial bond networks
        bond_network = self.get_bonding(coords)
        # Reorder the permutable atoms clockwise
        mapping, perm_atoms = self.clockwise_reorder(coords, bond_network)
        # No need to relabel, just add the placeholder nodes now
        return self.add_placeholders(bond_network, perm_atoms,
                                     mapping, coords.position)
    
    def get_bonding(self, coords: type) -> nx.Graph:
        """ Generate the bonding network for given molecule object, coords """
        # Update the neighbours for each atom before computing bonding
        nl = neighborlist.NeighborList(coords.natural_cutoffs,
                                       self_interaction=False)
        coords.atoms.set_positions(coords.position.reshape(coords.n_atoms, 3))
        nl.update(coords.atoms)
        # Analyse the bonding of the structure
        coords.analysis = Analysis(coords.position, nl=nl)
        uniq_bonds = coords.analysis.all_bonds[0]
        # Use the set of bonds to form the adjacency matrix
        adj_matrix = np.zeros((coords.n_atoms, coords.n_atoms), dtype=int)
        for c, i in enumerate(uniq_bonds):
            for j in i:
                adj_matrix[c, j] = 1
        # Make a networkx graph from the connectivity
        G = nx.from_numpy_array(adj_matrix)
        # Assign each node its corresponding element as attribute
        node_elements = dict(zip(range(coords.n_atoms), coords.atom_labels))
        nx.set_node_attributes(G, node_elements, name='element')
        return G

    def clockwise_reorder(self, coords: type, bond_network: nx.Graph) -> tuple:
        """ Reorder all the permutable atoms such that the atom index
            index increases with clockwise rotation """
        # First the permutable atoms, separated into those that can exchange
        # by rotation and those that cannot
        p_bonds, p_atoms, r_bonds, r_atoms = \
            self.get_reorderable_atoms(coords, bond_network)
        # Postprocess these set of bonds to remove repeats
        r_bonds, r_atoms = \
            self.postprocess_bonds(p_bonds, r_bonds, r_atoms, bond_network)
        # Set clockwise order for sets of permutable atoms related by rotation
        mapping1 = self.reorder_permutable_atoms(p_bonds, p_atoms,
                                                 bond_network, coords.position)
        # And those that are permutable, but not related by dihedral rotation
        mapping2 = self.reorder_rotatable_atoms(r_bonds, r_atoms, bond_network,
                                                coords.position)
        # Return combined mappings to get for full molecule
        return {**mapping1, **mapping2}, p_atoms+r_atoms

    def in_same_ring(self, atom1: int, atom2: int,
                     bond_network: nx.Graph) -> bool:
        """ Determine if two atoms belong to the same ring in the graph """
        loop_nodes = nx.cycle_basis(bond_network)
        for i in loop_nodes:
            if atom1 in i:
                if atom2 in i:
                    return True
        return False

    def get_connected_graphs(self, bond: list, perm_atoms: list,
                             bond_network: nx.Graph) -> list:
        """ Get the connected subgraph for each atom in perm_atoms, 
            assuming we have severed bond """
        connections = []
        # Go through each in turn to maintain order
        for i in perm_atoms:
            # If both atoms in a loop then treat separately
            if self.in_same_ring(bond[1], i, bond_network):
                conn = self.get_ring_connectivity(bond[1], i, perm_atoms,
                                                  bond_network)
            else:
                conn = self.get_connectivity(bond[1], i, bond_network)
            # Add the connections to the list
            connections.append(conn)
        return connections

    def get_connectivity(self, atom1: int, atom2: int,
                         G: nx.Graph) -> nx.Graph:
        """ Find the atoms connected to atom2 when we break the
            bond between atom1 and atom2 """
        H = G.copy()
        H.remove_edge(atom1, atom2)
        desc = nx.descendants(H, atom2)
        desc.add(atom2)
        return nx.induced_subgraph(H, desc)

    def get_ring_connectivity(self, atom1: int, atom2: int, perm_atoms: list,
                              G: nx.Graph) -> nx.Graph:
        """ Find the connected set of atoms given that atom2
            belongs to a loop """
        # Make the initial copy to cut edges from
        H = G.copy()
        # Get the loop atoms containing perm_atoms
        loop_nodes = nx.cycle_basis(G)
        # How many rings does each atom belong to
        membership = []
        for c, j in enumerate(loop_nodes):
            if atom1 in j and atom2 in j:
                membership.append(j)
        # Now test each loop in turn. We will run around the ring until we
        # return to the same point or hit another member of set
        for i in membership:
            # Get the indices for the current ring
            current_loop = i.copy()
            # Rotate the list so the reference atom is first
            idx2 = [c for c, j in enumerate(current_loop) if j == atom2][0]
            current_loop = np.roll(current_loop, -idx2)
            # Find the second element of the bond
            idx1 = [c for c, j in enumerate(current_loop) if j == atom1][0]
            # Determine the direction we will rotate
            if idx1 == 1:
                current_loop = np.roll(current_loop[::-1], 1)
            # Go around the ring checking for other permutable atoms
            for c, j in enumerate(current_loop[1:], 1):
                # Break the bond halfway around the loop
                if j in perm_atoms:
                    if c % 2 == 0:
                        idx1 = current_loop[int(c/2.0)]
                        idx2 = current_loop[int(c/2.0)-1]
                        if H.has_edge(idx1, idx2):
                            H.remove_edge(idx1, idx2)
                    else:
                        idx1 = current_loop[int((c/2.0)+0.5)]
                        idx2 = current_loop[int((c/2.0)-0.5)]
                        if H.has_edge(idx1, idx2):
                            H.remove_edge(idx1, idx2)
                    break
                # Break the bond before we get back to the start
                if j == atom1:
                    if H.has_edge(current_loop[c-1], current_loop[c-2]):
                        H.remove_edge(current_loop[c-1], current_loop[c-2])
                    break
        # Get the connections for each of these two options
        H.remove_edge(atom1, atom2)
        desc = nx.descendants(H, atom2)
        desc.add(atom2)
        return nx.induced_subgraph(H, desc)

    def get_reorderable_atoms(self, coords: type,
                              bond_network: nx.Graph) -> tuple:
        """ Find the set of locally permutable atoms, separated into those
            that can be related by dihedral rotation and those that cannot """
        # Get the rotatable bonds
        bonds = [i[1:3] for i in coords.rotatable_dihedrals]
        # Initialise lists to store known permutable information
        permutable_bonds = []
        permutable_atoms = []
        rotatable_bonds = []
        rotatable_atoms = []
        # Test each bond to see if connected atoms are permutable
        for i in bonds:
            # Forwards along the bond
            permutable, interchangeable, permutable_idxs = \
                self.get_permutable_information(i, bond_network)
            # If permutable then add to either permutable or rotatable
            if permutable:
                if interchangeable:
                    permutable_bonds.append(i)
                    permutable_atoms.append(permutable_idxs)
                else:
                    rotatable_bonds.append(i)
                    rotatable_atoms.append(permutable_idxs)
            # Then the reverse direction
            permutable, interchangeable, permutable_idxs = \
                self.get_permutable_information(i[::-1], bond_network)
            # If permutable then add to either permutable or rotatable
            if permutable:
                if interchangeable:
                    permutable_bonds.append(i[::-1])
                    permutable_atoms.append(permutable_idxs)
                else:
                    rotatable_bonds.append(i[::-1])
                    rotatable_atoms.append(permutable_idxs)
        return permutable_bonds, permutable_atoms,\
            rotatable_bonds, rotatable_atoms

    def get_permutable_atoms(self, bond: list, bond_network: nx.Graph) -> tuple:
        """ For a given bond, determine if atoms are permutable, and whether
            they can be related by rotation """
        # Find the neighbouring atoms for the final atom of the bond
        neighbours = [j for j in bond_network[bond[1]] if j != bond[0]]
        # Find the connected subsets for each neighbour
        connections = self.get_connected_graphs(bond, neighbours, bond_network)
        # Test which are the same on the basis of their connections
        permutable_atoms = self.get_permutable_sets(connections)
        return permutable_atoms, neighbours
    
    def get_permutable_information(self, bond:list,
                                   bond_network: nx.Graph) -> tuple:
        """ Return the initial information needed to assess if sets of 
            atoms are permutable when preparing network """
        perm_atoms, neighbours = self.get_permutable_atoms(bond, bond_network)
        if len(perm_atoms) == 0:
            return False, False, []
        # All connections are the same so permutable related by rotation
        if len(perm_atoms[0]) == len(neighbours):
            permutable_idxs = [neighbours[j] for j in perm_atoms[0]]
            return True, True, permutable_idxs
        # Not related by rotation, but still need to check if permutable
        else:
            for i in perm_atoms:
                # We do still have some permutable neighbours
                if len(i) > 1:
                    permutable_idxs = [neighbours[j] for j in i]
                    return True, False, permutable_idxs
            return False, False, []

    def postprocess_bonds(self, permutable_bonds: list, rotatable_bonds: list,
                          rotatable_atoms: list, bond_network: nx.Graph):
        """ Remove any repeated atoms that appear from two separate bonds """
        permutable_idxs = [i[1] for i in permutable_bonds]
        pruned_rotatable_bonds = []
        pruned_rotatable_atoms = []
        for c, i in enumerate(rotatable_bonds):
            allowed1 = False
            allowed2 = False
            # Check that it doesn't match any atom selected as permutable
            if i[1] not in permutable_idxs:
                allowed1 = True
            # Check current bonds
            pruned_idxs = [j[1] for j in pruned_rotatable_bonds]
            if i[1] not in pruned_idxs:
                allowed2 = True
            #    allowed2 = True
            if allowed1 and allowed2:
                pruned_rotatable_bonds.append(i)
                pruned_rotatable_atoms.append(rotatable_atoms[c])
        return pruned_rotatable_bonds, pruned_rotatable_atoms

    def reorder_permutable_atoms(self, bonds: list, atoms: list,
                                 bond_network: nx.Graph,
                                 position: NDArray) -> dict:
        """ Find the mapping that orders all permutable, and rotatable,
            atoms into a clockwise rotational direction """
        mapping = {}
        # Loop over the set of bonds, getting the clockwise permutation
        for c1, i in enumerate(bonds):
            atom_order = self.clockwise_order_perm(i, atoms[c1], bond_network,
                                                   position)
            # And combining them all into a single mapping
            for c2, j in enumerate(atoms[c1]):
                mapping[j] = atoms[c1][atom_order[c2]]
        return mapping

    def clockwise_order_perm(self, bond: list, atoms: list,
                             bond_network: nx.Graph,
                             position: NDArray) -> list:
        """ Find the clockwise permutation of permutable atoms """
        # Take any atom connected to the first atom in the bond
        idx1 = [j for j in bond_network[bond[0]] if j != bond[0]][0]
        # Get all the dihedrals
        dihedrals = []
        for i in atoms:
            dihedrals.append(self.get_dihedral(position,
                                               [idx1, bond[0], bond[1], i]))
        # Order by increasing angle
        return np.argsort(dihedrals)

    def reorder_rotatable_atoms(self, bonds: list, atoms: list,
                                bond_network: nx.Graph,
                                position: NDArray) -> dict:
        """ Find the mapping that orders all permutable, but not rotatable,
            atoms into a clockwise rotational direction """
        mapping = {}
        # Loop over the set of bonds, getting the clockwise permutation
        for c1, i in enumerate(bonds):
            atom_order = self.clockwise_order_rot(i, atoms[c1], bond_network,
                                                  position)
            # And combining them all into a single mapping
            for c2, j in enumerate(atoms[c1]):
                mapping[j] = atoms[c1][atom_order[c2]]
        return mapping

    def clockwise_order_rot(self, bond: list, atoms: list,
                            bond_network: nx.Graph, position: NDArray) -> list:
        """ Find the clockwise permutation of permutable atoms """
        # Get the neighbours and find the unique element
        neighbours = [j for j in bond_network[bond[1]] if j != bond[0]]
        idx_uniq = [j for j in neighbours if j not in atoms][0]
        # Get all the dihedrals
        dihedrals = []
        for i in atoms:
            dihedrals.append(self.get_dihedral(position,
                                               [idx_uniq, bond[0], bond[1], i]))
        # Order by increasing angle
        return np.argsort(dihedrals)

    def add_placeholders(self, bond_network: nx.Graph, p_atoms: list,
                         mapping: nx.Graph, position: NDArray) -> nx.Graph:
        """ Add extra nodes at each permutable atom to distinguish them in the 
            graph isomorphism. Allowing 3D structure to be maintained in 2D """
        # Get the initial number of nodes
        idx = bond_network.number_of_nodes()
        # Set all the placeholder values to False to distinguish
        nx.set_node_attributes(bond_network, False, 'placeholder')
        # Loop over each set of permutable atoms
        for i in p_atoms:
            # If we have three planar connections then no need
            if len(i) == 2:
                if self.check_planarity(i, bond_network, position):
                    continue
            # Add an increasing number of nodes for each atom
            for c, j in enumerate(i):
                for k in range(c):
                    bond_network.add_node(idx, placeholder=True)
                    bond_network.add_edge(idx, mapping[j])
                    idx += 1
        return bond_network

    def check_planarity(self, perm_atoms: list,
                        bond_network: nx.Graph,
                        position: NDArray) -> bool:
        """ Test if the neighbours of an atom are planar """
        # Get the mutual neighbour for these perm_atoms
        neighbours = []
        for i in perm_atoms:
            neighbours.append([j for j in bond_network[i]])
        central_atom = list(set.intersection(*map(set, [neighbours[0],
                                                        neighbours[1]])))[0]
        neighbour_atoms = [j for j in bond_network[central_atom]]
        # Remove any neighbours that are placeholders from previous
        n_atoms = int(position.size/3)
        neighbour_atoms = [j for j in neighbour_atoms if j < n_atoms]
        # Get their positions
        connected_atoms = [position[central_atom*3: central_atom*3+3]]
        for i in neighbour_atoms:
            connected_atoms.append(position[i*3:(i*3)+3])
        # Get the coordinates of the loop atoms
        connected_atoms = np.asarray(connected_atoms)
        # Pick three points to define a plane
        a, b, c = connected_atoms[0:3]
        normal_vec = np.cross((b-a), (c-a))
        normal_vec /= np.linalg.norm(normal_vec)
        for j in range(len(neighbour_atoms)-3):
            d = connected_atoms[j+3]
            diff_vec = (a-d) / np.linalg.norm(a-d)
            if np.abs(np.dot(diff_vec, normal_vec)) > 1e-1:
                return False
        return True

    def remove_placeholders(self, bond_network: nx.Graph) -> nx.Graph:
        """ Remove any additional nodes we have added, identifiable with
            attribute 'placeholder'=True """
        new_network = bond_network.copy()
        # Find the placeholder nodes and select them for removal
        placeholder_nodes = \
            [x for x, y in bond_network.nodes(data=True) if y['placeholder']]
        new_network.remove_nodes_from(placeholder_nodes)
        return new_network

    ### PREPARE AND ANALYSE THE NETWORK

    def match_graphs(self, network1: nx.Graph, network2: nx.Graph,
                     n_atoms: int) -> nx.Graph:
        """ Find all possible matching between networks accouting for
            atom elements """
        mappings = nx.vf2pp_all_isomorphisms(network1, network2,
                                             node_label='element')
        mappings = [i for i in mappings]
        return self.retain_unique_mappings(mappings, n_atoms)
        
    def retain_unique_mappings(self, mappings: list, n_atoms: int) -> list:
        """ Retain only the unique mappings that have keys within the
            atom indices """
        # First test that if all elements within the atom nodes are the same
        new_mappings = []
        for i in mappings:
            tmp_list = []
            for c in range(n_atoms):
                tmp_list.append(i[c])
            new_mappings.append(tmp_list)
        # Then get the unique set
        unique_mappings = set(tuple(i) for i in new_mappings)
        # Convert the unique elements back into dictionaries again
        final_mappings = []
        for i in unique_mappings:
            dict_elem = dict()
            for c, j in enumerate(i):
                dict_elem[c] = j
            final_mappings.append(dict_elem)
        return final_mappings

    def augment_network_edges(self, coords: type,
                              bond_network: nx.Graph) -> list:
        """ Analyse the bonding network graph to get the properties
            associated with each of the bonds """
        # Set the default attributes associated with bonds 
        for u, v in bond_network.edges:
            bond_network[u][v]["permutable"] = False
            bond_network[u][v]["rotatable"] = False
            bond_network[u][v]["atoms"] = [[], []]
            bond_network[u][v]["direction"] = []
        # Get the rotatable bonds as from ase
        bonds = [i[1:3] for i in coords.rotatable_dihedrals]
        permutable_bonds = []
        # And get the properties for each in turn
        for i in bonds:
            # Update the rotatable label
            bond_network[i[0]][i[1]]["rotatable"] = True
            # Find out which atoms are permutable
            perm_atoms1 = \
                self.get_permutable_idxs(i, bond_network)
            perm_atoms2 = \
                self.get_permutable_idxs(i[::-1], bond_network)
            # If multiple atoms are permutable update the labels
            if len(perm_atoms1) > 1 or len(perm_atoms2) > 1:
                bond_network.edges[i[0], i[1]]["permutable"] = True
                permutable_bonds.append([i[0], i[1]])
                # Depending on which side store the positive or negative bond
                if len(perm_atoms1) > 1:
                    bond_network.edges[i[0], i[1]]["direction"] = [i[0], i[1]]
                    bond_network.edges[i[0], i[1]]["atoms"] = [perm_atoms1,
                                                               perm_atoms2]
                if len(perm_atoms2) > 1:
                    bond_network.edges[i[0], i[1]]["direction"] = [i[1], i[0]]
                    bond_network.edges[i[0], i[1]]["atoms"] = [perm_atoms2,
                                                               perm_atoms1]
            else:
                bond_network.edges[i[0], i[1]]["direction"] = [i[0], i[1]]
                bond_network.edges[i[0], i[1]]["atoms"] = [perm_atoms1,
                                                           perm_atoms2]
        return permutable_bonds

    def get_connections(self, atom1: int, neighbours: list,
                        bonding_network: nx.Graph) -> list:
        """ Get the indices of all connected atoms, assuming breaking of atom1-
            neighbour bond, for each element of neighbours """
        connected_sets = []
        for i in neighbours:
            H = bonding_network.copy()
            H.remove_edge(atom1, i)
            tmp = list(nx.node_connected_component(H, i))
            connected_sets.append(tmp)
        return connected_sets

    def get_permutable_sets(self, connections: list) -> list:
        """ Find which of the sets of elements are the same """
        # Check which of these networks are isomorphic
        repeats = []
        for i in range(len(connections)-1):
            flat_rep = [x for xs in repeats for x in xs]
            if i in flat_rep:
                continue
            tmp_repeats = [i]
            for j in range(i+1, len(connections)):
                if nx.vf2pp_is_isomorphic(connections[i], connections[j],
                                          node_label='element'):
                    tmp_repeats.append(j)
            repeats.append(tmp_repeats)
        return repeats

    def get_permutable_idxs(self, bond: list, bond_network: list) -> list:
        """ Find the set of indices for the atoms that can be permuted.
            If all, store all. If not, store one with unique environment """
        # Get the permutable atoms and neighbours
        permutable_atoms, neighbours = \
            self.get_permutable_atoms(bond, bond_network)
        if permutable_atoms == []:
            return neighbours
        repeated = set([x for xs in permutable_atoms for x in xs])
        # All atoms are permutable so store all
        if len(repeated) == len(neighbours):
            permutable_idxs = [neighbours[j] for j in permutable_atoms[0]]
            return permutable_idxs
        else:
            # If first element is unique then return that
            if len(permutable_atoms[0]) == 1:
                return [neighbours[0]]
            # If not then find the later index that is not permutable
            else:
                range_set = set(range(len(neighbours)))
                missing = list(range_set.difference(repeated))
                return [neighbours[missing[0]]]

    ### ORDER THE BONDS INTO CONTAINING SETS

    def order_bonds(self, bonds: list, bond_network: nx.Graph) -> list:
        """ Reorder the bonds so they move outwards into the molecule, each to
            the right in the list is moved by rotating those to the left """
        # Get all the bonds that each bond contains in its connections
        contains = self.containing_bonds(bonds, bond_network)
        # Get the ordered bonds that form subsets
        connected_sets, connected_sizes = \
            self.get_connected_sets(bonds, contains)
        return self.get_ordered_sets(connected_sets, connected_sizes)

    def containing_bonds(self, bonds: list, bond_network: nx.Graph) -> list:
        """ For each bond find the other bonds contained in its
            connected atoms """
        contains = []
        # Loop over every bond
        for i in bonds:
            # Find the connected atoms to bonds[1] after broken
            H = bond_network.copy()
            H.remove_edge(i[0], i[1])
            connections = list(nx.node_connected_component(H, i[1]))
            tmp_contains = []
            # Check if all other bonds are within its connections
            for j in bonds:
                if set(j).issubset(set(connections)):
                    tmp_contains.append(j)
            # Add all containing bonds as element of the list
            contains.append(tmp_contains)
        return contains

    def get_connected_sets(self, bonds: list, contains: list) -> list:
        """ Generate the sets of bonds that depend upon each other """
        G = nx.Graph()
        for c1, i in enumerate(bonds):
            G.add_node(c1, bond=i)
        for c1, i in enumerate(contains):
            for j in i:
                # Find the matching bond idx
                idx = [c2 for c2, k in enumerate(bonds) if k == j][0]
                G.add_edge(c1, idx)
        # Returns subgraphs ordered by node connections
        components = list(nx.connected_components(G))
        connected_sets = []
        connected_sizes = []
        for i in components:
            tmp_bonds = []
            tmp_sizes = []
            for j in i:
                tmp_bonds.append(bonds[j])
                tmp_sizes.append(len(contains[j]))
            connected_sets.append(tmp_bonds)
            connected_sizes.append(tmp_sizes)
        return connected_sets, connected_sizes

    def get_ordered_sets(self, connected_sets: list,
                         connected_sizes: list) -> list:
        """ Order the sets such that rotating each bond moves
            all those at higher indices in the resulting list """
        sorted_sets = []
        for c, i in enumerate(connected_sizes):
            sorted_idxs = np.argsort(i)[::-1]
            sorted_set = [connected_sets[c][j] for j in sorted_idxs]
            sorted_sets.append(sorted_set)
        return sorted_sets

    ### GET MAPPINGS

    def get_mappings(self, ordered_sets: list, bond_network: nx.Graph) -> list:
        """ Generate all mappings given the sets of bonds that contain
            permutable atoms """
        set_mappings = []
        for i in ordered_sets:
            mappings = self.permute_set(i, bond_network)
            set_mappings.append(mappings)
        return set_mappings

    def permute_set(self, bond_set: list, bond_network: nx.Graph):
        """ Get all possible mappings for a given set of bonds with permutable
            atoms. Generate all cyclic permutations moving outwards """
        # Initial mapping is just each atom to itself
        mappings = [dict(zip(list(range(bond_network.number_of_nodes())),
                             list(range(bond_network.number_of_nodes()))))]
        # Iterate through the bonds to apply the permutation of each
        for i in bond_set:
            # Apply permutation to all previous mappings
            l_mappings = []
            for j in mappings:
                # Get the mapped network
                tmp_bond_network = bond_network.copy()
                tmp_bond_network = nx.relabel_nodes(tmp_bond_network, j)
                # Get the mapped bond and permutable atoms
                directed_bond, perm_atoms = \
                    self.get_bond_atoms(i, bond_network)
                mapped_bond = [j[k] for k in directed_bond]
                mapped_atoms = [j[k] for k in perm_atoms]
                # Get the possible mappings for this permutation
                tmp_mappings = self.get_permutation_mappings(mapped_bond,
                                                             mapped_atoms,
                                                             tmp_bond_network)
                if not tmp_mappings:
                    l_mappings = mappings
                    break
                # Update the old base mapping with the new and fill in
                tmp_mappings = \
                    self.update_mappings(tmp_mappings, j,
                                         bond_network.number_of_nodes())
                # Save each of these new mappings
                for k in tmp_mappings:
                    l_mappings.append(k)
            # Make these our reference to apply permutations to next iteration
            mappings = l_mappings
        return mappings

    def get_permutation_mappings(self, bond: list, perm_atoms: list,
                                 bond_network: list) -> list:
        """ Given a particular bond with permutable atoms, perm_atoms,
            find all the possible mapping that come from cyclically
            permuting these atoms, updating connected subgraphs """
        # Get all the connected subgraphs from perm_atoms
        connections = self.get_connected_graphs(bond, perm_atoms, bond_network)
        # Remove any placeholders on perm_atoms
        connections = self.remove_direct_placeholders(perm_atoms, connections,
                                                      bond_network)
        # Generate the set of allowed cyclic permutations
        permutations = self.get_cyclic_permutations(perm_atoms)
        mappings = []
        # Get the mappings for each possible permutation
        for i in permutations:
            connections2 = [connections[j] for j in i]
            # Call isomorphism all the way around to update each connection
            mapping = nx.vf2pp_isomorphism(connections[0], connections2[0],
                                           node_label='element')
            for j in range(1, len(permutations)):
                mapping.update(nx.vf2pp_isomorphism(connections[j],
                                                    connections2[j],
                                                    node_label='element'))
            # Store this mapping in the overall list
            mappings.append(mapping)
        return mappings

    def update_mappings(self, new_mappings: list, base_mapping: dict,
                        n_atoms: int) -> list:
        """ Fill in any missing parts of the mapping, if not present hasn't
            changed. Then update the base_mapping with each mapping in
            new_mappings to get complete set of new mappings """
        # Fill in the base mapping
        base_mapping = self.fill_mappings(base_mapping, n_atoms)
        # For each mapping we update the base and add updated
        updated_mappings = []
        for i in new_mappings:
            # Update the values
            updated_mappings.append(self.switch_labels(i, base_mapping))
        return updated_mappings

    def fill_mappings(self, mapping: dict, n_atoms: int) -> dict:
        """ Make sure the mapping has all elements, if any are missing
            add mapping to itself """
        for i in range(n_atoms):
            if i not in mapping:
                mapping[i] = i
        return mapping

    def switch_labels(self, mapping: dict, base_mapping: dict) -> dict:
        """ Update the base mapping given mapping to get the overall
            permutation vector """
        updated_mapping = base_mapping.copy()
        tmp_list = []
        for i in mapping:
            tmp_list.append(updated_mapping[mapping[i]])
        for c, i in enumerate(mapping):
            updated_mapping[i] = tmp_list[c]
        return updated_mapping

    def get_cyclic_permutations(self, permutable_atoms: list) -> list:
        """ Get all allowed cyclical permutations of a set of atoms """
        # Generate an initial list [0,1,2,...]
        base_permutation = list(range(len(permutable_atoms)))
        permutations = []
        # Roll it round as many times as necessary
        for i in range(len(permutable_atoms)):
            permutations.append(list(np.roll(base_permutation, i)))
        return permutations

    def get_bond_atoms(self, bond: list, bond_network: nx.Graph) -> list:
        """ Get the direction of the bond (towards any permutable atoms) 
            and its directly connected atoms """
        # Extract the information stored in the network edges
        conn_atoms = bond_network[bond[0]][bond[1]]["atoms"]
        directed_bond = bond_network[bond[0]][bond[1]]["direction"]
        # If first set of atoms are permutable then take these
        if len(conn_atoms[0]) > 1:
            perm_atoms = conn_atoms[0]
        else:
            perm_atoms = conn_atoms[1]
        return directed_bond, perm_atoms

    def remove_direct_placeholders(self, perm_atoms: list, connections: list,
                                   bond_network: nx.Graph) -> list:
        """ Remove any placeholder nodes directly connected to any
            element of perm_atoms """
        new_connections = []
        # Loop over each subgraph
        for i in connections:
            # Create a copy as we cannot modify original
            current = i.copy()
            # Find the placeholder nodes
            placeholder_nodes = \
                [x for x, y in i.nodes(data=True) if y['placeholder']]
            # Find which neighbours are in the graph
            direct = [j for j in i.nodes if j in perm_atoms][0]
            neighbours2 = list(i[direct])
            removals = \
                list(set(placeholder_nodes).intersection(set(neighbours2)))
            # Also account for any elements in the same ring as perm_atoms
            ring_atoms = []
            loop_nodes = nx.cycle_basis(bond_network)
            for j in loop_nodes:
                if set(j).intersection(set(perm_atoms)):
                    ring_atoms.append(j)
            ring_atoms = set([x for xs in ring_atoms for x in xs])
            ring_neighbours = []
            for j in ring_atoms:
                ring_neighbours.append(list(bond_network[j]))
            ring_neighbours = set([x for xs in ring_neighbours for x in xs])
            extra_removals = \
                list(set(placeholder_nodes).intersection(ring_neighbours))
            for j in extra_removals:
                removals.append(j)
            removals = list(set(removals))
            # Remove the relevant placeholder nodes from the network
            current.remove_nodes_from(removals)
            new_connections.append(current)
        return new_connections

    ### EVALUATE THE DIHEDRAL SIMILARITY

    def get_optimal_similarity(self, coords1: type, coords2: type,
                               base_mapping: dict,
                               mappings: list, bond_network: nx.Graph,
                               ordered_sets: list, n_paths: int) -> dict:
        """ Given a list of all allowed mappings find the one that has the minimal
            dihedral similarity """
        overall_mappings = []
        overall_dists = []
        computed_bonds = []
        # Loop over the set for each mapping
        for c, i in enumerate(mappings):
             # Set huge initial distances
            best_mappings = [None]*n_paths
            best_dists = np.full((n_paths), 1e30, dtype=float)
            bonds = self.get_modified_bonds(ordered_sets[c][0], coords1,
                                            bond_network)
            computed_bonds.append(bonds)
            # Loop over all possibilities for the given set and store the best
            for j in i:
                # Compute the dihedral angle difference
                delta_phi = self.get_dihedral_similarity(coords1, coords2,
                                                         base_mapping, j,
                                                         bond_network, bonds)

                # If better than seen then store as new best
                if delta_phi < best_dists[-1]:
                    best_dists, best_mappings = \
                        self.update_lists(best_dists, best_mappings,
                                          delta_phi, j)
            best_mappings, best_dists = self.prune_lists(best_mappings,
                                                         best_dists)
            # Keep just the atoms that are moved by the given mapping
            best_mappings = self.get_mapping_components(best_mappings,
                                                        ordered_sets[c][0],
                                                        bond_network)
            overall_mappings.append(best_mappings)
            overall_dists.append(best_dists)
        # Combine all the mappings of different parts together
        final_mappings, final_dists = \
            self.get_best_combinations(overall_mappings,
                                       overall_dists, n_paths)
        # Add the difference between rotatable bonds that have not yet
        # been considered as members of each bondset
        missing_bonds = self.find_missing_bonds(computed_bonds, coords1)
        delta_phi = self.get_dihedral_similarity(coords1, coords2,
                                                 base_mapping, range(coords1.n_atoms),
                                                 bond_network, missing_bonds)
        final_dists = [i+delta_phi for i in final_dists]
        # Combine the mappings that control different parts of molecule
        combined_mappings = \
            self.combine_mappings(final_mappings,
                                  bond_network.number_of_nodes())
        return combined_mappings, final_dists

    def update_lists(self, dist_list: NDArray, map_list: list,
                     current_dist: float, current_map: list) -> tuple:
        """ Add elements to the lists at the appropriate places """
        # Find where the elements should be added
        for c, i in enumerate(dist_list):
            if current_dist < i:
                idx = c
                break
        # Then update the lists
        update = dist_list[idx:-1].copy()
        dist_list[idx] = current_dist
        dist_list[idx+1:] = update
        update = map_list[idx:-1].copy()
        map_list[idx] = current_map
        map_list[idx+1:] = update
        return dist_list, map_list

    def prune_lists(self, list1: list, list2: list) -> tuple:
        """ Prune the lists if there are still None elements """
        for c, i in enumerate(list1):
            if i == None:
                return list1[:c], list2[:c]
        return list1, list2

    def get_mapping_components(self, mappings: dict, bond: list,
                               bond_network: nx.Graph) -> dict:
        """ Take the keys of mapping that are changed by rotating bond """
        # Get the bond the right way around
        directed_bond = bond_network[bond[0]][bond[1]]["direction"]
        neighbours = \
            [j for j in bond_network[directed_bond[1]] if j != directed_bond[0]]
        connections = self.get_connected_graphs(directed_bond, neighbours,
                                                bond_network)
        connected_nodes = []
        for i in range(len(connections)):
            connected_nodes.append(list(connections[i].nodes()))
        connected_atoms = list(set([x for xs in connected_nodes for x in xs]))
        new_mappings = []
        for j in mappings:
            new_mapping = {}
            # Keep each key/value of mapping if in the connections
            for i in j:
                if i in connected_atoms:
                    new_mapping[i] = j[i]
            new_mappings.append(new_mapping)
        return new_mappings

    def find_missing_bonds(self, all_bonds: list, coords1: type) -> list:
        """ Find the rotatable bonds present in coords1 that are not
            included in the nested list all_bonds """
        # Get the rotatable bonds
        rot_bonds = [i[1:3] for i in coords1.rotatable_dihedrals]
        # Flatten the bonds we've considered previously
        modified_bonds = []
        for i in all_bonds:
            for j in i:
                modified_bonds.append(j)
        missing_bonds = []
        for i in rot_bonds:
            if list(i) not in modified_bonds:
                missing_bonds.append(list(i))
        return missing_bonds

    def get_modified_bonds(self, base_bond: list, coords1: type,
                           bond_network: nx.Graph) -> list:
        """ Find all the rotatable bonds that are moved by rotating
            base_bond in the specified direction """
        # Find all connections for base_bond
        directed_bond = bond_network[base_bond[0]][base_bond[1]]["direction"]
        neighbours = \
            [j for j in bond_network[directed_bond[1]] if j != directed_bond[0]]
        connections = self.get_connected_graphs(directed_bond, neighbours,
                                                bond_network)
        connected_nodes = []
        for i in range(len(connections)):
            connected_nodes.append(list(connections[i].nodes()))
        connected_atoms = list(set([x for xs in connected_nodes for x in xs]))
        connected_atoms.append(directed_bond[1])
        # Check if all rotatable bonds are contained
        modified_bonds = [directed_bond]
        for i in coords1.rotatable_dihedrals:
            if i[1] in connected_atoms and i[2] in connected_atoms:
                modified_bonds.append([i[1], i[2]])
        return modified_bonds
    
    def combine_mappings(self, mappings: list, n_atoms: int) -> dict:
        """ Combine the mappings of different parts of the molecules """
        combined_mappings = []
        for j in mappings:
            final_mapping = {}
            # Add all key/value pairs to the same dict
            for i in j:
                final_mapping.update(i)
            # Then fill any gaps that are missing
            combined_mapping = self.fill_mappings(final_mapping, n_atoms)
            combined_mappings.append(combined_mapping)
        return combined_mappings

    def get_best_combinations(self, overall_mappings: list, overall_dists: list,
                              n_paths: int) -> tuple:
        """ Find the best combinations of all the different mappings """
        # Initialise the parameters
        n_sets = len(overall_mappings)
        best_set = [0]*n_sets
        final_mappings = []
        final_dists = []
        # Get the first element of the lists
        final_dists.append(np.sum([i[0] for i in overall_dists]))
        final_mappings.append([i[0] for i in overall_mappings])
        # Get the next shortest paths
        for i in range(n_paths-1):
            if i >= len(overall_mappings[0])-1:
                continue
            tmp_list = []
            for c, j in enumerate(best_set):
                tmp_list.append(overall_dists[c][j])
            idx = np.argmin(tmp_list)
            best_set[idx] += 1
            tmp_dist = 0.0
            tmp_mapping = []
            for c, j in enumerate(best_set):
                tmp_dist += overall_dists[c][j]
                tmp_mapping.append(overall_mappings[c][j])
            final_mappings.append(tmp_mapping)
            final_dists.append(tmp_dist)
        return final_mappings, final_dists        

    def get_dihedral_similarity(self, coords1: type, coords2: type,
                                base_mapping: dict, mapping: dict,
                                bond_network: nx.Graph, bonds: list) -> float:
        """ Get the overall dihedral similarity between the conformations in
            coords1 and coords2, given the two mappings applied to coords2 """
        # Loop over all rotatable bonds accumulating the rotational distance
        total_distance = 0.0
        for i in bonds:
            bond = [i[0], i[1]]
            # Get the reference atoms
            atoms1 = self.get_dihedral_atoms(bond, bond_network)
            atoms2 = self.get_dihedral_atoms(bond, bond_network)
            # Map the atoms from 1 to bonds in 2
            atoms1 = self.map_atoms(atoms1, base_mapping, mapping)
            # Then compute the angle and difference
            angle1 = self.get_dihedral(coords1.position, atoms1)
            angle2 = self.get_dihedral(coords2.position, atoms2)
            angle_diff = self.dihedral_difference(angle1, angle2)
            total_distance += np.abs(angle_diff)
        return total_distance

    def map_atoms(self, atoms: list, base_mapping: dict,
                  mapping: dict) -> list:
        """ Update the indices of the atoms based on the two consecutive
            mappings that have been applied to the network """
        mapped_atoms = []
        for i in atoms:
            mapped_atoms.append(mapping[base_mapping[i]])
        return mapped_atoms

    def get_dihedral(self, position: NDArray, idxs: list) -> float:
        """ Compute the dihedral angle between four sets of points """
        # Extract the positions
        p0 = position[idxs[0]*3:(idxs[0]*3)+3]
        p1 = position[idxs[1]*3:(idxs[1]*3)+3]
        p2 = position[idxs[2]*3:(idxs[2]*3)+3]
        p3 = position[idxs[3]*3:(idxs[3]*3)+3]
        # Compute dihedral angle in degrees with order idx1, idx2, idx3, idx4
        b0 = -1.0*(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        b1 /= np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1)*b1
        w = b2 - np.dot(b2, b1)*b1
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.degrees(np.arctan2(y, x))

    def get_rotation_matrix(self, angle: float) -> NDArray:
        """ Generate rotation matrix for rotation about x axis """
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, np.cos(angle), -1.0*np.sin(angle)],
                         [0.0, np.sin(angle), np.cos(angle)]])

    def get_dihedral_atoms(self, bond: list, bond_network: nx.Graph) -> list:
        """ Get the set of atoms that define the dihedral we will compare """
        ref_atoms = bond_network[bond[0]][bond[1]]["atoms"]
        ref_direction = bond_network[bond[0]][bond[1]]["direction"]
        if bond == ref_direction:
            return [ref_atoms[1][0], bond[0], bond[1], ref_atoms[0][0]]
        else:
            return [ref_atoms[0][0], bond[0], bond[1], ref_atoms[1][0]]

    def dihedral_difference(self, angle1: float, angle2: float) -> float:
        """ Compute the signed difference between two angles in degrees """
        # Compute the difference for each in turn
        rot_diff = np.radians(angle2) - np.radians(angle1)
        return np.rad2deg(np.arctan2(np.sin(rot_diff), np.cos(rot_diff)))

    def check_connectivity(self, atom1: int, atom2: int, G: nx.Graph) -> list:
        """ Find the atoms attached to atom2 if we break bond atom1-atom2 """
        H = G.copy()
        H.remove_edge(atom1, atom2)
        return list(nx.node_connected_component(H, atom2))

    ### UPDATE THE FINAL MAPPING

    def undo_mapping(self, new_mapping: dict,
                     base_mapping: dict, n_atoms: int) -> nx.Graph:
        """ Undo the effects of the both mappings in order to get the
            permutation vector for atom labels that transforms 1 -> 2 """
        updated_mapping = []
        # Undo the effects of two mappings
        for i in range(n_atoms):
            mapped_atom = new_mapping[base_mapping[i]]
            updated_mapping.append(mapped_atom)
        # Then reorder to get the final permutation
        best_mapping = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            best_mapping[updated_mapping[i]] = i
        return list(best_mapping)

    def plot_graph(self, network: nx.Graph, label: str):
        """ Just save the network as a figure """
        pos = nx.spring_layout(network, iterations=150, threshold=1e-6)
        fig, ax = plt.subplots(figsize=(8, 8))
        color_map = ['red'] + ['lawngreen']*6 + ['white']*12
        labels = {2:'1', 3:'1', 4:'1', 7:'2', 8:'2', 9:'2', 10:'3', 11:'3', 12:'3',
                  14:'4', 15:'4', 13:'4', 16:'5', 17:'5', 18:'5'}
        nx.draw_networkx(network, pos, node_color=color_map, with_labels=False, edgecolors='k',
                         node_size=1250)
        nx.draw_networkx_labels(network, pos, labels, font_size=25)
        ax.set_axis_off()
        fig.tight_layout()
        plt.savefig('Example%s.png' %label, dpi=500)

    def plot_graph2(self, network: nx.Graph, label: str):
        """ Just save the network as a figure """
        pos = nx.spring_layout(network, k=2.5, iterations=500, threshold=1e-9)
        fig, ax = plt.subplots(figsize=(8, 8))
        color_map = ['red'] + ['lawngreen']*6 + ['white']*12 + ['black']*15
        labels = {2:'1', 3:'1', 4:'1', 7:'2', 8:'2', 9:'2', 10:'3', 11:'3', 12:'3',
                  14:'4', 15:'4', 13:'4', 16:'5', 17:'5', 18:'5'}
        node_sizes = np.array([850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850,
                               850, 850, 850, 850, 850, 850, 850, 850,  
                               250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
                               250, 250])
        nx.draw_networkx(network, pos, node_color=color_map, with_labels=False, edgecolors='k',
                         node_size=node_sizes)
        #nx.draw_networkx_labels(network, pos, labels, font_size=25)
        ax.set_axis_off()
        fig.tight_layout()
        plt.savefig('Example%s.png' %label, dpi=500)

    def plot_graph3(self, network: nx.Graph, label: str):
        pos = nx.spring_layout(network, k=2.5, iterations=500, threshold=1e-9)
        fig, ax = plt.subplots(figsize=(8, 8))
        color_map = ['red'] + ['lawngreen']*6 + ['white']*12 + ['black']*15
        labels = {2:'1', 3:'1', 4:'1', 7:'2', 8:'2', 9:'2', 10:'3', 11:'3', 12:'3',
                  14:'4', 15:'4', 13:'4', 16:'5', 17:'5', 18:'5'}
        node_sizes = np.array([850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850,
                               850, 850, 850, 850, 850, 850, 850, 850,  
                               250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
                               250, 250])
        nx.draw_networkx(network, pos, with_labels=False)#, node_color=color_map, with_labels=False, edgecolors='k')#,
                         #node_size=node_sizes)
        #nx.draw_networkx_labels(network, pos, labels, font_size=25)
        ax.set_axis_off()
        fig.tight_layout()
        plt.savefig('Example%s.png' %label, dpi=500)