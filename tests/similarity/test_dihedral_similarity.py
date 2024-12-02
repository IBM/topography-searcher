import numpy as np
import networkx as nx
import pytest
import ase.io
from topsearch.data.coordinates import MolecularCoordinates
from topsearch.similarity.dihedral_similarity import DihedralSimilarity

comparer = DihedralSimilarity()


def test_get_bonding():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    assert set(bond_network.edges) == set([(0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 7),
                                  (2, 8), (2, 9), (3, 10), (3, 11), (3, 12),
                                  (4, 13), (4, 14), (4, 15), (5, 6), (6, 16), 
                                  (6, 17), (6, 18)])

def test_match_graphs():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    atoms = ase.io.read('test_data/pinacolone_rotated.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords2 = MolecularCoordinates(species, position)
    bond_network1 = comparer.get_bonding(coords)
    bond_network2 = comparer.get_bonding(coords2)
    match_network, mapping = comparer.match_graphs(bond_network1, bond_network2)
    assert mapping[0] == 0
    assert mapping[5] == 5
    assert mapping[1] == 1
    assert mapping[6] == 6

def test_match_graphs2():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    atoms = ase.io.read('test_data/pinacolone_permuted.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_p = MolecularCoordinates(species, position)
    bond_network1 = comparer.get_bonding(coords)
    bond_network2 = comparer.get_bonding(coords_p)
    mappings = comparer.match_graphs(bond_network1, bond_network2, 19)
    assert mappings[0][5] == 6
    assert mappings[0][6] == 5

def test_get_connections():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    atom1 = 1
    neighbours = [2, 3, 4]
    bond_network = comparer.get_bonding(coords)
    connected_sets = comparer.get_connections(atom1, neighbours,
                                              bond_network)
    assert connected_sets == [[8, 9, 2, 7], [11, 10, 3, 12], [4, 13, 14, 15]]

def test_get_connections2():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    atom1 = 5
    neighbours = [0, 6]
    bond_network = comparer.get_bonding(coords)
    connected_sets = comparer.get_connections(atom1, neighbours,
                                              bond_network)
    assert connected_sets == [[0], [16, 17, 18, 6]]

def test_get_permutable_sets():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond = [5, 1]
    perm_atoms = [2, 3, 4]
    bond_network = comparer.get_bonding(coords)
    connections = comparer.get_connected_graphs(bond, perm_atoms, bond_network)
    repeats = comparer.get_permutable_sets(connections)
    assert repeats == [[0, 1, 2]]

def test_get_permutable_sets2():
    atoms = ase.io.read('test_data/cymene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    connections = comparer.get_connected_graphs([1, 0], [3, 2, 10], bond_network)
    repeats = comparer.get_permutable_sets(connections)
    assert repeats == [[0, 1]]

def test_get_permutable_sets3():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    connections = comparer.get_connected_graphs([0, 5], [6, 1], bond_network)
    repeats = comparer.get_permutable_sets(connections)
    assert repeats == [[0]]

def test_get_permutable_sets4():
    atoms = ase.io.read('test_data/cymene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    connections = comparer.get_connected_graphs([1, 0], [10, 3, 2], bond_network)
    repeats = comparer.get_permutable_sets(connections)
    assert repeats == [[0], [1, 2]]

def test_get_permutable_idxs():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    idxs = comparer.get_permutable_idxs([5, 1], bond_network)
    assert idxs == [2, 3, 4]

def test_get_permutable_idxs2():
    atoms = ase.io.read('test_data/cymene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    idxs = comparer.get_permutable_idxs([1, 0], bond_network)
    assert idxs == [10]

def test_get_permutable_atoms():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    bond = [5, 6]
    atoms, neighbours = comparer.get_permutable_atoms(bond, bond_network)
    assert atoms == [[0, 1, 2]]
    assert neighbours == [16, 17, 18]

def test_get_permutable_atoms2():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    bond = [5, 1]
    atoms, neighbours = comparer.get_permutable_atoms(bond, bond_network)
    assert atoms == [[0, 1, 2]]
    assert neighbours == [2, 3, 4]

def test_get_permutable_atoms3():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    bond = [1, 5]
    atoms, neighbours = comparer.get_permutable_atoms(bond, bond_network)
    assert neighbours == [0, 6]
    assert atoms == [[0]]

def test_augment_network_edges():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    permutable_bonds = comparer.augment_network_edges(coords, bond_network)
    edge_permutable = nx.get_edge_attributes(bond_network, "permutable")
    edge_rotatable = nx.get_edge_attributes(bond_network, "rotatable")
    edge_atoms = nx.get_edge_attributes(bond_network, "atoms")
    edge_directions = nx.get_edge_attributes(bond_network, "direction")
    permutable_list = {(0, 5): False, (1, 2): True, (1, 3): True, (1, 4): True, 
                       (1, 5): True, (2, 7): False, (2, 8): False, (2, 9): False, 
                       (3, 10): False, (3, 11): False, (3, 12): False, (4, 13): False,
                       (4, 14): False, (4, 15): False, (5, 6): True, (6, 16): False,
                       (6, 17): False, (6, 18): False}
    rotatable_list = {(0, 5): False, (1, 2): True, (1, 3): True, (1, 4): True,
                      (1, 5): True, (2, 7): False, (2, 8): False, (2, 9): False, 
                      (3, 10): False, (3, 11): False, (3, 12): False, (4, 13): False,
                      (4, 14): False, (4, 15): False, (5, 6): True, (6, 16): False,
                      (6, 17): False, (6, 18): False}
    atoms_list = {(0, 5): [[], []], (1, 2): [[7, 8, 9], [5]], (1, 3): [[10, 11, 12], [5]], 
                  (1, 4): [[13, 14, 15], [5]], (1, 5): [[2, 3, 4], [0]], (2, 7): [[], []],
                  (2, 8): [[], []], (2, 9): [[], []], (3, 10): [[], []], (3, 11): [[], []],
                  (3, 12): [[], []], (4, 13): [[], []], (4, 14): [[], []], (4, 15): [[], []],
                  (5, 6): [[16, 17, 18], [0]], (6, 16): [[], []], (6, 17): [[], []],
                  (6, 18): [[], []]}
    directions_list = {(0, 5): [], (1, 2): [1, 2], (1, 3): [1, 3], (1, 4): [1, 4],
                       (1, 5): [5, 1], (2, 7): [], (2, 8): [], (2, 9): [], (3, 10): [],
                       (3, 11): [], (3, 12): [], (4, 13): [], (4, 14): [], (4, 15): [],
                       (5, 6): [5, 6], (6, 16): [], (6, 17): [], (6, 18): []}
    assert edge_rotatable == rotatable_list
    assert edge_permutable == permutable_list
    assert edge_atoms == atoms_list
    assert edge_directions == directions_list

def test_containing_bonds():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bonds = [[4, 13], [4, 14], [1, 3], [3, 10]]
    bond_network = comparer.get_bonding(coords)
    contains = comparer.containing_bonds(bonds, bond_network)
    assert contains == [[], [], [[3, 10]], []]

def test_containing_bonds2():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bonds = [[5, 1], [1, 3], [1, 4], [1, 2], [2, 7], [3, 11], [4, 13]]
    bond_network = comparer.get_bonding(coords)
    contains = comparer.containing_bonds(bonds, bond_network)
    assert contains == [[[1, 3], [1, 4], [1, 2], [2, 7], [3, 11], [4, 13]],
                        [[3, 11]], [[4, 13]], [[2, 7]], [], [], []]

def test_get_connected_sets():
    bonds = [[4, 13], [4, 14], [1, 3], [3, 10]]
    contains = [[], [], [[3, 10]], []]
    connected, sizes = comparer.get_connected_sets(bonds, contains)
    assert connected == [[[4, 13]], [[4, 14]], [[1, 3], [3, 10]]]
    assert sizes == [[0], [0], [1, 0]]

def test_get_connected_sets2():
    bonds = [[5, 1], [1, 3], [1, 4], [1, 2], [2, 7], [3, 11], [4, 13]]
    contains = [[[1, 3], [1, 4], [1, 2], [2, 7], [3, 11], [4, 13]],
                [[3, 11]], [[4, 13]], [[2, 7]], [], [], []]
    connected, sizes = comparer.get_connected_sets(bonds, contains)
    assert connected == [[[5, 1], [1, 3], [1, 4], [1, 2], [2, 7], [3, 11], [4, 13]]]
    assert sizes == [[6, 1, 1, 1, 0, 0, 0]]

def test_get_connected_sets3():
    bonds = [[4, 13], [3, 11], [2, 7], [1, 4], [1, 2], [5, 1], [1, 3]]
    contains = [[], [], [], [[4, 13]], [[2, 7]],
                [[1, 3], [1, 4], [1, 2], [2, 7], [3, 11], [4, 13]], [[3, 11]]]
    connected, sizes = comparer.get_connected_sets(bonds, contains)
    assert connected == [[[4, 13], [3, 11], [2, 7], [1, 4], [1, 2], [5, 1], [1, 3]]]
    assert sizes == [[0, 0, 0, 1, 1, 6, 1]]

def test_get_ordered_sets():
    connected = [[[4, 13]], [[4, 14]], [[1, 3], [3, 10]]]
    sizes = [[0], [0], [1, 0]]
    sorted_sets = comparer.get_ordered_sets(connected, sizes)
    assert sorted_sets == [[[4, 13]], [[4, 14]], [[1, 3], [3, 10]]]

def test_get_ordered_sets2():
    connected = [[[4, 13], [3, 11], [2, 7], [1, 4], [1, 2], [5, 1], [1, 3]]]
    sizes = [[0, 0, 0, 1, 1, 6, 1]]
    sorted_sets = comparer.get_ordered_sets(connected, sizes)
    assert sorted_sets == [[[5, 1], [1, 3], [1, 2], [1, 4], [2, 7], [3, 11], [4, 13]]]

def test_order_bonds():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    bonds = [[5, 1], [1, 4], [5, 6], [1, 2], [1, 3]]
    bond_sets = comparer.order_bonds(bonds, bond_network) 
    assert bond_sets == [[[5, 1], [1, 3], [1, 2], [1, 4]], [[5, 6]]]

def test_order_bonds2():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    bonds = [[4, 13], [4, 14], [1, 3], [3, 10]] 
    bond_sets = comparer.order_bonds(bonds, bond_network)
    assert bond_sets == [[[4, 13]], [[4, 14]], [[1, 3], [3, 10]]]

def test_order_bonds3():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    bonds = [[5, 1], [1, 3], [1, 4], [1, 2], [2, 7], [3, 11], [4, 13]]
    bond_sets = comparer.order_bonds(bonds, bond_network)
    assert bond_sets == [[[5, 1], [1, 2], [1, 4], [1, 3], [4, 13],
                          [3, 11], [2, 7]]]

def test_get_cyclic_permutations():
    perm_atoms = [13, 14, 15]
    permutations = comparer.get_cyclic_permutations(perm_atoms)
    assert permutations == [[0, 1, 2], [2, 0, 1], [1, 2, 0]]

def test_get_cyclic_permutations2():
    perm_atoms = [1, 3]
    permutations = comparer.get_cyclic_permutations(perm_atoms)
    assert permutations == [[0, 1], [1, 0]]

def test_get_bond_atoms():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    permutable_bonds = comparer.augment_network_edges(coords, bond_network)
    bond = [1, 5]
    directed_bond, p_atoms = comparer.get_bond_atoms(bond, bond_network)
    assert directed_bond == [5, 1]
    assert p_atoms == [2, 3, 4]

def test_get_bond_atoms2():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    permutable_bonds = comparer.augment_network_edges(coords, bond_network)
    bond = [5, 1]
    directed_bond, p_atoms = comparer.get_bond_atoms(bond, bond_network)
    assert directed_bond == [5, 1]
    assert p_atoms == [2, 3, 4]

def test_get_connectivity():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    atom1 = 1
    atom2 = 2
    bond_network = comparer.get_bonding(coords)
    c_graph = comparer.get_connectivity(atom1, atom2, bond_network)
    assert list(c_graph.nodes) ==  [8, 9, 2, 7]
    assert list(c_graph.edges) ==  [(8, 2), (9, 2), (2, 7)]

def test_get_connectivity2():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    atom1 = 5
    atom2 = 1
    bond_network = comparer.get_bonding(coords)
    c_graph = comparer.get_connectivity(atom1, atom2, bond_network)
    assert list(c_graph.nodes) ==  [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert list(c_graph.edges) ==  [(1, 2), (1, 3), (1, 4), (2, 7), (2, 8), (2, 9),
                              (3, 10), (3, 11), (3, 12), (4, 13), (4, 14), (4, 15)]

def test_get_connected_graphs():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond = [5, 1]
    perm_atoms = [2, 3, 4]
    bond_network = comparer.get_bonding(coords) 
    connections = comparer.get_connected_graphs(bond, perm_atoms, bond_network)
    assert set(connections[0].nodes) == {8, 9, 2, 7}
    assert set(connections[1].nodes) == {3, 10, 11, 12}
    assert set(connections[2].nodes) == {4, 13, 14, 15}

def test_fill_mappings():
    mapping = {2: 2, 7: 8, 8: 9, 9: 7, 3: 3, 10: 10, 11: 11, 12: 12,
               4: 4, 13: 13, 14: 14, 15: 15}
    mapping = comparer.fill_mappings(mapping, 19)
    assert mapping == {2: 2, 7: 8, 8: 9, 9: 7, 3: 3, 10: 10, 11: 11, 12: 12,
                       4: 4, 13: 13, 14: 14, 15: 15, 0: 0, 1: 1, 5: 5, 6: 6,
                       16: 16, 17: 17, 18: 18}

def test_update_mappings():
    base_mapping = {2: 4, 7: 13, 8: 14, 9: 15, 3: 2, 10: 7, 11: 9, 12: 8,
                    4: 3, 13: 12, 14: 11, 15: 10}
    new_mappings = [{13: 13, 14: 14, 15: 15}, {13: 15, 14: 13, 15: 14},
                    {13: 14, 14: 15, 15: 13}]
    new_mappings = comparer.update_mappings(new_mappings, base_mapping, 19)
    assert new_mappings[0] == {2: 4, 7: 13, 8: 14, 9: 15, 3: 2, 10: 7, 11: 9, 12: 8,
                               4: 3, 13: 12, 14: 11, 15: 10, 0: 0, 1: 1, 5: 5, 6: 6,
                               16: 16, 17: 17, 18: 18}
    assert new_mappings[1] == {2: 4, 7: 13, 8: 14, 9: 15, 3: 2, 10: 7, 11: 9, 12: 8,
                               4: 3, 13: 10, 14: 12, 15: 11, 0: 0, 1: 1, 5: 5, 6: 6,
                               16: 16, 17: 17, 18: 18}
    assert new_mappings[2] == {2: 4, 7: 13, 8: 14, 9: 15, 3: 2, 10: 7, 11: 9, 12: 8,
                               4: 3, 13: 11, 14: 10, 15: 12, 0: 0, 1: 1, 5: 5, 6: 6,
                               16: 16, 17: 17, 18: 18}

def test_switch_labels():
    base_mapping = {2: 4, 7: 13, 8: 14, 9: 15, 3: 2, 10: 7, 11: 9, 12: 8,
                    4: 3, 13: 12, 14: 11, 15: 10}
    mapping = {13: 15, 14: 13, 15: 14}
    base_mapping = comparer.switch_labels(mapping, base_mapping)
    assert base_mapping == {2: 4, 7: 13, 8: 14, 9: 15, 3: 2, 10: 7, 11: 9,
                            12: 8, 4: 3, 13: 10, 14: 12, 15: 11}

def test_get_permutation_mappings():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond = [5, 1]
    perm_atoms = [2, 3, 4]
    bond_network = comparer.prepare_network(coords)
    mappings = comparer.get_permutation_mappings(bond, perm_atoms, bond_network)
    assert mappings[0] == {2: 2, 9: 9, 8: 8, 7: 7, 28: 28, 29: 29, 30: 30, 3: 3,
                           12: 12, 10: 10, 11: 11, 31: 31, 32: 32, 33: 33, 4: 4,
                           13: 13, 14: 14, 15: 15, 23: 24, 24: 23, 22: 22} 
    assert mappings[1] == {2: 4, 9: 13, 8: 14, 7: 15, 28: 22, 29: 24, 30: 23, 3: 2,
                           12: 9, 10: 8, 11: 7, 31: 28, 32: 29, 33: 30, 4: 3, 13: 12,
                           14: 10, 15: 11, 23: 32, 24: 33, 22: 31}
    assert mappings[2] ==  {2: 3, 9: 12, 8: 10, 7: 11, 28: 31, 29: 32, 30: 33, 3: 4,
                            12: 13, 10: 14, 11: 15, 31: 22, 32: 24, 33: 23, 4: 2, 13: 9,
                            14: 8, 15: 7, 23: 29, 24: 30, 22: 28}

def test_permute_set():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.prepare_network(coords)
    bond_network = comparer.remove_placeholders(bond_network)
    permutable_bonds = comparer.augment_network_edges(coords, bond_network)
    bond_set = [[5, 1], [1, 2], [1, 3], [1, 4]]
    mappings = comparer.permute_set(bond_set, bond_network)
    assert len(mappings) == 81

def test_get_mappings():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.prepare_network(coords)
    bond_network = comparer.remove_placeholders(bond_network)
    permutable_bonds = comparer.augment_network_edges(coords, bond_network)
    ordered_sets = [[[5, 6]], [[5, 1], [1, 2], [1, 3], [1, 4]]]
    mappings = comparer.get_mappings(ordered_sets, bond_network)
    assert len(mappings[0]) == 3
    assert len(mappings[1]) == 81
    assert mappings[0][0] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                              9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
                              16: 16, 17: 17, 18: 18}
    assert mappings[0][1] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                              9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
                              16: 18, 17: 16, 18: 17}
    assert mappings[0][2] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                              9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
                              16: 17, 17: 18, 18: 16}

def test_get_rotation_matrix():
    rot_mat = comparer.get_rotation_matrix(np.pi/2.0)
    assert np.all(rot_mat == pytest.approx(np.array([[1.0, 0.0, 0.0],
                                                     [0.0, 0.0, -1.0],
                                                     [0.0, 1.0, 0.0]])))

def test_dihedral_difference():
    angle1 = 30.0
    angle2 = 45.0
    angle_diff = comparer.dihedral_difference(angle1, angle2)
    assert angle_diff == pytest.approx(15.0)

def test_dihedral_difference2():
    angle1 = 45.0
    angle2 = 30.0
    angle_diff = comparer.dihedral_difference(angle1, angle2)
    assert angle_diff == pytest.approx(-15.0)

def test_dihedral_difference3():
    angle1 = -45.0
    angle2 = 30.0
    angle_diff = comparer.dihedral_difference(angle1, angle2)
    assert angle_diff == pytest.approx(75.0)

def test_dihedral_difference4():
    angle1 = 45.0
    angle2 = -30.0
    angle_diff = comparer.dihedral_difference(angle1, angle2)
    assert angle_diff == pytest.approx(-75.0)

def test_get_dihedral():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    angle = comparer.get_dihedral(coords.position, [6, 5, 1, 4])
    assert angle == pytest.approx(179.1, 0.1)

def test_get_dihedral2():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    angle = comparer.get_dihedral(coords.position, [18, 6, 5, 1])
    assert angle == pytest.approx(57.4, 0.1)

def test_get_dihedral3():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    angle = comparer.get_dihedral(coords.position, [17, 6, 5, 1])
    assert angle == pytest.approx(-65.2, 0.1)

def test_get_dihedral_atoms():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond = [1, 5]
    bond_network = comparer.get_bonding(coords)
    permutable_bonds = comparer.augment_network_edges(coords, bond_network)
    dihedral_atoms = comparer.get_dihedral_atoms(bond, bond_network)
    assert dihedral_atoms == [2, 1, 5, 0]

def test_get_dihedral_atoms2():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond = [5, 1]
    bond_network = comparer.get_bonding(coords)
    permutable_bonds = comparer.augment_network_edges(coords, bond_network)
    dihedral_atoms = comparer.get_dihedral_atoms(bond, bond_network)
    assert dihedral_atoms == [0, 5, 1, 2]

def test_get_dihedral_similarity():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.prepare_network(coords)
    atoms = ase.io.read('test_data/pinacolone_rotated.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords2 = MolecularCoordinates(species, position)
    bond_network2 = comparer.prepare_network(coords2)
    network = comparer.remove_placeholders(bond_network)
    permutable_bonds = comparer.augment_network_edges(coords, network)
    nx.set_edge_attributes(bond_network, network.edges)
    base_mapping = dict(zip(list(range(bond_network.number_of_nodes())),
                            list(range(bond_network.number_of_nodes()))))
    mapping = dict(zip(list(range(bond_network.number_of_nodes())),
                       list(range(bond_network.number_of_nodes()))))
    bonds = [[i[1], i[2]] for i in coords.rotatable_dihedrals]
    distance = comparer.get_dihedral_similarity(coords, coords2, base_mapping,
                                                mapping, bond_network, bonds)
    assert distance < 1e-10

def test_get_dihedral_similarity2():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    atoms = ase.io.read('test_data/pinacolone_rotated.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords2 = MolecularCoordinates(species, position)
    permutable_bonds = comparer.augment_network_edges(coords, bond_network)
    base_mapping = dict(zip(list(range(bond_network.number_of_nodes())),
                            list(range(bond_network.number_of_nodes()))))
    mapping = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11,
               12:12, 13:13, 14:14, 15:15, 16:17, 17:18, 18:16}
    bonds = [[i[1], i[2]] for i in coords.rotatable_dihedrals]
    distance = comparer.get_dihedral_similarity(coords, coords2, base_mapping,
                                                mapping, bond_network, bonds)
    assert distance == pytest.approx(118.83079031972898)

def test_get_dihedral_similarity3():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    atoms = ase.io.read('test_data/pinacolone_rotated.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords2 = MolecularCoordinates(species, position)
    permutable_bonds = comparer.augment_network_edges(coords, bond_network)
    base_mapping = dict(zip(list(range(bond_network.number_of_nodes())),
                            list(range(bond_network.number_of_nodes()))))
    mapping = {0:0, 1:1, 2:3, 3:4, 4:2, 5:5, 6:6, 7:11, 8:10, 9:12, 10:13, 11:14,
               12:15, 13:8, 14:7, 15:9, 16:16, 17:17, 18:18}
    bonds = [[i[1], i[2]] for i in coords.rotatable_dihedrals]
    distance = comparer.get_dihedral_similarity(coords, coords2, base_mapping,
                                                mapping, bond_network, bonds)
    assert distance == pytest.approx(146.35876126772698, 1e-3)

def test_get_optimal_similarity():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.prepare_network(coords)
    atoms = ase.io.read('test_data/pinacolone_rotated.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords2 = MolecularCoordinates(species, position)
    network = comparer.remove_placeholders(bond_network)
    permutable_bonds = comparer.augment_network_edges(coords, network)
    nx.set_edge_attributes(bond_network, network.edges)
    ordered_sets = [[[5, 6]], [[5, 1], [1, 2], [1, 3], [1, 4]]]
    mappings = comparer.get_mappings(ordered_sets, bond_network)
    base_mapping = dict(zip(list(range(bond_network.number_of_nodes())),
                            list(range(bond_network.number_of_nodes()))))
    best_mapping, best_dists = comparer.get_optimal_similarity(coords, coords2, base_mapping,
                                                               mappings,
                                                               bond_network, ordered_sets, 1)
    assert best_mapping[0] == {6: 6, 16: 16, 17: 17, 18: 18, 25: 25, 26: 26, 27: 27, 1: 1,
                               2: 2, 3: 3, 4: 4, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
                               13: 13, 14: 14, 15: 15, 19: 19, 20: 20, 21: 21, 22: 22,
                               23: 24, 24: 23, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32,
                               33: 33, 0: 0, 5: 5}

def test_undo_mapping():
    new_mapping = {0:1, 1:2, 2:0, 3:3, 4:4}
    base_mapping = {0:0, 1:3, 2:2, 3:1, 4:4}
    final_mapping = comparer.undo_mapping(new_mapping, base_mapping, 4)
    assert final_mapping == [2, 0, 3, 1]

def test_get_permutable_information():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond = [5, 1]
    bond_network = comparer.get_bonding(coords)
    permutable, perm, permutable_atoms = comparer.get_permutable_information(bond, bond_network)
    assert permutable == True
    assert perm == True
    assert permutable_atoms == [2, 3, 4]

def test_get_reorderable_atoms():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    p_bonds, p_atoms, r_bonds, r_atoms = comparer.get_reorderable_atoms(coords, bond_network)
    assert p_bonds == [(5, 1), (1, 4), (5, 6), (1, 2), (1, 3)]
    assert p_atoms ==  [[2, 3, 4], [13, 14, 15], [16, 17, 18], [7, 8, 9], [10, 11, 12]]
    assert r_bonds ==  [(4, 1), (2, 1), (3, 1)]
    assert r_atoms ==  [[2, 3], [3, 4], [2, 4]]

def test_postprocess_bonds():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    p_bonds = [(5, 1), (1, 4), (5, 6), (1, 2), (1, 3)]
    r_bonds = [(4, 1), (2, 1), (3, 1)]
    r_atoms = [[2, 3], [3, 4], [2, 4]]
    r_bonds, r_atoms = comparer.postprocess_bonds(p_bonds, r_bonds, r_atoms, bond_network)
    assert r_bonds == []
    assert r_atoms == []

def test_clockwise_order_perm():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    bond = [5, 1]
    atoms = [2, 3, 4]
    order = comparer.clockwise_order_perm(bond, atoms, bond_network, coords.position)
    assert list(order) == [0, 2, 1]

def test_reorder_permutable_atoms():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    p_bonds = [(5, 1), (1, 4), (5, 6), (1, 2), (1, 3)]
    p_atoms = [[2, 3, 4], [13, 14, 15], [16, 17, 18], [7, 8, 9], [10, 11, 12]] 
    mapping = comparer.reorder_permutable_atoms(p_bonds, p_atoms, bond_network, coords.position)
    assert mapping == {2: 2, 3: 4, 4: 3, 13: 15, 14: 14, 15: 13, 16: 18,
                       17: 16, 18: 17, 7: 7, 8: 8, 9: 9, 10: 11, 11: 10, 12: 12}

def test_clockwise_reorder():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    mapping, perm_atoms = comparer.clockwise_reorder(coords, bond_network)
    assert mapping == {2: 2, 3: 4, 4: 3, 13: 15, 14: 14, 15: 13, 16: 18, 17: 16, 18: 17,
                       7: 7, 8: 8, 9: 9, 10: 11, 11: 10, 12: 12}
    assert perm_atoms == [[2, 3, 4], [13, 14, 15], [16, 17, 18], [7, 8, 9], [10, 11, 12]]

def test_add_placeholders():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    perm_atoms = [[2, 3, 4], [13, 14, 15], [16, 17, 18], [7, 8, 9], [10, 11, 12]]
    mapping = {2: 2, 3: 4, 4: 3, 13: 15, 14: 14, 15: 13, 16: 18, 17: 16, 18: 17,
               7: 7, 8: 8, 9: 9, 10: 11, 11: 10, 12: 12}
    bond_network = comparer.add_placeholders(bond_network, perm_atoms, mapping, coords.position)
    assert list(bond_network.edges) == [(0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 7),
                                        (2, 8), (2, 9), (3, 10), (3, 11), (3, 12), (3, 20),
                                        (3, 21), (4, 13), (4, 14), (4, 15), (4, 19), (5, 6),
                                        (6, 16), (6, 17), (6, 18), (8, 28), (9, 29), (9, 30),
                                        (10, 31), (12, 32), (12, 33), (13, 23), (13, 24),
                                        (14, 22), (16, 25), (17, 26), (17, 27)]

def test_remove_placeholders():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords)
    old_network = bond_network.copy()
    perm_atoms = [[2, 3, 4], [13, 14, 15], [16, 17, 18], [7, 8, 9], [10, 11, 12]]
    mapping = {2: 2, 3: 4, 4: 3, 13: 15, 14: 14, 15: 13, 16: 18, 17: 16, 18: 17,
               7: 7, 8: 8, 9: 9, 10: 11, 11: 10, 12: 12}
    bond_network = comparer.add_placeholders(bond_network, perm_atoms, mapping, coords.position)
    new_network = comparer.remove_placeholders(bond_network)
    assert old_network.edges == new_network.edges

def test_prepare_network():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond_network = comparer.prepare_network(coords)
    assert list(bond_network.edges) == [(0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 7),
                                        (2, 8), (2, 9), (3, 10), (3, 11), (3, 12), (3, 20),
                                        (3, 21), (4, 13), (4, 14), (4, 15), (4, 19), (5, 6),
                                        (6, 16), (6, 17), (6, 18), (8, 28), (9, 29), (9, 30),
                                        (10, 31), (12, 32), (12, 33), (13, 23), (13, 24),
                                        (14, 22), (16, 25), (17, 26), (17, 27)]

def test_remove_direct_placeholders():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    bond = [5, 1]
    perm_atoms = [2, 3, 4]
    bond_network = comparer.prepare_network(coords)
    connections = comparer.get_connected_graphs(bond, perm_atoms, bond_network)
    connections = comparer.remove_direct_placeholders(perm_atoms, connections, bond_network)
    assert set(list(connections[0].nodes)) == set([2, 7, 8, 9, 28, 29, 30])
    assert set(list(connections[1].nodes)) == set([32, 33, 3, 10, 11, 12, 31])
    assert set(list(connections[2].nodes)) == set([4, 13, 14, 15, 22, 23, 24])


def test_find_best_alignments():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    coords_rot = MolecularCoordinates(species, position)
    bond = [5, 1]
    bond_network = comparer.get_bonding(coords_rot)
    moved_atoms = coords_rot.get_movable_atoms(bond, 'dihedral', bond_network)
    coords_rot.rotate_dihedral(bond, 120.0, moved_atoms)
    best_mapping, best_dist = comparer.find_best_alignments(coords, coords_rot)
    assert best_mapping[0] == [0, 1, 3, 4, 2, 5, 6, 11, 10, 12, 13, 14, 15, 8, 7, 9, 16, 17, 18]

def test_find_best_alignments2():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    coords_rot = MolecularCoordinates(species, position)
    bond = [5, 1]
    bond_network = comparer.get_bonding(coords_rot)
    moved_atoms = coords_rot.get_movable_atoms(bond, 'dihedral', bond_network)
    coords_rot.rotate_dihedral(bond, 120.0, moved_atoms)
    bond = [1, 4]
    moved_atoms = coords_rot.get_movable_atoms(bond, 'dihedral', bond_network)
    coords_rot.rotate_dihedral(bond, 100.0, moved_atoms)  
    best_mappings, best_dists = comparer.find_best_alignments(coords, coords_rot, 2)
    assert best_mappings[0] == [0, 1, 3, 4, 2, 5, 6, 11, 10, 12, 14, 15, 13, 8, 7, 9, 16, 17, 18]

def test_find_best_alignments3():
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    atoms = ase.io.read('test_data/pinacolone_perm.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_rot = MolecularCoordinates(species, position)
    bond = [5, 1]
    bond_network = comparer.get_bonding(coords_rot)
    moved_atoms = coords_rot.get_movable_atoms(bond, 'dihedral', bond_network)
    coords_rot.rotate_dihedral(bond, 120.0, moved_atoms)
    bond = [1, 4]
    moved_atoms = coords_rot.get_movable_atoms(bond, 'dihedral', bond_network)
    coords_rot.rotate_dihedral(bond, 100.0, moved_atoms)
    best_mappings, best_dists = comparer.find_best_alignments(coords, coords_rot)
    assert best_mappings[0] == [0, 1, 3, 4, 2, 5, 6, 11, 10, 12, 14, 15, 13, 8, 7, 9, 16, 18, 17]

def test_find_best_alignments4():    
    atoms = ase.io.read('test_data/pinacolone.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    atoms = ase.io.read('test_data/pinacolone_perm.xyz') 
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_rot = MolecularCoordinates(species, position)
    bond = [5, 1] 
    bond_network = comparer.get_bonding(coords_rot)
    moved_atoms = coords_rot.get_movable_atoms(bond, 'dihedral', bond_network)
    coords_rot.rotate_dihedral(bond, 120.0, moved_atoms)
    bond = [1, 4]
    moved_atoms = coords_rot.get_movable_atoms(bond, 'dihedral', bond_network)
    coords_rot.rotate_dihedral(bond, 100.0, moved_atoms)
    bond = [5, 6]
    moved_atoms = coords_rot.get_movable_atoms(bond, 'dihedral', bond_network)
    coords_rot.rotate_dihedral(bond, 120.0, moved_atoms)
    best_mappings, best_dists = comparer.find_best_alignments(coords, coords_rot)
    assert best_mappings[0] == [0, 1, 3, 4, 2, 5, 6, 11, 10, 12, 14, 15, 13, 8, 7, 9, 17, 16, 18]

def test_find_best_alignments5():
    atoms = ase.io.read('test_data/pentanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    coords_rot = MolecularCoordinates(species, position)
    bond = [2, 4]
    coords_rot.rotate_dihedral(bond, 120.0, [11, 12, 13])
    best_mappings, best_dists = comparer.find_best_alignments(coords, coords_rot, 1)
    assert best_mappings[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 11, 12, 14, 15, 16, 17]

def test_optimal_alignment5():
    atoms = ase.io.read('test_data/pentanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    coords_rot = MolecularCoordinates(species, position)
    bond = [2, 4]
    coords_rot.rotate_dihedral(bond, 120.0, [11, 12, 13])
    dist, c1, c2, perm = comparer.optimal_alignment(coords, coords_rot.position)
    assert perm == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 11, 12, 14, 15, 16, 17]
    assert np.abs(dist - 1.9076837625270002) < 0.1

def test_closest_distance5():
    atoms = ase.io.read('test_data/pentanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    coords_rot = MolecularCoordinates(species, position)
    bond = [2, 4]
    coords_rot.rotate_dihedral(bond, 120.0, [11, 12, 13])
    dist = comparer.closest_distance(coords, coords_rot.position)
    assert np.abs(dist - 1.9076837625270002) < 0.1

def test_test_same5():
    atoms = ase.io.read('test_data/pentanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    coords_rot = MolecularCoordinates(species, position)
    bond = [2, 4]
    coords_rot.rotate_dihedral(bond, 120.0, [11, 12, 13])
    is_same = comparer.test_same(coords, coords_rot.position, 1.0, 1.0)
    assert is_same == True

def test_find_best_alignments6():
    atoms = ase.io.read('test_data/pentanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    # Switch 9-10
    atoms = ase.io.read('test_data/pentanol_perm2.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_rot = MolecularCoordinates(species, position)
    best_mappings, best_dists = comparer.find_best_alignments(coords, coords_rot, 3)
    assert best_mappings[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11, 12, 13, 14, 15, 16, 17]

def test_find_best_alignments7():
    atoms = ase.io.read('test_data/pentanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords = MolecularCoordinates(species, position)
    # Switch 6-7, 9-10
    atoms = ase.io.read('test_data/pentanol_perm.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_rot = MolecularCoordinates(species, position)
    bond = [2, 4]
    coords_rot.rotate_dihedral(bond, 120.0, [11, 12, 13])
    best_mappings, best_dists = comparer.find_best_alignments(coords, coords_rot, 7)
    assert best_mappings[0] == [0, 1, 2, 3, 4, 5, 7, 6, 8, 10, 9, 13, 11, 12, 14, 15, 16, 17]

def test_find_best_alignments8():
    atoms = ase.io.read('test_data/ethylbenzene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_ben = MolecularCoordinates(species, position)
    coords_rot = MolecularCoordinates(species, position)
    coords_rot.rotate_dihedral([1, 0], 170.0, [9, 8, 12, 14, 13, 4])
    best_mappings, best_dists = comparer.find_best_alignments(coords_ben, coords_rot)
    assert best_mappings[0] == [0, 1, 3, 2, 4, 6, 5, 7, 8, 9, 11, 10, 12, 13, 14, 16, 15, 17]

def test_find_best_alignments9():
    atoms = ase.io.read('test_data/cymene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_ben = MolecularCoordinates(species, position)
    coords_rot = MolecularCoordinates(species, position)
    coords_rot.rotate_dihedral([1, 0], 180.0, [10, 2, 3, 14, 15, 16, 11, 12, 13])
    coords_rot.rotate_dihedral([6, 9], 120.0, [21, 22, 23])
    best_mappings, best_dists = comparer.find_best_alignments(coords_ben, coords_rot)
    assert best_mappings[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                            14, 15, 16, 17, 18, 19, 20, 23, 21, 22]

def test_get_connectivity3():
    atoms = ase.io.read('test_data/toluene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_tol = MolecularCoordinates(species, position)
    bond_network = comparer.prepare_network(coords_tol)
    connection = comparer.get_connectivity(3, 0, bond_network)
    assert list(connection.nodes()) == [0, 1, 2, 4, 5, 6, 7, 8, 12, 13, 14]

def test_get_ring_connectivity():
    atoms = ase.io.read('test_data/toluene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_tol = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords_tol)
    connection = comparer.get_ring_connectivity(0, 1, [1, 2], bond_network)
    assert set(list(connection.nodes())) == set([1, 4, 12, 7])

def test_get_ring_connectivity2():
    atoms = ase.io.read('test_data/napthalene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_nap = MolecularCoordinates(species, position)
    bond_network = comparer.prepare_network(coords_nap)
    connection = comparer.get_ring_connectivity(0, 2, [2, 4], bond_network)
    assert set(list(connection.nodes())) == set([2, 3, 6, 7, 10, 11, 14, 15])

def test_get_ring_connectivity3():
    atoms = ase.io.read('test_data/oxepane.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_oxe = MolecularCoordinates(species, position)
    bond_network = comparer.get_bonding(coords_oxe)
    connection = comparer.get_ring_connectivity(0, 5, [5, 6], bond_network)
    assert set(list(connection.nodes())) == set([1, 3, 5, 7, 8, 11, 12, 15, 16])

def test_retain_unique_mappings():
    atoms = ase.io.read('test_data/pentane.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_pent = MolecularCoordinates(species, position)
    coords_pent2 = MolecularCoordinates(species, position)
    bond_network = comparer.prepare_network(coords_pent)
    bond_network2 = comparer.prepare_network(coords_pent2)
    isomorphs = nx.vf2pp_all_isomorphisms(bond_network, bond_network2, node_label='element')
    mappings = [i for i in isomorphs]
    new_mappings = comparer.retain_unique_mappings(mappings, 17)
    assert new_mappings == [{0: 0, 1: 2, 2: 1, 3: 4, 4: 3, 5: 5, 6: 6, 7: 9, 8: 10, 9: 7,
                             10: 8, 11: 15, 12: 16, 13: 14, 14: 13, 15: 11, 16: 12},
                            {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
                             10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16}]

def test_retain_unique_mappings2():
    atoms = ase.io.read('test_data/pentanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_pent = MolecularCoordinates(species, position)
    coords_pent2 = MolecularCoordinates(species, position)
    bond_network = comparer.prepare_network(coords_pent)
    bond_network2 = comparer.prepare_network(coords_pent2)
    isomorphs = nx.vf2pp_all_isomorphisms(bond_network, bond_network2, node_label='element')
    mappings = [i for i in isomorphs]
    new_mappings = comparer.retain_unique_mappings(mappings, 18)
    assert new_mappings == [{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
                             10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17}]

def test_match_graphs():
    atoms = ase.io.read('test_data/pentane.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_pent = MolecularCoordinates(species, position)
    coords_pent2 = MolecularCoordinates(species, position)
    bond_network = comparer.prepare_network(coords_pent)
    bond_network2 = comparer.prepare_network(coords_pent2)
    new_mappings = comparer.match_graphs(bond_network, bond_network2, 17)
    assert new_mappings == [{0: 0, 1: 2, 2: 1, 3: 4, 4: 3, 5: 5, 6: 6, 7: 9, 8: 10, 9: 7,
                             10: 8, 11: 15, 12: 16, 13: 14, 14: 13, 15: 11, 16: 12},
                             {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
                             10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16}]

def test_update_lists():
    dist_list = np.array([0.1, 0.3, 0.5, 0.6, 0.9])
    map_list = [{1:1}, {2:2}, {3:3}, {4:5}, {5:4}]
    current_dist = 0.2
    current_map = {7:7}
    dist_list, map_list = comparer.update_lists(dist_list, map_list, current_dist, current_map)
    assert np.all(dist_list == [0.1, 0.2, 0.3, 0.5, 0.6])
    assert map_list == [{1:1}, {7:7}, {2:2}, {3:3}, {4:5}]

def test_prune_lists():
    list1 = [{1:1}, {2:2}, None, None]
    list2 = [3.0, 4.0, 1e30, 1e30]
    list1, list2 = comparer.prune_lists(list1, list2)
    assert list1 == [{1:1}, {2:2}]
    assert list2 == [3.0, 4.0]

def test_find_best_alignments10(): 
    atoms = ase.io.read('test_data/pentane.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_pen = MolecularCoordinates(species, position)
    coords_pen2 = MolecularCoordinates(species, position)
    best_mappings, best_dists = comparer.find_best_alignments(coords_pen, coords_pen2, 5)
    assert best_mappings[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                14, 15, 16]
    assert best_dists[0] == pytest.approx(0.0)

def test_get_best_combinations():
    overall_mappings = [[{14: 15, 15: 16, 16: 14, 20: 20, 21: 21, 22: 22},
                         {14: 14, 15: 15, 16: 16, 20: 20, 21: 21, 22: 22},
                         {14: 16, 15: 14, 16: 15, 20: 20, 21: 21, 22: 22}],
                         [{11: 13, 12: 11, 13: 12, 17: 17, 18: 18, 19: 19},
                          {11: 11, 12: 12, 13: 13, 17: 17, 18: 18, 19: 19},
                          {11: 12, 12: 13, 13: 11, 17: 17, 18: 18, 19: 19}]]
    overall_dists = [np.array([125.2584682 , 244.70665131, 244.99565395]), np.array([125.27136587, 244.70665131, 245.01868641])]
    final_mappings, final_dists = \
        comparer.get_best_combinations(overall_mappings, overall_dists, 3)
    assert final_mappings[0] == [{14: 15, 15: 16, 16: 14, 20: 20, 21: 21, 22: 22},
                                 {11: 13, 12: 11, 13: 12, 17: 17, 18: 18, 19: 19}]
    assert final_mappings[2] == [{14: 14, 15: 15, 16: 16, 20: 20, 21: 21, 22: 22},
                                 {11: 11, 12: 12, 13: 13, 17: 17, 18: 18, 19: 19}]
    assert np.all(final_dists == pytest.approx(np.array([250.52983407, 369.97801718, 489.41330262])))

def test_combine_mappings():
    mappings = [[{14: 15, 15: 16, 16: 14, 20: 20, 21: 21, 22: 22}, {11: 13, 12: 11, 13: 12, 17: 17, 18: 18, 19: 19}],
                [{14: 14, 15: 15, 16: 16, 20: 20, 21: 21, 22: 22}, {11: 13, 12: 11, 13: 12, 17: 17, 18: 18, 19: 19}],
                [{14: 14, 15: 15, 16: 16, 20: 20, 21: 21, 22: 22}, {11: 11, 12: 12, 13: 13, 17: 17, 18: 18, 19: 19}],
                [{14: 16, 15: 14, 16: 15, 20: 20, 21: 21, 22: 22}, {11: 11, 12: 12, 13: 13, 17: 17, 18: 18, 19: 19}]]
    combined_mappings = comparer.combine_mappings(mappings, 17)
    assert combined_mappings == [{14: 15, 15: 16, 16: 14, 20: 20, 21: 21, 22: 22, 11: 13, 12: 11, 13: 12, 17: 17,
                                  18: 18, 19: 19, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10},
                                 {14: 14, 15: 15, 16: 16, 20: 20, 21: 21, 22: 22, 11: 13, 12: 11, 13: 12, 17: 17,
                                  18: 18, 19: 19, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10},
                                 {14: 14, 15: 15, 16: 16, 20: 20, 21: 21, 22: 22, 11: 11, 12: 12, 13: 13, 17: 17,
                                  18: 18, 19: 19, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10},
                                 {14: 16, 15: 14, 16: 15, 20: 20, 21: 21, 22: 22, 11: 11, 12: 12, 13: 13, 17: 17,
                                  18: 18, 19: 19, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}]

def test_find_best_alignments11():
    atoms = ase.io.read('test_data/toluene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_tol = MolecularCoordinates(species, position)
    coords_rot = MolecularCoordinates(species, position)
    coords_rot.rotate_dihedral([3, 0], 180.0, [1, 2, 4, 5, 6, 7, 12, 14, 13, 8])
    coords_rot.rotate_dihedral([0, 3], 120.0, [9, 10, 11])
    best_mappings, best_dists = comparer.find_best_alignments(coords_tol, coords_rot, 4)
    assert best_mappings[0] == [0, 2, 1, 3, 5, 4, 6, 8, 7, 11, 9, 10, 13, 12, 14]

def test_optimal_alignment11():
    atoms = ase.io.read('test_data/toluene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_tol = MolecularCoordinates(species, position)
    coords_rot = MolecularCoordinates(species, position)
    coords_rot.rotate_dihedral([3, 0], 180.0, [1, 2, 4, 5, 6, 7, 12, 14, 13, 8])
    coords_rot.rotate_dihedral([0, 3], 120.0, [9, 10, 11])
    dist, c1, c2, perm = comparer.optimal_alignment(coords_tol, coords_rot.position)
    assert perm == [0, 2, 1, 3, 5, 4, 6, 8, 7, 11, 9, 10, 13, 12, 14]
    assert np.abs(dist - 0.46401545271011035) < 0.5

def test_closest_distance11():
    atoms = ase.io.read('test_data/toluene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_tol = MolecularCoordinates(species, position)
    coords_rot = MolecularCoordinates(species, position)
    coords_rot.rotate_dihedral([3, 0], 180.0, [1, 2, 4, 5, 6, 7, 12, 14, 13, 8])
    coords_rot.rotate_dihedral([0, 3], 120.0, [9, 10, 11])
    dist = comparer.closest_distance(coords_tol, coords_rot.position)
    assert np.abs(dist - 0.46401545271011035) < 0.5

def test_test_same11():
    atoms = ase.io.read('test_data/toluene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_tol = MolecularCoordinates(species, position)
    coords_rot = MolecularCoordinates(species, position)
    coords_rot.rotate_dihedral([3, 0], 180.0, [1, 2, 4, 5, 6, 7, 12, 14, 13, 8])
    coords_rot.rotate_dihedral([0, 3], 120.0, [9, 10, 11])
    is_same = comparer.test_same(coords_tol, coords_rot.position, 4.0, 4.0)
    assert is_same == True

def test_test_same11_2():
    atoms = ase.io.read('test_data/toluene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_tol = MolecularCoordinates(species, position)
    coords_rot = MolecularCoordinates(species, position)
    coords_rot.rotate_dihedral([3, 0], 180.0, [1, 2, 4, 5, 6, 7, 12, 14, 13, 8])
    coords_rot.rotate_dihedral([0, 3], 120.0, [9, 10, 11])
    is_same = comparer.test_same(coords_tol, coords_rot.position, 4.0, 4.5)
    assert is_same == False

def test_find_best_alignments12():
    atoms = ase.io.read('test_data/cymene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_cy = MolecularCoordinates(species, position)
    atoms = ase.io.read('test_data/cymene_perm.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_perm = MolecularCoordinates(species, position)
    best_mappings, best_dists = comparer.find_best_alignments(coords_cy, coords_perm, 4)
    assert best_mappings[0] == [7, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23]

def test_find_best_alignments13():
    atoms = ase.io.read('test_data/cymene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_cy = MolecularCoordinates(species, position)
    atoms = ase.io.read('test_data/cymene_perm2.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_perm = MolecularCoordinates(species, position)
    best_mappings, best_dists = comparer.find_best_alignments(coords_cy, coords_perm, 4)
    assert best_mappings[0] == [0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23]

def test_find_best_alignments14():
    atoms = ase.io.read('test_data/cymene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_cy = MolecularCoordinates(species, position)
    atoms = ase.io.read('test_data/cymene_perm3.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_perm = MolecularCoordinates(species, position)
    best_mappings, best_dists = comparer.find_best_alignments(coords_cy, coords_perm, 4)
    assert best_mappings[0] == [7, 1, 3, 2, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 16, 15, 14,
                                17, 18, 19, 20, 21, 22, 23]

def test_find_best_alignments15():
    atoms = ase.io.read('test_data/cymene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_cy = MolecularCoordinates(species, position)
    atoms = ase.io.read('test_data/cymene_perm3.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_perm = MolecularCoordinates(species, position)
    coords_perm.rotate_dihedral([6, 9],  90.0, [21, 22, 23])
    coords_perm.rotate_dihedral([7, 2], 140.0, [14, 15, 16])
    best_mappings, best_dists = comparer.find_best_alignments(coords_cy, coords_perm, 4)
    assert best_mappings[0] == [7, 1, 3, 2, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 15, 14, 16,
                                17, 18, 19, 20, 23, 21, 22]

def test_find_best_alignments16():
    atoms = ase.io.read('test_data/cymene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_cy = MolecularCoordinates(species, position)
    atoms = ase.io.read('test_data/cymene_perm3.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    coords_perm = MolecularCoordinates(species, position)
    coords_perm.rotate_dihedral([1, 7], 150.0, [2, 3, 10, 14, 15, 16, 11, 12, 13])
    coords_perm.rotate_dihedral([6, 9],  90.0, [21, 22, 23])
    coords_perm.rotate_dihedral([7, 2], 140.0, [14, 15, 16])
    best_mappings, best_dists = comparer.find_best_alignments(coords_cy, coords_perm, 4)
    assert best_mappings[0] == [7, 1, 3, 2, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 14, 16, 15,
                                17, 18, 19, 20, 23, 21, 22]
