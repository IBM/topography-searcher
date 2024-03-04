""" Testing for the StandardCoordinates class """

import pytest
import ase
import numpy as np
import os.path
from topsearch.data.coordinates import StandardCoordinates, \
    AtomicCoordinates, MolecularCoordinates
from topsearch.potentials.force_fields import MMFF94

########### TWO DIMENSIONS - EVEN BOUNDS ###############

def test_initialisation():
    # Generate the coordinates object
    coords = StandardCoordinates(ndim=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
    assert coords.ndim == 2
    assert np.all(coords.lower_bounds == np.array([0.0, 0.0]))
    assert np.all(coords.upper_bounds == np.array([1.0, 1.0]))
    assert coords.bounds == [(0.0, 1.0), (0.0, 1.0)]

def test_generate_random_point():
    coords = StandardCoordinates(ndim=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
    assert coords.position.size == coords.ndim
    assert coords.at_bounds() == False

def test_check_bounds():
    coords = StandardCoordinates(ndim=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
    # All bounds
    coords.position = np.array([1.0, 1.0])
    bounds_array1 = coords.check_bounds()
    # At bounds
    coords.position = np.array([0.5, 1.0])
    bounds_array2 = coords.check_bounds()
    # Not at bounds
    coords.position = np.array([0.5, 0.5])
    bounds_array3 = coords.check_bounds()
    assert np.all(bounds_array1 == np.array([True, True]))
    assert np.all(bounds_array2 == np.array([False, True]))
    assert np.all(bounds_array3 == np.array([False, False]))

def test_at_bounds():
    coords = StandardCoordinates(ndim=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
    # All bounds
    coords.position = np.array([1.0, 1.0])
    assert coords.at_bounds() == True
    # At bounds
    coords.position = np.array([0.5, 1.0])
    assert coords.at_bounds() == True
    # Not at bounds
    coords.position = np.array([0.5, 0.5])
    assert coords.at_bounds() == False

def test_all_bounds():
    coords = StandardCoordinates(ndim=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
    # All bounds
    coords.position = np.array([1.0, 1.0])
    assert coords.all_bounds() == True
    # At bounds
    coords.position = np.array([0.5, 1.0])
    assert coords.all_bounds() == False
    # Not at bounds
    coords.position = np.array([0.5, 0.5])
    assert coords.all_bounds() == False

def test_active_bounds():
    coords = StandardCoordinates(ndim=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
    # All bounds
    coords.position = np.array([-0.1, 1.1])
    below, above = coords.active_bounds()
    assert np.all(below == np.array([True, False]))
    assert np.all(above == np.array([False, True]))
    # At bounds
    coords.position = np.array([0.5, 0.1])
    below, above = coords.active_bounds()
    assert np.all(below == np.array([False, False]))
    assert np.all(above == np.array([False, False]))
    # Not at bounds
    coords.position = np.array([1.5, 0.0])
    below, above = coords.active_bounds()
    assert np.all(below == np.array([False, True]))
    assert np.all(above == np.array([True, False]))

def test_move_to_bounds():
    coords = StandardCoordinates(ndim=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
    # Both above
    coords.position = np.array([1.2, 1.3])
    coords.move_to_bounds()
    assert np.all(coords.position == np.array([1.0, 1.0]))
    # One below
    coords.position = np.array([0.5, -1.0])
    coords.move_to_bounds()
    assert np.all(coords.position == np.array([0.5, 0.0]))
    # Inside
    coords.position = np.array([0.2, 0.4])
    coords.move_to_bounds()
    assert np.all(coords.position == np.array([0.2, 0.4]))

########### THREE DIMENSIONS - UNEVEN BOUNDS ###############

def test_initialisation2():
    # Generate the coordinates object
    coords = StandardCoordinates(ndim=3, bounds=[(0.0, 1.0),
                                                 (0.0, 0.5),
                                                 (-0.5, 0.0)])
    assert coords.ndim == 3
    assert np.all(coords.lower_bounds == np.array([0.0, 0.0, -0.5]))
    assert np.all(coords.upper_bounds == np.array([1.0, 0.5, 0.0]))
    assert coords.bounds == [(0.0, 1.0), (0.0, 0.5), (-0.5, 0.0)]

def test_generate_random_point2():
    coords = StandardCoordinates(ndim=3, bounds=[(0.0, 1.0),
                                                 (0.0, 0.5),
                                                 (-0.5, 0.0)])
    assert coords.position.size == coords.ndim
    assert coords.at_bounds() == False

def test_check_bounds2():
    coords = StandardCoordinates(ndim=3, bounds=[(0.0, 1.0),
                                                 (0.0, 0.5),
                                                 (-0.5, 0.0)])
    # All bounds
    coords.position = np.array([1.0, 1.0, 1.0])
    bounds_array1 = coords.check_bounds()
    # At bounds
    coords.position = np.array([0.5, 0.2, -0.3])
    bounds_array2 = coords.check_bounds()
    # Not at bounds
    coords.position = np.array([0.5, 0.6, -0.6])
    bounds_array3 = coords.check_bounds()
    assert np.all(bounds_array1 == np.array([True, True, True]))
    assert np.all(bounds_array2 == np.array([False, False, False]))
    assert np.all(bounds_array3 == np.array([False, True, True]))

def test_at_bounds2():
    coords = StandardCoordinates(ndim=3, bounds=[(0.0, 1.0),
                                                 (0.0, 0.5),
                                                 (-0.5, 0.0)])
    # All bounds
    coords.position = np.array([1.0, 1.0, 1.0])
    assert coords.at_bounds() == True
    # At bounds
    coords.position = np.array([0.5, 0.2, -0.3])
    assert coords.at_bounds() == False
    # Not at bounds
    coords.position = np.array([0.5, 0.6, -0.6])
    assert coords.at_bounds() == True

def test_all_bounds2():
    coords = StandardCoordinates(ndim=3, bounds=[(0.0, 1.0),
                                                 (0.0, 0.5),
                                                 (-0.5, 0.0)])
    # All bounds
    coords.position = np.array([1.0, 1.0, 1.0])
    assert coords.all_bounds() == True
    # At bounds
    coords.position = np.array([0.5, 0.2, -0.3])
    assert coords.all_bounds() == False
    # Not at bounds
    coords.position = np.array([0.5, 0.6, -0.6])
    assert coords.all_bounds() == False

def test_active_bounds2():
    coords = StandardCoordinates(ndim=3, bounds=[(0.0, 1.0),
                                                 (0.0, 0.5),
                                                 (-0.5, 0.0)])
     # All bounds
    coords.position = np.array([1.0, 1.0, 1.0])
    below, above = coords.active_bounds()
    assert np.all(below == np.array([False, False, False]))
    assert np.all(above == np.array([True, True, True]))
    # At bounds
    coords.position = np.array([0.5, 0.2, -0.3])
    below, above = coords.active_bounds()
    assert np.all(below == np.array([False, False, False]))
    assert np.all(above == np.array([False, False, False]))
    # Not at bounds
    coords.position = np.array([0.5, 0.6, -0.6])
    below, above = coords.active_bounds()
    assert np.all(below == np.array([False, False, True]))
    assert np.all(above == np.array([False, True, False]))

def test_move_to_bounds2():
    coords = StandardCoordinates(ndim=3, bounds=[(0.0, 1.0),
                                                 (0.0, 0.5),
                                                 (-0.5, 0.0)])
    # All bounds
    coords.position = np.array([1.0, 1.0, 1.0])
    coords.move_to_bounds()
    assert np.all(coords.position == np.array([1.0, 0.5, 0.0]))
    # At bounds
    coords.position = np.array([0.5, 0.2, -0.3])
    coords.move_to_bounds()
    assert np.all(coords.position == np.array([0.5, 0.2, -0.3]))
    # Not at bounds
    coords.position = np.array([0.5, 0.6, -0.6])
    coords.move_to_bounds()
    assert np.all(coords.position == np.array([0.5, 0.5, -0.5]))


####### TEST ATOMIC COORDINATES CLASS
    
def test_get_atom():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                       -0.7430002647, -0.2647604843, 0.0468569750,
                        0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','C','C']
    coords = AtomicCoordinates(atom_labels, position)
    atom0 = coords.get_atom(0)
    assert np.all(atom0 == np.array([0.7430002202, 0.2647603899,
                                     -0.0468575389]))
    atom4 = coords.get_atom(4)
    assert np.all(atom4 == np.array([-0.1822009635, 0.5970484122,
                                     0.4844363476]))

def test_write_xyz():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                       -0.7430002647, -0.2647604843, 0.0468569750,
                        0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','C','C']
    coords = AtomicCoordinates(atom_labels, position)
    coords.write_xyz('min')
    assert os.path.exists('coordsmin.xyz') == True
    os.remove('coordsmin.xyz')

def test_write_extended_xyz():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                       -0.7430002647, -0.2647604843, 0.0468569750,
                        0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','C','C']
    coords = AtomicCoordinates(atom_labels, position)
    coords.write_extended_xyz(energy=-6.0000, grad=np.zeros((18), dtype=float))
    assert os.path.exists('coords.xyz') == True
    os.remove('coords.xyz')

def test_same_bonds_atomic():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                        0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','C','C']
    coords = AtomicCoordinates(atom_labels, position)
    connected = coords.same_bonds()
    assert connected == True

def test_same_bonds_atomic2():
    position = np.array([9.814092288762296, 9.227490230720699, -10.038780475529002,
                         0.8092010989322692, -0.16947705869253502, -0.9655624073859348,
                         0.26589614724764465, 0.9736503616532832, -0.19023873584003642,
                         9.025775160466077, 10.023895940544458, -9.705304241267015,
                         -10.730951910669928, -9.633300369407152, 10.271102362179427,
                         -10.080332241326632, -10.514264109213222, 10.426436148917208,
                         0.8963194565882777, 0.09200500439447187, 0.20234734892535222])
    atom_labels = ['C','C','C','C','C','C','C']
    coords = AtomicCoordinates(atom_labels, position)
    connected = coords.same_bonds()
    assert connected == False

def test_check_atom_clashes():
    position = np.array([9.814092288762296, 9.227490230720699, -10.038780475529002,
                         0.8092010989322692, -0.16947705869253502, -0.9655624073859348,
                         0.26589614724764465, 0.9736503616532832, -0.19023873584003642,
                         9.025775160466077, 10.023895940544458, -9.705304241267015,
                         -10.730951910669928, -9.633300369407152, 10.271102362179427,
                         -10.080332241326632, -10.514264109213222, 10.426436148917208,
                         0.8963194565882777, 0.09200500439447187, 0.20234734892535222])
    atom_labels = ['C','C','C','C','C','C','C']
    coords = AtomicCoordinates(atom_labels, position)
    clashes = coords.check_atom_clashes()
    assert clashes == False

def test_check_atom_clashes2():
    position = np.array([9.814092288762296, 9.227490230720699, -10.038780475529002,
                         0.8092010989322692, -0.16947705869253502, -0.9655624073859348,
                         0.86589614724764465, 0.0736503616532832, -0.99023873584003642,
                         9.025775160466077, 10.023895940544458, -9.705304241267015,
                         -10.730951910669928, -9.633300369407152, 10.271102362179427,
                         -10.080332241326632, -10.514264109213222, 10.426436148917208,
                         0.8963194565882777, 0.09200500439447187, 0.20234734892535222])
    atom_labels = ['C','C','C','C','C','C','C']
    coords = AtomicCoordinates(atom_labels, position)
    clashes = coords.check_atom_clashes()
    assert clashes == True

def test_remove_atom_clashes_atomic():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                       -0.7430002647, -0.2647604843, 0.0468569750,
                        0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','C','C']
    coords = AtomicCoordinates(atom_labels, position)
    coords.remove_atom_clashes()
    assert np.all(coords.position == position)

def test_remove_atom_clashes_atomic2():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                         0.6430002647, -0.2647604843, 0.0468569750,
                        0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','C','C']
    coords = AtomicCoordinates(atom_labels, position)
    coords.remove_atom_clashes()
    unclashed = np.array([0.93500027, 0.36404554, -0.06442912, 0.79750033,
                          -0.36404567, 0.06442834, 0.18525043, -0.61149277,
                           0.8558963, -0.35850121, 0.611493, -0.85589594,
                          -0.33715136, 0.82094157, 0.66609998, 0.16390207,
                          -0.82094167, -0.66609956])
    assert np.all(coords.position == pytest.approx(unclashed, abs=1e-5))

####### TEST MOLECULAR COORDINATES CLASS

def test_mol_initialisation():
    atom_labels = ['C','C','O','H','H','H','H','H','H']
    position = np.array([0.0072, -0.5687, 0.0,
                         -1.2854, 0.2499, 0.0,
                         1.1304, 0.3147, 0.0,
                         0.0392, -1.1972, 0.89,
                         0.0392, -1.1972, -0.89,
                         -1.3175, 0.8784, 0.89,
                         -1.3175, 0.8784, -0.89,
                         -2.1422, -0.4239, 0.0,
                          1.9857, -0.1365, 0.0])
    coords = MolecularCoordinates(atom_labels, position)
    bonds = []
    for u, v in coords.reference_bonds.edges():
        bonds.append([u, v])
    assert bonds == [[0, 1], [0, 2], [0, 3], [0, 4], [1, 5],
                     [1, 6], [1, 7], [2, 8]]

def test_get_bonds():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    bonds = coords.get_bonds()
    current_bonds = []
    for u, v in bonds.edges():
        current_bonds.append([u, v])
    assert current_bonds == [[0, 1], [0, 2], [0, 3], [0, 4], [1, 5],
                             [1, 6], [1, 7], [2, 8]]

def test_get_bonds2():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    coords.position[0:3] = np.array([-1.2854, 0.2499, 0.0])
    coords.position[3:6] = np.array([0.0072, -0.5687, 0.0])
    bonds = coords.get_bonds()
    current_bonds = []
    for u, v in bonds.edges():
        current_bonds.append([u, v])
    assert current_bonds == [[0, 1], [0, 5], [0, 6], [0, 7],
                             [1, 2], [1, 3], [1, 4], [2, 8]]

def test_get_bonds3():
    atoms = ase.io.read('test_data/azobenzene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    bonds = coords.get_bonds()
    current_bonds = []
    for u, v in bonds.edges():
        current_bonds.append([u, v])
    assert current_bonds == [[0, 1], [0, 5], [0, 6], [1, 2],
                             [5, 4], [6, 7], [2, 3], [3, 4],
                             [7, 8], [8, 9], [8, 13], [9, 10],
                             [13, 12], [10, 11], [11, 12]]

def test_get_bonds4():
    # dissociation
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    coords.position[6:9] += np.array([4.0, 4.0, 4.0])
    coords.position[-3:] += np.array([4.0, 4.0, 4.0])
    bonds = coords.get_bonds()
    current_bonds = []
    for u, v in bonds.edges():
        current_bonds.append([u, v])
    assert current_bonds == [[0, 1], [0, 3], [0, 4], [1, 5],
                             [1, 6], [1, 7], [2, 8]]

def test_same_bonds():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    coords.position[0:3] = np.array([-1.2854, 0.2499, 0.0])
    coords.position[3:6] = np.array([0.0072, -0.5687, 0.0])
    same = coords.same_bonds()
    assert same == True

def test_same_bonds2():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    same = coords.same_bonds()
    assert same == True

def test_same_bonds3():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    coords.position[6:9] += np.array([4.0, 4.0, 4.0])
    coords.position[-3:] += np.array([4.0, 4.0, 4.0])
    same = coords.same_bonds()
    assert same == False

def test_get_planar_rings():
    atoms = ase.io.read('test_data/azobenzene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    rigid_atoms = coords.get_planar_rings()
    assert rigid_atoms == [11, 10, 9, 8, 13, 12, 1, 2, 3, 4, 5, 0]

def test_get_planar_rings2():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    rigid_atoms = coords.get_planar_rings()
    assert rigid_atoms == []

def test_get_planar_rings3():
    atoms = ase.io.read('test_data/benzene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    rigid_atoms = coords.get_planar_rings()
    assert rigid_atoms == [0, 1, 2, 3, 4, 5]

def test_get_movable_atoms():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    movable_atoms = coords.get_movable_atoms([0, 1], 'dihedral', coords.reference_bonds)
    assert movable_atoms == [1, 5, 6, 7]
    movable_atoms = coords.get_movable_atoms([0, 2], 'dihedral', coords.reference_bonds)
    assert movable_atoms == [8, 2]

def test_get_movable_atoms2():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    movable_atoms = coords.get_movable_atoms([1, 0, 2], 'angle', coords.reference_bonds)
    assert movable_atoms == [8, 2]
    movable_atoms = coords.get_movable_atoms([3, 0, 4], 'angle', coords.reference_bonds)
    assert movable_atoms == [4]

def test_get_movable_atoms3():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    movable_atoms = coords.get_movable_atoms([0, 1], 'length', coords.reference_bonds)
    assert movable_atoms == [1, 5, 6, 7]
    movable_atoms = coords.get_movable_atoms([0, 2], 'length', coords.reference_bonds)
    assert movable_atoms == [8, 2]

def test_get_movable_atoms4():
    atoms = ase.io.read('test_data/benzene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    movable_atoms = coords.get_movable_atoms([0, 1], 'dihedral', coords.reference_bonds)
    assert movable_atoms == [7]
    movable_atoms = coords.get_movable_atoms([1, 2], 'dihedral', coords.reference_bonds)
    assert movable_atoms == [8]

def test_get_movable_atoms5():
    atoms = ase.io.read('test_data/benzene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    movable_atoms = coords.get_movable_atoms([0, 1, 2], 'angle', coords.reference_bonds)
    assert movable_atoms == [2, 8]
    movable_atoms = coords.get_movable_atoms([4, 5, 11], 'angle', coords.reference_bonds)
    assert movable_atoms == [11]

def test_get_movable_atoms6():
    atoms = ase.io.read('test_data/benzene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    movable_atoms = coords.get_movable_atoms([4, 10], 'length', coords.reference_bonds)
    assert movable_atoms == [10]
    movable_atoms = coords.get_movable_atoms([3, 4], 'length', coords.reference_bonds)
    assert movable_atoms == [4, 10]

def test_get_movable_atoms7():
    atoms = ase.io.read('test_data/cyclopentane.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    movable_atoms = coords.get_movable_atoms([2, 5], 'dihedral', coords.reference_bonds)
    assert movable_atoms == [9, 10]
    movable_atoms = coords.get_movable_atoms([2, 7], 'dihedral', coords.reference_bonds)
    assert movable_atoms == [7]

def test_get_movable_atoms8():
    atoms = ase.io.read('test_data/cyclopentane.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    movable_atoms = coords.get_movable_atoms([7, 2, 5], 'angle', coords.reference_bonds)
    assert movable_atoms == [7]
    movable_atoms = coords.get_movable_atoms([2, 5, 8], 'angle', coords.reference_bonds)
    assert movable_atoms == [8, 11, 12]

def test_get_movable_atoms9():
    atoms = ase.io.read('test_data/cyclopentane.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    movable_atoms = coords.get_movable_atoms([2, 5], 'length', coords.reference_bonds)
    assert movable_atoms == [5, 9, 10]
    movable_atoms = coords.get_movable_atoms([2, 7], 'length', coords.reference_bonds)
    assert movable_atoms == [7]

def test_get_movable_atoms10():
    atoms = ase.io.read('test_data/azobenzene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    movable_atoms = coords.get_movable_atoms([6, 7], 'dihedral', coords.reference_bonds)
    assert movable_atoms == [7, 8, 9, 10, 11, 12, 13]
    movable_atoms = coords.get_movable_atoms([3, 4], 'dihedral', coords.reference_bonds)
    assert movable_atoms == []

def test_get_movable_atoms11():
    atoms = ase.io.read('test_data/azobenzene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    movable_atoms = coords.get_movable_atoms([6, 7, 8], 'angle', coords.reference_bonds)
    assert movable_atoms == [8, 9, 10, 11, 12, 13]
    movable_atoms = coords.get_movable_atoms([3, 4, 5], 'angle', coords.reference_bonds)
    assert movable_atoms == [5]
    movable_atoms = coords.get_movable_atoms([5, 0, 6], 'angle', coords.reference_bonds)
    assert movable_atoms == [6, 7, 8, 9, 10, 11, 12, 13]
    movable_atoms = coords.get_movable_atoms([6, 0, 5], 'angle', coords.reference_bonds)
    assert movable_atoms == [6, 7, 8, 9, 10, 11, 12, 13]

def test_get_movable_atoms12():
    atoms = ase.io.read('test_data/azobenzene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    movable_atoms = coords.get_movable_atoms([6, 7], 'length', coords.reference_bonds)
    assert movable_atoms == [7, 8, 9, 10, 11, 12, 13]
    movable_atoms = coords.get_movable_atoms([3, 4], 'length', coords.reference_bonds)
    assert movable_atoms == [4]

def test_get_repeat_dihedrals():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    dihedrals = [[1, 0, 2, 8], [2, 0, 1, 5], [2, 0, 1, 6],
                 [2, 0, 1, 7], [3, 0, 1, 5], [3, 0, 1, 6],
                 [3, 0, 1, 7], [3, 0, 2, 8], [4, 0, 1, 5],
                 [4, 0, 1, 6], [4, 0, 1, 7], [4, 0, 2, 8]]
    repeats = coords.get_repeat_dihedrals(dihedrals)
    assert repeats == [[2, 0, 1, 6], [2, 0, 1, 7], [3, 0, 1, 5],
                       [3, 0, 1, 6], [3, 0, 1, 7], [3, 0, 2, 8],
                       [4, 0, 1, 5], [4, 0, 1, 6], [4, 0, 1, 7],
                       [4, 0, 2, 8]]

def test_get_rotatable_dihedrals():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    assert coords.rotatable_dihedrals == [(1, 0, 2, 8), (2, 0, 1, 5)]
    assert coords.rotatable_atoms[0] == [8, 2]
    assert coords.rotatable_atoms[1] == [1, 5, 6, 7]

def test_get_rotatable_dihedrals2():
    atoms = ase.io.read('test_data/benzene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    assert coords.rotatable_dihedrals == []
    assert coords.rotatable_atoms == []

def test_get_rotatable_dihedrals3():
    atoms = ase.io.read('test_data/azobenzene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    assert coords.rotatable_dihedrals == [(0, 6, 7, 8), (6, 7, 8, 9),
                                          (1, 0, 6, 7)]
    assert coords.rotatable_atoms[0] == [7, 8, 9, 10, 11, 12, 13]
    assert coords.rotatable_atoms[1] == [8, 9, 10, 11, 12, 13]
    assert coords.rotatable_atoms[2] == [6, 7, 8, 9, 10, 11, 12, 13]

def test_get_rotation_matrix():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    rot_matrix = coords.get_rotation_matrix(np.pi/2.0)
    assert np.all(rot_matrix == pytest.approx(np.array([[1.0, 0.0, 0.0],
                                                        [0.0, 0.0, -1.0],
                                                        [0.0, 1.0, 0.0]])))

def test_get_rotation_matrix2():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    rot_matrix = coords.get_rotation_matrix(-np.pi/2.0)
    assert np.all(rot_matrix == pytest.approx(np.array([[1.0, 0.0, 0.0],
                                                        [0.0, 0.0, 1.0],
                                                        [0.0, -1.0, 0.0]])))

def test_rotate_dihedral():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    coords.rotate_dihedral([0, 1], 120.0, [1, 5, 6, 7])
    atoms_rot = ase.io.read('test_data/ethanol_rot120.xyz')
    rotated = atoms_rot.get_positions()
    assert np.all(rotated.flatten() == pytest.approx(coords.position))

def test_rotate_dihedral2():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    coords.rotate_dihedral([0, 1], 60.0, [1, 5, 6, 7])
    atoms_rot = ase.io.read('test_data/ethanol_rot60.xyz')
    rotated = atoms_rot.get_positions()
    assert np.all(rotated.flatten() == pytest.approx(coords.position))

def test_rotate_angle():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    coords.rotate_angle([8, 2, 0], 45.0, [8])
    atoms_rot = ase.io.read('test_data/ethanol_angle.xyz')
    rotated = atoms_rot.get_positions()
    assert np.all(rotated.flatten() == pytest.approx(coords.position))

def test_rotate_angle2():
    atoms = ase.io.read('test_data/cyclopentane.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    coords.rotate_angle([2, 5, 8], 10.0, [8, 11, 12])
    atoms_rot = ase.io.read('test_data/cyclopentane_angle.xyz')
    rotated = atoms_rot.get_positions()
    assert np.all(rotated.flatten() == pytest.approx(coords.position))

def test_change_bond_length():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    coords.change_bond_length([2, 8], 1.0, [8])
    atoms_bond = ase.io.read('test_data/ethanol_bond.xyz')
    bond_change = atoms_bond.get_positions()
    assert np.all(bond_change.flatten() == pytest.approx(coords.position))

def test_change_bond_length2():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    coords.change_bond_length([0, 1], 1.0, [1, 5, 6, 7])
    atoms_bond = ase.io.read('test_data/ethanol_bond2.xyz')
    bond_change = atoms_bond.get_positions()
    assert np.all(bond_change.flatten() == pytest.approx(coords.position))

def test_change_bond_lengths():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    bonds = [[0, 1], [0, 3], [0, 4], [1, 7]]
    step = [1.0, 0.5, 0.5, 0.5]
    coords.change_bond_lengths(bonds, step, coords.reference_bonds)
    atoms_bond = ase.io.read('test_data/ethanol_expand.xyz')
    bond_change = atoms_bond.get_positions()
    assert np.all(bond_change.flatten() == pytest.approx(coords.position))

def test_change_bond_lengths2():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    bonds = [[0, 1], [0, 3], [0, 4], [1, 7]]
    step = [-0.3, -0.5, -0.5, -0.5]
    coords.change_bond_lengths(bonds, step, coords.reference_bonds)
    atoms_bond = ase.io.read('test_data/ethanol_compress.xyz')
    bond_change = atoms_bond.get_positions()
    assert np.all(bond_change.flatten() == pytest.approx(coords.position))

def test_change_bond_angles():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    angles = [[2, 0, 1], [8, 2, 0], [5, 1, 6]]
    step = [-30.0, 50.0, 40.0]
    coords.change_bond_angles(angles, step, coords.reference_bonds)
    atoms_bond = ase.io.read('test_data/ethanol_multiangle.xyz')
    bond_change = atoms_bond.get_positions()
    assert np.all(bond_change.flatten() == pytest.approx(coords.position))

def test_change_dihedral_angles():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    dihedrals = [[1, 0, 2, 8], [2, 0, 1, 5]]
    step = [180.0, 60.0]
    coords.change_dihedral_angles(dihedrals, step, coords.reference_bonds)
    atoms_bond = ase.io.read('test_data/ethanol_dihedral.xyz')
    bond_change = atoms_bond.get_positions()
    assert np.all(bond_change.flatten() == pytest.approx(coords.position))

def test_get_bond_angle_info():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    information = coords.get_bond_angle_info()
    assert information[0] == [[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [1, 7], [2, 8]]
    assert information[1] == [1.5300067712268468, 1.4289764868604382, 1.0900166283135317,
                              1.0900166283135317, 1.090019568631683, 1.090019568631683,
                              1.090005816498242, 0.9670157858070363]
    assert information[2] == [[1, 0, 2], [1, 0, 3], [1, 0, 4], [2, 0, 3], [2, 0, 4],
                              [0, 1, 5], [0, 1, 6], [0, 1, 7], [5, 1, 6], [5, 1, 7], [0, 2, 8]]
    assert information[3] == [109.46888152765341, 109.46912025014315, 109.46912025014315,
                              109.47402157616285, 109.47402157616285, 109.47377577610308,
                              109.47377577610308, 109.47200676667457, 109.47172120558963,
                              109.4680219897502, 114.00166889964733]
    assert information[4] == [(1, 0, 2, 8), (2, 0, 1, 5)]
    assert information[5] == [180.0, 299.99812858558613]

def test_get_bond_angle_info2():
    atoms = ase.io.read('test_data/benzene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    information = coords.get_bond_angle_info()
    assert information[0] == [[0, 1], [0, 5], [0, 6], [1, 2], [1, 7], [2, 3], [2, 8],
                              [3, 4], [3, 9], [4, 5], [4, 10], [5, 11]]
    assert information[1] == [1.3969675336241714, 1.3969675336241714, 1.0839999999999999,
                              1.397, 1.0840246491662449, 1.3969675336241714, 1.0840246491662449,
                              1.3969675336241714, 1.0839999999999999, 1.397, 1.0840246491662449,
                              1.0840246491662449]
    assert information[2] == [[1, 0, 5], [1, 0, 6], [0, 1, 2], [0, 1, 7], [1, 2, 3],
                              [1, 2, 8], [2, 3, 4], [2, 3, 9], [3, 4, 5], [3, 4, 10],
                              [0, 5, 4], [0, 5, 11]]
    assert information[3] == [119.99846240774873, 120.00076879612564, 120.00076879612564,
                              119.99998338673893, 120.00076879612564, 119.99924781713545,
                              119.99846240774873, 120.00076879612564, 120.00076879612564,
                              119.99998338673893, 120.00076879612564, 119.99998338673893]
    assert information[4] == []
    assert information[5] == []

def test_get_bond_angle_info3():
    atoms = ase.io.read('test_data/cyclopentane.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    information = coords.get_bond_angle_info()
    assert information[0] == [[0, 1], [0, 8], [0, 13], [0, 14],
                              [1, 2], [1, 3], [1, 4], [2, 5],
                              [2, 6], [2, 7], [5, 8], [5, 9],
                              [5, 10], [8, 11], [8, 12]]
    assert information[1] == [1.5409482405324326, 1.5417236587663823, 1.1021277013123298,
                              1.1027180419309373, 1.5392686737538708, 1.1015017203799546,
                              1.1035691459985641, 1.5383986056935959, 1.1039334989029004,
                              1.1012469432420684, 1.540094198417746, 1.1010999046408096,
                              1.1039341782914414, 1.1031420126166893, 1.1017811670200213]
    assert information[2] == [[1, 0, 8], [1, 0, 13], [1, 0, 14], [8, 0, 13], [8, 0, 14],
                              [0, 1, 2], [0, 1, 3], [0, 1, 4], [2, 1, 3], [2, 1, 4],
                              [1, 2, 5], [1, 2, 6], [1, 2, 7], [5, 2, 6], [5, 2, 7],
                              [2, 5, 8], [2, 5, 9], [2, 5, 10], [8, 5, 9], [8, 5, 10],
                              [0, 8, 5], [0, 8, 11], [0, 8, 12], [5, 8, 11], [5, 8, 12]]
    assert information[3] == [107.40877470546506, 110.78842406821003, 110.81956424369591,
                              110.98926065549517, 110.59010235738948, 106.75991608213614,
                              111.4421489307025, 110.46997083346221, 111.3706302549835,
                              110.4775004839204, 106.1302122965746, 110.38454074240173,
                              111.66724424323061, 110.45311861830031, 111.68721940032029,
                              106.33656780418664, 111.59722472538715, 110.41086681893229,
                              111.58335694333499, 110.43820332939484, 107.1881448468413,
                              110.41940784021584, 111.1485715322863, 110.59738547819364,
                              111.06990831334431]
    assert information[4] == [(1, 0, 8, 5), (1, 2, 5, 8), (2, 1, 0, 8),
                              (0, 8, 5, 2), (0, 1, 2, 5)]
    assert information[5] == [4.023407170780608, 25.221246320105575, 11.569715813197046,
                              341.9236462777772, 337.25629573038157]

def test_remove_repeat_angles():
    atoms = ase.io.read('test_data/cyclopentane.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    angles = [[0, 1, 2], [0, 1, 3], [1, 0, 3], [7, 1, 3], [2, 0, 4], [6, 5, 4]]
    final_angles = coords.remove_repeat_angles(angles)
    assert set(tuple(i) for i in final_angles) == \
        set(tuple(i) for i in [[0, 1, 2], [0, 1, 3], [1, 0, 3], [6, 5, 4]])

def test_get_specific_bond_angle_info():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    # same as get_bond_angle_info
    bonds = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [1, 7], [2, 8]]
    angles = [[1, 0, 2], [1, 0, 3], [1, 0, 4], [2, 0, 3], [2, 0, 4],
              [0, 1, 5], [0, 1, 6], [0, 1, 7], [5, 1, 6], [5, 1, 7], [0, 2, 8]]
    dihedrals = [(1, 0, 2, 8), (2, 0, 1, 5)]
    coords.position[0:3] = np.array([-1.2854, 0.2499, 0.0])
    coords.position[3:6] = np.array([0.0072, -0.5687, 0.0])
    coords.position[9:12] = np.array([1.9857, -0.1365, 0.0])
    coords.position[24:27] = np.array([0.0392, -1.1972, 0.89])
    permutation = [1, 0, 2, 8, 4, 5, 6, 7, 3]
    information = coords.get_specific_bond_angle_info(bonds, angles, dihedrals, permutation)
    assert information[0] == [1.5300067712268468, 1.4289764868604382, 1.0900166283135317,
                              1.0900166283135317, 1.090019568631683, 1.090019568631683,
                              1.090005816498242, 0.9670157858070363]
    assert information[1] == [109.46888152765341, 109.46912025014315, 109.46912025014315,
                              109.47402157616285, 109.47402157616285, 109.47377577610308,
                              109.47377577610308, 109.47200676667457, 109.47172120558963,
                              109.4680219897502, 114.00166889964733]
    assert information[2] == [180.0, 299.99812858558613]

def test_get_specific_bond_angle_info2():
    atoms = ase.io.read('test_data/cyclopentane.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    # same as get_bond_angle_info
    bonds = [[0, 1], [0, 8], [0, 13], [0, 14], [1, 2], [1, 3],
             [1, 4], [2, 5], [2, 6], [2, 7], [5, 8], [5, 9],
             [5, 10], [8, 11], [8, 12]]
    angles = [[1, 0, 8], [1, 0, 13], [1, 0, 14], [8, 0, 13], [8, 0, 14],
              [0, 1, 2], [0, 1, 3], [0, 1, 4], [2, 1, 3], [2, 1, 4],
              [1, 2, 5], [1, 2, 6], [1, 2, 7], [5, 2, 6], [5, 2, 7],
              [2, 5, 8], [2, 5, 9], [2, 5, 10], [8, 5, 9], [8, 5, 10],
              [0, 8, 5], [0, 8, 11], [0, 8, 12], [5, 8, 11], [5, 8, 12]]
    dihedrals = [(1, 0, 8, 5), (1, 2, 5, 8), (2, 1, 0, 8), (0, 8, 5, 2), (0, 1, 2, 5)]
    coords.position[0:3] = np.array([1.2384, 0.3351, 0.2162])
    coords.position[6:9] = np.array([-0.8201, -1.0104, -0.1068])
    coords.position[15:18] = np.array([0.0767, 1.2934, -0.0981])
    coords.position[9:12] = np.array([0.1607, 1.6760, -1.1302])
    coords.position[18:21] = np.array([-1.9775, 0.7710, -0.6688])
    permutation = [2, 1, 5, 6, 4, 0, 3, 7, 8, 9, 10, 11, 12, 13, 14]
    information = coords.get_specific_bond_angle_info(bonds, angles, dihedrals, permutation)
    # Check same answer as for get_bond_angle_info
    assert information[0] == [1.5409482405324326, 1.5417236587663823, 1.1021277013123298,
                              1.1027180419309373, 1.5392686737538708, 1.1015017203799546,
                              1.1035691459985641, 1.5383986056935959, 1.1039334989029004,
                              1.1012469432420684, 1.540094198417746, 1.1010999046408096,
                              1.1039341782914414, 1.1031420126166893, 1.1017811670200213]
    assert information[1] == [107.40877470546506, 110.78842406821003, 110.81956424369591,
                              110.98926065549517, 110.59010235738948, 106.75991608213614,
                              111.4421489307025, 110.46997083346221, 111.3706302549835,
                              110.4775004839204, 106.1302122965746, 110.38454074240173,
                              111.66724424323061, 110.45311861830031, 111.68721940032029,
                              106.33656780418664, 111.59722472538715, 110.41086681893229,
                              111.58335694333499, 110.43820332939484, 107.1881448468413,
                              110.41940784021584, 111.1485715322863, 110.59738547819364,
                              111.06990831334431]
    assert information[2] == [4.023407170780608, 25.221246320105575, 11.569715813197046,
                              341.9236462777772, 337.25629573038157]

def test_remove_atom_clashes():
    atoms = ase.io.read('test_data/hexane.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    ff = MMFF94('test_data/hexane.xyz')
    coords.rotate_dihedral([1, 2], 180.0, [3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    coords.rotate_dihedral([2, 3], 180.0, [4, 5, 13, 14, 15, 16, 17, 18, 19])
    coords.rotate_dihedral([3, 4], 180.0, [5, 15, 16, 17, 18, 19])
    coords.remove_atom_clashes(ff)
    unclashed_position = coords.position.copy()
    coords.remove_atom_clashes(ff)
    assert np.all(coords.position == unclashed_position)
