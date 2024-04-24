import pytest
import numpy as np
import ase
from topsearch.similarity.molecular_similarity import MolecularSimilarity
from topsearch.data.coordinates import AtomicCoordinates, MolecularCoordinates

### ATOMIC SYSTEMS

def test_permutational_alignment():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                       -0.7430002647, -0.2647604843, 0.0468569750,
                        0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','Ag','C','Ag','C','Ag']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05)
    coords_b = coords.position.copy()
    coords_b[0:3] = np.array([0.1977276118, -0.4447220146, 0.6224700350])
    coords_b[6:9] = np.array([0.7430002202, 0.2647603899, -0.0468575389])
    perm_coords, permutation = similarity.permutational_alignment(coords, coords_b)
    assert np.all(permutation == np.array([2, 1, 0, 3, 4, 5]))
    assert np.all(perm_coords == coords.position)

def test_permutational_alignment2():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                       -0.7430002647, -0.2647604843, 0.0468569750,
                        0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','Ag','Ag']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05)
    coords_b = np.array([-0.1977281310, 0.4447221826, -0.6224697723,
                         0.7430002202, 0.2647603899, -0.0468575389,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1822015272, -0.5970484858, -0.4844360463,
                        -0.1822009635, 0.5970484122, 0.4844363476])
    perm_coords, permutation = similarity.permutational_alignment(coords, coords_b)
    assert np.all(permutation == np.array([1, 3, 2, 0, 5, 4]))
    assert np.all(perm_coords == pytest.approx(coords.position))

def test_get_permutable_groups():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                       -0.7430002647, -0.2647604843, 0.0468569750,
                        0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','Ag','Ag']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05)
    coords_b = np.array([-0.1977281310, 0.4447221826, -0.6224697723,
                         0.7430002202, 0.2647603899, -0.0468575389,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1822015272, -0.5970484858, -0.4844360463,
                        -0.1822009635, 0.5970484122, 0.4844363476])
    p1, p2 = similarity.get_permutable_groups(coords, coords_b)
    if p1[0] == [4, 5]:
        assert p1 == [[4, 5], [0, 1, 2, 3]]
        assert p2 == [[4, 5], [0, 1, 2, 3]]
    else:
        assert p1 == [[0, 1, 2, 3], [4, 5]]
        assert p2 == [[0, 1, 2, 3], [4, 5]]

def test_random_rotation():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    init_position = position.copy()
    atom_labels = ['C','C','C','C','C','C']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05)
    rotated = similarity.random_rotation(coords.position)
    coords.position = rotated
    bond_lengths = []
    atom0 = coords.get_atom(0)
    for i in range(1, 6):
        atom_i = coords.get_atom(i)
        bond_lengths.append(np.linalg.norm(atom0 - atom_i))
    assert bond_lengths == pytest.approx([1.5803076306378077, 1.1174465987013884,
                            1.1174459811655348, 1.1174460854628294,
                            1.1174455970848058])
    assert not np.all(init_position == coords.position)

def test_centre():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    position += 0.1
    atom_labels = ['C','C','C','C','O','O']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05)
    centered = similarity.centre(position, coords.atom_weights)
    centered2 = similarity.centre(centered, coords.atom_weights)
    assert np.all(centered == pytest.approx(centered2))
    assert not np.all(centered2 == position)

def test_centre_weighted():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    position -= 0.31
    atom_labels = ['C','C','C','C','O','O']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05, weighted=True)
    centered = similarity.centre(position, coords.atom_weights)
    centered2 = similarity.centre(centered, coords.atom_weights)
    assert np.all(centered == pytest.approx(centered2))
    assert not np.all(centered2 == position)

def test_distance():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    init_position = position.copy()
    position[0:12] -= 0.1
    position[12:] -= 0.2
    atom_labels = ['C','C','C','C','O','O']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05)
    dist = similarity.distance(coords.position, init_position)
    assert dist == pytest.approx(0.6)

def test_rotational_alignment():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    init_position = position.copy()
    atom_labels = ['C','C','C','C','C','C']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05)
    for i in range(5):
        rotated = similarity.random_rotation(coords.position)
        dist, c2 = similarity.rotational_alignment(coords, rotated)
        assert dist < 1e-5
    
def test_align():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','O','O']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05, weighted=True)
    # Rotate, permute and displace by a very small amount
    coords2 = np.array([-0.1977281310, 0.4447221826, -0.6224697723,
                         0.7430002202, 0.2647603899, -0.0468575389,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1822015272, -0.5970484858, -0.4844360463,
                        -0.1822009635, 0.5970484122, 0.4844363476])
    coords2 = similarity.random_rotation(coords2)
    dist, c2, permutation = similarity.align(coords, coords2)
    assert dist < 1e-5
    assert np.all(c2 == pytest.approx(coords.position, abs=1e-5))

def test_test_exact_same():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','O','O']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05, weighted=True)
    # Rotate, permute and displace by a very small amount
    coords2 = np.array([-0.1977281310, 0.4447221826, -0.6224697723,
                         0.7430002202, 0.2647603899, -0.0468575389,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1822015272, -0.5970484858, -0.4844360463,
                        -0.1822009635, 0.5970484122, 0.4844363476])
    coords2 = similarity.random_rotation(coords2)
    dist, c2, permutation = similarity.test_exact_same(coords, coords2)
    assert dist < 1e-5

def test_test_exact_same2():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','O','O']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05, weighted=True)
    # Rotate, permute and displace by a very small amount
    coords2 = np.array([-0.2077281310, 0.4447221826, -0.6224697723,
                         0.7530002202, 0.2647603899, -0.0468575389,
                         0.1977276118, -0.4347220146, 0.6224700350,
                        -0.7430002647, -0.2647604843, 0.1468569750,
                         0.1822015272, -0.5970484858, -0.4844360463,
                        -0.1822009635, 0.5970484122, 0.4844363476])
    coords2 = similarity.random_rotation(coords2)
    dist, c2, permutation = similarity.test_exact_same(coords, coords2)
    assert dist == pytest.approx(0.21154343240370782, abs=1e-4)

def test_optimal_alignment():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','O','O']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05, weighted=True)
    # Rotate, permute and displace by a very small amount
    coords2 = np.array([-0.1977281310, 0.4447221826, -0.6224697723,
                         0.7430002202, 0.2647603899, -0.0468575389,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1822015272, -0.5970484858, -0.4844360463,
                        -0.1822009635, 0.5970484122, 0.4844363476])
    coords2 = similarity.random_rotation(coords2)
    coords2 -= 0.4
    dist, c1, c2, permutation = similarity.optimal_alignment(coords, coords2)
    assert dist < 1e-5
    assert np.all(c2 == pytest.approx(coords.position, abs=1e-3))

def test_optimal_alignment2():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','O','O']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05, weighted=False)
    # Rotate, permute and displace by a very small amount
    coords2 = np.array([-0.2077281310, 0.4447221826, -0.6224697723,
                         0.7430002202, 0.2747603899, -0.0468575389,
                         0.1977276118, -0.4447220146, 0.7224700350,
                        -0.7430002647, -0.2647604843, 0.1468569750,
                         0.1822015272, -0.6970484858, -0.4844360463,
                        -0.2822009635, 0.5970484122, 0.4844363476])
    coords2 = similarity.random_rotation(coords2)
    coords2 -= 0.4
    dist, c1, c2, permutation = similarity.optimal_alignment(coords, coords2)
    assert dist == pytest.approx(0.15335115564097054, abs=1e-4)

def test_optimal_alignment3():
    position = np.array([-0.34730965898408783, -1.116779267503695, -0.09399409205649237,
                         -0.25853048505296805, -0.2833641287780657, -0.8396822072934766,
                          0.3927268992562973, -0.2979222137330398, 0.06781518125803959,
                         -0.048923655113916845, 0.6715072867111124, -0.30098510758122643,
                         -0.7114398756865784, -0.094695953958119, 0.1696865603760323,
                          0.023600228355890873, 0.4615816191371775, 0.7991148142576938,
                          0.949876547225363, 0.6596726581246294, 0.19804485103942987])
    atom_labels = ['C','C','C','C','C','C', 'C']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05, weighted=False, allow_inversion=True)
    # Rotate, permute and displace by a very small amount
    coords2 = np.array([ 0.38713985736237416, -0.365316094325874, 0.7542320614929241,
                        -0.32461053888530045, 0.6158843587799965, -0.6062289562915606,
                         0.5268025388623481, 0.03400401773758548, -1.0478443691044557,
                         0.6031723038313309, 0.4243471089169785, -0.00449757252090606,
                        -0.10452889267045866, -0.41886913028599604, -0.2474132794600202,
                        -0.400792479161798, 0.3866310244850648, 0.48351773270378035,
                        -0.6871827893384962, -0.6766812853077553, 0.6682343831802381])
    coords2 = coords2 + 0.4
    dist, c1, c2, permutation = similarity.optimal_alignment(coords, coords2)
    assert dist < 1e-3

def test_test_same():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','O','O']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05, weighted=True)
    # Rotate, permute and displace by a very small amount
    coords2 = np.array([-0.1977281310, 0.4447221826, -0.6224697723,
                         0.7430002202, 0.2647603899, -0.0468575389,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1822015272, -0.5970484858, -0.4844360463,
                        -0.1822009635, 0.5970484122, 0.4844363476])
    coords2 = similarity.random_rotation(coords2)
    coords2 -= 0.4
    same = similarity.test_same(coords, coords2, 1.0, 1.0)
    assert same == True
    same = similarity.test_same(coords, coords2, 1.0, 2.0)
    assert same == False

def test_test_same2():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','O','O']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.01, 0.05, weighted=False)
    # Rotate, permute and displace by a very small amount
    coords2 = np.array([-0.2077281310, 0.4447221826, -0.6224697723,
                         0.7430002202, 0.2747603899, -0.0468575389,
                         0.1977276118, -0.4447220146, 0.7224700350,
                        -0.7430002647, -0.2647604843, 0.1468569750,
                         0.1822015272, -0.6970484858, -0.4844360463,
                        -0.2822009635, 0.5970484122, 0.4844363476])
    coords2 = similarity.random_rotation(coords2)
    coords2 -= 0.4
    same = similarity.test_same(coords, coords2, 1.0, 1.0)
    assert same == False

def test_closest_distance():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','O','O']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.1, 0.05, weighted=True)
    # Rotate, permute and displace by a very small amount
    coords2 = np.array([-0.1977281310, 0.4447221826, -0.6224697723,
                         0.7430002202, 0.2647603899, -0.0468575389,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1822015272, -0.5970484858, -0.4844360463,
                        -0.1822009635, 0.5970484122, 0.4844363476])
    coords2 = similarity.random_rotation(coords2)
    coords2 -= 0.4
    dist = similarity.closest_distance(coords, coords2)
    assert dist < 1e-5

def test_closest_distance2():
    position = np.array([0.7430002202, 0.2647603899, -0.0468575389,
                        -0.7430002647, -0.2647604843, 0.0468569750,
                         0.1977276118, -0.4447220146, 0.6224700350,
                        -0.1977281310, 0.4447221826, -0.6224697723,
                        -0.1822009635, 0.5970484122, 0.4844363476,
                         0.1822015272, -0.5970484858, -0.4844360463])
    atom_labels = ['C','C','C','C','O','O']
    coords = AtomicCoordinates(atom_labels, position)
    similarity = MolecularSimilarity(0.01, 0.05, weighted=False)
    # Rotate, permute and displace by a very small amount
    coords2 = np.array([-0.2077281310, 0.4447221826, -0.6224697723,
                         0.7430002202, 0.2747603899, -0.0468575389,
                         0.1977276118, -0.4447220146, 0.7224700350,
                        -0.7430002647, -0.2647604843, 0.1468569750,
                         0.1822015272, -0.6970484858, -0.4844360463,
                        -0.2822009635, 0.5970484122, 0.4844363476])
    coords2 = similarity.random_rotation(coords2)
    coords2 -= 0.4
    dist = similarity.closest_distance(coords, coords2)
    assert dist == pytest.approx(0.15335115564097054, abs=1e-4)

def test_generate_pairs():
    similarity = MolecularSimilarity(0.01, 0.05, weighted=False)
    pairs = similarity.generate_pairs([1, 2], [0, 3, 4])
    assert pairs == [[1, 0], [1, 3], [1, 4], [2, 0], [2, 3], [2, 4]]

######## MOLECULAR TESTING

def test_permutational_alignment_mol():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    init_position = position.copy()
#    coords.position[0:3] = np.array([-1.2854, 0.2499, 0.0])
#    coords.position[3:6] = np.array([0.0072, -0.5687, 0.0])
    coords.position[9:12] = np.array([-1.3175, 0.8784, 0.89])
    coords.position[15:18] = np.array([0.0392, -1.1972, 0.89])
    similarity = MolecularSimilarity(0.01, 0.05, weighted=False)
    p_coords, permutation = similarity.permutational_alignment(coords,
                                                               init_position.flatten())
    assert np.all(permutation == np.array([0, 1, 2, 5, 4, 3, 6, 7, 8]))
    assert np.all(p_coords == pytest.approx(coords.position))

def test_permutational_alignment_mol2():
    atoms = ase.io.read('test_data/ethanol_rot.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    init_position = position.copy()
#    coords.position[0:3] = np.array([-1.2854, 0.2499, 0.0])
#    coords.position[3:6] = np.array([0.0072, -0.5687, 0.0])
    coords.position[9:12] = np.array([-1.3175, 0.8784, 0.89])
    coords.position[15:18] = np.array([0.0392, -1.1972, 0.89])
    coords.position[12:15] = np.array([-1.3175, 0.8783999999999994, -0.8899999999999996])
    coords.position[18:21] = np.array([0.039200000000000026, -1.1972, -0.8899999999999996])
    similarity = MolecularSimilarity(0.01, 0.05, weighted=False)
    p_coords, permutation = similarity.permutational_alignment(coords,
                                                               init_position.flatten())
    assert np.all(permutation == np.array([0, 1, 2, 5, 6, 3, 4, 7, 8]))
    assert np.all(p_coords == pytest.approx(coords.position))

def test_get_permutable_groups2():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    init_position = position.copy()
    coords.position[9:12] = np.array([-1.3175, 0.8784, 0.89])
    coords.position[15:18] = np.array([0.0392, -1.1972, 0.89])
    similarity = MolecularSimilarity(0.01, 0.05, weighted=False)
    p1, p2 = similarity.get_permutable_groups(coords, init_position.flatten())
    p1_set = set([tuple(x) for x in p1])
    p2_set = set([tuple(x) for x in p2])
    assert p1_set == {(3, 4, 5, 6, 7), (2,), (8,), (1,), (0,)}
    assert p2_set == {(3, 4, 5, 6, 7), (2,), (8,), (1,), (0,)}
    for i in range(len(p1)):
        assert p1[i] == p2[i]

def test_get_permutable_groups3():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    init_position = position.copy()
    coords.position[0:3] = np.array([-1.2854, 0.2499, 0.0])
    coords.position[3:6] = np.array([0.0072, -0.5687, 0.0])
    coords.position[9:12] = np.array([-1.3175, 0.8784, 0.89])
    coords.position[15:18] = np.array([0.0392, -1.1972, 0.89])
    similarity = MolecularSimilarity(0.01, 0.05, weighted=False)
    p1, p2 = similarity.get_permutable_groups(coords, init_position.flatten())
    p1_set = set([tuple(x) for x in p1])
    p2_set = set([tuple(x) for x in p2])
    assert p1_set == {(3, 4, 5, 6, 7), (2,), (8,), (1,), (0,)}
    assert p2_set == {(3, 4, 5, 6, 7), (2,), (8,), (1,), (0,)}
    for i in range(len(p1)):
        print(i, p1[i], p2[i])
        if p1[i] == [0]:
            assert p2[i] == [1]
        elif p1[i] == [1]:
            assert p2[i] == [0]
        else:
            assert p1[i] == p2[i]

def test_get_permutable_groups4():
    atoms = ase.io.read('test_data/benzene.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    init_position = position.copy()
    similarity = MolecularSimilarity(0.01, 0.05, weighted=False)
    p1, p2 = similarity.get_permutable_groups(coords, init_position.flatten())
    p1_set = set([tuple(x) for x in p1])
    p2_set = set([tuple(x) for x in p2])
    assert p1_set == {(6, 7, 8, 9, 10, 11), (0, 1, 2, 3, 4, 5)}
    assert p2_set == {(6, 7, 8, 9, 10, 11), (0, 1, 2, 3, 4, 5)}

def test_rotational_alignment_mol():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    position2 = position.copy()
    similarity = MolecularSimilarity(0.01, 0.05, weighted=False)
    position2 = similarity.random_rotation(position2)
    dist, r_coords = similarity.rotational_alignment(coords, position2)
    assert dist < 1e-5
    assert np.all(r_coords == pytest.approx(coords.position))

def test_rotational_alignment_mol2():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    atom2 = ase.io.read('test_data/ethanol_rot.xyz')
    position2 = atom2.get_positions().flatten()
    similarity = MolecularSimilarity(0.01, 0.05, weighted=True)
    position2 = similarity.random_rotation(position2)
    dist, r_coords = similarity.rotational_alignment(coords, position2)
    assert dist == pytest.approx(1.473644922799864)

def test_test_exact_same_mol():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    position2 = position.copy()
#    coords.position[0:3] = np.array([-1.2854, 0.2499, 0.0])
#    coords.position[3:6] = np.array([0.0072, -0.5687, 0.0])
    coords.position[9:12] = np.array([-1.3175, 0.8784, 0.89])
    coords.position[15:18] = np.array([0.0392, -1.1972, 0.89])
    similarity = MolecularSimilarity(0.01, 0.05, weighted=False)
    position2 = similarity.random_rotation(position2)
    dist, aligned, permutation = similarity.test_exact_same(coords, position2)
    assert dist < 1e-5
    assert np.all(aligned == pytest.approx(coords.position))

def test_test_exact_same_mol2():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    atom2 = ase.io.read('test_data/ethanol_rot.xyz')
    position2 = atom2.get_positions().flatten()
    similarity = MolecularSimilarity(0.01, 0.05, weighted=True)
#    coords.position[0:3] = np.array([-1.2854, 0.2499, 0.0])
#    coords.position[3:6] = np.array([0.0072, -0.5687, 0.0])
    coords.position[9:12] = np.array([-1.3175, 0.8784, 0.89])
    coords.position[15:18] = np.array([0.0392, -1.1972, 0.89])
    position2 = similarity.random_rotation(position2)
    dist, aligned, permutation = similarity.test_exact_same(coords, position2)
    assert dist == pytest.approx(1.473644922799864)

def test_optimal_alignment_mol():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    position2 = position.copy()
#    coords.position[0:3] = np.array([-1.2854, 0.2499, 0.0])
#    coords.position[3:6] = np.array([0.0072, -0.5687, 0.0])
    coords.position[9:12] = np.array([-1.3175, 0.8784, 0.89])
    coords.position[15:18] = np.array([0.0392, -1.1972, 0.89])
    similarity = MolecularSimilarity(0.01, 0.05, weighted=False)
    position2 = similarity.random_rotation(position2)
    dist, coords1, aligned, permutation = similarity.optimal_alignment(coords, position2)
    assert dist < 1e-5
    assert np.all(aligned == pytest.approx(coords.position))
    assert np.all(permutation == np.array([0, 1, 2, 5, 4, 3, 6, 7, 8]))

def test_optimal_alignment_mol2():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    atom2 = ase.io.read('test_data/ethanol_rot.xyz')
    position2 = atom2.get_positions().flatten()
    similarity = MolecularSimilarity(0.01, 0.05, weighted=True)
#    coords.position[0:3] = np.array([-1.2854, 0.2499, 0.0])
#    coords.position[3:6] = np.array([0.0072, -0.5687, 0.0])
    coords.position[9:12] = np.array([-1.3175, 0.8784, 0.89])
    coords.position[15:18] = np.array([0.0392, -1.1972, 0.89])
    position2 = similarity.random_rotation(position2)
    dist, coords1, aligned, permutation = similarity.optimal_alignment(coords, position2)
    assert dist == pytest.approx(1.4388154363315437)
    assert np.all(permutation == np.array([0, 1, 2, 5, 4, 3, 6, 7, 8]))

def test_optimal_alignment_mol3():
    atoms = ase.io.read('test_data/salicylic1.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    atom2 = ase.io.read('test_data/salicylic2.xyz')
    position2 = atom2.get_positions().flatten()
    similarity = MolecularSimilarity(0.01, 0.05, weighted=True)
    position2 = similarity.random_rotation(position2)
    dist, coords1, aligned, permutation = similarity.optimal_alignment(coords, position2)
    assert dist == pytest.approx(1.7485847210878422)
    assert np.all(permutation == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                           10, 11, 12, 13, 14, 15]))

def test_optimal_alignment_mol4():
    atoms = ase.io.read('test_data/salicylic1.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    atom2 = ase.io.read('test_data/salicylic2.xyz')
    position2 = atom2.get_positions().flatten()
    similarity = MolecularSimilarity(0.01, 0.05, weighted=True)
    # 5-7, 14-15, 11-12
    coords.position[15:18] = np.array([-1.423224299360272, 1.493631603364975, 0.5400770471453962])
    coords.position[21:24] = np.array([-0.3761305247798132, 1.3327601638416506, -0.344503208155016])
    coords.position[33:36] = np.array([-1.9798193775845712, 2.4251806736519903, 0.5708316568640939])
    coords.position[36:39] = np.array([-1.2933881725400491, -1.5795415133446002, 2.016322624644797])
    coords.position[42:45] = np.array([2.8216721489040713, -0.9991029618138099, -2.0899512548049763])
    coords.position[45:48] = np.array([1.354615845998053, -2.1293969457066315, -0.14640952180641303])
    position2 = similarity.random_rotation(position2)
    dist, coords1, aligned, permutation = similarity.optimal_alignment(coords, position2)
    assert dist == pytest.approx(1.7485847210878422)
    assert np.all(permutation == np.array([0, 1, 2, 3, 4, 7, 6, 5, 8, 9,
                                           10, 12, 11, 13, 15, 14]))

def test_optimal_alignment_mol5():
    atoms = ase.io.read('test_data/paracetamol1.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    coords.rotate_dihedral([0, 1], 10.0, [11, 12, 13])
    atom2 = ase.io.read('test_data/paracetamol2.xyz')
    position2 = atom2.get_positions().flatten()
    similarity = MolecularSimilarity(0.01, 0.05, weighted=True)
    position2 = similarity.random_rotation(position2)
    dist, coords1, aligned, permutation = similarity.optimal_alignment(coords, position2)
    assert dist == pytest.approx(10.56782808628859, abs=1e-2)

def test_optimal_alignment_mol6():
    atoms = ase.io.read('test_data/paracetamol1.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    coords.rotate_dihedral([0, 1], 10.0, [11, 12, 13])
    atom2 = ase.io.read('test_data/paracetamol2.xyz')
    position2 = atom2.get_positions().flatten()
    similarity = MolecularSimilarity(0.01, 0.05, weighted=True)
    position2 = similarity.random_rotation(position2)
    # 15-19, 5-9, 11-12
    coords.position[15:18] = np.array([0.13014163299164325, -2.01395243474366, -0.2876252523210924])
    coords.position[27:30] = np.array([-1.1731039901799258, 0.32461176830414684, -1.0313700030846433])
    coords.position[33:36] = np.array([-0.07356836578529435, 1.2450524263551377, 1.9981394509216563])
    coords.position[36:39] = np.array([1.6380757520911244, 0.8284606072010503, 2.12956481843749])
    coords.position[45:48] = np.array([1.8275606179952912, -0.7542999101540282, 0.11210399009400124])
    coords.position[57:60] = np.array([-1.6808653883470404, 1.2449409619057872, -1.309108774449683])
    coords.write_xyz('yolo')
    dist, coords1, aligned, permutation = similarity.optimal_alignment(coords, position2)
    assert dist == pytest.approx(10.56782808628859, abs=1e-2)
    print("permutation = ", permutation)
    assert np.all(permutation == np.array([0, 1, 2, 3, 4, 9, 6, 7, 8, 5,
                                           10, 12, 13, 11, 14, 19, 16, 17, 18, 15]))

def test_get_furthest_from_centre():
    atoms = ase.io.read('test_data/ethanol.xyz')
    position = atoms.get_positions()
    similarity = MolecularSimilarity(0.01, 0.05, weighted=True)
    furthest = similarity.get_furthest_from_centre(position.flatten())
    assert furthest == [7]

def test_get_furthest_perpendicular():
    atoms = ase.io.read('test_data/ethanol.xyz')
    position = atoms.get_positions()
    similarity = MolecularSimilarity(0.01, 0.05, weighted=True)
    furthest = similarity.get_furthest_perpendicular(position.flatten(), 7)
    assert furthest == [5, 6]
