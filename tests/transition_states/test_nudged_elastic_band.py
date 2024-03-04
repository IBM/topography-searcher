import pytest
import numpy as np
import matplotlib.pyplot as plt
import ase.io
from topsearch.transition_states.nudged_elastic_band import NudgedElasticBand
from topsearch.potentials.test_functions import Camelback
from topsearch.data.coordinates import StandardCoordinates, MolecularCoordinates
from topsearch.similarity.molecular_similarity import MolecularSimilarity

def test_perpendicular_component():
    camel = Camelback()
    double_ended = NudgedElasticBand(camel, 10.0, 50, 10, 1e-2)
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    p_comp = double_ended.perpendicular_component(vec1, vec2)
    assert np.all(p_comp == pytest.approx(np.array([1.0, 0.0, 0.0])))
    vec1 = np.array([0.4, 0.3, 0.1])
    vec2 = np.array([0.8, 0.6, 0.2])
    p_comp = double_ended.perpendicular_component(vec1, vec2)
    assert np.all(p_comp == pytest.approx(np.array([0.0, 0.0, 0.0])))
    vec1 = np.array([0.4, 0.3])
    vec2 = np.array([0.8, 0.1])
    p_comp = double_ended.perpendicular_component(vec1, vec2)
    assert np.all(p_comp == pytest.approx(np.array([-0.0307692307692308,
                                                    0.2461538462])))

def test_linear_interpolation():
    # Initialise class instances
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 50.0, 4.0, 50, 1e-2)
    coords.position = np.array([-0.0898, 0.7126])
    coords2 = np.array([0.0898, -0.7126])
    interpolation = double_ended_search.linear_interpolation(coords, coords2)
    assert np.all(interpolation == pytest.approx(np.array([[-0.0898,  0.7126],
                                                    [-0.06984444,  0.55424444],
                                                    [-0.04988889,  0.39588889],
                                                    [-0.02993333,  0.23753333],
                                                    [-0.00997778,  0.07917778],
                                                    [ 0.00997778, -0.07917778],
                                                    [ 0.02993333, -0.23753333],
                                                    [ 0.04988889, -0.39588889],
                                                    [ 0.06984444, -0.55424444],
                                                    [ 0.0898, -0.7126]])))

def test_linear_interpolation2():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 50.0, 1000.0, 15, 1e-2)
    coords.position = np.array([-0.0898, 0.7126])
    coords2 = np.array([0.0898, -0.7126])
    interpolation = double_ended_search.linear_interpolation(coords, coords2)
    assert np.all(interpolation == pytest.approx(np.array([[-0.0898, 0.7126],
                                                    [-0.07697143, 0.6108],
                                                    [-0.06414286, 0.509],
                                                    [-0.05131429, 0.4072],
                                                    [-0.03848571,  0.3054],
                                                    [-0.02565714,  0.2036],
                                                    [-0.01282857,  0.1018],
                                                    [0.0, 0.0],
                                                    [0.01282857, -0.1018],
                                                    [0.02565714, -0.2036],
                                                    [0.03848571, -0.3054],
                                                    [0.05131429, -0.4072],
                                                    [0.06414286, -0.509],
                                                    [0.07697143, -0.6108],
                                                    [0.0898, -0.7126]])))

def test_band_potential_function():
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 50.0, 1000.0, 15, 1e-2)
    band = np.array([[-0.0898,  0.7126], [-0.06984444,  0.55424444],
                    [-0.04988889,  0.39588889], [-0.02993333,  0.23753333],
                    [-0.00997778,  0.07917778], [ 0.00997778, -0.07917778],
                    [ 0.02993333, -0.23753333], [ 0.04988889, -0.39588889],
                    [ 0.06984444, -0.55424444], [ 0.0898, -0.7126]])
    total_energy, energies = double_ended_search.band_potential_function(band)
    assert np.all(energies == pytest.approx(np.array([-1.03162842,
                                               -0.87054029,
                                               -0.53846523,
                                               -0.21648236,
                                               -0.02531109,
                                               -0.02531109,
                                               -0.21648236,
                                               -0.53846523,
                                               -0.87054029,
                                               -1.03162842])))
    assert total_energy == pytest.approx(np.sum(energies))
    
def test_get_force_constants():
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 50.0, 1000.0, 15, 1e-2)
    double_ended_search.n_images = 6
    double_ended_search.get_force_constants()
    assert np.all(double_ended_search.force_constants ==
                  np.array([50.0, 50.0, 50.0, 50.0, 50.0]))

def test_find_ts_candidates():
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 50.0, 1000.0, 15, 1e-2)
    double_ended_search.n_images = 10
    band = np.array([[-0.0898,  0.7126], [-0.06984444,  0.55424444],
                    [-0.04988889,  0.39588889], [-0.02993333,  0.23753333],
                    [-0.00997778,  0.07917778], [ 0.00997778, -0.07917778],
                    [ 0.02993333, -0.23753333], [ 0.04988889, -0.39588889],
                    [ 0.06984444, -0.55424444], [ 0.0898, -0.7126]])
    candidates, positions = double_ended_search.find_ts_candidates(band)
    assert np.all(candidates == np.array([4, 5]))
    assert np.all(positions[0, :] == np.array([-0.00997778,  0.07917778]))
    assert np.all(positions[1, :] == np.array([0.00997778, -0.07917778]))

def test_update_image_density():
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 50.0, 1000.0, 15, 1e-2)
    double_ended_search.update_image_density(1)
    assert double_ended_search.image_density == 1500.0

def test_revert_image_density():
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 50.0, 1000.0, 15, 1e-2)
    double_ended_search.image_density = 300.0
    double_ended_search.revert_image_density()
    assert double_ended_search.image_density == 1000.0

def test_find_tangent_differences():
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 50.0, 1000.0, 15, 1e-2)
    band = np.array([[-0.9,  0.7], [-0.8,  0.55],
                     [-0.4,  0.4], [-0.1,  0.2],
                     [1.0,  1.0]])
    energies = np.array([-1.0, 0.5, 0.5, 0.5, -0.45])
    double_ended_search.n_images = 5
    tangents = double_ended_search.find_tangent_differences(band, energies)
    assert np.all(tangents == pytest.approx(np.array([[-0.93632918,  0.35112344],
                                                      [-0.93632918,  0.35112344],
                                                      [-0.83205029,  0.5547002 ]])))

def test_find_tangent_differences2():
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 50.0, 1000.0, 15, 1e-2)
    band = np.array([[-0.9,  0.7], [-0.8,  0.55],
                     [-0.4,  0.4], [-0.1,  0.2],
                     [1.0,  1.0]])
    energies = np.array([-1.0, 0.5, -0.5, -0.2, -0.45])
    double_ended_search.n_images = 5
    tangents = double_ended_search.find_tangent_differences(band, energies)
    assert np.all(tangents == pytest.approx(np.array([[-0.88147997,  0.47222141],
                                                      [-0.91914503,  0.3939193 ],
                                                      [-0.90532466, -0.42472021]])))

def test_find_tangent_differences3():
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 50.0, 1000.0, 15, 1e-2)
    band = np.array([[-0.9,  0.7], [-0.8,  0.55],
                     [-0.4,  0.4], [-0.1,  0.2],
                     [1.0,  1.0]])
    energies = np.array([-1.0, -0.4, -0.3, -0.4, -0.45])
    double_ended_search.n_images = 5
    tangents = double_ended_search.find_tangent_differences(band, energies)
    assert np.all(tangents == pytest.approx(np.array([[-0.93632918,  0.35112344],
                                                      [-0.89442719,  0.4472136 ],
                                                      [-0.83205029,  0.5547002 ]])))

def test_band_function_gradient():
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 10.0, 1000.0, 15, 1e-2)
    band = np.array([[-0.0898,  0.7126], [-0.06984444,  0.55424444],
                    [-0.04988889,  0.39588889], [-0.02993333,  0.23753333],
                    [-0.00997778,  0.07917778], [ 0.00997778, -0.07917778],
                    [ 0.02993333, -0.23753333], [ 0.04988889, -0.39588889],
                    [ 0.06984444, -0.55424444], [ 0.0898, -0.7126]])
    double_ended_search.n_images = 10
    double_ended_search.get_force_constants()
    f_val, grad = double_ended_search.band_function_gradient(band)
    assert f_val == pytest.approx(-4.218492976699833)
    assert np.all(grad == pytest.approx(np.array([0.0, 0.0,
                                                  -0.22239304,
                                                  -0.02802527,
                                                  -0.27805839,
                                                  -0.03504032,
                                                  -0.21451777,
                                                  -0.02703285,
                                                  -0.07945314,
                                                  -0.01001259,
                                                  0.07945314,
                                                  0.01001259,
                                                  0.21451777,
                                                  0.02703285,
                                                  0.27805839,
                                                  0.03504032,
                                                  0.22239304,
                                                  0.02802527,
                                                  0.0, 0.0])))

def test_band_function_gradient2():
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 10.0, 1000.0, 15, 1e-2)
    band = np.array([[-0.9,  0.7], [-0.8,  0.55],
                     [-0.4,  0.4], [-0.1,  0.2],
                     [1.0,  1.0]])
    double_ended_search.n_images = 5
    double_ended_search.get_force_constants()
    f_val, grad = double_ended_search.band_function_gradient(band)
    assert f_val == pytest.approx(14.877512333333335)
    assert np.all(grad == pytest.approx(np.array([0.0, 0.0,
                                                  -0.76359233,
                                                  -4.08529069,
                                                  -1.75237221,
                                                  -2.77494042,
                                                   8.63760731,
                                                   5.11539738,
                                                   0.0, 0.0])))

def test_minimise_interpolation():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 10.0, 1000.0, 15, 1e-4)
    coords.position = np.array([-1.7036, 0.79608])
    coords2 = np.array([1.7036, -0.79608])
    interpolation = double_ended_search.linear_interpolation(coords, coords2)
    double_ended_search.get_force_constants()
    min_band = double_ended_search.minimise_interpolation(interpolation)
    # Check we have correct curvature
    # First few image should have gradient (1, 0)
    vec1 = min_band[3, :] - min_band[0, :]
    vec1 /= np.linalg.norm(vec1)
    angle = np.arccos(np.dot(vec1, np.array([1.0, 0.0])))*(180.0/np.pi)
    assert angle < 1.0
    # Middle images around (0, 1)
    vec2 = min_band[5, :] - min_band[7, :]
    vec2 /= np.linalg.norm(vec2)
    angle = np.arccos(np.dot(vec2, np.array([0.0, 1.0])))*(180.0/np.pi)
    assert angle < 25.0
    # And final few images (1, 0) again
    vec3 = min_band[-1, :] - min_band[-4, :]
    vec3 /= np.linalg.norm(vec3)
    angle = np.arccos(np.dot(vec3, np.array([1.0, 0.0])))*(180.0/np.pi)
    assert angle < 1.0

def test_initial_interpolation():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 50.0, 4.0, 50, 1e-2)
    coords.position = np.array([-0.0898, 0.7126])
    coords2 = np.array([0.0898, -0.7126])
    interpolation = double_ended_search.initial_interpolation(coords, coords2, 0, None)
    assert np.all(interpolation == pytest.approx(np.array([[-0.0898,  0.7126],
                                                    [-0.06984444,  0.55424444],
                                                    [-0.04988889,  0.39588889],
                                                    [-0.02993333,  0.23753333],
                                                    [-0.00997778,  0.07917778],
                                                    [ 0.00997778, -0.07917778],
                                                    [ 0.02993333, -0.23753333],
                                                    [ 0.04988889, -0.39588889],
                                                    [ 0.06984444, -0.55424444],
                                                    [ 0.0898, -0.7126]])))
    assert np.all(double_ended_search.force_constants == np.full((9), 50.0))
    assert double_ended_search.n_images == 10
    assert double_ended_search.image_density == 4.0
    assert double_ended_search.band_bounds == [(-3.0, 3.0), (-2.0, 2.0)]*10

def test_initial_interpolation2():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 50.0, 4.0, 20, 1e-2)
    coords.position = np.array([-0.0898, 0.7126])
    coords2 = np.array([0.0898, -0.7126])
    interpolation = double_ended_search.initial_interpolation(coords, coords2, 2, None)
    assert np.all(double_ended_search.force_constants == np.full((16), 50.0))
    assert double_ended_search.n_images == 17
    assert double_ended_search.image_density == 4.0
    assert np.all(interpolation == pytest.approx(np.array([[-0.0898, 0.7126],
                                                           [-0.078575, 0.623525],
                                                           [-0.06735, 0.53445 ],
                                                           [-0.056125, 0.445375],
                                                           [-0.0449, 0.3563],
                                                           [-0.033675, 0.267225],
                                                           [-0.02245, 0.17815 ],
                                                           [-0.011225, 0.089075],
                                                           [ 0.0, 0.0],
                                                           [ 0.011225, -0.089075],
                                                           [ 0.02245, -0.17815 ],
                                                           [ 0.033675, -0.267225],
                                                           [ 0.0449, -0.3563],
                                                           [ 0.056125, -0.445375],
                                                           [ 0.06735, -0.53445 ],
                                                           [ 0.078575, -0.623525],
                                                           [ 0.0898,  -0.7126  ]])))

def test_double_ended_ts_search():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camel = Camelback()
    double_ended_search = NudgedElasticBand(camel, 10.0, 1000.0, 15, 1e-4)
    coords.position = np.array([-1.7036, 0.79608])
    coords2 = np.array([1.7036, -0.79608])
    candidates, positions = double_ended_search.run(coords, coords2, 0)
    ts1 = np.array([-1.109205336676489795, 0.7682680961466419323])
    ts2 = np.array([0.0, 0.0])
    ts3 = np.array([1.109205336676489795, -0.7682680961466419323])
    assert np.linalg.norm(positions[0, :] - ts1) < 0.1
    assert np.linalg.norm(positions[1, :] - ts2) < 0.05
    assert np.linalg.norm(positions[2, :] - ts3) < 0.1

def test_dihedral_interpolation():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    atoms2 = ase.io.read('test_data/ethanol_rot.xyz')
    position2 = atoms2.get_positions().flatten()
    coords1 = MolecularCoordinates(species, position)
    double_ended_search = NudgedElasticBand(None, 10.0, 1000.0, 15, 1e-4)
    permutation = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    band = double_ended_search.dihedral_interpolation(coords1, position2, permutation)
    assert np.all(band[0, :] == pytest.approx(np.array([7.20000000e-03, -5.68700000e-01, 0.00000000e+00, -1.28540000e+00,
                                                        2.49900000e-01, 0.00000000e+00, 1.13040000e+00, 3.14700000e-01,
                                                        0.00000000e+00, 3.92000000e-02, -1.19720000e+00, 8.90000000e-01,
                                                        3.92000000e-02, -1.19720000e+00, -8.90000000e-01, -1.31750000e+00,
                                                        8.78400000e-01, 8.90000000e-01, -1.31750000e+00, 8.78400000e-01,
                                                        -8.90000000e-01, -2.14220000e+00, -4.23900000e-01, 0.00000000e+00,
                                                        1.98570000e+00, -1.36500000e-01, 0.00000000e+00])))
    assert np.all(band[2, :] == pytest.approx(np.array([ 7.20000000e-03, -5.68700000e-01, 0.00000000e+00, -1.28540000e+00,
                                                          2.49900000e-01, 0.00000000e+00, 1.13040000e+00, 3.14700000e-01,
                                                         -3.30940311e-17, 3.92000000e-02, -1.19720000e+00, 8.90000000e-01,
                                                          3.92000000e-02, -1.19720000e+00, -8.90000000e-01, -1.31750000e+00,
                                                          8.78400000e-01, 8.90000000e-01, -1.31750000e+00, 8.78400000e-01,
                                                         -8.90000000e-01, -2.14220000e+00, -4.23900000e-01, 0.00000000e+00,
                                                          1.96143729e+00, -1.05651149e-01, -2.60387140e-01])))
    assert np.all(band[5, :] == pytest.approx(np.array([7.20000000e-03, -5.68700000e-01, 0.00000000e+00, -1.28540000e+00,
                                                        2.49900000e-01, 0.00000000e+00, 1.13040000e+00, 3.14700000e-01,
                                                        -6.61880622e-17, 3.92000000e-02, -1.19720000e+00, 8.90000000e-01,
                                                        3.92000000e-02, -1.19720000e+00, -8.90000000e-01, -1.31750000e+00,
                                                        8.78400000e-01, 8.90000000e-01, -1.31750000e+00, 8.78400000e-01,
                                                        -8.90000000e-01, -2.14220000e+00, -4.23900000e-01, 0.00000000e+00,
                                                        1.83991351e+00, 4.88604119e-02, -6.00865564e-01])))
    assert np.all(band[10, :] == pytest.approx(np.array([7.20000000e-03, -5.68700000e-01, 0.00000000e+00, -1.28540000e+00,
                                                         2.49900000e-01, 0.00000000e+00, 1.13040000e+00, 3.14700000e-01,
                                                         -6.61880622e-17, 3.92000000e-02, -1.19720000e+00, 8.90000000e-01,
                                                         3.92000000e-02, -1.19720000e+00, -8.90000000e-01, -1.31750000e+00,
                                                         8.78400000e-01, 8.90000000e-01, -1.31750000e+00, 8.78400000e-01,
                                                         -8.90000000e-01, -2.14220000e+00, -4.23900000e-01, 1.23259516e-32,
                                                         1.48038889e+00, 5.05978418e-01, -8.80931252e-01])))
    assert np.all(band[14, :] == pytest.approx(np.array([7.20000000e-03, -5.68700000e-01, 0.00000000e+00, -1.28540000e+00,
                                                        2.49900000e-01, 0.00000000e+00, 1.13040000e+00, 3.14700000e-01,
                                                        -6.61880622e-17, 3.92000000e-02, -1.19720000e+00, 8.90000000e-01,
                                                        3.92000000e-02, -1.19720000e+00, -8.90000000e-01, -1.31750000e+00,
                                                        8.78400000e-01, 8.90000000e-01, -1.31750000e+00, 8.78400000e-01,
                                                        -8.90000000e-01, -2.14220000e+00, -4.23900000e-01, 2.46519033e-32,
                                                        1.16651562e+00, 9.05052978e-01, -7.65048073e-01])))

def test_dihedral_interpolation2():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    atoms2 = ase.io.read('test_data/ethanol_rot.xyz')
    position2 = atoms2.get_positions().flatten()
    coords1 = MolecularCoordinates(species, position)
    coords1.change_bond_length([0, 2], 0.3, [2, 8])
    coords1.rotate_angle([0, 2, 8], 45.0, [8])
    double_ended_search = NudgedElasticBand(None, 10.0, 1000.0, 15, 1e-4)
    permutation = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    band = double_ended_search.dihedral_interpolation(coords1, position2, permutation)
    assert np.all(band[0, :] == pytest.approx(np.array([7.20000000e-03, -5.68700000e-01, 0.00000000e+00, -1.28540000e+00,
                                                        2.49900000e-01, 0.00000000e+00, 1.46736000e+00, 5.79720000e-01,
                                                        0.00000000e+00, 3.92000000e-02, -1.19720000e+00, 8.90000000e-01,
                                                        3.92000000e-02, -1.19720000e+00, -8.90000000e-01, -1.31750000e+00,
                                                        8.78400000e-01, 8.90000000e-01, -1.31750000e+00, 8.78400000e-01,
                                                        -8.90000000e-01, -2.14220000e+00, -4.23900000e-01, 0.00000000e+00,
                                                        1.75310185e+00, -3.44115010e-01, 0.00000000e+00])))
    assert np.all(band[2, :] == pytest.approx(np.array([7.20000000e-03, -5.68700000e-01, 0.00000000e+00, -1.28540000e+00,
                                                        2.49900000e-01, -3.08148791e-33, 1.37930621e+00, 5.10465441e-01,
                                                        -1.60403306e-17, 3.92000000e-02, -1.19720000e+00, 8.90000000e-01,
                                                        3.92000000e-02, -1.19720000e+00, -8.90000000e-01, -1.31750000e+00,
                                                        8.78400000e-01, 8.90000000e-01, -1.31750000e+00, 8.78400000e-01,
                                                        -8.90000000e-01, -2.14220000e+00, -4.23900000e-01, -3.08148791e-33,
                                                        1.74098298e+00, -3.42884974e-01, -2.75866840e-01])))
    assert np.all(band[5, :] == pytest.approx(np.array([7.20000000e-03, -5.68700000e-01, 0.00000000e+00, -1.28540000e+00,
                                                        2.49900000e-01, -1.23259516e-32, 1.25708096e+00, 4.14334935e-01,
                                                        -4.67076575e-17, 3.92000000e-02, -1.19720000e+00, 8.90000000e-01,
                                                        3.92000000e-02, -1.19720000e+00, -8.90000000e-01, -1.31750000e+00,
                                                        8.78400000e-01, 8.90000000e-01, -1.31750000e+00, 8.78400000e-01,
                                                        -8.90000000e-01, -2.14220000e+00, -4.23900000e-01, -6.16297582e-33,
                                                        1.62840999e+00, -1.92135492e-01, -6.55307486e-01])))
    assert np.all(band[10, :] == pytest.approx(np.array([7.20000000e-03, -5.68700000e-01, 0.00000000e+00, -1.28540000e+00,
                                                        2.49900000e-01, -3.23556231e-32, 1.07708440e+00, 2.72767131e-01,
                                                        3.93443160e-17, 3.92000000e-02, -1.19720000e+00, 8.90000000e-01,
                                                        3.92000000e-02, -1.19720000e+00, -8.90000000e-01, -1.31750000e+00,
                                                        8.78400000e-01, 8.90000000e-01, -1.31750000e+00, 8.78400000e-01,
                                                        -8.90000000e-01, -2.14220000e+00, -4.23900000e-01, -1.69481835e-32,
                                                        1.26783044e+00, 3.32584708e-01, -9.46127548e-01])))
    assert np.all(band[14, :] == pytest.approx(np.array([7.20000000e-03, -5.68700000e-01, 0.00000000e+00, -1.28540000e+00,
                                                        2.49900000e-01, -2.46519033e-32, 9.51938078e-01, 1.74339190e-01,
                                                        9.69538233e-17, 3.92000000e-02, -1.19720000e+00, 8.90000000e-01,
                                                        3.92000000e-02, -1.19720000e+00, -8.90000000e-01, -1.31750000e+00,
                                                        8.78400000e-01, 8.90000000e-01, -1.31750000e+00, 8.78400000e-01,
                                                        -8.90000000e-01, -2.14220000e+00, -4.23900000e-01, -7.39557099e-32,
                                                        9.88053694e-01, 7.64692168e-01, -7.65048073e-01])))

def test_dihedral_interpolation3():
    comparer = MolecularSimilarity(0.01, 0.05)
    atoms = ase.io.read('test_data/ethanol_scramble.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    atoms2 = ase.io.read('test_data/ethanol_rot.xyz')
    position2 = atoms2.get_positions().flatten()
    coords1 = MolecularCoordinates(species, position)
    d, coords1a, coords2, permutation = comparer.optimal_alignment(coords1, position2)
    permutation = np.array([0, 1, 2, 5, 4, 3, 8, 7, 6])
    double_ended_search = NudgedElasticBand(None, 10.0, 1000.0, 15, 1e-4)
    band = double_ended_search.dihedral_interpolation(coords1, position2, permutation)
    assert np.all(band[0, :] == pytest.approx(np.array([3.13481814e-01, -4.41511985e-01, -2.14301869e-06, -9.79118186e-01,
                                                        3.77088015e-01, -2.14301869e-06, 1.77364181e+00, 7.06908015e-01,
                                                        -2.14301869e-06, -8.64934576e-01, 1.23657530e+00, 6.60593320e-01,
                                                        3.45481814e-01, -1.07001198e+00, -8.90002143e-01, 3.45481814e-01,
                                                        -1.07001198e+00, 8.89997857e-01, 2.05938366e+00, -2.16926994e-01,
                                                        -2.14301869e-06, -1.80275935e+00, -2.44352942e-01, 3.51479083e-01,
                                                        -1.19065881e+00, 7.22244558e-01, -1.01205954e+00])))
    assert np.all(band[2, :] == pytest.approx(np.array([3.13481814e-01, -4.41511985e-01, -2.14301869e-06, -9.79118186e-01,
                                                        3.77088015e-01, -2.14301869e-06, 1.68558802e+00, 6.37653456e-01,
                                                        -2.14301869e-06, -8.83075601e-01, 1.20792995e+00, 6.99011567e-01,
                                                        3.45481814e-01, -1.07001198e+00, -8.90002143e-01, 3.45481814e-01,
                                                        -1.07001198e+00, 8.89997857e-01, 2.04726479e+00, -2.15696959e-01,
                                                        -2.75868983e-01, -1.81149076e+00, -2.58140165e-01, 3.02906653e-01,
                                                        -1.16378686e+00, 7.64676380e-01, -1.00190803e+00])))
    assert np.all(band[5, :] == pytest.approx(np.array([3.13481814e-01, -4.41511985e-01, -2.14301869e-06, -9.79118186e-01,
                                                        3.77088015e-01, -2.14301869e-06, 1.56336277e+00, 5.41522950e-01,
                                                        -2.14301869e-06, -9.12151161e-01, 1.16201855e+00, 7.53351780e-01,
                                                        3.45481814e-01, -1.07001198e+00, -8.90002143e-01, 3.45481814e-01,
                                                        -1.07001198e+00, 8.89997857e-01, 1.93469181e+00, -6.49474765e-02,
                                                        -6.55309629e-01, -1.82213279e+00, -2.74944325e-01, 2.28674244e-01,
                                                        -1.12406985e+00, 8.27391015e-01, -9.82019904e-01])))
    assert np.all(band[10, :] == pytest.approx(np.array([3.13481814e-01, -4.41511985e-01, -2.14301869e-06, -9.79118186e-01,
                                                         3.77088015e-01, -2.14301869e-06, 1.38336622e+00, 3.99955147e-01,
                                                         -2.14301869e-06, -9.65172310e-01, 1.07829617e+00, 8.34416991e-01,
                                                         3.45481814e-01, -1.07001198e+00, -8.90002143e-01, 3.45481814e-01,
                                                         -1.07001198e+00, 8.89997857e-01, 1.57411225e+00, 4.59772723e-01,
                                                         -9.46129691e-01, -1.83318595e+00, -2.92397689e-01, 1.02319985e-01,
                                                         -1.05999614e+00, 9.28565798e-01, -9.36737789e-01])))
    assert np.all(band[14, :] == pytest.approx(np.array([3.13481814e-01, -4.41511985e-01, -2.14301869e-06, -9.79118186e-01,
                                                         3.77088015e-01, -2.14301869e-06, 1.25821989e+00, 3.01527205e-01,
                                                         -2.14301869e-06, -1.01121819e+00, 1.00558802e+00, 8.89997857e-01,
                                                         3.45481814e-01, -1.07001198e+00, -8.90002143e-01, 3.45481814e-01,
                                                         -1.07001198e+00, 8.89997857e-01, 1.29433551e+00, 8.91880183e-01,
                                                         -7.65050216e-01, -1.83591819e+00, -2.96711985e-01, -2.14301869e-06,
                                                         -1.01121819e+00, 1.00558802e+00, -8.90002143e-01])))

def test_initial_interpolation3():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    atoms2 = ase.io.read('test_data/ethanol_rot.xyz')
    position2 = atoms2.get_positions().flatten()
    coords1 = MolecularCoordinates(species, position)
    double_ended_search = NudgedElasticBand(None, 10.0, 1000.0, 15, 1e-4)
    permutation = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    band = double_ended_search.initial_interpolation(coords1, position2, 0, permutation)
    assert np.all(band[5, :] == pytest.approx(np.array([7.20000000e-03, -5.68700000e-01, 0.00000000e+00, -1.28540000e+00,
                                                        2.49900000e-01, 0.00000000e+00, 1.13040000e+00, 3.14700000e-01,
                                                        -6.61880622e-17, 3.92000000e-02, -1.19720000e+00, 8.90000000e-01,
                                                        3.92000000e-02, -1.19720000e+00, -8.90000000e-01, -1.31750000e+00,
                                                        8.78400000e-01, 8.90000000e-01, -1.31750000e+00, 8.78400000e-01,
                                                        -8.90000000e-01, -2.14220000e+00, -4.23900000e-01, 0.00000000e+00,
                                                        1.83991351e+00, 4.88604119e-02, -6.00865564e-01])))
    assert np.all(band[10, :] == pytest.approx(np.array([7.20000000e-03, -5.68700000e-01, 0.00000000e+00, -1.28540000e+00,
                                                         2.49900000e-01, 0.00000000e+00, 1.13040000e+00, 3.14700000e-01,
                                                         -6.61880622e-17, 3.92000000e-02, -1.19720000e+00, 8.90000000e-01,
                                                         3.92000000e-02, -1.19720000e+00, -8.90000000e-01, -1.31750000e+00,
                                                         8.78400000e-01, 8.90000000e-01, -1.31750000e+00, 8.78400000e-01,
                                                         -8.90000000e-01, -2.14220000e+00, -4.23900000e-01, 1.23259516e-32,
                                                         1.48038889e+00, 5.05978418e-01, -8.80931252e-01])))
