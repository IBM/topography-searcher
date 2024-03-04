import pytest
import numpy as np
from topsearch.potentials.test_functions import Camelback
from topsearch.potentials.atomic import LennardJones
from topsearch.data.coordinates import StandardCoordinates, AtomicCoordinates

def test_check_valid_minimum():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camelback = Camelback()
    coords.position = np.array([0.0898, -0.7126])
    valid_min = camelback.check_valid_minimum(coords)
    assert valid_min == True
    coords.position = np.array([5.0, 5.0])
    valid_min = camelback.check_valid_minimum(coords)
    assert valid_min == True
    coords.position = np.array([0.0, 0.0])
    valid_min = camelback.check_valid_minimum(coords)
    assert valid_min == False

def test_check_valid_ts():
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    camelback = Camelback()
    coords.position = np.array([0.0898, -0.7126])
    valid_ts = camelback.check_valid_ts(coords)
    assert valid_ts == False
    coords.position = np.array([5.0, 1.0])
    valid_ts = camelback.check_valid_ts(coords)
    assert valid_ts == True
    coords.position = np.array([0.0, 0.0])
    valid_ts = camelback.check_valid_ts(coords)
    assert valid_ts == True

def test_check_valid_min_atomistic():
    position = np.array([-0.34730965898408783, -1.116779267503695, -0.09399409205649237,
                         -0.25853048505296805, -0.2833641287780657, -0.8396822072934766,
                          0.3927268992562973, -0.2979222137330398, 0.06781518125803959,
                         -0.048923655113916845, 0.6715072867111124, -0.30098510758122643,
                         -0.7114398756865784, -0.094695953958119, 0.1696865603760323,
                          0.023600228355890873, 0.4615816191371775, 0.7991148142576938,
                          0.949876547225363, 0.6596726581246294, 0.19804485103942987])
    atom_labels = ['C','C','C','C','C','C', 'C']
    coords = AtomicCoordinates(atom_labels, position)
    lennard_jones = LennardJones()
    valid_min = lennard_jones.check_valid_minimum(coords)
    assert valid_min == True
    coords.position = np.array([-0.03028673324421371, -0.6993777792942178, -0.0869428831075663,
                          0.26278568715849937, -0.08252196441013813, -0.9868547902050976,
                          0.8698739912206924, -0.0499490487429821, -0.06199890340665235,
                         -0.4079003047908625, 0.3411477490188449, -0.21836120468253423,
                         -0.910045635915035, -0.34449892192451365, 0.5055093492092153,
                          0.10424270883745425, 0.06810078143584047, 0.7333260570774981,
                          0.4446812400887467, 0.9551650938893821, 0.12220599655290978])
    valid_min = lennard_jones.check_valid_minimum(coords)
    assert valid_min == False

def test_check_valid_ts_atomistic():
    position = np.array([-0.34730965898408783, -1.116779267503695, -0.09399409205649237,
                         -0.25853048505296805, -0.2833641287780657, -0.8396822072934766,
                          0.3927268992562973, -0.2979222137330398, 0.06781518125803959,
                         -0.048923655113916845, 0.6715072867111124, -0.30098510758122643,
                         -0.7114398756865784, -0.094695953958119, 0.1696865603760323,
                          0.023600228355890873, 0.4615816191371775, 0.7991148142576938,
                          0.949876547225363, 0.6596726581246294, 0.19804485103942987])
    atom_labels = ['C','C','C','C','C','C', 'C']
    coords = AtomicCoordinates(atom_labels, position)
    lennard_jones = LennardJones()
    valid_ts = lennard_jones.check_valid_ts(coords)
    assert valid_ts == False
    coords.position = np.array([-0.03028673324421371, -0.6993777792942178, -0.0869428831075663,
                          0.26278568715849937, -0.08252196441013813, -0.9868547902050976,
                          0.8698739912206924, -0.0499490487429821, -0.06199890340665235,
                         -0.4079003047908625, 0.3411477490188449, -0.21836120468253423,
                         -0.910045635915035, -0.34449892192451365, 0.5055093492092153,
                          0.10424270883745425, 0.06810078143584047, 0.7333260570774981,
                          0.4446812400887467, 0.9551650938893821, 0.12220599655290978])
    valid_ts = lennard_jones.check_valid_ts(coords)
    assert valid_ts == True
