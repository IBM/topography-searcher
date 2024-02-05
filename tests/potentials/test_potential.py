import pytest
import numpy as np
from topsearch.potentials.test_functions import Schwefel, Camelback 
from topsearch.data.coordinates import StandardCoordinates

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
