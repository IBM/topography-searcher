import pytest
import numpy as np
import ase.io
from topsearch.potentials.force_fields import MMFF94
from topsearch.minimisation import lbfgs
from topsearch.data.coordinates import MolecularCoordinates

def test_function():
    atoms = ase.io.read('test_data/ethanol.xyz')
    position = atoms.get_positions().flatten()
    ff = MMFF94('test_data/ethanol.xyz')
    energy = ff.function(position)
    assert energy == pytest.approx(-0.3243009121732608)

def test_gradient():
    atoms = ase.io.read('test_data/ethanol.xyz')
    position = atoms.get_positions().flatten()
    ff = MMFF94('test_data/ethanol.xyz')
    grad = ff.gradient(position)
    assert np.all(grad == pytest.approx(np.array([-4.52011926, -5.47800931, 0.0,
                                                  -13.14840751, 10.28990225, 0.0,
                                                   13.02685832, -12.90189822, 0.0,
                                                   -2.99873219, 1.49786098, -2.21369722,
                                                   -2.99873219, 1.49786098, 2.21369722,
                                                    1.83651213, -4.19251896, -0.26984793,
                                                    1.83651213, -4.19251896, 0.26984793,
                                                    3.40363344, -0.43160148, 0.0,
                                                    3.56247514, 13.9109227, 0.0])))

def test_function_gradient():
    atoms = ase.io.read('test_data/ethanol.xyz')
    position = atoms.get_positions().flatten()
    ff = MMFF94('test_data/ethanol.xyz')
    energy, grad = ff.function_gradient(position)
    assert energy == pytest.approx(-0.3243009121732608)
    assert np.all(grad == pytest.approx(np.array([-4.52011926, -5.47800931, 0.0,
                                                  -13.14840751, 10.28990225, 0.0,
                                                   13.02685832, -12.90189822, 0.0,
                                                   -2.99873219, 1.49786098, -2.21369722,
                                                   -2.99873219, 1.49786098, 2.21369722,
                                                    1.83651213, -4.19251896, -0.26984793,
                                                    1.83651213, -4.19251896, 0.26984793,
                                                    3.40363344, -0.43160148, 0.0,
                                                    3.56247514, 13.9109227, 0.0])))

def test_min():
    atoms = ase.io.read('test_data/ethanol.xyz')
    position = atoms.get_positions().flatten()
    ff = MMFF94('test_data/ethanol.xyz')
    min_position, energy, results_dict = \
        lbfgs.minimise(func_grad=ff.function_gradient,
                       initial_position=position,
                       bounds=None,
                       conv_crit=1e-3)
    atoms = ase.io.read('test_data/ethanol2.xyz')
    position = atoms.get_positions().flatten()
    assert energy == pytest.approx(-1.5170975759813006)
    assert np.all(position == pytest.approx(min_position, abs=1e-5))

def test_function_gradient2():
    atoms = ase.io.read('test_data/ethanol2.xyz')
    position = atoms.get_positions().flatten()
    ff = MMFF94('test_data/ethanol.xyz')
    energy, grad = ff.function_gradient(position)
    assert energy == pytest.approx(-1.5170975759813006)
    assert np.all(grad < 1e-3)
