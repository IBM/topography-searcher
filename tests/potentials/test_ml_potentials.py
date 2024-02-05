import pytest
import numpy as np
import ase.io
from topsearch.potentials.ml_potentials import MachineLearningPotential

def test_ani_function():
    atoms = ase.io.read('ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    mlp = MachineLearningPotential(atom_labels=species,
                                   calculator_type='torchani')
    energy = mlp.function(position)
    assert energy == pytest.approx(-154.98837187806524)

def test_ani_gradient():
    atoms = ase.io.read('ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    mlp = MachineLearningPotential(atom_labels=species,
                                   calculator_type='torchani')
    grad = mlp.gradient(position)
    assert np.all(np.abs(grad) < 5e-2)

def test_ani_function_gradient():
    atoms = ase.io.read('ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    mlp = MachineLearningPotential(atom_labels=species,
                                   calculator_type='torchani')
    energy, grad = mlp.function_gradient(position)
    assert energy == pytest.approx(-154.98837187806524)
    assert np.all(np.abs(grad) < 5e-2)
