import pytest
import numpy as np
import ase
from topsearch.potentials.atomic import LennardJones, BinaryGupta

def test_lj_energy():
    position = np.array([-0.2604720088,  0.7363147287,  0.4727061929,
                          0.2604716550, -0.7363150782, -0.4727063011,
                         -0.4144908003, -0.3652598516,  0.3405559620,
                         -0.1944131041,  0.2843471802, -0.5500413671,
                          0.6089042582,  0.0809130209,  0.2094855133])
    lennard_jones = LennardJones()
    energy = lennard_jones.function(position)
    assert energy == pytest.approx(-9.103852415681363)

def test_lj_gradient():
    position = np.array([-0.2604720088,  0.7363147287,  0.4727061929,
                          0.2604716550, -0.7363150782, -0.4727063011,
                         -0.4144908003, -0.3652598516,  0.3405559620,
                         -0.1944131041,  0.2843471802, -0.5500413671,
                          0.6089042582,  0.0809130209,  0.2094855133])
    lennard_jones = LennardJones()
    grad = lennard_jones.gradient(position)
    assert np.all(np.abs(grad) < 1e-4)

def test_lj_energy_gradient():
    position = np.array([-0.2604720088,  0.7363147287,  0.4727061929,
                          0.2604716550, -0.7363150782, -0.4727063011,
                         -0.4144908003, -0.3652598516,  0.3405559620,
                         -0.1944131041,  0.2843471802, -0.5500413671,
                          0.6089042582,  0.0809130209,  0.2094855133])
    lennard_jones = LennardJones()
    energy, grad = lennard_jones.function_gradient(position)
    assert energy == pytest.approx(-9.103852415681363)
    assert np.all(np.abs(grad) < 1e-4)

def test_gupta_energy():
    position = np.array([-0.2604720088,  0.7363147287,  0.4727061929,
                          0.2604716550, -0.7363150782, -0.4727063011,
                         -0.4144908003, -0.3652598516,  0.3405559620,
                         -0.1944131041,  0.2843471802, -0.5500413671])
    species = ['Au', 'Ag', 'Au', 'Ag']
    gupta = BinaryGupta(species=species)
    energy = gupta.function(position)
    assert energy == pytest.approx(850.0455834675192)
