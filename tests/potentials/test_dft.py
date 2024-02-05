import pytest
import numpy as np
import ase
from topsearch.potentials.dft import DensityFunctionalTheory
from topsearch.potentials.force_fields import MMFF94

def test_dft_energy():
    atoms = ase.io.read('ethanol.xyz')
    ff = MMFF94('ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    options = {"method": 'pbe',
               "basis": 'def2-svp',
               "threads": 8}
    dft = DensityFunctionalTheory(atom_labels=species,
                                  options=options,
                                  force_field=ff)
    energy = dft.function(position)
    assert energy == pytest.approx(-4210.120823638515)

def test_dft_energy2():
    atoms = ase.io.read('ethanol.xyz')
    ff = MMFF94('ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    options = {"method": 'b3lyp',
               "basis": '6-311g_d_p_',
               "threads": 8}
    dft = DensityFunctionalTheory(atom_labels=species,
                                  options=options,
                                  force_field=ff)
    energy = dft.function(position)
    assert energy == pytest.approx(-4220.118524903629)

def test_dft_energy3():
    atoms = ase.io.read('ethanol.xyz')
    ff = MMFF94('ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    options = {"method": 'b3lyp',
               "basis": '6-311g_d_p_',
               "threads": 8}
    dft = DensityFunctionalTheory(atom_labels=species,
                                  options=options,
                                  force_field=ff)
    position[0:3] = np.array([-1.2854, 0.2499, 0.0])
    energy = dft.function(position)
    assert energy == pytest.approx(1000.0)

def test_reset_options():
    atoms = ase.io.read('ethanol.xyz')
    ff = MMFF94('ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    options = {"method": 'pbe',
               "basis": 'def2-svp',
               "threads": 8}
    dft = DensityFunctionalTheory(atom_labels=species,
                                  options=options,
                                  force_field=ff)
    energy = dft.function(position)
    assert energy == pytest.approx(-4210.120823638515)
    options2 = {"method": 'b3lyp',
                "basis": '6-311g_d_p_',
                "threads": 8}
    dft.reset_options(options2)
    energy = dft.function(position)
    assert energy == pytest.approx(-4220.118524903629)

def test_dft_gradient():
    atoms = ase.io.read('ethanol.xyz')
    ff = MMFF94('ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    options = {"method": 'b3lyp',
               "basis": '6-311g_d_p_',
               "threads": 8}
    dft = DensityFunctionalTheory(atom_labels=species,
                                  options=options,
                                  force_field=ff)
    grad = dft.gradient(position)
    assert np.all(np.abs(grad < 0.7))

def test_dft_gradient2():
    atoms = ase.io.read('ethanol.xyz')
    ff = MMFF94('ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    options = {"method": 'b3lyp',
               "basis": '6-311g_d_p_',
               "threads": 8}
    dft = DensityFunctionalTheory(atom_labels=species,
                                  options=options,
                                  force_field=ff)
    position[0:3] = np.array([-1.2854, 0.2499, 0.0])
    grad = dft.gradient(position)
    assert np.all(grad > 1e20)

def test_dft_energy_gradient():
    atoms = ase.io.read('ethanol.xyz')
    ff = MMFF94('ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    options = {"method": 'b3lyp',
               "basis": '6-311g_d_p_',
               "threads": 8}
    dft = DensityFunctionalTheory(atom_labels=species,
                                  options=options,
                                  force_field=ff)
    energy, grad = dft.function_gradient(position)
    assert energy == pytest.approx(-4220.118524903629)
    assert np.all(np.abs(grad) < 0.7)

def test_dft_energy_gradient2():
    atoms = ase.io.read('ethanol.xyz')
    ff = MMFF94('ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    options = {"method": 'b3lyp',
               "basis": '6-311g_d_p_',
               "threads": 8}
    dft = DensityFunctionalTheory(atom_labels=species,
                                  options=options,
                                  force_field=ff)
    position[0:3] = np.array([-1.2854, 0.2499, 0.0])
    energy, grad = dft.function_gradient(position)
    assert energy == pytest.approx(1000.0)
    assert np.all(grad > 1e20)
