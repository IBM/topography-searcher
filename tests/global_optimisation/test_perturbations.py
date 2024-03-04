import pytest
import numpy as np
from topsearch.data.coordinates import StandardCoordinates, \
    AtomicCoordinates, MolecularCoordinates
from topsearch.global_optimisation.perturbations import StandardPerturbation, \
    AtomicPerturbation, MolecularPerturbation
import ase

def test_step_size():
    coords = StandardCoordinates(ndim=3, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0),
                                                 (-500.0, 500.0)])
    step_taking = StandardPerturbation(max_displacement=0.1)
    step_sizes = step_taking.set_step_sizes(coords)
    assert np.all(step_sizes == np.array([0.1, 0.1, 0.1]))

def test_proportional_step_size():
    coords = StandardCoordinates(ndim=3, bounds=[(-50.0,  50.0),
                                                 (-500.0, 500.0),
                                                 (-400.0, 400.0)])
    step_taking = StandardPerturbation(max_displacement=0.01,
                                        proportional_distance=True)
    step_sizes = step_taking.set_step_sizes(coords)
    assert np.all(step_sizes == np.array([1.0, 10.0, 8.0]))

def test_step_size_perturbations():
    coords = StandardCoordinates(ndim=3, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0),
                                                 (-500.0, 500.0)])
    coords.position = np.array([0.0, 0.0, 0.0])
    step_taking = StandardPerturbation(max_displacement=0.1)
    for i in range(150):
        initial_position = coords.position.copy()
        step_taking.perturb(coords)
        final_position = coords.position.copy()
        step = final_position - initial_position
        assert np.all(step != np.array([0.0, 0.0, 0.0]))
        assert np.all(step <= np.full(3, 0.1))

def test_proportional_step_size_perturbations():
    coords = StandardCoordinates(ndim=3, bounds=[(-50.0,  50.0),
                                                 (-500.0, 500.0),
                                                 (-400.0, 400.0)])
    coords.position = np.array([0.0, 0.0, 0.0])
    step_taking = StandardPerturbation(max_displacement=0.01,
                                        proportional_distance=True)
    for i in range(150):
        initial_position = coords.position.copy()
        step_taking.perturb(coords)
        final_position = coords.position.copy()
        step = final_position - initial_position
        assert np.all(step != np.array([0.0, 0.0, 0.0]))
        assert np.all(step <= np.array([1.0, 10.0, 8.0]))

def test_edge_perturbations():
    coords = StandardCoordinates(ndim=3, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0),
                                                 (-400.0, 400.0)])
    step_taking = StandardPerturbation(max_displacement=1.0)
    for i in range(150):
        coords.position = np.array([-500.0, -500.0, -400.0])
        step_taking.perturb(coords)
        assert np.all(coords.position >= np.array([-500.0, -500.0, -400.0]))

#### ATOMIC
        
def test_perturb():
    atom_labels = ['C','C','C','C','C']
    init_position = np.array([-0.2604720088, 0.7363147287, 0.4727061929,
                          0.2604716550, -0.7363150782, -0.4727063011,
                         -0.4144908003, -0.3652598516, 0.3405559620,
                         -0.1944131041, 0.2843471802, -0.5500413671,
                          0.6089042582, 0.0809130209, 0.2094855133])
    coords = AtomicCoordinates(atom_labels, init_position.copy())
    step_taking = AtomicPerturbation(0.1, 3)
    for i in range(50):
        step_taking.perturb(coords)
        diff = np.subtract(coords.position, init_position)
        moved_atoms = 0
        for j in range(5):
            if np.all(diff.reshape(-1, 3)[j] == np.array([0.0, 0.0, 0.0])):
               moved_atoms += 1
        coords.position = init_position.copy()
        assert np.all(diff < 0.1)
        assert moved_atoms == 2
    
### MOLECULAR

def test_perturb2():
    atoms = ase.io.read('test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions()
    coords = MolecularCoordinates(species, position.flatten())
    step_taking = MolecularPerturbation(180.0, 1)
    step_taking.perturb(coords)
    # Check that we have rotated 1 dihedral
    coords.atoms.set_positions(coords.position.reshape(9, 3))
    angle1 = coords.atoms.get_dihedral(1, 0, 2, 8)
    angle2 = coords.atoms.get_dihedral(2, 0, 1, 5)
    diff1 = 180.0 - angle1
    diff2 = 300.0 - angle2
    if diff1 != 0.0:
        assert diff2 < 1e-2
    else:
        assert diff1 < 1e-2
