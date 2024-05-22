import pytest
import numpy as np
import ase.io
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.similarity.molecular_similarity import MolecularSimilarity
from topsearch.potentials.test_functions import Schwefel, Camelback
from topsearch.potentials.dft import DensityFunctionalTheory
from topsearch.potentials.force_fields import MMFF94
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.data.coordinates import StandardCoordinates, MolecularCoordinates
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.global_optimisation.perturbations import StandardPerturbation, \
        MolecularPerturbation
from topsearch.analysis.minima_properties import get_minima_energies

def test_run_camelback():
    step_taking = StandardPerturbation(max_displacement=0.7,
                                       proportional_distance=True)
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    camelback = Camelback()
    basin_hopping = BasinHopping(ktn=ktn, potential=camelback,
                                 similarity=similarity,
                                 step_taking=step_taking)
    basin_hopping.run(coords=coords, n_steps=1500,
                      temperature=1.0, conv_crit=1e-6)
    energies = get_minima_energies(ktn)
    # Check that we locate all six minima of the function within bounds
    assert sorted(energies)[0] == pytest.approx(-1.0316284534898734)
    assert sorted(energies)[1] == pytest.approx(-1.0316284534898585)

def test_run_schwefel():
    step_taking = StandardPerturbation(max_displacement=0.3,
                                       proportional_distance=True)
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=3, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0),
                                                 (-500.0, 500.0)])
    ktn = KineticTransitionNetwork()
    schwefel = Schwefel()
    basin_hopping = BasinHopping(ktn=ktn, potential=schwefel,
                                 similarity=similarity,
                                 step_taking=step_taking)
    basin_hopping.run(coords=coords, n_steps=2500,
                      temperature=100.0, conv_crit=1e-5)
    energies = get_minima_energies(ktn)
    min_energy = min(energies)
    assert min_energy < 1e-3

def test_metropolis():
    step_taking = StandardPerturbation(max_displacement=0.3,
                                       proportional_distance=True)
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0),
                                                 (-2.0, 2.0)])
    coords.position = np.array([0.1, -0.7])
    ktn = KineticTransitionNetwork()
    camelback = Camelback()
    basin_hopping = BasinHopping(ktn=ktn, potential=camelback,
                                 similarity=similarity,
                                 step_taking=step_taking)
    accept = basin_hopping.metropolis(10.0, 9.0, 1.0)
    assert accept == True
    accept = basin_hopping.metropolis(10.0, 8.0, 10.0)
    assert accept == True
    accept = basin_hopping.metropolis(10.0, 11.0, np.inf)
    assert accept == True
    accept = basin_hopping.metropolis(10.0, 11.0, 1e-5)
    assert accept == False

def test_prepare_initial_coordinates():
    step_taking = StandardPerturbation(max_displacement=0.3,
                                       proportional_distance=True)
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0),
                                                 (-2.0, 2.0)])
    coords.position = np.array([0.1, -0.7])
    ktn = KineticTransitionNetwork()
    camelback = Camelback()
    basin_hopping = BasinHopping(ktn=ktn, potential=camelback,
                                 similarity=similarity,
                                 step_taking=step_taking)
    energy = basin_hopping.prepare_initial_coordinates(coords, 1e-6)
    assert energy == pytest.approx(-1.031628422)
    assert np.all(coords.position == pytest.approx(np.array([0.0898420385,
                                                             -0.712656397])))
