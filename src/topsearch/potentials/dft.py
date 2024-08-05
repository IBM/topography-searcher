""" Module containing the class for performing electronic structure
    calculations to compute the potential energy of atomistic systems """

import numpy as np
from nptyping import NDArray
from ase.calculators.psi4 import Psi4
from ase import Atoms
from ase.units import Hartree, Bohr
import warnings
import traceback
import contextlib
from .potential import Potential


class DensityFunctionalTheory(Potential):
    """
    Description
    ---------------
    Evaluate the energy and force of a molecular system using the Psi4
    package via its ase calculators. The energy is computed using
    density functional theory

    Attributes
    ----------------
    atom_labels : list
        Labels containing the species of each atom in the system
    calculator_type : string
        ase calculator to use for energy and force. Default is 'psi4'
    options : dict
        keyword arguments that will be passed to the ase calculator
    force_field : class
        DFT calls can fail with significant atomic overlap so the force_field
        object is used in these cases to remove clashes without failure
    """

    def __init__(self, atom_labels: list, options: dict,
                 force_field: type, calculator_type: str = 'psi4') -> None:
        self.atomistic = True
        self.calculator_type = calculator_type
        self.atom_labels = atom_labels
        self.options = options
        self.force_field = force_field
        # Make a placeholder atomic configuration for initialising calculator
        n_atoms = len(self.atom_labels)
        init_position = np.ones((n_atoms, 3), dtype=float) * \
            np.arange(0, n_atoms, dtype=float).reshape(-1, 1)
        # Create the atoms object for computing the energy from coordinates
        self.atoms = \
            Atoms(''.join(self.atom_labels),
                  positions=init_position)
        # Calculate reference properties to retain
        self.calculator_type = calculator_type
        # Set up the Psi4 calculator
        calc = Psi4(atoms=self.atoms,
                    method=self.options["method"],
                    memory='500MB',
                    basis=self.options["basis"],
                    num_threads=self.options["threads"])
        del self.options["method"], self.options["basis"], \
            self.options["threads"]
        calc.psi4.set_options(self.options)
        self.atoms.calc = calc

    def reset_options(self, options: dict) -> None:
        """ Reset the input options to the ase calculator """
        self.options = options
        self.atoms.calc.psi4.core.clean_options()
        calc = Psi4(atoms=self.atoms,
                    method=self.options["method"],
                    memory='500MB',
                    basis=self.options["basis"],
                    num_threads=self.options["threads"])
        del self.options["method"], self.options["basis"], \
            self.options["threads"]
        calc.psi4.set_options(self.options)
        self.atoms.calc = calc

    def function(self, position: NDArray) -> float:
        """ Compute the electronic potential energy """
        self.atoms.set_positions(position.reshape(-1, 3))
        try:
            energy = self.atoms.get_potential_energy()
        except Exception:
            return 1000.0
        return energy

    def function_gradient(self, position: NDArray) -> tuple:
        """ Compute the electronic potential energy and its forces """
        self.atoms.set_positions(position.reshape(-1, 3))
        try:
            energy = self.atoms.get_potential_energy()
        except Exception:
            return 1000.0, np.full((position.size), np.inf, dtype=float)
        forces = self.atoms.get_forces().flatten()
        gradient = -1.0 * np.array(forces.tolist())
        return energy, gradient

    def gradient(self, position: NDArray) -> NDArray:
        """ Compute the analytical gradient from the ASE calculator """
        self.atoms.set_positions(position.reshape(-1, 3))
        try:
            forces = self.atoms.get_forces().flatten()
        except Exception:
            return np.full((position.size), np.inf, dtype=float)
        return -1.0 * np.array(forces.tolist())
    
    def hessian(self, position: NDArray,
                displacement: float = 1e-4) -> NDArray:
        """ Override numerical hessian calculation with internal analytic version for DFT."""
        
        if displacement != 1e-4:
            warnings.warn("Displacement is not used in DFT hessian calculation, ignoring input.",
                          UserWarning)
        self.atoms.set_positions(position.reshape(-1, 3))
        calc = self.atoms.calc
        calc.set_psi4()
        calc.psi4.core.set_output_file(calc.label + '.dat',
                                       False)
        method = calc.parameters['method']
        basis = calc.parameters['basis']
        try:
            with contextlib.redirect_stdout(None): # silence chatty Psi4
                hess = calc.psi4.driver.hessian(f'{method}/{basis}',
                                                molecule=calc.molecule,).to_array()
            hess *= Bohr**2/Hartree
        except Exception:
            traceback.print_exc()
            hess = np.full((position.size, position.size), np.inf, dtype=float)
        
        return hess
