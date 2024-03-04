""" Module that contains the class for evaluating machine-learned
    potentials through ase """

import numpy as np
from nptyping import NDArray
from ase import Atoms
from .potential import Potential


class MachineLearningPotential(Potential):
    """
    Description
    ---------------
    Evaluate the energy of a molecular system using machine learning potentials
    called through ase

    Attributes
    ---------------
    atom_labels : list
        List of atomic species for each atom in the molecule.
    calculator_type : str
        Type of calculator to use. Options are 'mace' or 'torchani'
    """

    def __init__(self, atom_labels: list,
                 calculator_type: str = 'torchani') -> None:
        self.atomistic = True
        self.calculator_type = calculator_type
        self.atom_labels = atom_labels
        # Make a placeholder atomic configuration for initialising calculator
        n_atoms = len(self.atom_labels)
        init_position = np.ones((n_atoms, 3), dtype=float) * \
            np.arange(0, n_atoms, dtype=float).reshape(-1, 1)
        # Create the atoms object for the molecule
        self.atoms = Atoms(''.join(self.atom_labels),
                           positions=init_position)
        self.calculator_type = calculator_type
        # Set up the calculator based on the specified type
        if self.calculator_type == 'torchani':
            import torchani
            self.model = torchani.models.ANI2x(periodic_table_index=True)
        elif self.calculator_type == 'mace':
            from mace.calculators import MACECalculator
            self.atoms.calc = \
                MACECalculator(model_paths='MACE_model_swa.model',
                               device='cpu')

    def function(self, position: NDArray) -> float:
        """ Compute the electronic potential energy """
        self.atoms.set_positions(position.reshape(-1, 3))
        if self.calculator_type == 'mace':
            energy = self.atoms.get_potential_energy()
        elif self.calculator_type == 'torchani':
            import torch
            species = torch.tensor(self.atoms.get_atomic_numbers(),
                                   dtype=torch.int64).unsqueeze(0)
            coordinates = torch.tensor(position.reshape(-1, 3),
                                       dtype=torch.float32,
                                       requires_grad=True).unsqueeze(0)
            energy = self.model((species, coordinates)).energies.item()
        return energy

    def function_gradient(self, position: NDArray) -> tuple:
        """ Compute the electronic potential energy and its forces """
        self.atoms.set_positions(position.reshape(-1, 3))

        if self.calculator_type == 'mace':
            # Calculate energy and forces using Psi4
            energy = self.atoms.get_potential_energy()
            forces = self.atoms.get_forces().flatten()
            gradient = -1.0 * np.array(forces.tolist())
        elif self.calculator_type == 'torchani':
            import torch
            species = torch.tensor(self.atoms.get_atomic_numbers(),
                                   dtype=torch.int64).unsqueeze(0)
            coordinates = torch.tensor(position.reshape(-1, 3),
                                       dtype=torch.float32,
                                       requires_grad=True).unsqueeze(0)
            energy_torch = self.model((species, coordinates)).energies
            gradient_torch = torch.autograd.grad(energy_torch.sum(),
                                                 coordinates)[0]
            energy = energy_torch.item()
            gradient = np.array(gradient_torch.flatten().tolist())
        return energy, gradient

    def gradient(self, position: NDArray) -> NDArray:
        """ Compute the analytical gradient from the ASE calculator """
        self.atoms.set_positions(position.reshape(-1, 3))
        if self.calculator_type == 'mace':
            forces = self.atoms.get_forces().flatten()
            gradient = -1.0 * np.array(forces.tolist())
        elif self.calculator_type == 'torchani':
            import torch
            species = torch.tensor(self.atoms.get_atomic_numbers(),
                                   dtype=torch.int64).unsqueeze(0)
            coordinates = torch.tensor(position.reshape(-1, 3),
                                       dtype=torch.float32,
                                       requires_grad=True).unsqueeze(0)
            energy_torch = self.model((species, coordinates)).energies
            gradient_torch = torch.autograd.grad(energy_torch.sum(),
                                                 coordinates)[0]
            gradient = np.array(gradient_torch.flatten().tolist())
        return gradient
