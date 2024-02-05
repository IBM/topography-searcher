""" Perturbations module contains methods to take steps around
    a given space. These moves each have a class, and are
    passed to the global optimisation instance to drive its step-taking """

import random
import numpy as np
from nptyping import NDArray


class StandardPerturbation:

    """
    Description
    ------------

    Class to perturb a position for use in global optimisation.
    For non-atomic systems this is implemented as random displacements

    Attributes
    -----------

    max_displacement : float
        Maximum allowed perturbation size from the given point
    proportional_distance : logical
        Determines whether max_displacement is an absolute value (if False)
        or a proportion of the bounds (if True)
    """

    def __init__(self, max_displacement: float,
                 proportional_distance: bool = False) -> None:
        self.max_displacement = max_displacement
        self.proportional_distance = proportional_distance

    def set_step_sizes(self, coords: type) -> NDArray:
        """ Generate an appropriate step size for the given coordinates """
        # Generate a uniform step size
        step_sizes = np.full(coords.ndim, self.max_displacement)
        # Scale appropriately to bounds range
        if self.proportional_distance:
            step_sizes = (coords.upper_bounds - coords.lower_bounds)*step_sizes
        return step_sizes

    def perturb(self, coords: type) -> None:
        """ Displace the given position by randomly up to specified size """
        step_sizes = self.set_step_sizes(coords)
        # Find perturbations relative to the total size of the space
        perturbations = (np.random.rand(coords.ndim)-0.5)*step_sizes
        # Apply the perturbations to the initial point, x
        coords.position += perturbations
        # If outside bounds then move back to corresponding bound
        coords.move_to_bounds()


class AtomicPerturbation():
    """
    Description
    ------------
    Simple displacement routine applicable to atomic systems
    Perturbs up to max_atoms atoms with a displacement up to max_displacement

    Attributes
    -----------
    max_displacement : float
        The maximum allowed displacement in a given direction for an atom
    max_atoms : int
        The maximum number of atoms that can be perturbed at each call
    """

    def __init__(self, max_displacement: float, max_atoms: int) -> None:
        self.max_displacement = max_displacement
        self.max_atoms = max_atoms

    def perturb(self, coords: type) -> None:
        """ Returns coords perturbed by a displacement to up to max_atoms """
        # Select the max_atoms which are to be displaced
        perturbed_atoms = random.sample(
            range(1, int(coords.ndim/3)), self.max_atoms)
        # Make a displacement vector drawn from uniform distribution
        perturbations = (np.random.rand(self.max_atoms, 3) *
                         self.max_displacement)-0.5*self.max_displacement
        # Add perturbations to the atoms that need to be moved
        for i in range(self.max_atoms):
            coords.position[(perturbed_atoms[i]*3):(perturbed_atoms[i]*3)+3] \
                += perturbations[i]


class MolecularPerturbation():

    """
    Description
    ------------
    Displacement routine applicable to molecular systems
    Performs rotations of flexible bond dihedrals through a random angle
    up to a specified limit

    Attributes
    -----------
    max_displacement : float
        The maximum allowed angle change of a given dihedral
    max_bonds : int
        The maximum number of dihedrals that can be perturbed at each call
    """

    def __init__(self, max_displacement: float, max_bonds: int) -> None:
        self.max_displacement = max_displacement
        self.max_bonds = max_bonds

    def perturb(self, coords: type) -> None:
        """ Rotate about randomly selected bonds within a molecular system """

        chosen_bonds = random.sample(coords.rotatable_dihedrals,
                                     self.max_bonds)
        for i in chosen_bonds:
            bond_index = coords.rotatable_dihedrals.index(i)
            # Generate a random angle
            random_angle = ((random.random()*2.0)-1.0) * self.max_displacement
            # Find the atoms that need to be rotated
            moved_atoms = coords.rotatable_atoms[bond_index]
            # Apply the rotation
            coords.rotate_dihedral([i[1], i[2]], random_angle, moved_atoms)
