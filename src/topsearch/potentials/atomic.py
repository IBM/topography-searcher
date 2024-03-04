""" Module to store interatomic potentials that can be used
    for atomistic systems """

import numpy as np
from nptyping import NDArray
from .potential import Potential


class LennardJones(Potential):
    """
    Description
    ------------
    The Lennard-Jones pair potential for a set of atoms
    Simple but fast representation of interatomic potentials

    Attributes
    -----------
    epsilon : float
        The well depth of the pair potential
    sigma : float
        The length scale of the pair potential
    """

    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0):
        self.atomistic = True
        self.epsilon = epsilon
        self.sigma = sigma

    def get_atom(self, position: NDArray, atom_ind: int) -> NDArray:
        """ Get the Cartesian position of the atom given index atom_ind """
        return position[atom_ind*3:(atom_ind*3)+3]

    def pair_potential(self, atom1: NDArray, atom2: NDArray) -> tuple:
        """ Return the value of the pair potential and terms needed for
            the gradient """
        dist = self.squared_distance(atom1, atom2)
        r6_term = self.sigma/(dist**3)
        potential_energy = 4*self.epsilon*r6_term*(r6_term-1.0)
        return potential_energy, r6_term, dist

    def squared_distance(self, atom1: NDArray, atom2: NDArray) -> float:
        """ Return the squared distance between two atoms """
        dist = (atom1[0] - atom2[0])**2 + (atom1[1] - atom2[1])**2 + \
            (atom1[2] - atom2[2])**2
        return dist

    def function(self, position: NDArray) -> float:
        """ Return the potential energy of a set of atoms evaluated
            using the Lennard Jones potential """
        pot_energy_total = 0.0
        for i in range(int(position.size/3)-1):
            for j in range(i+1, int(position.size/3)):
                v_ij, r6_term, dist = self.pair_potential(
                    self.get_atom(position, i), self.get_atom(position, j))
                pot_energy_total += v_ij
        return pot_energy_total

    def gradient(self, position: NDArray) -> NDArray:
        """ Return the gradient vector at configuration position """
        grad = np.zeros(position.size, dtype=float)
        for i in range(int(position.size/3)-1):
            for j in range(i+1, int(position.size/3)):
                v_ij, r6_term, dist = self.pair_potential(
                    self.get_atom(position, i), self.get_atom(position, j))
                g_factor = (24.0 * self.epsilon *
                            (r6_term - 2.0 * (r6_term**2))) / dist
                diff = self.get_atom(position, i) - self.get_atom(position, j)
                grad[i*3:(i*3)+3] += diff*g_factor
                grad[j*3:(j*3)+3] -= diff*g_factor
        return grad

    def function_gradient(self, position: NDArray) -> tuple:
        """ Return both the function and gradient at point position """
        grad = np.zeros(position.size, dtype=float)
        pot_energy_total = 0.0
        for i in range(int(position.size/3)-1):
            for j in range(i+1, int(position.size/3)):
                v_ij, r6_term, dist = self.pair_potential(
                    self.get_atom(position, i), self.get_atom(position, j))
                pot_energy_total += v_ij
                g_factor = (24.0 * self.epsilon *
                            (r6_term - 2.0 * (r6_term**2))) / dist
                diff = self.get_atom(position, i) - self.get_atom(position, j)
                grad[i*3:(i*3)+3] += diff*g_factor
                grad[j*3:(j*3)+3] -= diff*g_factor
        return pot_energy_total, grad


class BinaryGupta(Potential):
    """
    Description
    ------------
    Evaluate energy of binary metal clusters with the Gupta potential
    Here only contains the parameters for Au and Ag.

    Attributes
    -----------
    atom_labels : array
        Denotes the species of atoms, can be 'Au' or 'Ag'
    a_repulsive : array
        Repulsive coefficient
    zeta : array
        Attractive coefficient
    p_range : array
        Range of interaction
    q_range : array
        Range of interaction
    r_eq : array
        Equilibrium bond lengths
    """

    def __init__(self, species: list):
        self.atomistic = True
        self.species = species
        self.atom_labels = [0 if i == 'Au' else 1 for i in self.species]
        # Initialise the parameters of the potential
        self.a_repulsive = \
            np.array(((0.1028, 0.149), (0.149, 0.2061)), dtype=float)
        self.zeta = np.array(((1.178, 1.4874), (1.4874, 1.790)), dtype=float)
        self.p_range = \
            np.array(((10.928, 10.494), (10.494, 10.229)), dtype=float)
        self.q_range = np.array(((3.139, 3.607), (3.607, 4.036)), dtype=float)
        self.r_eq = np.array(((2.8885, 2.8864), (2.8864, 2.8843)), dtype=float)

    def function(self, position: NDArray) -> float:
        """ Return the potential energy of the system at position """

        #  Initialise arrays
        n_atoms = int(position.size/3)
        rho = np.zeros(n_atoms, dtype=float)
        local_potentials = np.zeros(n_atoms, dtype=float)

        #  Loop over atoms to get energy of each
        for i in range(n_atoms-1):
            species1 = self.atom_labels[i]
            for j in range(i+1, n_atoms):
                species2 = self.atom_labels[j]
                # Get the distance between atoms
                dist = np.linalg.norm(position[i*3:(i*3)+3] -
                                      position[j*3:(j*3)+3])
                # And the relevant potential parameters
                a_tmp = self.a_repulsive[species1, species2]
                p_tmp = self.p_range[species1, species2]
                r_tmp = self.r_eq[species1, species2]
                zeta_tmp = self.zeta[species1, species2]
                q_tmp = -2.0*self.q_range[species1, species2]
                # Repulsive potential between atoms
                repulsive = a_tmp*np.exp(dist*(-1.0*p_tmp/r_tmp) + p_tmp)
                local_potentials[i] += repulsive
                local_potentials[j] += repulsive
                # Attractive part of interatomic potential
                attractive = (zeta_tmp**2)*np.exp(dist*(q_tmp/r_tmp) - q_tmp)
                rho[i] += attractive
                rho[j] += attractive

        #  Accumulate the potential energy
        local_potentials -= np.sqrt(rho)
        return np.sum(local_potentials)
