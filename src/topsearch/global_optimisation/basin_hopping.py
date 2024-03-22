""" basin_hopping module contains the BasinHopping class which performs
    global optimisation of a given function """

from timeit import default_timer as timer
import numpy as np
from topsearch.minimisation import lbfgs
from topsearch.data.coordinates import MolecularCoordinates, AtomicCoordinates
from topsearch.potentials.dft import DensityFunctionalTheory


class BasinHopping:

    """
    Description
    ------------

    Class to perform basin-hopping global optimisation algorithm
    Basin-hopping algorithm described in doi:10.1021/jp970984n
    One iteration is described by:
    1) Perturb current position
    2) Minimisation to get corresponding local minimum
    3) Accept/reject based on Metropolis-like criterion

    Attributes
    ----------

    ktn : class instance
        An instance of the kinetic transition network class that is used
        to store the unique minima we locate
    potential : class instance
        Class containing the function that we are aiming to optimise
    similarity: class instance
        Class that determines if two minima are the same or unique
    step_taking : class instance
        Instance that perturbs the current position using a given set
        of moves specified in the class
    """

    def __init__(self,
                 ktn: type,
                 potential: type,
                 similarity: type,
                 step_taking: type) -> None:
        self.ktn = ktn
        self.potential = potential
        self.similarity = similarity
        self.step_taking = step_taking

    def run(self, coords: type, n_steps: int, conv_crit: float,
            temperature: float) -> None:
        """ Method to perform basin-hopping from a given start point """
        start_time = timer()
        self.write_initial_information(n_steps, temperature)
        energy = self.prepare_initial_coordinates(coords, conv_crit)
        # coords/energy - coordinates after each perturbation
        # markov_coords/energy - current minimum in the Markov chain
        markov_coords = coords.position.copy()
        markov_energy = energy
        # Main loop for repeated perturbation, minimisation, and acceptance
        for i in range(n_steps):
            #  Perturb coordinates
            self.step_taking.perturb(coords)
            # Test for and remove atom clashes if density functional theory
            if isinstance(self.potential, DensityFunctionalTheory):
                coords.remove_atom_clashes(self.potential.force_field)
            elif isinstance(coords, AtomicCoordinates):
                coords.remove_atom_clashes()
            # Perform local minimisation
            min_position, energy, results_dict = \
                lbfgs.minimise(func_grad=self.potential.function_gradient,
                               initial_position=coords.position,
                               bounds=coords.bounds,
                               conv_crit=conv_crit)
            # Failed minimisation so don't accept
            if results_dict['warnflag'] != 0 or \
                    results_dict['task'] == 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH':
                with open('logfile', 'a', encoding="utf-8") as outfile:
                    outfile.write(f"Step {i+1}: Could not converge \n")
                # Revert to previous coordinates
                coords.position = markov_coords
                continue
            coords.position = min_position
            # Check that the bonds have not changed if molecular
            if isinstance(coords, (AtomicCoordinates, MolecularCoordinates)):
                # Bonds have been changed so reject step
                if not coords.same_bonds():
                    # Revert coordinates and reject step
                    coords.position = markov_coords
                    continue
            # Check the Metropolis criterion for acceptance of new state
            if self.metropolis(markov_energy, energy, temperature):
                # Check similarity and add if different to all existing minima
                self.similarity.test_new_minimum(self.ktn, coords, energy)
                # Update the current minimum in the Markov chain
                markov_coords = coords.position
                markov_energy = energy
            else:
                # Check if minimum should be added even if not accepted
                self.similarity.test_new_minimum(self.ktn, coords, energy)
                coords.position = markov_coords
                energy = markov_energy
        end_time = timer()
        self.write_final_information(start_time, end_time)

    def prepare_initial_coordinates(self, coords: type,
                                    conv_crit: float) -> float:
        """ Generate the initial minimum and set its energy and position """
        min_position, energy, results_dict = \
            lbfgs.minimise(func_grad=self.potential.function_gradient,
                           initial_position=coords.position,
                           bounds=coords.bounds,
                           conv_crit=conv_crit)
        coords.position = min_position
        # After minimisation add to the existing stationary point network
        if results_dict['warnflag'] == 0:
            self.similarity.test_new_minimum(self.ktn, coords, energy)
        return energy

    def metropolis(self, energy1: float, energy2: float,
                   temperature: float) -> bool:
        """
        Compute the acceptance probability using the Metropolis criterion
        energy2 - proposed minimum, energy1 - current state in Markov chain
        """

        if energy2 < energy1:
            return True
        boltzmann_factor = np.exp(-(energy2-energy1)/temperature)
        uniform_random = np.random.random()
        return bool(boltzmann_factor > uniform_random)

    def write_initial_information(self, n_steps: int,
                                  temperature: float) -> None:
        """ Print out the initial basin-hopping """

        with open('logfile', 'a', encoding="utf-8") as outfile:
            outfile.write('-------BASIN-HOPPING-------\n')
            outfile.write(f"Steps = {n_steps},"
                          f"Temperature = {temperature}\n")

    def write_final_information(self, start_time: float,
                                end_time: float) -> None:
        """ Print out the final basin-hopping information """

        with open('logfile', 'a', encoding="utf-8") as outfile:
            outfile.write(f"Located {self.ktn.n_minima} distinct minima\n")
        with open('logfile', 'a', encoding="utf-8") as outfile:
            outfile.write(f"Basin-hopping completed in "
                          f"{end_time - start_time}\n\n")
