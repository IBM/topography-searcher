""" The basin_hopping module contains the BasinHopping class which performs
    global optimisation of a given function """

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import logging
from timeit import default_timer as timer
import numpy as np
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.global_optimisation.perturbations import AtomicPerturbation, MolecularPerturbation, StandardPerturbation
from topsearch.minimisation import lbfgs, psi4_internal
from topsearch.data.coordinates import MolecularCoordinates, AtomicCoordinates, StandardCoordinates
from topsearch.potentials.dft import DensityFunctionalTheory
from topsearch.potentials.potential import Potential
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.utils.parallel import run_parallel
from tqdm.auto import tqdm, trange

class BasinHopping:

    """
    Description
    ------------

    Class to perform basin-hopping global optimisation algorithm
    as described in https://doi.org/10.1021/jp970984n
    One iteration is described by:
    1) Perturb current position
    2) Minimisation to get corresponding local minimum
    3) Accept/reject based on Metropolis-like criterion

    Attributes
    ----------

    ktn : class instance
        An instance of the KineticTransitionNetwork class that is used
        to store the unique minima we locate
    potential : class instance
        Object containing the function that we are aiming to optimise
    similarity: class instance
        Object that determines if two minima are the same or unique
    step_taking : class instance
        Instance that perturbs the current position using a given set
        of moves specified in the class
    """

    def __init__(self,
                 ktn: KineticTransitionNetwork,
                 potential: Potential,
                 similarity: StandardSimilarity,
                 step_taking: StandardPerturbation | AtomicPerturbation | MolecularPerturbation,
                 opt_method: str = 'scipy',
                 ignore_relreduc: bool = False) -> None:
        self.ktn = ktn
        self.potential = potential
        self.similarity = similarity
        self.step_taking = step_taking
        self.logger = logging.getLogger("basin-hopping")
        self.opt_method = opt_method
        self.ignore_relreduc = ignore_relreduc

    def run_batch(self, initial_positions: np.ndarray, coords: type, n_steps: int, conv_crit: float,
                temperature: float, num_proc=8) -> None:
        self.logger.debug(f"Running basin hopping for {len(initial_positions)} starting points with {num_proc} processes")

        run_step = partial(self.run_single, coords=coords, n_steps=n_steps, conv_crit=conv_crit, temperature=temperature)

        for position, ktn in run_parallel(run_step, initial_positions, processes=num_proc, return_input=True):
                for m in range(ktn.n_minima):
                    try:
                        coords.position = ktn.get_minimum_coords(m)
                        self.logger.debug(f"New minimum value {ktn.get_minimum_energy(m)}")
                        self.similarity.test_new_minimum(self.ktn, coords, ktn.get_minimum_energy(m))
                    except KeyError:
                        self.logger.error(f"Missing minimum index {m}. n_minima: {ktn.n_minima} Nodes in graph: {len(ktn.G.nodes)}.\n Keys: {ktn.G.nodes.keys()} ")
                        ktn.dump_network("bad_minima")
                
                self.ktn.add_attempted_position(position)
                self.ktn.dump_network()
           
    def run_single(self, position: np.ndarray, coords: type, n_steps: int, conv_crit: float,
                temperature: float):
            
            coords.position = position
            self.logger.debug(f"Starting position: {position}")
            self.logger.debug(f"Current KTN with n_minima: {self.ktn.n_minima} nodes in graph: {len(self.ktn.G.nodes)}")
            self.ktn.reset_network() # Clear the KTN to aovid returning a massive one
            self.run(coords=coords, n_steps=n_steps, conv_crit=conv_crit, temperature=temperature)
            self.logger.debug(f"Returning KTN with n_minima: {self.ktn.n_minima} nodes in graph: {len(self.ktn.G.nodes)}")
            return self.ktn

    def run(self, coords: StandardCoordinates | MolecularCoordinates | AtomicCoordinates, n_steps: int, conv_crit: float,
            temperature: float) -> None:
        """ Method to perform basin-hopping from a given start point """
        start_time = timer()
        self.write_initial_information(n_steps, temperature, conv_crit)
        energy = self.prepare_initial_coordinates(coords, conv_crit)
        # coords/energy - coordinates after each perturbation
        # markov_coords/energy - current minimum in the Markov chain
        markov_coords = coords.position.copy()
        markov_energy = energy
        # Main loop for repeated perturbation, minimisation, and acceptance
        for i in trange(n_steps, desc="Basin hopping - steps"):
            self.logger.debug(f"Step {i} of {n_steps}")
            #  Perturb coordinates
            self.step_taking.perturb(coords)
            # Test for and remove atom clashes if density functional theory
            if isinstance(self.potential, DensityFunctionalTheory):
                coords.remove_atom_clashes(self.potential.force_field)
            elif isinstance(coords, AtomicCoordinates):
                try:
                    coords.remove_atom_clashes()
                except:
                    pass # TODO: this was giving me an error with the MACE
            # Perform local minimisation
            if self.opt_method == 'scipy':
                min_position, energy, results_dict = \
                    lbfgs.minimise(func_grad=self.potential.function_gradient,
                                initial_position=coords.position,
                                bounds=coords.bounds,
                                conv_crit=conv_crit)
            elif self.opt_method == 'ase':
                min_position, energy, results_dict = \
                    lbfgs.minimise_ase(self.potential,
                                       initial_position=coords.position,
                                       numbers=coords.atoms.numbers,
                                       conv_crit=conv_crit)
            elif self.opt_method == 'psi4':
                if not isinstance(self.potential, DensityFunctionalTheory):
                    raise ValueError("Psi4 optimisation method requires DFT potential")
                min_position, energy, results_dict = \
                psi4_internal.minimise(self.potential,
                                initial_position=coords.position,
                                conv_crit=conv_crit)
            else:
                raise ValueError("Invalid optimisation method, options are 'scipy' or 'psi4'")
            # Failed minimisation so don't accept
            if results_dict['warnflag'] != 0 or \
                    (results_dict['task'] == 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH' \
                        and not self.ignore_relreduc):
                
                self.logger.info(f"Step {i+1}: Could not converge with {results_dict} \n")
                # Revert to previous coordinates
                coords.position = markov_coords
                continue
            coords.position = min_position
            # coords.write_xyz(f"_bh_{i}_")
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

    def prepare_initial_coordinates(self, coords: StandardCoordinates,
                                    conv_crit: float) -> float:
        """ Generate the initial minimum and set its energy and position """
        self.logger.debug("Initial minimisation")
        if self.opt_method == 'scipy':
            min_position, energy, results_dict = \
                lbfgs.minimise(func_grad=self.potential.function_gradient,
                            initial_position=coords.position,
                            bounds=coords.bounds,
                            conv_crit=conv_crit)
            self.logger.debug(f"Initial minima: {min_position}, energy: {energy}, results_dict: {results_dict}")
        elif self.opt_method == 'ase':
                min_position, energy, results_dict = \
                    lbfgs.minimise_ase(self.potential,
                                       initial_position=coords.position,
                                       numbers=coords.atoms.numbers,
                                       conv_crit=conv_crit)
        elif self.opt_method == 'psi4':
            if not isinstance(self.potential, DensityFunctionalTheory):
                raise ValueError("Psi4 optimisation method requires DFT potential")
            min_position, energy, results_dict = \
            psi4_internal.minimise(self.potential,
                            initial_position=coords.position,
                            conv_crit=conv_crit)
        else:
            raise ValueError("Invalid optimisation method, options are 'scipy' or 'psi4'")
        coords.position = min_position
        # After minimisation add to the existing stationary point network
        if results_dict['warnflag'] == 0:
            self.similarity.test_new_minimum(self.ktn, coords, energy)
        elif results_dict['warnflag'] == 1:
            self.logger.info("Initial minimisation did not converge")
             
        return energy

    def metropolis(self, energy1: float, energy2: float,
                   temperature: float) -> bool:
        """
        Compute the acceptance probability using the Metropolis criterion.
        energy2 - proposed minimum, energy1 - current state in Markov chain
        """

        if energy2 < energy1:
            return True
        boltzmann_factor = np.exp(-(energy2-energy1)/temperature)
        uniform_random = np.random.random()
        return bool(boltzmann_factor > uniform_random)

    def write_initial_information(self, n_steps: int,
                                  temperature: float, conv_crit: float) -> None:
        """ Print out the initial basin-hopping information """

        self.logger.info(f"Steps = {n_steps}, Temperature = {temperature}, conv_crit = {conv_crit}")

    def write_final_information(self, start_time: float,
                                end_time: float) -> None:
        """ Print out the final basin-hopping information """

        self.logger.info(f"Located {self.ktn.n_minima} distinct minima\n")
        self.logger.info(f"Basin-hopping completed in "
                          f"{end_time - start_time}\n\n")
