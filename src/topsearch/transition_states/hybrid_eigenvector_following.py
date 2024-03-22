""" The single_ended_search module employs different methods for locating
    transition states starting from a single point """

from math import inf
from copy import deepcopy
import numpy as np
from nptyping import NDArray
from topsearch.minimisation import lbfgs
from topsearch.data.coordinates import AtomicCoordinates, MolecularCoordinates


class HybridEigenvectorFollowing:

    """
    Description
    ------------

    Class for single-ended transition state searches, starting
    from a given position the hybrid eigenvector-following algorithm
    is used to take steps uphill until a Hessian index one point is found.
    i) Finds eigenvector corresponding to the lowest eigenvalue
    ii) Take a step along that eigenvector
    iii) Minimise in the remaining subspace
    iv) Repeat until convergence

    Attributes
    -----------

    potential : class instance
        Function on which we are attempting to find transition states
    conv_crit : float
        The value the norm of the gradient must be less than before the
        transition state is considered converged
    ts_steps : int
        Allowed number of steps before giving up on transition state
    max_uphill_step_size : float
        Maximum size of a step that goes in the uphill direction following
        the eigenvector corresponding to the negative eigenvalue
    positive_eigenvalue_step : float
        The size of a step along the eigenvector corresponding to the smallest
        eigenvalue, when it is still positive
    pushoff : float
        The size of the displacement along the eigenvector when finding
        connected minima for each transition state
    eig_bounds : list
        Bounds placed on the eigenvector during minimisation, updated
        throughout the search based on the bounds coords is at
    failure : string
        Contains the reason for failure to find a transition state to report
    remove_trans_rot : bool
        Flag as to whether to remove eigenvectors that correspond to overall
        translation and rotation, needed for atomic and molecular systems
    """

    def __init__(self,
                 potential: type,
                 ts_conv_crit: float,
                 ts_steps: int,
                 pushoff: float,
                 steepest_descent_conv_crit: float = 1e-6,
                 min_uphill_step_size: float = 1e-7,
                 max_uphill_step_size: float = 1.0,
                 positive_eigenvalue_step: float = 0.1,
                 eigenvalue_conv_crit: float = 1e-5) -> None:
        self.potential = potential
        self.ts_conv_crit = ts_conv_crit
        self.ts_steps = ts_steps
        self.steepest_descent_conv_crit = steepest_descent_conv_crit
        self.max_uphill_step_size = max_uphill_step_size
        self.min_uphill_step_size = min_uphill_step_size
        self.positive_eigenvalue_step = positive_eigenvalue_step
        self.eigenvalue_conv_crit = eigenvalue_conv_crit
        self.pushoff = pushoff
        self.eigenvector_bounds = None
        self.failure = None
        self.remove_trans_rot = None

    def run(self, coords: type) -> tuple:
        """ Perform a single-ended transition state search starting from coords
            Returns the information about transition states and their connected
            minima that is needed for testing similarity and adding to
            stationary point network """

        # Set whether we should remove overall translation and rotation
        # Necessary for atomic and molecular systems
        if isinstance(coords, (AtomicCoordinates, MolecularCoordinates)):
            self.remove_trans_rot = True
        else:
            self.remove_trans_rot = False

        # Generate initial vector from which to find eigenvector
        eigenvector = self.generate_random_vector(coords.ndim)
        self.eigenvector_bounds = [(-inf, inf)]*coords.ndim
        # Set the initial failure reporting
        self.failure = None

        # Set appropriate initial eigenvector bounds
        lower_bounds, upper_bounds = coords.active_bounds()
        self.update_eigenvector_bounds(lower_bounds, upper_bounds)

        # Take steps to find transition states until convergence or too many
        for n_steps in range(self.ts_steps):
            # Find the smallest eigenvalue and associated eigenvector
            eigenvector, eigenvalue, eig_steps = \
                self.get_smallest_eigenvector(eigenvector,
                                              coords,
                                              lower_bounds,
                                              upper_bounds)
            # Invalid eigenvector so return
            if eigenvector is None:
                return None, None, None, None, None, None, None
            # Take a step following the eigenvector uphill
            self.take_uphill_step(coords, eigenvector, eigenvalue)
            # If eigenvalue is below zero then minimise in orthogonal subspace
            if eigenvalue < 0.0 and eig_steps < 5:
                subspace_pos, energy, results_dict = \
                    self.subspace_minimisation(coords, eigenvector)
                coords.position = subspace_pos
            # Reset appropriate eigenvector bounds
            lower_bounds, upper_bounds = coords.active_bounds()
            self.update_eigenvector_bounds(lower_bounds, upper_bounds)
            # Test for convergence to a transition state
            if self.test_convergence(coords.position, lower_bounds,
                                     upper_bounds):
                # Converged to transition state so recalculate eigenvector
                eigenvector, eigenvalue, eig_steps = \
                    self.get_smallest_eigenvector(eigenvector,
                                                  coords,
                                                  lower_bounds,
                                                  upper_bounds)
                # If not valid TS then return
                if eigenvector is None:
                    return None, None, None, None, None, None, None
                # Find the steepest descent paths from transition state
                plus_minimum, e_plus, minus_minimum, e_minus = \
                    self.steepest_descent(coords, eigenvector)
                # If either minimisation does not converge then discard
                if (plus_minimum is None) or (minus_minimum is None):
                    if self.failure != 'pushoff':
                        self.failure = 'SDpaths'
                    return None, None, None, None, None, None, None
                # Calculate the energy of the transition state
                e_ts = self.potential.function(coords.position)
                # Return successful transition state and connected minima
                return coords.position, e_ts, plus_minimum, e_plus, \
                    minus_minimum, e_minus, eigenvector

        # Didn't find transition state in allowed steps
        self.failure = 'steps'
        return None, None, None, None, None, None, None

    def get_smallest_eigenvector(self, initial_vector: NDArray,
                                 coords: type,
                                 lower_bounds: NDArray,
                                 upper_bounds: NDArray) -> NDArray:
        """ Routine to find the smallest eigenvalue and the associated
            eigenvector of the Hessian at a given point coords. Method
            tests for validity and deals with bounds """

        # Minimise to get smallest eigenvalue and its eigenvector
        eigenvector, eigenvalue, results_dict = \
            lbfgs.minimise(func_grad=self.rayleigh_ritz_function_gradient,
                           initial_position=initial_vector,
                           args=coords.position,
                           bounds=self.eigenvector_bounds,
                           conv_crit=self.eigenvalue_conv_crit)
        bfgs_steps = results_dict['nit']
        # Normalise eigenvector
        if np.linalg.norm(eigenvector) != 0.0:
            eigenvector /= np.linalg.norm(eigenvector)
        # If the eigenvector is not valid then return
        if not self.check_valid_eigenvector(eigenvector, eigenvalue, coords):
            return None, None, None
        eigenvector = self.check_eigenvector_direction(eigenvector,
                                                       coords.position)
        # If true direction points out of bounds then project onto bounds
        if np.any(lower_bounds) or np.any(upper_bounds):
            eigenvector = self.project_onto_bounds(eigenvector,
                                                   lower_bounds,
                                                   upper_bounds)
            # Recompute eigenvalue with the restriction of this plane
            eigenvalue = \
                self.rayleigh_ritz_function_gradient(
                    eigenvector, *coords.position.tolist())[0]
        return eigenvector, eigenvalue, bfgs_steps

    def check_eigenvector_direction(self, eigenvector: NDArray,
                                    position: NDArray) -> NDArray:
        """ Check the eigenvector is pointing in the uphill
            direction, and if not, flip it so it is """

        # Calculate if we need eigenvector or its negative
        grad = self.potential.gradient(position)
        # Find projection of gradient onto search direction
        proj = self.parallel_component(grad, eigenvector)
        # If signs do not match then need negative as uphill direction
        if np.sign(proj)[0] != np.sign(eigenvector)[0]:
            eigenvector *= -1.0
        return eigenvector

    def check_valid_eigenvector(self, eigenvector: NDArray,
                                eigenvalue: float,
                                coords: type) -> bool:
        """ Test that an eigenvector is valid. It cannot contain
            only zeros or any nans """
        if (not np.any(eigenvector)) or (np.any(np.isnan(eigenvector))):
            self.failure = 'eigenvector'
            return False
        if eigenvalue == 0.0:
            self.failure = 'eigenvalue'
            return False
        lower_bounds, upper_bounds = coords.active_bounds()
        all_bounds = np.column_stack((lower_bounds, upper_bounds))
        # Point is at all bounds and cannot be meaningful maximum
        if np.all(np.any(all_bounds, axis=0)):
            self.failure = 'bounds'
            return False
        return True

    def parallel_component(self, vec1: NDArray, vec2: NDArray) -> NDArray:
        """ Return parallel component of vec1 relative to vec2 """
        vec2_magnitude = np.sum(vec2**2)
        if vec2_magnitude < 1e-13:
            return np.zeros(np.size(vec1))
        return (np.dot(vec1, vec2)/vec2_magnitude)*vec2

    def perpendicular_component(self, vec1: NDArray, vec2: NDArray) -> NDArray:
        """ Return perpendicular component of vector vec1 relative to vec2 """
        return vec1 - self.parallel_component(vec1, vec2)

    def subspace_minimisation(self, coords: type,
                              eigenvector: NDArray) -> tuple:
        """ Perform minimisation in the subspace orthogonal to eigenvector.
            The distance is limited to be at most 1% of the range of the
            space to avoid very large initial steps in scipy.lbfgs """
        subspace_bounds = self.get_local_bounds(coords)
        position, energy, results_dict = \
            lbfgs.minimise(func_grad=self.subspace_function_gradient,
                           initial_position=coords.position,
                           args=eigenvector,
                           bounds=subspace_bounds,
                           conv_crit=self.steepest_descent_conv_crit)
        return position, energy, results_dict

    def get_local_bounds(self, coords: type) -> list:
        """ Create a box around the given point that ensures that subspace
            minimisation is bounded to a small region to avoid moving to
            a different basin. Still respect total function bounds """
        # Molecular systems have arbitrarily large bounds so absolute value
        if isinstance(coords, (AtomicCoordinates, MolecularCoordinates)):
            step_sizes = np.array([0.05 for i in coords.bounds])
        else:
            step_sizes = np.array([(i[1]-i[0])*0.02 for i in coords.bounds])
        upper_limit = coords.position + step_sizes
        lower_limit = coords.position - step_sizes
        limits = np.row_stack((lower_limit, upper_limit))
        limits = np.clip(limits, coords.lower_bounds, coords.upper_bounds)
        return list(zip(limits[0], limits[1]))

    def analytic_step_size(self, grad: NDArray, eigenvector: NDArray,
                           eigenvalue: float) -> float:
        """Analytical expression to compute an optimal step length
           from coords along eigenvector, given its eigenvalue
           Return the appropriate step length in the uphill direction """
        overlap = np.dot(grad, eigenvector)
        denominator = (1.0 + np.sqrt(1.0+(4.0*(overlap/eigenvalue)**2)))
        # Check that eigenvalue is non-zero or divide by zero
        if eigenvalue != 0.0:
            step_length = (2.0*overlap) / (np.abs(eigenvalue)*denominator)
        else:
            step_length = 0.0
        # Check step length is within the allowed upper and lower limits
        if np.abs(step_length) > self.max_uphill_step_size:
            return self.max_uphill_step_size
        # If step length is too small then set to specified lower limit
        if np.abs(step_length) < self.min_uphill_step_size:
            return self.min_uphill_step_size
        return step_length

    def take_uphill_step(self, coords: type,
                         eigenvector: NDArray,
                         eigenvalue: float) -> NDArray:
        """ Take uphill step of appropriate length given an eigenvector
            that defines the step direction, and an eigenvalue that
            determines if the step is uphill or not
            Return coords after step is taken """
        if eigenvalue >= 0.0:
            coords.position += self.positive_eigenvalue_step*eigenvector
        else:
            grad = self.potential.gradient(coords.position)
            step_size = self.analytic_step_size(grad, eigenvector,
                                                eigenvalue)
            coords.position += step_size*eigenvector
        coords.move_to_bounds()

    def steepest_descent(self, transition_state: type,
                         eigenvector: NDArray) -> tuple:
        """
        Perturb away from the transition state along the eigenvector
        corresponding to the negative eigenvalue and minimise
        to produce the two connected minima and their energies
        Returns the coordinates and energy of both connected minima
        """
        # Find a suitable pushoff that leads to a decrease in energy
        positive_x, negative_x = self.find_pushoff(transition_state,
                                                   eigenvector)
        # Couldn't find an appropriate pushoff
        if (positive_x is None) or (negative_x is None):
            return None, None, None, None
        # Steepest descent paths to get connected minima
        connected_minimum = deepcopy(transition_state)
        # First positive step
        connected_minimum.position = positive_x
        plus_min, plus_energy, d_plus = \
            self.steepest_descent_paths(connected_minimum)
        # Second negative step
        connected_minimum.position = negative_x
        minus_min, minus_energy, d_minus = \
            self.steepest_descent_paths(connected_minimum)
        return plus_min, plus_energy, minus_min, minus_energy

    def steepest_descent_paths(self, coords: type) -> tuple:
        """ Perform local minimisation to get the result of a steepest
            descent path beginning from position. To avoid very large
            initial steps in LBFGS we constrain the local minimisation
            into several short steps """
        # For molecules the default step size is fine
        if isinstance(coords, (AtomicCoordinates, MolecularCoordinates)):
            coords.position, current_energy, current_dict = \
                lbfgs.minimise(func_grad=self.potential.function_gradient,
                               initial_position=coords.position,
                               bounds=coords.bounds,
                               conv_crit=self.steepest_descent_conv_crit)
            return coords.position, current_energy, current_dict
        # For ML this can span the whole normalised feature space
        # so to prevent errant paths we do it repeatedly over small range
        for i in range(50):
            # Get local bounds centered on current position to limit the LBFGS
            local_bounds = self.get_local_bounds(coords)
            # Perform local minimisation with these bounds
            coords.position, current_energy, current_dict = \
                lbfgs.minimise(func_grad=self.potential.function_gradient,
                               initial_position=coords.position,
                               bounds=local_bounds,
                               conv_crit=self.steepest_descent_conv_crit)
            # Check if converged at edges of local bounds or inside
            local_bounds = np.asarray(local_bounds)
            l_below_bounds = np.invert(coords.position > local_bounds[:, 0])
            l_above_bounds = np.invert(coords.position < local_bounds[:, 1])
            # Inside bounds so accept the converged point
            if not np.any(l_below_bounds | l_above_bounds) and \
                    np.all(np.abs(current_dict['grad']) <
                           self.steepest_descent_conv_crit):
                return coords.position, current_energy, current_dict
            # Is at bounds, but could be allowed if the bounds are just
            # those imposed by coordinates
            else:
                # Get the active bounds w.r.t the coordinates
                below_bounds, above_bounds = coords.active_bounds()
                # If active bounds are all at total bounds then accept
                if np.all(below_bounds == l_below_bounds) and \
                        np.all(above_bounds == l_above_bounds) and \
                        (np.any(l_below_bounds) or np.any(l_above_bounds)):
                    return coords.position, current_energy, current_dict
        return coords.position, current_energy, current_dict

    def find_pushoff(self, transition_state: type,
                     eigenvector: NDArray) -> tuple[NDArray, NDArray]:
        """ Given the transition state location and direction of its single
            negative eigenvalue. We compute a step length in this direction
            before beginning a steepest-descent path.
            Returns the points to begin steepest-descent paths in the forwards
            and backwards direction """
        #  Find energy at transition state
        ts_energy = self.potential.function(transition_state.position)
        ts_position = transition_state.position.copy()
        # Set increment to a small percentage of the original pushoff
        increment = self.pushoff/10.0
        # Get eigenvector in both forwards and backwards directions
        # For atomistic systems we make pushoff equal to max atom displacement
        if isinstance(transition_state, (AtomicCoordinates,
                                         MolecularCoordinates)):
            max_displacement = \
                np.max(np.linalg.norm(eigenvector.reshape(-1, 3), axis=1))
            plus_eigenvector = eigenvector*(self.pushoff/max_displacement)
        else:
            plus_eigenvector = eigenvector
        neg_eigenvector = -1.0*eigenvector
        # Find pushoff where energy decreases
        found_pushoff = False
        for i in range(10):
            transition_state.position = self.do_pushoff(ts_position,
                                                        plus_eigenvector,
                                                        increment, i)
            transition_state.move_to_bounds()
            current_energy, current_grad = \
                self.potential.function_gradient(transition_state.position)
            if (ts_energy > current_energy) and \
               (np.max(current_grad) > 5.0*self.steepest_descent_conv_crit):
                positive_x = transition_state.position.copy()
                found_pushoff = True
                break
        # May be due to imprecise transition state location, but a
        # small push may still allow SD paths to get both connected minima
        if not found_pushoff:
            transition_state.position = self.do_pushoff(ts_position,
                                                        plus_eigenvector,
                                                        increment, 20)
            transition_state.move_to_bounds()
            positive_x = transition_state.position.copy()
            self.failure = 'pushoff'
        # Then the backwards direction
        found_pushoff = False
        for i in range(10):
            transition_state.position = self.do_pushoff(ts_position,
                                                        neg_eigenvector,
                                                        increment, i)
            transition_state.move_to_bounds()
            current_energy, current_grad = \
                self.potential.function_gradient(transition_state.position)
            if (ts_energy > current_energy) and \
               (np.max(current_grad) > 5.0*self.steepest_descent_conv_crit):
                negative_x = transition_state.position.copy()
                found_pushoff = True
                break
        if not found_pushoff:
            transition_state.position = self.do_pushoff(ts_position,
                                                        neg_eigenvector,
                                                        increment, 20)
            transition_state.move_to_bounds()
            negative_x = transition_state.position.copy()
            self.failure = 'pushoff'
        # Reset to initial position before return
        transition_state.position = ts_position
        return positive_x, negative_x

    def do_pushoff(self, ts_position: type, eigenvector: NDArray,
                   increment: float, iteration: int) -> NDArray:
        """ Take a step along the pushoff.
            Return the new position after performing pushoff """
        return ts_position + (increment*iteration)*eigenvector

    def generate_random_vector(self, ndim: int) -> NDArray:
        """ Return random uniform vector on sphere """
        rand_vec = np.random.rand(ndim)-0.5
        return rand_vec / np.linalg.norm(rand_vec)

    def project_onto_bounds(self, vector: NDArray, lower_bounds: NDArray,
                            upper_bounds: NDArray) -> NDArray:
        """ Project vector back to within the bounds if pointing outside """
        # If at upper bounds, and positive then make 0
        for i in range(vector.size):
            if lower_bounds[i] and vector[i] < 0.0:
                vector[i] = 0.0
            if upper_bounds[i] and vector[i] > 0.0:
                vector[i] = 0.0
        return vector / np.linalg.norm(vector)

    def test_convergence(self, position: NDArray, lower_bounds: NDArray,
                         upper_bounds: NDArray) -> bool:
        """ Test for convergence accounting for active bounds
            Only consider the gradient components in dimensions not at bounds
            Return True if points passes convergence test """
        grad = self.potential.gradient(position)
        # Find the directions at bounds
        all_bounds = np.column_stack((lower_bounds, upper_bounds))
        # Get the gradient for just the dimensions not at bounds
        non_bounds_dimensions = np.where(np.any(all_bounds, axis=0))[0]
        if non_bounds_dimensions.size > 0:
            grad[non_bounds_dimensions] = 0.0
        if np.max(np.abs(grad)) < self.ts_conv_crit:
            return True
        return False

    def update_eigenvector_bounds(self, lower_bounds: NDArray,
                                  upper_bounds: NDArray) -> None:
        """ Update the bounds on eigenvector to ensure it remains in certain
            portion of the sphere that points within the function bounds """
        for i in range(lower_bounds.size):
            if upper_bounds[i]:
                self.eigenvector_bounds[i] = (-inf, 0.0)
            elif lower_bounds[i]:
                self.eigenvector_bounds[i] = (0.0, inf)
            else:
                self.eigenvector_bounds[i] = (-inf, inf)

    def rayleigh_ritz_function_gradient(self, vec: NDArray,
                                        *args: list) -> tuple[float, NDArray]:
        """ Evaluate the Rayleigh-Ritz ratio to find the value of the
            eigenvalue for a given eigenvector vec computed at point
            coords, and the gradient with respect to changes in vec """
        displacement = 1e-3
        central_point = np.array(args)
        if np.any(np.isnan(vec)):
            return 0.0, np.zeros(np.size(vec), dtype=float)
        if np.linalg.norm(vec) == 0.0:
            return 0.0, np.zeros(np.size(vec), dtype=float)
        if self.remove_trans_rot:
            vec /= np.linalg.norm(vec)
            vec = self.remove_zero_eigenvectors(vec, central_point)
        vec /= np.linalg.norm(vec)
        # Gradient of the true potential with small displacements along y
        grad_plus = self.potential.gradient(central_point+(displacement*vec))
        grad_minus = self.potential.gradient(central_point-(displacement*vec))
        delta_grad = grad_plus - grad_minus
        # Rayleigh Ritz function
        f_val = np.dot(delta_grad, vec)/(2.0*displacement)
        # Rayleigh ritz gradient
        grad = (delta_grad/displacement) - (2.0*f_val*vec)
        # If specified remove overall translation and rotation eigevectors
        if self.remove_trans_rot:
            grad = self.remove_zero_eigenvectors(grad, central_point)
        return f_val, grad

    def remove_zero_eigenvectors(self, vec: NDArray,
                                 position: NDArray) -> NDArray:
        """ Make vec orthogonal to eigenvectors of the Hessian corresponding
            to overall translations and rotations at given position """
        vec = vec.reshape(-1, 3)
        n_atoms = float(vec.shape[0])
        centre_of_mass = position.reshape(-1, 3).sum(0) / n_atoms
        centered_coords = position.reshape(-1, 3) - centre_of_mass
        vdot = np.zeros(6, dtype=float)
        # Attempt to orthogonalise w.r.t overall translation and rotation
        for attempts in range(100):
            vdot[:] = 0.0
            # Translations first
            for i in range(3):
                vec_com = vec[:, i].sum() / n_atoms
                vdot[i] = vec_com * np.sqrt(n_atoms)
                vec[:, i] -= vec_com
            # Then rotations
            sign = [1.0, -1.0, 1.0]
            # The set of axes to use x(0), y(1) and z(2)
            xyz = [[1, 2], [0, 2], [0, 1]]
            for i in range(3):
                vec1 = vec[:, xyz[i][0]] * centered_coords[:, xyz[i][1]]
                vec2 = vec[:, xyz[i][1]] * centered_coords[:, xyz[i][0]]
                term1 = sign[i] * np.sum(vec1 - vec2)
                term2 = np.sum((centered_coords[:, xyz[i][0]])**2 +
                               (centered_coords[:, xyz[i][1]])**2)
                if term2 > 0.0:
                    vdot[3+i] = np.abs(term1) / np.sqrt(term2)
                    term3 = term1 / term2
                    vec[:, xyz[i][0]] -= sign[i] * term3 * \
                        centered_coords[:, xyz[i][1]]
                    vec[:, xyz[i][1]] += sign[i] * term3 * \
                        centered_coords[:, xyz[i][0]]
            # Successfully orthogonalised so exit
            if np.max(vdot) <= 1e-6:
                break
        return vec.reshape(-1)

    def subspace_function_gradient(self, position: NDArray,
                                   *args: list) -> tuple[float, NDArray]:
        """ Return the function value and the component of the gradient
            perpendicular to self.eigenvector at point coords """
        eigenvector = np.array(args)
        f_val, grad = self.potential.function_gradient(position)
        subspace_grad = self.perpendicular_component(grad, eigenvector)
        return f_val, subspace_grad
