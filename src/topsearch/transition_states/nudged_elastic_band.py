""" Module that performs double ended transition states by the nudged
    elastic band. Double ended searches take two endpoint minima and
    aim to locate a minimum energy pathway between them, the maxima
    on this path are candidates for transition states. """

import numpy as np
from nptyping import NDArray
from topsearch.minimisation import lbfgs
from topsearch.data.coordinates import MolecularCoordinates


class NudgedElasticBand:

    """
    Description
    ------------

    Takes in the coordinates of two endpoint minima and perform a double-ended
    transition state search using the nudged elastic band
    Produces an initial linear interpolation with linear_interpolation
    Optimise this initial guess using minimise_interpolation
    And pick out transition state candidates using find_ts_candidates
    The whole process is performed by run

    Attributes
    ----------

    potential : class instance
        The function which we are finding minimum energy paths on
    minimiser : class instance
        The local minimiser used for optimising the band
    force_constant : float
        The value of the force constant used in harmonic part of band energy
    image_density : float
        The density of images, per distance unit, in the initial band
        interpolations
    original_image_density : float
        Store of image_density to return image_density to value once updated
    max_images : int
        The maximum allowed number of images in the nudged elastic band
    force_constants : np array
        An array of force constants for the nudged elastic band, one per image
    n_images : int
        The number of images in the given band after initial interpolation
    band_bounds : list
        The bounds on the function range that can be passed to optimiser
        Same as potential bounds, but one per image
    """

    def __init__(self, potential: type,
                 force_constant: float,
                 image_density: float,
                 max_images: int,
                 neb_conv_crit: float) -> None:
        self.potential = potential
        self.force_constant = force_constant
        self.image_density = image_density
        self.original_image_density = image_density
        self.max_images = max_images
        self.neb_conv_crit = neb_conv_crit
        self.force_constants = None
        self.n_images = None
        self.band_bounds = None
        self.neb_count = 0

    def run(self, coords1: type, coords2: NDArray, attempts: int = 0,
            permutation: NDArray = None) -> tuple[NDArray, NDArray]:
        """ Complete double-ended transition state search when provided with
            two minima. Returns an array of transition state candidates taken
            as maxima on the approximate minimum energy path """
        # Get the initial band
        band = self.initial_interpolation(coords1, coords2, attempts,
                                          permutation)
        # Minimise the initial band to get approximate minimum energy path
        band = self.minimise_interpolation(band)
        # Find local maxima of the band as transition state candidates
        candidates, positions = self.find_ts_candidates(band)
        # Return the transition state candidates and their positions
        self.neb_count += 1
        return candidates, positions

    def minimise_interpolation(self, band: NDArray) -> NDArray:
        """ Perform minimisation of the given interpolation between
            endpoints using box-contrained LBFGS and return the
            optimised band """
        # Require a 1d array for scipy lbfgs
        band = band.flatten()
        optimised_band, f_val, r_dict = \
            lbfgs.minimise(func_grad=self.band_function_gradient,
                           initial_position=band,
                           bounds=self.band_bounds,
                           conv_crit=self.neb_conv_crit)
        # Reshape into 2d array, one image per row
        return np.reshape(optimised_band, (self.n_images, -1))

    def update_image_density(self, attempts: int) -> None:
        """ Update the image density parameter """
        self.image_density = self.original_image_density*1.5*attempts

    def revert_image_density(self) -> None:
        """ Revert the image density to its original value """
        self.image_density = self.original_image_density

    def initial_interpolation(self, coords1: type, coords2: NDArray,
                              attempts: int, permutation: NDArray) -> NDArray:
        """ Return initial interpolation band using the appropriate number
            of images based on previous attempts. Set force constants and
            bounds attributes for use in other methods """

        # If retrying connection then use increased image density
        if attempts > 0:
            self.update_image_density(attempts)
        # Perform the type of interpolation specified by number of attempts
        if isinstance(coords1, MolecularCoordinates):
            band = self.dihedral_interpolation(coords1, coords2, permutation)
        else:
            band = self.linear_interpolation(coords1, coords2)
        # If changed parameters then revert back
        if attempts > 0:
            self.revert_image_density()
        # And initialise bounds array given self.n_images is now set
        self.band_bounds = coords1.bounds*self.n_images
        # With the band known calculate the force constants
        self.get_force_constants()
        return band

    def linear_interpolation(self, coords1: type, coords2: NDArray) -> NDArray:
        """ Produce a linear interpolation between two points with a number of
            images specified by self.image_density """
        # Set the number of images based on given image density
        dist = np.linalg.norm(coords1.position-coords2)
        self.n_images = int(self.image_density*dist)
        # Can't have too few images
        if self.n_images < 10:
            self.n_images = 10
        # Don't allow band to have above the maximum specified images
        elif self.n_images > self.max_images:
            self.n_images = self.max_images
        # Make the linear interpolation
        band = np.zeros((self.n_images, coords1.ndim), dtype=float)
        direction_vec = (coords2-coords1.position)/(self.n_images-1)
        for i in range(self.n_images):
            band[i, :] = coords1.position+(direction_vec*i)
        return band

    def dihedral_interpolation(self, coords1: type, coords2: NDArray,
                               permutation: NDArray) -> NDArray:
        """ Interpolate linearly in the space of dihedrals, angles and
            bond lengths, which will be much more appropriate for molecules """
        # Set the number of images based on given image density
        dist = np.linalg.norm(coords1.position-coords2)
        self.n_images = int(self.image_density*dist)
        # Can't have too few images
        if self.n_images < 10:
            self.n_images = 10
        # Don't allow band to have above the maximum specified images
        elif self.n_images > self.max_images:
            self.n_images = self.max_images
        # Get the angular representation of the conformation and its bonds
        represent1 = coords1.get_bond_angle_info()
        bond_network = coords1.get_bonds()
        # Get the angular representation of same bonds, including permutations
        initial_coords1 = coords1.position.copy()
        coords1.position = coords2
        represent2 = \
            coords1.get_specific_bond_angle_info(represent1[0],
                                                 represent1[2],
                                                 represent1[4],
                                                 permutation)
        coords1.position = initial_coords1
        # Calculate the direction differences to create the band
        bond_differences = (np.asarray(represent2[0]) -
                            np.asarray(represent1[1]))/(self.n_images-1)
        angle_differences = (np.asarray(represent2[1]) -
                             np.asarray(represent1[3]))/(self.n_images-1)
        dihedral_differences = []
        # Accounting for periodicity in dihedrals
        for i in range(len(represent1[5])):
            plus_dist = np.abs(represent2[2][i] - represent1[5][i])
            minus_dist = np.abs(represent2[2][i] + 360.0 - represent1[5][i])
            if plus_dist < minus_dist:
                dihedral_differences.append(represent2[2][i]-represent1[5][i])
            else:
                dihedral_differences.append(represent2[2][i] + 360.0
                                            - represent1[5][i])
        dihedral_differences = \
            np.asarray(dihedral_differences)/(self.n_images-1)
        # Make the linear interpolation
        band = np.zeros((self.n_images, coords1.ndim), dtype=float)
        band[0, :] = coords1.position
        for i in range(1, self.n_images):
            coords1.change_bond_lengths(represent1[0],
                                        bond_differences,
                                        bond_network)
            coords1.change_bond_angles(represent1[2],
                                       -1.0*angle_differences,
                                       bond_network)
            coords1.change_dihedral_angles(represent1[4],
                                           dihedral_differences,
                                           bond_network)
            band[i, :] = coords1.position.flatten()
        coords1.position = initial_coords1
        return band

    def get_force_constants(self) -> None:
        """ Get the array of force constants for the harmonic potential """
        self.force_constants = np.zeros((self.n_images-1), dtype=float)
        self.force_constants.fill(self.force_constant)

    def find_ts_candidates(self, band: NDArray) -> tuple[NDArray, NDArray]:
        """ Find the local maxima of the nudged elastic band and return the
        indices of these candidate transition states and their coordinates """
        # Compute the energy of each image in the band
        band_potential_energies = self.band_potential_function(band)[1]
        # Initialise arrays to store any transition state candidates we find
        candidates = np.empty((0), dtype=int)
        positions = np.empty((0, 0), dtype=float)
        # Search through the 1D array of energies to find local maxima
        for i in range(1, self.n_images-1):
            if (band_potential_energies[i] >= band_potential_energies[i+1] and
                  band_potential_energies[i] >= band_potential_energies[i-1]):
                # Store index and coordinates of transition state candidate
                candidates = np.append(candidates, i)
                positions = np.append(positions, band[i, :])
        return candidates, \
            np.reshape(positions, (np.size(candidates), band[0, :].size))

    def find_tangent_differences(self, band: NDArray,
                                 energies: NDArray) -> NDArray:
        """ Compute the tangent vectors between consecutive images
            in the band, these are calculated from lower to higher
            connected images in the Upwind scheme. Returns array
            of tangents, one for each image (apart from endpoints) """

        # Create the 2D array to place the tangents in
        tangents = np.zeros((self.n_images-2, band[0, :].size), dtype=float)
        # Only use a tangent to the higher connected image
        position_differences = -1.0*np.diff(band, axis=0)
        # Compute the differences between energies of images
        energy_differences = np.diff(energies, axis=0)
        # Compute the signs of energy differences to show changing energy
        energy_change_sign = np.sign(energy_differences).astype(int)
        for i in range(1, self.n_images-1):
            # Band has increasing energy so select following image
            if energy_change_sign[i-1]+energy_change_sign[i] >= 1:
                tangents[i-1, :] = position_differences[i]
            # Band has decreasing energy so select previous image
            elif energy_change_sign[i-1]+energy_change_sign[i] <= -1:
                tangents[i-1, :] = position_differences[i-1]
            # Band has local maximum, local minimum or it is flat
            elif energy_change_sign[i-1]+energy_change_sign[i] == 0:
                #  Band is flat
                if 0 in (energy_change_sign[i-1], energy_change_sign[i]):
                    tangents[i-1, :] = position_differences[i-1]
                # Local minimum or local maximum
                else:
                    vec1 = position_differences[i]
                    vec2 = position_differences[i-1]
                    abs_energy_differences = \
                        np.abs(energy_differences[i-1:i+1])
                    v_max = max(abs_energy_differences)
                    v_min = min(abs_energy_differences)
                    if energies[i+1] >= energies[i-1]:
                        tangents[i-1, :] = vec1*v_max + vec2*v_min
                    else:
                        tangents[i-1, :] = vec1*v_min + vec2*v_max
        # Normalise the tangent vectors
        row_sums = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = np.divide(tangents, row_sums, out=np.zeros_like(
            tangents), where=row_sums != 0)
        return tangents

    def band_potential_function(self, band: NDArray) -> tuple[float, NDArray]:
        """ Return the total energy of the band, and an array giving the
            true function evaluated at each image of the band """
        n_images = int(np.size(band, axis=0))
        band_potential_energies = np.zeros((n_images, 1), dtype=float)
        band_potential_energies = \
            np.apply_along_axis(self.potential.function, 1, band)
        potential_function = np.sum(band_potential_energies)
        return potential_function, band_potential_energies

    def band_function_gradient(self, band: NDArray) -> tuple[float, NDArray]:
        """ Return the total energy and the gradient of the nudged elastic
            band, this involves the summation of the true function and
            a series of harmonic terms between adjacent images """

        # Reshape the band into its images
        band = np.reshape(band, (self.n_images, -1))
        # Compute the components of the true potential
        potential_gradient = \
            np.zeros((self.n_images, band[0, :].size), dtype=float)
        band_potential_energies = np.zeros((self.n_images, 1), dtype=float)
        for i in range(self.n_images):
            f_val, grad = self.potential.function_gradient(band[i, :])
            potential_gradient[i, :] = grad
            band_potential_energies[i] = f_val
        # Calculate the function value as sum over all images
        potential_function = np.sum(band_potential_energies)
        # Ensure no true gradient on the endpoint images
        potential_gradient[0, :].fill(0.0)
        potential_gradient[-1, :].fill(0.0)

        # Find the tangents to the band
        tau = self.find_tangent_differences(band, band_potential_energies)

        # Compute the components of the harmonic potential parallel to tangents
        g_parallel = np.zeros((self.n_images, band[0, :].size), dtype=float)
        # Get separation vector between adjacent images
        differences = -1.0*np.diff(band, axis=0)
        # Compute the distances
        distances = np.linalg.norm(differences, axis=1)
        # Calculate the function
        harmonic_function = 0.5*np.sum((distances**2)*self.force_constants)
        # And its gradient parallel to the tangent
        g_parallel[1:-1, :] = tau * \
            (-1.0*np.diff(distances)*self.force_constants[:-1]).reshape(-1, 1)

        # Combine the perpendicular component of the true potential
        # and the parallel component of the harmonic potential
        band_gradient = np.zeros((self.n_images, band[0, :].size), dtype=float)
        for i in range(1, self.n_images-1):
            band_gradient[i, :] = g_parallel[i, :] + \
                self.perpendicular_component(potential_gradient[i, :],
                                             tau[i-1, :])
        return harmonic_function+potential_function, band_gradient.flatten()

    def perpendicular_component(self, vec1: NDArray, vec2: NDArray) -> NDArray:
        """ Return perpendicular component of vector vec1 relative to vec2 """
        vec2_magnitude = np.sum(vec2**2)
        if vec2_magnitude < 1e-13:
            return np.zeros(np.size(vec1))
        return vec1 - (np.dot(vec1, vec2)/vec2_magnitude)*vec2
