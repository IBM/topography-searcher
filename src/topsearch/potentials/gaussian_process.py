""" Module that contains the Gaussian process potential class """

import warnings
import sys
import numpy as np
from nptyping import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, Kernel, \
      StationaryKernelMixin, NormalizedKernelMixin, Hyperparameter, _check_length_scale
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform, cdist, pdist
from scipy.spatial.transform import Rotation as R
from .potential import Potential
warnings.filterwarnings('ignore')


class GaussianProcess(Potential):

    """
    Description
    -------------

    The log-marginal likelihood for Gaussian process regression.
    Need to specify the kernel used to calculate covariance, and provide
    training data

    Attributes
    -----------
    model_data : class
        The class containing the training and response data we will fit
    kernel_choice : str
        The choice of kernel, can be 'RBF' or 'Matern'
    kernel_bounds : list
        Limits on the kernel lengthscales and noise (final element)
    standardise_training : bool
        Choose whether to standardise the training data before GP fit
    standardise_response : bool
        Choose whether to standardise the response data before GP fit
    limit_highest_data : logical
        Specifies if we should limit the largest response value
        Useful in molecular applications where the steep repuslive wall
        gives huge values
    matern_nu : float
        The nu parameter of the Matern kernel
    gpr : class
        The sklearn gaussian process object
    """

    def __init__(self, model_data: type,
                 kernel_choice: str,
                 kernel_bounds: list,
                 standardise_training: bool = False,
                 standardise_response: bool = True,
                 limit_highest_data: bool = False,
                 matern_nu: float = None,
                 coords: type = None,
                 reference: NDArray = None) -> None:
        self.atomistic = False
        self.model_data = model_data
        self.kernel_choice = kernel_choice
        self.kernel_bounds = kernel_bounds
        self.standardise_training = standardise_training
        self.standardise_response = standardise_response
        self.limit_highest_data = limit_highest_data
        self.matern_nu = matern_nu
        self.coords = coords
        self.reference = reference
        self.prepare_training_data()
        self.initialise_gaussian_process()

    def prepare_training_data(self):
        """ Modify the training data that is provided to the Gaussian process
            to normalise training, response and limit their values """
        if self.standardise_training:
            self.model_data.standardise_training()
        if self.limit_highest_data:
            self.model_data.limit_response_maximum(
                5.0*np.abs(np.mean(self.model_data.response)))
        if self.standardise_response:
            self.model_data.standardise_response()

    def initialise_gaussian_process(self, n_restarts: int = 50) -> None:
        """ Initialise the Gaussian process from sklearn
            Returns a gpr object as an attribute of this class """
        kernel = self.initialise_kernel()
        self.gpr = GaussianProcessRegressor(
            kernel=kernel, normalize_y=False,
            n_restarts_optimizer=n_restarts, random_state=0)
        self.gpr.fit(self.model_data.training, self.model_data.response)

    def initialise_kernel(self) -> None:
        """ Create a specified kernel for use in a Gaussian process """
        initial_lengthscales = np.zeros(len(self.kernel_bounds)-1, dtype=float)
        for i, bounds in enumerate(self.kernel_bounds[:-1]):
            initial_lengthscales[i] = 0.5 * \
                (bounds[1]-bounds[0]) + bounds[0]
        initial_noise = 0.5 * \
            (self.kernel_bounds[-1][1]-self.kernel_bounds[-1][0]) + \
            self.kernel_bounds[-1][0]
        if self.kernel_choice == 'RBF':
            kernel = RBF(initial_lengthscales, self.kernel_bounds[:-1]) + \
                     WhiteKernel(initial_noise, self.kernel_bounds[-1])
        elif self.kernel_choice == 'Matern':
            kernel = Matern(initial_lengthscales,
                            self.kernel_bounds[:-1], self.matern_nu) + \
                     WhiteKernel(initial_noise, self.kernel_bounds[-1])
        elif self.kernel_choice == 'PermInvariant':
            kernel = ClusterCartKernel2(initial_lengthscales,
                                        self.kernel_bounds[:-1],
                                        coords=self.coords,
                                        reference=self.reference) + \
                     WhiteKernel(initial_noise, self.kernel_bounds[-1])
        else:
            sys.exit(0)
        return kernel

    def add_data(self, new_training: NDArray,
                 new_response: NDArray) -> None:
        """ Add data to the model data accounting for standardisation """
        if self.standardise_training:
            self.model_data.unstandardise_training()
        if self.standardise_response:
            self.model_data.unstandardise_response()
        self.model_data.append_data(new_training, new_response)
        if self.standardise_training:
            self.model_data.standardise_training()
        if self.standardise_response:
            self.model_data.standardise_response()

    def lowest_point(self) -> float:
        """ Find the lowest point in the current dataset """
        if self.standardise_response:
            self.model_data.unstandardise_response()
        lowest = np.min(self.model_data.response)
        if self.standardise_response:
            self.model_data.standardise_response()
        return lowest

    def rescale_high_energies(self, upper_limit: float, scale: float) -> None:
        """ Call the rescaling of high energy routine with appropriate
            normalisation """
        if self.standardise_response:
            self.model_data.unstandardise_response()
        self.model_data.scale_response_maximum(upper_limit, scale)
        if self.standardise_response:
            self.model_data.standardise_response()

    def write_fit(self) -> None:
        """ Write the hyperparameters of the best GP fit """
        with open('logfile', 'a', encoding="utf-8") as outfile:
            outfile.write("Best GP fit: ")
            for i in self.gpr.kernel_.theta:
                outfile.write(f"{i} ")
            outfile.write('\n')

    def get_score(self) -> float:
        """ Get the R^2 score of the gp fit """
        return self.gpr.score(self.model_data.training,
                              self.model_data.response)

    def function(self, position: NDArray) -> float:
        """ Return the mean of the GP fit at position """
        return self.gpr.predict(position.reshape(1, -1))[0]

    def function_and_std(self, position: NDArray) -> float:
        """ Return the variance of the GP fit at position """
        return self.gpr.predict(position.reshape(1, -1), return_std=True)

    def refit_model(self, n_restarts: int = 50) -> None:
        """ Refit the GP model based on the current model_data """
        self.gpr.n_restarts_optimizer = n_restarts
        self.gpr.fit(self.model_data.training, self.model_data.response)

    def update_bounds(self, scaling: float) -> None:
        """ Change the lengthscale bounds for kernel """
        length_bounds = np.exp(self.gpr.kernel.bounds)[:-1, :]
        length_bounds[:, 0] *= 1-scaling
        length_bounds[:, 1] *= 1+scaling
        lengthscale_bounds = [tuple(i) for i in length_bounds.tolist()]
        kernel_params = {"k1__length_scale_bounds": lengthscale_bounds}
        self.gpr.kernel.set_params(**kernel_params)


# class ClusterKernel(Kernel):
#     """ An RBF kernel in which the distance is computed
#         acccounting for permutation in atomistic systems """
#     def __init__(self, length_scale=1.0):
#              self.length_scale = length_scale
#     def __call__(self, X, Y=None):
#         if Y is None:
#             Y = X
#         return np.inner(X, X if Y is None else Y) ** 2
#     def diag(self, X):
#         return np.ones(X.shape[0])
#     def is_stationary(self):
#         return True


class ClusterKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Radial basis function kernel (aka squared-exponential kernel).

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.


    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * RBF(1.0)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8354..., 0.03228..., 0.1322...],
           [0.7906..., 0.0652..., 0.1441...]])
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 atom_array=None, coords=None):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.coords = coords
        self.atom_array = atom_array
        self.get_permutable_atoms()

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
#        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            # Compute the pairwise distances
            dists = self.p_pairwise(X)
            print("dists = ", dists)
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            print("Cov matrix = ", K)
#            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = self.c_pairwise(X, Y)
            K = np.exp(-0.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale**2
                )
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0]
            )
        
    def permutational_alignment(self, x1: NDArray, x2: NDArray) -> tuple:
        """ Hungarian algorithm adapted to solve the linear assignment problem
            for each atom of each distinct element. Finds the optimal
            permutation of all atoms respecting the atomic species """

        # Get coordinates to apply permutation to
        permuted_x2 = x2.copy()
        # Distance for permutation of first species
        coords1_element = np.take(x1.reshape(-1, 3),
                                  self.permutable_atoms1, axis=0)
        coords2_element = np.take(x2.reshape(-1, 3),
                                  self.permutable_atoms1, axis=0)
#        print("Permutable_atoms = ", self.permutable_atoms1)
#        print("Coords1 subset: ", coords1_element)
#        print("Coords2 subset: ", coords2_element)
        # Calculate distance matrix for these subset of atoms
        dist_matrix = distance_matrix(coords1_element,
                                      coords2_element)
#        print("Dist matrix = ", dist_matrix)
        # Optimal permutational alignment
        col_ind = linear_sum_assignment(dist_matrix**2)[1]
#        print("Col_ind: ", col_ind)
        # Modify the coordinates to nearest permutation
        for idx, atom in enumerate(self.permutable_atoms1):
#            print("idx, atom, col_ind[idx] = ", idx, atom, col_ind[idx])
            permuted_x2[atom*3:(atom*3)+3] = \
                x2[self.permutable_atoms1[col_ind[idx]]*3:
                   (self.permutable_atoms1[col_ind[idx]]*3)+3]

        # Distance for permutation of second species
        coords1_element = np.take(x1.reshape(-1, 3),
                                  self.permutable_atoms2, axis=0)
        coords2_element = np.take(x2.reshape(-1, 3),
                                  self.permutable_atoms2, axis=0)
        # Calculate distance matrix for these subset of atoms
        dist_matrix = distance_matrix(coords1_element,
                                      coords2_element)
        # Optimal permutational alignment
        col_ind = linear_sum_assignment(dist_matrix**2)[1]
        # Modify coordinates to nearest permutation
        for idx, atom in enumerate(self.permutable_atoms2):
            permuted_x2[atom*3:(atom*3)+3] = \
                x2[self.permutable_atoms2[col_ind[idx]]*3:
                   (self.permutable_atoms2[col_ind[idx]]*3)+3]
        return permuted_x2

    def get_permutable_atoms(self) -> None:
        """ Set the permutable atoms for use in the permutationally-aligned
            distance """
        # Get atoms of element 0
        self.permutable_atoms1 = \
            [i for i, x in enumerate(self.atom_array) if x == 0]
        # Get atoms of element 1
        self.permutable_atoms2 = \
            [i for i, x in enumerate(self.atom_array) if x == 1]

    def get_aligned_coordinates(self, x1, x2):
        """ Get the second z matrix permutationally aligned to the first """
        cart1 = self.coords.zmatrix_to_cartesian(x1)
        cart2 = self.coords.zmatrix_to_cartesian(x2)
#        print("Cart1: ", cart1)
#        print("Cart2: ", cart2)
        perm_cart2 = self.permutational_alignment(cart1, cart2)
        return self.coords.cartesian_to_zmatrix(perm_cart2)

    def p_pairwise(self, X):
        """ Compute the distance between array of points and itself """
        cov_matrix = np.zeros((X.shape[0], X.shape[0]), dtype=float)
        print("Cov matrix: ", cov_matrix)
        for i in range(0, X.shape[0]-1):
            for j in range(i+1, X.shape[0]):
                # Get the aligned coordinates
                perm_j = self.get_aligned_coordinates(X[i], X[j])
                # Compute the distance
                print("perm_j: ", perm_j)
                print("X[i] = ", X[i])
                cov_matrix[i, j] = np.linalg.norm((X[i] / self.length_scale) -
                                                  (perm_j / self.length_scale))**2
                cov_matrix[j, i] = cov_matrix[i, j]
                print("element = ", cov_matrix[i, j])
        print("cov_matrix = ", cov_matrix)
        return cov_matrix

    def c_pairwise(self, X, Y):
        """ Compute the pairwise distance between two arrays """
        cov_matrix = np.zeros((X.shape[0], Y.shape[0]), dtype=float)
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                # Get the aligned coordinates
                perm_j = self.get_aligned_coordinates(X[i], Y[j])
                # Compute the distance
                cov_matrix[i, j] = np.linalg.norm((X[i] / self.length_scale) -
                                                  (perm_j / self.length_scale))**2
        return cov_matrix


class ClusterCartKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Radial basis function kernel (aka squared-exponential kernel) """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 atom_array=None, coords=None):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.atom_array = atom_array
        self.get_permutable_atoms()
        self.coords = coords

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient """
        X = np.atleast_2d(X)
#        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            # Compute the pairwise distances
            dists = self.p_pairwise(X)
#            print("dists = ", dists)
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
#            print("Cov matrix = ", K)
#            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = self.c_pairwise(X, Y)
            K = np.exp(-0.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale**2
                )
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0]
            )

    def permutational_alignment(self, x1: NDArray, x2: NDArray) -> tuple:
        """ Hungarian algorithm adapted to solve the linear assignment problem
            for each atom of each distinct element. Finds the optimal
            permutation of all atoms respecting the atomic species """

        # Get the full Cartesian coordinates
        c1 = self.coords.hybrid_to_cartesian(x1)
        c2 = self.coords.hybrid_to_cartesian(x2)
        # Get coordinates to apply permutation to
        permuted_c2 = c2.copy()
        # Distance for permutation of first species
        coords1_element = np.take(c1.reshape(-1, 3),
                                  self.permutable_atoms1, axis=0)
        coords2_element = np.take(c2.reshape(-1, 3),
                                  self.permutable_atoms1, axis=0)
        # Calculate distance matrix for these subset of atoms
        dist_matrix = distance_matrix(coords1_element,
                                      coords2_element)
        # Optimal permutational alignment
        col_ind = linear_sum_assignment(dist_matrix**2)[1]
#        print("col_ind1 = ", col_ind)
        # Modify the coordinates to nearest permutation
        for idx, atom in enumerate(self.permutable_atoms1):
            permuted_c2[atom*3:(atom*3)+3] = \
                c2[self.permutable_atoms1[col_ind[idx]]*3:
                   (self.permutable_atoms1[col_ind[idx]]*3)+3]

        # Distance for permutation of second species
        coords1_element = np.take(c1.reshape(-1, 3),
                                  self.permutable_atoms2, axis=0)
        coords2_element = np.take(c2.reshape(-1, 3),
                                  self.permutable_atoms2, axis=0)
        # Calculate distance matrix for these subset of atoms
        dist_matrix = distance_matrix(coords1_element,
                                      coords2_element)
        # Optimal permutational alignment
        col_ind = linear_sum_assignment(dist_matrix**2)[1]
#        print("col_ind2 = ", col_ind)
        # Modify coordinates to nearest permutation
        for idx, atom in enumerate(self.permutable_atoms2):
            permuted_c2[atom*3:(atom*3)+3] = \
                c2[self.permutable_atoms2[col_ind[idx]]*3:
                   (self.permutable_atoms2[col_ind[idx]]*3)+3]
        return self.coords.cartesian_to_hybrid(permuted_c2)

    def get_permutable_atoms(self) -> None:
        """ Set the permutable atoms for use in the permutationally-aligned
            distance """
        # Get atoms of element 0
        self.permutable_atoms1 = \
            [i for i, x in enumerate(self.atom_array) if x == 0 and i != 0]
        # Get atoms of element 1
        self.permutable_atoms2 = \
            [i for i, x in enumerate(self.atom_array) if x == 1 and i != 0]

    def p_pairwise(self, X):
        """ Compute the distance between array of points and itself """
        cov_matrix = np.zeros((X.shape[0], X.shape[0]), dtype=float)
#        print("Cov matrix: ", cov_matrix)
        for i in range(0, X.shape[0]-1):
            for j in range(i+1, X.shape[0]):
                # Get the aligned coordinates
                perm_j = self.permutational_alignment(X[i], X[j])
                # Compute the distance
#                print("perm_j: ", perm_j)
#                print("X[i] = ", X[i])
                cov_matrix[i, j] = np.linalg.norm((X[i] / self.length_scale) -
                                                  (perm_j / self.length_scale))**2
                cov_matrix[j, i] = cov_matrix[i, j]
#                print("element = ", cov_matrix[i, j])
#        print("cov_matrix = ", cov_matrix)
        return cov_matrix

    def c_pairwise(self, X, Y):
        """ Compute the pairwise distance between two arrays """
        cov_matrix = np.zeros((X.shape[0], Y.shape[0]), dtype=float)
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                # Get the aligned coordinates
                perm_j = self.permutational_alignment(X[i], X[j])
                # Compute the distance
                cov_matrix[i, j] = np.linalg.norm((X[i] / self.length_scale) -
                                                  (perm_j / self.length_scale))**2
        return cov_matrix

class ClusterCartKernel_Original(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Radial basis function kernel (aka squared-exponential kernel)
       ALIGN ALL TO FIXED REFERENCE IN THIS """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 coords=None, reference=None):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.coords = coords
        self.get_permutable_atoms()
        self.reference = reference

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        # Align all X to the given reference
        for i in range(X.shape[0]):
            perm_i = self.permutational_alignment(self.reference, X[i, :])
            X[i,:] = perm_i
#        print("X' = ", X)
#        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            # Compute the pairwise distances
            dists = pdist(X / length_scale, metric='sqeuclidean')
#            print("dists = ", dists)
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
#            print("Cov matrix = ", K)
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale**2
                )
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0]
            )

    def permutational_alignment(self, x1: NDArray, x2: NDArray) -> tuple:
        """ Hungarian algorithm adapted to solve the linear assignment problem
            for each atom of each distinct element. Finds the optimal
            permutation of all atoms respecting the atomic species """

        # Get the full Cartesian coordinates
        c1 = self.coords.hybrid_to_cartesian(x1)
        c2 = self.coords.hybrid_to_cartesian(x2)
        # Get coordinates to apply permutation to
        permuted_c2 = c2.copy()
        # Distance for permutation of first species
        coords1_element = np.take(c1.reshape(-1, 3),
                                  self.permutable_atoms1, axis=0)
        coords2_element = np.take(c2.reshape(-1, 3),
                                  self.permutable_atoms1, axis=0)
        # Calculate distance matrix for these subset of atoms
        dist_matrix = distance_matrix(coords1_element,
                                      coords2_element)
        # Optimal permutational alignment
        col_ind = linear_sum_assignment(dist_matrix**2)[1]
#        print("col_ind1 = ", col_ind)
        # Modify the coordinates to nearest permutation
        for idx, atom in enumerate(self.permutable_atoms1):
            permuted_c2[atom*3:(atom*3)+3] = \
                c2[self.permutable_atoms1[col_ind[idx]]*3:
                   (self.permutable_atoms1[col_ind[idx]]*3)+3]

        # Distance for permutation of second species
        coords1_element = np.take(c1.reshape(-1, 3),
                                  self.permutable_atoms2, axis=0)
        coords2_element = np.take(c2.reshape(-1, 3),
                                  self.permutable_atoms2, axis=0)
        # Calculate distance matrix for these subset of atoms
        dist_matrix = distance_matrix(coords1_element,
                                      coords2_element)
        # Optimal permutational alignment
        col_ind = linear_sum_assignment(dist_matrix**2)[1]
#        print("col_ind2 = ", col_ind)
        # Modify coordinates to nearest permutation
        for idx, atom in enumerate(self.permutable_atoms2):
            permuted_c2[atom*3:(atom*3)+3] = \
                c2[self.permutable_atoms2[col_ind[idx]]*3:
                   (self.permutable_atoms2[col_ind[idx]]*3)+3]
        return self.coords.cartesian_to_hybrid(permuted_c2)

    def get_permutable_atoms(self) -> None:
        """ Set the permutable atoms for use in the permutationally-aligned
            distance """
        elements = list(set(self.coords.atom_labels))
        el1 = elements[0]
        el2 = elements[1]
        # Get atoms of element 0
        self.permutable_atoms1 = \
            [i for i, x in enumerate(self.coords.atom_labels) if x == el1 and i not in [0, 1, 2]]
        # Get atoms of element 1
        self.permutable_atoms2 = \
            [i for i, x in enumerate(self.coords.atom_labels) if x == el2 and i not in [0, 1, 2]]

class ClusterCartKernel2(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Radial basis function kernel (aka squared-exponential kernel).
       Project each data point onto the nearest isomer to fixed conformation """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 coords=None, reference=None):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.coords = coords
        self.get_permutable_atoms()
        self.reference = reference

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        # Align all X to the given reference
        for i in range(X.shape[0]):
            perm_i = self.alignment(self.reference, X[i, :])
            X[i, :] = perm_i
#        print("X' = ", X)
#        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            # Compute the pairwise distances
            dists = pdist(X / length_scale, metric='sqeuclidean')
#            print("dists = ", dists)
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
#            print("Cov matrix = ", K)
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
#            for i in range(Y.shape[0]):
#                perm_i = self.alignment(self.reference, Y[i, :])
#                Y[i, :] = perm_i
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale**2
                )
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0]
            )

    def alignment(self, x1: NDArray, x2: NDArray) -> tuple:
        """ Hungarian algorithm adapted to solve the linear assignment problem
            for each atom of each distinct element. Finds the optimal
            permutation of all atoms respecting the atomic species """

        # Get the full Cartesian coordinates
        c1 = self.coords.hybrid_to_cartesian(x1)
        c2 = self.coords.hybrid_to_cartesian(x2)
        # Get coordinates to apply permutation to
        initial_c2 = c2.copy()

        # Compute best distance from four initial reflections
        dist, c_ind1, c_ind2 = self.permutational_alignment(c1, c2)
        best_dist = dist
        best_c_ind1 = c_ind1
        best_c_ind2 = c_ind2
        best_c2 = initial_c2.copy()
        # Apply reflection
        c2 = self.apply_reflection(initial_c2)
        dist, c_ind1, c_ind2 = self.permutational_alignment(c1, c2)
        if dist < best_dist:
            best_dist = dist
            best_c_ind1 = c_ind1
            best_c_ind2 = c_ind2
            best_c2 = c2.copy()
        # Apply rotation
        c2 = self.apply_rotation(initial_c2)
        dist, c_ind1, c_ind2 = self.permutational_alignment(c1, c2)
        if dist < best_dist:
            best_dist = dist
            best_c_ind1 = c_ind1
            best_c_ind2 = c_ind2
            best_c2 = c2.copy()
        # Apply rotation and reflection
        c2 = self.apply_rotation(initial_c2)
        c2 = self.apply_reflection(c2)
        dist, c_ind1, c_ind2 = self.permutational_alignment(c1, c2)
        if dist < best_dist:
            best_dist = dist
            best_c_ind1 = c_ind1
            best_c_ind2 = c_ind2
            best_c2 = c2.copy()

        # Modify the coordinates to best permutation
        permuted_c2 = best_c2.copy()
        for idx, atom in enumerate(self.permutable_atoms1):
            permuted_c2[atom*3:(atom*3)+3] = \
                best_c2[self.permutable_atoms1[best_c_ind1[idx]]*3:
                        (self.permutable_atoms1[best_c_ind1[idx]]*3)+3]

        # Modify coordinates to best permutation
        for idx, atom in enumerate(self.permutable_atoms2):
            permuted_c2[atom*3:(atom*3)+3] = \
                best_c2[self.permutable_atoms2[best_c_ind2[idx]]*3:
                        (self.permutable_atoms2[best_c_ind2[idx]]*3)+3]
        return self.coords.cartesian_to_hybrid(permuted_c2)

    def permutational_alignment(self, c1: NDArray, c2: NDArray) -> float:
        """ Find the distance between the closest permutation """

        # Distance for permutation of first species
        coords1_element = np.take(c1.reshape(-1, 3),
                                  self.permutable_atoms1, axis=0)
        coords2_element = np.take(c2.reshape(-1, 3),
                                  self.permutable_atoms1, axis=0)
        # Calculate distance matrix for these subset of atoms
        dist_matrix = distance_matrix(coords1_element,
                                      coords2_element)
        # Optimal permutational alignment
        row_ind1, col_ind1 = linear_sum_assignment(dist_matrix**2)
        # Find the permuted distance
        dist1 = dist_matrix[row_ind1, col_ind1].sum()

        # Distance for permutation of second species
        coords1_element = np.take(c1.reshape(-1, 3),
                                  self.permutable_atoms2, axis=0)
        coords2_element = np.take(c2.reshape(-1, 3),
                                  self.permutable_atoms2, axis=0)
        # Calculate distance matrix for these subset of atoms
        dist_matrix = distance_matrix(coords1_element,
                                      coords2_element)
        # Optimal permutational alignment
        row_ind2, col_ind2 = linear_sum_assignment(dist_matrix**2)
        # Find the permuted distance
        dist2 = dist_matrix[row_ind2, col_ind2].sum()

        # Return the total separation and permutations
        return dist1 + dist2, col_ind1, col_ind2

    def permutational_alignment_old(self, x1: NDArray, x2: NDArray) -> tuple:
        """ Hungarian algorithm adapted to solve the linear assignment problem
            for each atom of each distinct element. Finds the optimal
            permutation of all atoms respecting the atomic species """

        # Get the full Cartesian coordinates
        c1 = self.coords.hybrid_to_cartesian(x1)
        c2 = self.coords.hybrid_to_cartesian(x2)
        # Get coordinates to apply permutation to
        permuted_c2 = c2.copy()
        # Distance for permutation of first species
        coords1_element = np.take(c1.reshape(-1, 3),
                                  self.permutable_atoms1, axis=0)
        coords2_element = np.take(c2.reshape(-1, 3),
                                  self.permutable_atoms1, axis=0)
        # Calculate distance matrix for these subset of atoms
        dist_matrix = distance_matrix(coords1_element,
                                      coords2_element)
        # Optimal permutational alignment
        col_ind = linear_sum_assignment(dist_matrix**2)[1]
        # Modify the coordinates to nearest permutation
        for idx, atom in enumerate(self.permutable_atoms1):
            permuted_c2[atom*3:(atom*3)+3] = \
                c2[self.permutable_atoms1[col_ind[idx]]*3:
                   (self.permutable_atoms1[col_ind[idx]]*3)+3]

        # Distance for permutation of second species
        coords1_element = np.take(c1.reshape(-1, 3),
                                  self.permutable_atoms2, axis=0)
        coords2_element = np.take(c2.reshape(-1, 3),
                                  self.permutable_atoms2, axis=0)
        # Calculate distance matrix for these subset of atoms
        dist_matrix = distance_matrix(coords1_element,
                                      coords2_element)
        # Optimal permutational alignment
        col_ind = linear_sum_assignment(dist_matrix**2)[1]
        # Modify coordinates to nearest permutation
        for idx, atom in enumerate(self.permutable_atoms2):
            permuted_c2[atom*3:(atom*3)+3] = \
                c2[self.permutable_atoms2[col_ind[idx]]*3:
                   (self.permutable_atoms2[col_ind[idx]]*3)+3]
        return self.coords.cartesian_to_hybrid(permuted_c2)

    def apply_rotation(self, position: NDArray) -> NDArray:
        """ Reflect the Cartesian coordinates about plane """
        new_position = position.copy()
        n_atoms = int(position.size/3)
        norm_factor = 1.0 / np.sqrt(3)
        # Rodrigues formula for rotation about (1,1,1)
        w = np.array([[0.0, -1.0*norm_factor, norm_factor],
                      [norm_factor, 0.0, -1.0*norm_factor],
                      [-1.0*norm_factor, norm_factor, 0.0]])
        rot_mat = np.identity(3, dtype=float) + 2.0*np.matmul(w, w)
        for i in range(2, n_atoms):
            atom = position[(i*3):(i*3)+3]
            new_atom = np.matmul(rot_mat, atom)
            new_position[(i*3): (i*3)+3] = new_atom
        return new_position
    
    def apply_reflection(self, position: NDArray) -> NDArray:
        """ Flip the atoms about x=y """
        new_position = position.copy()
        n_atoms = int(position.size/3)
        for i in range(3, n_atoms):
            new_position[(i*3)] = position[(i*3)+1]
            new_position[(i*3)+1] = position[(i*3)]
        return new_position

    def get_permutable_atoms(self) -> None:
        """ Set the permutable atoms for use in the permutationally-aligned
            distance """
        elements = list(set(self.coords.atom_labels))
        el1 = elements[0]
        el2 = elements[1]
        # Get atoms of element 0
        self.permutable_atoms1 = \
            [i for i, x in enumerate(self.coords.atom_labels) if x == el1 and i not in [0, 1, 2]]
        # Get atoms of element 1
        self.permutable_atoms2 = \
            [i for i, x in enumerate(self.coords.atom_labels) if x == el2 and i not in [0, 1, 2]]
