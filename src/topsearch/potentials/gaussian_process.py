""" Module that contains classes to implement Gaussian process regression
    and evaluate the resulting model """

import logging
import warnings
import sys
import numpy as np
from nptyping import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

from topsearch.data.model_data import ModelData
from .potential import Potential
warnings.filterwarnings('ignore')


class GaussianProcess(Potential):

    """
    Description
    -------------
    Fit and evaluate a Gaussian process regression model to a given dataset.
    The function we evaluate is the value of the regression model at any
    point in feature space

    Attributes
    -----------
    model_data : ModelData instance
        The object containing the training and response data we will fit
    kernel_choice : str
        The choice of kernel, can be 'RBF' or 'Matern'
    kernel_bounds : list
        Limits on the kernel lengthscales and noise (final element)
        hyperparameters
    standardise_training : bool
        Choose whether to standardise the training data before GP fit
    standardise_response : bool
        Choose whether to standardise the response data before GP fit
    limit_highest_data : logical
        Specifies if we should limit the largest response value.
        Useful in molecular applications where the steep repuslive wall
        gives huge values
    matern_nu : float
        The nu parameter of the Matern kernel
    gpr : class
        The sklearn gaussian process object
    """

    def __init__(self, model_data: ModelData,
                 kernel_choice: str,
                 kernel_bounds: list,
                 standardise_training: bool = False,
                 standardise_response: bool = True,
                 limit_highest_data: bool = False,
                 matern_nu: float = None) -> None:
        self.atomistic = False
        self.model_data = model_data
        self.kernel_choice = kernel_choice
        self.kernel_bounds = kernel_bounds
        self.standardise_training = standardise_training
        self.standardise_response = standardise_response
        self.limit_highest_data = limit_highest_data
        self.matern_nu = matern_nu
        self.prepare_training_data()
        self.initialise_gaussian_process()
        self.logger = logging.getLogger('guassian')

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
        else:
            sys.exit(0)
        return kernel

    def add_data(self, new_training: NDArray,
                 new_response: NDArray) -> None:
        """ Add data to the model data, accounting for standardisation """
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

    def write_fit(self) -> None:
        """ Write the hyperparameters of the best GP fit """
        self.logger.info("Best GP fit: ")
        for i in self.gpr.kernel_.theta:
            self.logger.info(f"{i} ")
        self.logger.info('\n')

    def get_score(self) -> float:
        """ Get the R^2 score of the gp fit """
        return self.gpr.score(self.model_data.training,
                              self.model_data.response)

    def function(self, position: NDArray) -> float:
        """ Return the mean of the GP fit at position """
        return self.gpr.predict(position.reshape(1, -1))[0]

    def function_and_std(self, position: NDArray) -> float:
        """ Return the mean and variance of the GP fit at position """
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
