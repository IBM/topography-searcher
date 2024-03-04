""" Module that contains the classes for performing interpolation or
    regression of a provided dataset and then exploring the fitted function """

import numpy as np
from nptyping import NDArray
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate, KFold
from scipy.interpolate import RBFInterpolator
from .potential import Potential


class DatasetInterpolation(Potential):
    """
    Class to compute and evaluate a regression or interpolation model fitted
    to a given dataset.

    Attributes
    -------------

    model_data: class
        The dataset which we interpolate or regress
    smoothness: float
        Smoothness parameter for the kernel in RBF interpolation
    model: class
        The scipy interpolation model that can be fit and queried
    """
    def __init__(self, model_data: type, smoothness: float = 0.0) -> None:
        self.atomistic = False
        self.model_data = model_data
        self.smoothness = smoothness
        self.model = None
        self.initialise_model()

    def initialise_model(self) -> None:
        """ Initialise the interpolation model, a radial basis function
            interpolation with a thin-plate kernel as implemented in scipy """
        self.model = RBFInterpolator(self.model_data.training,
                                     self.model_data.response,
                                     smoothing=self.smoothness)

    def function(self, position: NDArray) -> float:
        """ Evaluate the value of the regression/interpolation model """
        return float(self.model(position.reshape(1, -1)))

    def refit_model(self) -> None:
        """ Refit the interpolation model for the current model_data """
        self.model = RBFInterpolator(self.model_data.training,
                                     self.model_data.response,
                                     smoothing=self.smoothness)


class DatasetRegression(Potential):
    """
    Class to compute and evaluate a regression or interpolation model fitted
    to a given dataset.

    Attributes
    -------------

    model_data: class
        The dataset which we interpolate or regress
    model_rand: int
        Random seed passed to MLP fitting
    model: class
        The sklearn regression model that can be fit and queried
    cv_results: dict
        The results of cross-validation fitting of MLP
    """
    def __init__(self, model_data: type, model_rand: int = 1) -> None:
        self.atomistic = False
        self.model_data = model_data
        self.model_rand = model_rand
        self.model = None
        self.cv_results = None
        self.initialise_model()

    def initialise_model(self) -> None:
        """ Initialise the regression model, a multi-layer perceptron
            as implemented in sklearn """
        self.model = MLPRegressor(random_state=self.model_rand, max_iter=1000)
        self.model.fit(self.model_data.training, self.model_data.response)
        cv = KFold(n_splits=5, shuffle=True, random_state=self.model_rand)
        self.cv_results = cross_validate(self.model,
                                         self.model_data.training,
                                         self.model_data.response,
                                         cv=cv,
                                         scoring=('neg_mean_squared_error'),
                                         n_jobs=None,
                                         return_estimator=True)

    def function(self, position: NDArray) -> float:
        """ Evaluate the value of the regression/interpolation model """
        return float(self.model.predict(position.reshape(1, -1)))

    def refit_model(self) -> None:
        """ Refit model and recalculate error based on current training """
        self.model.fit(self.model_data.training, self.model_data.response)
        cv = KFold(n_splits=5, shuffle=True, random_state=self.model_rand)
        self.cv_results = cross_validate(self.model,
                                         self.model_data.training,
                                         self.model_data.response,
                                         cv=cv,
                                         scoring=('neg_mean_squared_error'),
                                         n_jobs=None,
                                         return_estimator=True)

    def get_model_error(self) -> float:
        """ Return the model error of current fit """
        best_estimator = \
            np.argmin(-self.cv_results['test_score'])
        return -1.0*self.cv_results['test_score'][best_estimator]
