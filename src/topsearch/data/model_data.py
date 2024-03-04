""" Module that contains the ModelData class that stores and operates on
    a dataset for use in Potential classes """

import numpy as np
from nptyping import NDArray
from scipy.spatial import Delaunay


class ModelData:

    """
    Class to store the data associated with a machine learning model,
    and perform the methods to modify the data.

    Attributes
    -------------
    training : NDArray
        All training data points of the dataset
    response : NDArray
        The corresponding response values for each data point
    n_points : int
        The number of data points in the dataset
    resp_props : dict
        Dictionary containing the statistics of the response array
    train_props : dict
        Dictionary containing the statistics of the training array
    hull : NDArray

    """

    def __init__(self,
                 training_file: str,
                 response_file: str) -> None:
        self.training = None
        self.response = None
        # Read in the training and response arrays
        self.read_data(training_file, response_file)
        # Set the number of data points
        self.n_points = self.training.shape[0]
        self.n_dims = self.training.shape[1]
        self.resp_props = {'std':  0.0,
                           'mean': 0.0,
                           'min':  0.0,
                           'max':  0.0}
        self.train_props = {'std':  np.zeros(self.n_dims, dtype=float),
                            'mean': np.zeros(self.n_dims, dtype=float),
                            'min':  np.zeros(self.n_dims, dtype=float),
                            'max':  np.zeros(self.n_dims, dtype=float)}
        self.hull = None

    def read_data(self, training_file: str, response_file: str) -> None:
        """ Read in the training and repsonse values needed for an ML model.
            Stored in class attributes to be accessed through potential """
        self.training = np.loadtxt(f"{training_file}", dtype=float, ndmin=2)
        self.response = np.loadtxt(f"{response_file}", dtype=float)
        # Reset number of points and dimensions
        self.n_points = self.training.shape[0]
        self.n_dims = self.training.shape[1]

    def write_data(self, training_file: str, response_file: str) -> None:
        """ Writes the training and response attributes into the
            specified files """
        np.savetxt(f"{training_file}", self.training)
        np.savetxt(f"{response_file}", self.response)

    def append_data(self, new_training: NDArray,
                    new_response: NDArray) -> None:
        """ Add additional training and response data to the attributes
            that store these within the class """
        self.training = np.append(self.training, new_training, axis=0)
        self.response = np.append(self.response, new_response)
        self.n_points = self.training.shape[0]

    def limit_response_maximum(self, upper_limit: float) -> None:
        """ Limits the maximum allowed response value """
        self.response = np.clip(self.response, None, upper_limit)

    def standardise_response(self) -> None:
        """ Standardises response values, enforcing a mean of 0,
            and a standard deviation of 1 """
        self.resp_props['mean'] = np.mean(self.response)
        self.resp_props['std'] = np.std(self.response)
        self.response = (self.response - self.resp_props['mean']) / \
            self.resp_props['std']

    def normalise_response(self) -> None:
        """ Returns the normalised response values,
            scaled to lie in the range (0,1) """
        self.resp_props['min'] = self.response.min(axis=0)
        self.resp_props['max'] = self.response.max(axis=0)
        self.response = (self.response - self.resp_props['min']) / \
            (self.resp_props['max'] - self.resp_props['min'])

    def standardise_training(self) -> None:
        """ Standardises each feature of the training data to have
            mean 0, standard deviation 1 """
        self.train_props['mean'] = np.mean(self.training, axis=0)
        self.train_props['std'] = np.std(self.training, axis=0)
        self.training = (self.training - self.train_props['mean']) / \
            self.train_props['std']

    def normalise_training(self) -> None:
        """ Limit all features to lie within the range (0, 1) """
        self.train_props['min'] = self.training.min(axis=0)
        self.train_props['max'] = self.training.max(axis=0)
        self.training = (self.training - self.train_props['min']) / \
            (self.train_props['max'] - self.train_props['min'])

    def unstandardise_response(self):
        """ Undo the normalisation of the response array """
        self.response = (self.response * self.resp_props['std']) + \
            self.resp_props['mean']

    def unnormalise_response(self):
        """ Undo the normalisation of the training array """
        self.response = self.response * \
            (self.resp_props['max'] - self.resp_props['min']) \
            + self.resp_props['min']

    def unstandardise_training(self):
        """ Undo the normalisation of the response array """
        self.training = (self.training * self.train_props['std']) + \
            self.train_props['mean']

    def unnormalise_training(self):
        """ Undo the normalisation of the training array """
        self.training = self.training * \
            (self.train_props['max'] - self.train_props['min']) \
            + self.train_props['min']

    def remove_duplicates(self, dist_cutoff: float = 1e-7) -> None:
        """ Remove any minima within dist_cutoff from each other, retaining
            only the first """
        repeated_points = []
        for i in range(self.n_points-1):
            for j in range(i+1, self.n_points):
                d = np.linalg.norm(self.training[i, :] - self.training[j, :])
                if d < dist_cutoff:
                    repeated_points.append(j)
                    break
        self.training = np.delete(self.training, repeated_points, axis=0)
        self.response = np.delete(self.response, repeated_points, axis=0)
        self.n_points = self.training.shape[0]

    def feature_subset(self, features: list) -> None:
        """ Get a subset of the features of the training data """
        self.training = self.training[:, features]
        self.n_dims = len(features)

    def convex_hull(self) -> None:
        """ Compute the convex hull for the training data """
        self.hull = Delaunay(self.training)

    def point_in_hull(self, point: NDArray) -> bool:
        """ Determine if point is within convex hull of training data """
        return self.hull.find_simplex(point) >= 0
