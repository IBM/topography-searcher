from typing import List
import numpy as np
from scipy.stats import multivariate_normal
from topsearch.potentials.potential import Potential
from topsearch.analysis.roughness import RoughnessContribution

class LocalRoughness(Potential):

    def __init__(self, contributions: List[RoughnessContribution], 
                 parallel_scaling: float,
                 orthogonal_scaling: float,
                 prefactor_scaling: float,
                 distance_to_ts: float = 0.75):
        self.atomistic = False

        self.vectors = []
        self.means = []
        self.prefactors = []
        self.distributions = []
        self.scalings = []
        self.n_dim = contributions[0].minimum.size

        for vector in contributions:
            self.vectors.append(vector.ts - vector.minimum)
            self.means.append(vector.minimum + distance_to_ts*(vector.ts-vector.minimum))
            self.prefactors.append(vector.frustration*prefactor_scaling)

    ## Repeat for all frustration vectors
        for i in range(len(self.vectors)):

            ## Construct original covariance matrix
            # Empty covariance matrix
            cov_matrix = np.zeros((self.n_dim, self.n_dim), dtype=float)
            # Add all diagonals apart from x
            for j in range(1, self.n_dim):
                cov_matrix[j, j] = orthogonal_scaling*np.linalg.norm(self.vectors[i])
            # Make x direction proportional to vector length
            # Scale the covariance so that value is at 10% at the end of vector
            cov_matrix[0, 0] = (-1.0*np.linalg.norm(0.5*self.vectors[i])**2)/(2.0*np.log(parallel_scaling))

            ## Find rotation of x onto vectors[i]
            # Set up unit vector along first dimension
            u = np.zeros((self.n_dim), dtype=float)
            u[0] = 1.0
            # Get and normalise desired vector
            v = self.vectors[i] / np.linalg.norm(self.vectors[i])

            # Compute the rotation matrix
            identity = np.identity(self.n_dim, dtype=float)
            vec_sum = (u + v)
            second_term = np.outer((vec_sum / (1.0 + np.dot(u, v))), vec_sum)
            third_term = 2.0*np.outer(v, u)
            rot_matrix = identity - second_term + third_term

            # Update the covariance matrix to reflect the rotation
            new_cov = rot_matrix @ cov_matrix @ rot_matrix.T

            # Construct the corresponding multivariate normal distribution
            rv = multivariate_normal(mean=self.means[i], cov=new_cov)
            self.distributions.append(rv)
            self.scalings.append(rv.pdf(x=self.means[i]))

    def function(self, position):
        f_val = 0.0
        for c, i in enumerate(self.distributions, 0):
            f_val += (self.prefactors[c]*i.pdf(position))/self.scalings[c]
        # Flip the potential before basin-hopping to get highest roughness as minima
        return -1.0*f_val