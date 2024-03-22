""" Potential module contains the base Potential class. This class
    provides the function that can be explored and gives the
    Each class provides the function value, first and second
    derivatives """

from typing import Tuple
import numpy as np
from nptyping import NDArray


class Potential:

    """
    Description
    ------------

    Class to specify the optimisable function, and provide methods
    to compute its value, first and second derivatives given a position
    in space, and any necessary information such as training data

    Attributes
    -----------

    """

    def __init__(self) -> None:
        self.atomistic = False

    def function(self, position: NDArray) -> float:
        """ Evaluate the function at a given point """
        return 0.0

    def gradient(self, position: NDArray,
                 displacement: float = 1e-6) -> NDArray:
        """
        Evaluate the numerical gradient of the function a point coords
        Useful default to fall on if no analytical derivatives
        Returns 1d array with gradient w.r.t all components
        """
        ndim = position.size
        grad = np.zeros(ndim, dtype=float)
        for i in range(ndim):
            position[i] += displacement
            f_plus = self.function(position)
            position[i] -= 2.0*displacement
            f_minus = self.function(position)
            position[i] += displacement
            grad[i] = (f_plus - f_minus)/(2.0*displacement)
        return grad

    def function_gradient(self, position: NDArray) -> Tuple[float, NDArray]:
        """
        Evaluate both function and derivative in single method for optimiser
        Returns function value and gradient vector
        """
        function_val = self.function(position)
        grad = self.gradient(position)
        return function_val, grad

    def hessian(self, position: NDArray,
                displacement: float = 1e-4) -> NDArray:
        """ Compute the matrix of second derivatives numerically
            Returns 2d array of second derivatives """
        ndim = position.size
        hess = np.zeros((ndim, ndim), dtype=float)
        for i in range(ndim):
            position[i] -= displacement
            grad_minus = self.gradient(position)
            position[i] += 2.0*displacement
            grad_plus = self.gradient(position)
            position[i] -= displacement
            for j in range(i, ndim):
                hess[i, j] = (grad_plus[j]-grad_minus[j])/(2.0*displacement)
                hess[j, i] = hess[i, j]
        return hess

    def check_valid_minimum(self, coords: type) -> bool:
        """ Determines if a minimum is allowed based on eigenvalues """
        if not coords.at_bounds():
            # Find the eigenvalues of the Hessian matrix
            hess = self.hessian(coords.position)
            eigs = np.linalg.eigvalsh(hess)
            # If atomistic will have six zero eigenvalues
            if self.atomistic:
                if eigs[0] > -1.0 and np.all(eigs[6:] > 1e-6):
                    return True
                return False
            else:
                if np.all(eigs > 1e-9):
                    return True
                return False
        return True

    def check_valid_ts(self, coords: type) -> bool:
        """ Check if transition state one negative eigenvalue """
        if not coords.at_bounds():
            # Compute the eigenvalue spectrum
            hess = self.hessian(coords.position)
            eigs = np.linalg.eigvalsh(hess)
            # If smallest eigenvalue is non-negative then not transition state
            if self.atomistic:
                if eigs[0] < -1e-3 and np.all(eigs[7:] > 1e-6):
                    return True
                return False
            else:
                if eigs[0] < -1e-5 and np.all(eigs[1:] > 1e-9):
                    return True
                return False
        return True
