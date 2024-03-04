""" Module that includes two simple test functions that can be used
    for validation and testing of the methodology """

import numpy as np
from nptyping import NDArray
from .potential import Potential


class Camelback(Potential):

    """
    Description
    ------------
    Six-hump camel function, sfu.ca/~ssurjano/camel6.html

    Attributes
    -----------

    """

    def __init__(self):
        self.atomistic = False

    def function(self, position: NDArray) -> float:
        """ Return function evaluated at position """
        x, y = position
        term1 = (4.0 - (2.1*(x**2)) + ((x**4)/3.0)) * (x**2)
        term2 = x*y
        term3 = (-4.0 + 4.0*(y**2))*(y**2)
        f_val = term1 + term2 + term3
        return f_val

    def gradient(self, position: NDArray) -> NDArray:
        """ Return gradient vector evaluated at position """
        x, y = position
        grad = np.zeros(2, dtype=float)
        grad[0] = y + (8.0*x) - (8.4*(x**3)) + (2.0*(x**5))
        grad[1] = x + (16.0*(y**3)) - (8.0*y)
        return grad

    def hessian(self, position: NDArray) -> NDArray:
        """ Construct the 2x2 Hessian matrix of second derivatives """
        x, y = position
        hess = np.zeros((2, 2), dtype=float)
        hess[0, 1] = 1.0
        hess[1, 0] = 1.0
        hess[0, 0] = (x**2) * (-4.2+(4.0*(x**2))) + (4*x) * \
                     ((-4.2*x) + ((4.0/3.0)*(x**3))) + \
                     (4.0-(2.1*(x**2)) + ((x**4)/3.0))*2.0
        hess[1, 1] = (40.0*(y**2)) + 2.0*(-4.0+(4.0*(y**2)))
        return hess


class Schwefel(Potential):

    """
    Description
    ------------
    Schwefel function, https://www.sfu.ca/~ssurjano/schwef.html

    Attributes
    ------------

    """

    def __init__(self):
        self.atomistic = False

    def function(self, position: NDArray) -> NDArray:
        """ Returns the Schwefel function evaluated at position """
        const_term = 418.9829*position.size
        sum_term = 0.0
        for i in position:
            sum_term += i * np.sin(np.sqrt(np.abs(i)))
        return const_term-sum_term


class Quadratic(Potential):

    """
    Description
    -------------
    Simple quadratic centered at (0, 0, 0, ... ). Used only for testing

    Attributes
    --------------

    """

    def __init__(self):
        self.atomistic = False

    def function(self, position: NDArray) -> float:
        """ Return the quadratic function value """
        f_val = 0.0
        for i in position:
            f_val += i**2
        return f_val
