""" Module to contain local minimisers. Global optimisation, single and
    double-ended transition state searches all rely on local minimisation """

import scipy
from nptyping import NDArray


def minimise(func_grad, initial_position: NDArray, bounds: list,
             conv_crit: float = 1e-6, history_size: int = 5,
             n_steps: int = 200,
             args: list = None) -> tuple[NDArray, float, dict]:
    """ Wrapper for the scipy box-constrained LBFGS implementation. Takes
        in a function, gradient and initial position and performs local
        minimisation subject to the specified bounds """

    if args is None:
        args = []
    # Do a normal single local minimisation
    min_coords, f_val, results_dict = scipy.optimize.fmin_l_bfgs_b(
                                func=func_grad,
                                x0=initial_position,
                                bounds=bounds,
                                m=history_size,
                                args=args,
                                factr=1e-30,
                                pgtol=conv_crit,
                                maxiter=n_steps,
                                maxls=40)
    return min_coords, f_val, results_dict
