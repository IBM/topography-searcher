""" Local minimisation routines. Global optimisation, single and
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


def minimise_ase(potential, initial_position: NDArray, 
                 numbers: NDArray,
                conv_crit: float = 1e-3,
                n_steps: int = 200,
                args: list = None,
                output_level=0) -> tuple[NDArray, float, dict]:
    """Wrapper for ASE implementation of LBFGS. 

    Parameters
    -----------
    potential: MachineLearningPotential or any other Potential type with an ase calculator associated
    initial_position: 1D NDArray with positions to be optimised
    numbers: atomic numbers
    conv_crit: float, maximum force to define convergence. Note that this differs from scipy version of function
    n_steps: int, maximum number of steps
    args: list, not used currently, there for compatibility
    
    Returns
    ---------
    min_coords: minimised coordinates as 1D array
    f_val: energy value
    results_dict: info on the success or otherwise of optimisation"""
    
    atoms = Atoms(positions=initial_position.reshape(-1, 3), numbers=numbers, cell=None)
    atoms.calc = potential.atoms.calc
    
    if potential.calculator_type == 'aimnet2':
        potential.atoms.calc.calculate(atoms, properties=['energy', 'forces']) # hack needed to reset AIMNet internally
    
    if output_level > 0:
        logfile = '-'
    else:
        logfile='/dev/null'
        
    opt = LBFGS(atoms, 
                logfile=logfile, 
                trajectory=None)
    is_converged = opt.run(fmax=conv_crit, steps=n_steps)
    
    # check convergence
    results_dict = {'warnflag': 0, 'task': 'complete'}
    if not is_converged:
        results_dict['warnflag'] = 1
        results_dict['task'] = f'failed to converge in {n_steps} steps'
    
    min_coords = atoms.positions.reshape(-1)
    f_val = atoms.get_potential_energy()

    return min_coords, f_val, results_dict
