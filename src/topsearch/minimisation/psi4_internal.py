"Use fast internal Optking routine to optimise geometry if doing DFT with Psi4."

from nptyping import NDArray
import numpy as np
from ase.units import Bohr

def minimise(potential,
             initial_position: NDArray,
             conv_crit: float = 1e-6,
             n_steps: int = 50,):
    from psi4.driver import OptimizationConvergenceError
    from psi4.driver.p4util.exceptions import SCFConvergenceError
    atoms = potential.atoms
    atoms.set_positions(initial_position.reshape(-1, 3))
    calc = potential.atoms.calc
    psi4 = calc.psi4
    method = calc.parameters['method']
    basis = calc.parameters['basis']
    calc.set_psi4()
    calc.psi4.core.set_output_file(calc.label + '.dat',
                                       False)
    
    # using only force convergence for now
    try:
        e, hist = psi4.opt(f'{method}/{basis}',
                        molecule=calc.molecule,
                        return_history=True,
                        optimizer_keywords={"MAX_FORCE_G_CONVERGENCE": conv_crit,
                            "GEOM_MAXITER": n_steps,
                            "MAX_ENERGY_G_CONVERGENCE": 1e2,
                            "RMS_FORCE_G_CONVERGENCE": 1e2,
                            "MAX_DISP_G_CONVERGENCE": 1e2,
                            "RMS_DISP_G_CONVERGENCE": 1e2})
    
        min_coords = hist['coordinates'][-1].flatten()*Bohr
        results_dict = {'task': None,
                        'warnflag': 0,
                    }
        
    except (SCFConvergenceError, OptimizationConvergenceError):
        print("Optimisation failed to converge")
        min_coords = np.full((initial_position.size), np.nan, dtype=float)
        e = np.nan
        # dummy to mimic scipy results_dict
        results_dict = {'task': None,
                        'warnflag': 1,
                    }
    return min_coords, e, results_dict
    