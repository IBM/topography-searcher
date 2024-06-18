"Use internal Optking routines to optimise geometry if using Psi4 for DFT."

from nptyping import NDArray

def minimise(potential,
             initial_position: NDArray,
             conv_crit: float = 1e-6,
             n_steps: int = 50,):

    atoms = potential.atoms
    atoms.set_positions(initial_position.reshape(-1, 3))
    calc = potential.atoms.calc
    psi4 = calc.psi4
    method = calc.parameters['method']
    basis = calc.parameters['basis']
    calc.set_psi4()
    
    # using only force convergence for now
    e, hist = psi4.optimize(f'{method}/{basis}',
                    molecule=calc.molecule,
                    return_history=True,
                    optimizer_keywords={"MAX_FORCE_G_CONVERGENCE": conv_crit,
                        "GEOM_MAXITER": n_steps,
                        "MAX_ENERGY_G_CONVERGENCE": 1e2,
                        "RMS_FORCE_G_CONVERGENCE": 1e2,
                        "MAX_DISP_G_CONVERGENCE": 1e2,
                        "RMS_DISP_G_CONVERGENCE": 1e2})
    
    return e, hist
    