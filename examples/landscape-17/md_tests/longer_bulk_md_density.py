from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.atoms import Atoms
from ase.io.trajectory import Trajectory

from ase.data.pubchem import (pubchem_atoms_search, 
                              pubchem_atoms_conformer_search, 
                              pubchem_search)
import py3Dmol
from ase.io import write, read
import matplotlib.pyplot as plt
import numpy as np
from ase.md.verlet import VelocityVerlet


from topsearch.potentials.ml_potentials import MachineLearningPotential
from ase.md import MDLogger
from ase.calculators.mixing import SumCalculator
# from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator


from sys import argv

def printenergy(atoms: Atoms) -> None:
    """Function to print the potential, kinetic and total energy"""
    epot = atoms.get_potential_energy()[0] / len(atoms)
    ekin = atoms.get_kinetic_energy() / len(atoms)
    temperature = ekin / (1.5 * units.kB)

    print(f'Energy per atom: Epot = {epot:.3f}eV  Ekin = {ekin:.3f}eV '
          f'(T={temperature:3.0f}K)  Etot = {epot+ekin:.3f}eV')


species = ['H']
mlp = MachineLearningPotential(species, 'aimnet2', '/u/jdm/models/aimnet2_b973c_0.jpt', device=None)

import weakref
from typing import IO, Any, Union
from ase import Atoms, units
from ase.parallel import world
from ase.utils import IOContext
class MyMDLogger(MDLogger):
    def __init__(self,
                 dyn: Any,  # not fully annotated so far to avoid a circular import
                atoms: Atoms,
                logfile: Union[IO, str],
                header: bool = True,
                stress: bool = False,
                peratom: bool = False,
                mode: str = "a",
                rmsd: bool = True,
                reset_step: bool = True):
        
        super().__init__(dyn, atoms, logfile, False, stress, peratom, mode)
        self.fmt = self.fmt.strip("\n")
        self.hdr = self.hdr.strip("\n")
        if stress: # calculate pressure here
            self.fmt += "%12.4f"
            self.hdr += "  %12s" % ("P[GPa]",)
        self.fmt += "%12.4f"
        self.hdr += "%12s" % ("Vol/N",)
        self.fmt += " %12d"
        self.hdr += "%12s" % ("Step",)
        self.fmt += " %12.4f \n"
        self.hdr += "  %12s \n" % ("RMSD/N[Å]",)
        self.rmsd = 0
        self.last_step_pos = atoms.get_positions()
        if 'rmsd' in self.atoms.info:
            self.rmsd = self.atoms.info['rmsd']
        if 'step' in self.atoms.info:
            if reset_step:
                print(f"resetting dyn step to {self.atoms.info['step']}")
                self.dyn.nsteps = self.atoms.info['step']
                self.dyn.timeelapsed = self.atoms.info['step'] * self.dyn.dt
        else:
            self.atoms.info['step'] = 0
        
        if header:
            self.logfile.write(self.hdr + "\n")

    
    def save_last_step(self):
        self.last_step_pos = self.atoms.get_positions()
        
    def __call__(self):
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = self.atoms.get_temperature()
        self.rmsd += np.sqrt(((self.atoms.get_positions() - self.last_step_pos)**2).flatten().mean())
        self.save_last_step()
        global_natoms = self.atoms.get_global_number_of_atoms()
        if self.peratom:
            epot /= global_natoms
            ekin /= global_natoms
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000 * units.fs)
            dat = (t,)
        else:
            dat = ()
        dat += (epot + ekin, epot, ekin, temp)
        if self.stress:
            dat += tuple(self.atoms.get_stress(
                include_ideal_gas=True) / units.GPa)
            dat += (-1*self.atoms.get_stress(
                include_ideal_gas=True)[:3].mean(), )
        dat += (self.atoms.get_volume()/len(self.atoms), )
        self.atoms.info['step'] = int(self.dyn.nsteps)
        self.atoms.info['rmsd'] = self.rmsd
        dat += (self.atoms.info['step'], )
        dat += (self.rmsd, )
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()
        
def create_log(dyn, file):
    
    log = MyMDLogger(dyn, at, file, 
        header=True, stress=True, peratom=True, mode='a')
    return log

# ats = read('big_bulk-3-4k.xyz', ':')
debug_fac=1
rundir = f'longer_bulk_md_density_{argv[1]}_normd3_scratch'
# rundir = f'longer_bulk_md_density_{argv[1]}_normd3_tt-sil2only_prevsett-0'

# ats = [read('bulk_md_diffusion/sil-3-293K-NVT-50ps.traj', '-1')] # start from end of well-equilibrated previous traj
ats = [read(f'bulk_molecules-{argv[1]}.xyz')] # start from initial structure
# ats = [read(f'longer_bulk_md_density_{argv[1]}/master-0.traj', '-1')] # start from end of well-equilibrated previous traj
ats[0].info['step'] = 0
ats[0].info['rmsd'] = 0
ndump=500 // debug_fac
ndumplog=100 // debug_fac
if ndump % ndumplog:
    raise RuntimeError("Please use ndumlog that divides ndump")
for i, at in enumerate(ats):
    try:
        at = read(f'{rundir}/master-{i}.traj', '-1')
        print(f'Restarting on step {at.info["step"]}')
    except:
        print(f"No restart found, starting from scratch on {i}")
    master_traj = Trajectory(f'{rundir}/master-{i}.traj', 'a', at)
    totstep=0    
    mlp = MachineLearningPotential(species, 'aimnet2', '/u/jdm/models/aimnet2_b973c_0.jpt', 'cuda')
    calc = mlp.atoms.calc
    calc.base_calc.cutoff_lr = 20
    # calc2 = TorchDFTD3Calculator(atoms=at, device="cuda", xc='b97-3c', damping="bj", cutoff=12)
    # calc = SumCalculator([calc1, calc2])
    
    at.calc = calc
    
    dyn_lang = Langevin(at, 0.5*units.fs, temperature_K=800, friction=5e-3)
    log_lang = create_log(dyn_lang, f'{rundir}/equil_nvt_log.out')
    traj = Trajectory(f'{rundir}/sil-{i}-303K-NPT.traj', 'a', at)

    print('Starting dynamics')
    
    totstep += 100000 // debug_fac
    diff = totstep - at.info['step']
    offset = (ndump - (diff % ndump)) % ndump
    if at.info['step'] < totstep:
        print('running ', diff, ' after ', offset)
        dyn_lang.run(offset)    
        dyn_lang.attach(log_lang, interval=ndumplog)
        dyn_lang.attach(master_traj, interval=ndump)
        dyn_lang.run(diff//2)
        dyn_npt = NPT(at, 0.5*units.fs, temperature_K=303, externalstress=0, 
                     ttime=100*units.fs, pfactor=(100*units.fs)**2 * 0.6e-2)
        dyn_npt.set_fraction_traceless(0)
        log_lang = create_log(dyn_npt, f'{rundir}/equil_nvt_log.out')
        dyn_npt.attach(log_lang, interval=ndumplog)
        dyn_npt.attach(master_traj, interval=ndump)
        dyn_npt.run(diff//2)
    else:
        print (f"continuing from {at.info['step']}")
        
    print("equilibration finished!")
    
    dyn_npt = NPT(at, 0.5*units.fs, temperature_K=303, externalstress=0, ttime=100*units.fs, pfactor=(100*units.fs)**2 * 0.6e-2)
    dyn_npt.set_fraction_traceless(0)
    log_npt = create_log(dyn_npt, f'{rundir}/md_log_303K_NPT_log.out')
    print('Internal number of steps for dn: ', dyn_npt.nsteps)
        
    totstep += 300000 // debug_fac
    diff = totstep - at.info['step']
    offset = (ndump - (diff % ndump)) % ndump
    if at.info['step'] < totstep:
        print('running ', diff, ' ', offset)
        dyn_lang.run(offset) 
        dyn_npt.attach(log_npt, interval=ndumplog)
        dyn_npt.attach(master_traj, interval=ndump)
        dyn_npt.attach(traj.write, interval=ndump)
        dyn_npt.run(diff)
    print("Done NPT!")
    traj.close()
    
    
    ts = [372, 423, 473]
    for temperature in ts:
        print(f"\n\n\nStarting at T={temperature}\n\n\n")
        trajnpt = Trajectory(f'{rundir}/sil-{i}-{temperature}K-NPT.traj', 'a', at)
        dyn_npt_new = NPT(at, 0.5*units.fs, temperature_K=temperature, externalstress=0, ttime=100*units.fs, pfactor=(100*units.fs)**2 * 0.6e-2)
        dyn_npt_new.set_fraction_traceless(0)
        log = create_log(dyn_npt_new, f'{rundir}/md_log-{i}-{temperature}K.out')
        totstep += 50000 // debug_fac
        diff = totstep - at.info['step']
        offset = (ndump - (diff % ndump)) % ndump
        if at.info['step'] < totstep:
            print('running ', diff, ' ', offset)
            dyn_npt_new.run(offset)
            dyn_npt_new.attach(log, interval=ndumplog)
            dyn_npt_new.attach(master_traj, interval=ndump)
            dyn_npt_new.attach(trajnpt, interval=ndump)
            dyn_npt_new.run(diff)
        else:
            print (f"continuing from {at.info['step']}")
            
        totstep += 300000 // debug_fac
        diff = totstep - at.info['step']
        offset = (ndump - (diff % ndump)) % ndump
        if at.info['step'] < totstep:
            print('running ', diff, ' ', offset)
            dyn_npt_new.run(offset)
            dyn_npt_new.attach(master_traj, interval=ndump)
            dyn_npt_new.attach(trajnpt, interval=ndump)
            dyn_npt_new.run(diff)
        else:
            print (f"continuing from {at.info['step']}")
        trajnpt.close()
        
    master_traj.close()