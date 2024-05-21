""" Module that contains the classes for evaluating classical molecular
    force fields to compute the energy of atomistic systems """

import numpy as np
from nptyping import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from .potential import Potential


class MMFF94(Potential):
    """ 
    Description
    ---------------
    Evaluate the energy and force of a molecular system using the
    MMFF94 empirical force field through RDKit.

    Attributes
    ----------------
    xyz_file : string
        The filename for the xyz file that contains the molecular or atomic
        system
    molecule : class instance
        rdkit object that stores the information about the molecule
    conf : class instance
        stores the particular atomic positions for the molecule
    n_atoms : int
        Store the number of atoms in the molecule or cluster
    force_field : class instance
        the force field object that contains parameters to allow evaluation
        of the potential energy from the molecule object
        """

    def __init__(self, xyz_file: str):
        self.atomistic = True
        self.xyz_file = xyz_file
        # Initialise the molecule from xyz_file
        mol = Chem.MolFromXYZFile(xyz_file)
        # And determine its bonding
        self.molecule = Chem.Mol(mol)
        rdDetermineBonds.DetermineBonds(self.molecule)
        self.n_atoms = self.molecule.GetNumAtoms()
        Chem.SanitizeMol(self.molecule)
        self.conf = self.molecule.GetConformer()
        mol_properties = AllChem.MMFFGetMoleculeProperties(self.molecule)
        self.force_field = AllChem.MMFFGetMoleculeForceField(self.molecule,
                                                             mol_properties)

    def function(self, position: NDArray) -> float:
        """ Compute the potential energy """
        self.set_xyz(position)
        mol_properties = AllChem.MMFFGetMoleculeProperties(self.molecule)
        self.force_field = AllChem.MMFFGetMoleculeForceField(self.molecule,
                                                             mol_properties)
        return self.force_field.CalcEnergy()

    def function_gradient(self, position: NDArray) -> tuple:
        """ Compute the potential energy and its gradient """
        self.set_xyz(position)
        mol_properties = AllChem.MMFFGetMoleculeProperties(self.molecule)
        self.force_field = AllChem.MMFFGetMoleculeForceField(self.molecule,
                                                             mol_properties)
        energy = self.force_field.CalcEnergy()
        forces = self.force_field.CalcGrad()
        return energy, np.asarray(forces)

    def gradient(self, position: NDArray) -> NDArray:
        """ Compute the atomic forces in the molecule """
        self.set_xyz(position)
        mol_properties = AllChem.MMFFGetMoleculeProperties(self.molecule)
        self.force_field = AllChem.MMFFGetMoleculeForceField(self.molecule,
                                                             mol_properties)
        forces = self.force_field.CalcGrad()
        return np.asarray(forces)

    def set_xyz(self, position: NDArray) -> None:
        """ Put the xyz coordinates provided in position into the
            molecule object """
        for i in range(self.n_atoms):
            x, y, z = position[(i*3):(i*3)+3]
            self.conf.SetAtomPosition(i, Point3D(x, y, z))
