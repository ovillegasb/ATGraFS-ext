"""Submodule containing functions and classes for working with SBUs."""

import os
import logging
import pickle

import ase
from ase import Atoms
import numpy as np

from atgrafsE.utils import __data__
from atgrafsE.utils.io import read_sbu

from atgrafsE.utils import symmetry
from atgrafsE.utils.mmanalysis import analyze_mm

logger = logging.getLogger(__name__)


# Skeleton of dummy atoms formed a geometry
d = 0.8
skeleton_X = {
    "D*h": Atoms(
        "XX",
        positions=[(-d/8, 0, 0), (d/8, 0, 0)]
    ),
    "D4h": Atoms(
        "XXXX",
        positions=[(d, 0, 0), (0, d, 0), (-d, 0, 0), (0, -d, 0)]
    ),
    "D2h": Atoms(
        "XXXX",
        positions=[
            (d*0.86602540, d*-0.5, 0),
            (d*0.86602540, d*0.5, 0),
            (d*-0.86602540, d*-0.5, 0),
            (d*-0.86602540, d*0.5, 0)
        ]
    ),
    "D3h": Atoms(
        "XXX",
        positions=[
            (0, d, 0),
            (0, -d*np.sin(np.deg2rad(30.0)), -d*np.cos(np.deg2rad(30.0))),
            (0, -d*np.sin(np.deg2rad(30.0)), d*np.cos(np.deg2rad(30.0)))
        ]
    ),
    "Td": Atoms(
        "XXXX",
        positions=[
            (0, 0, d),
            (0, d*-0.94280904, d*-0.33333333),
            (d*0.81649658, d*0.47140452, d*-0.33333333),
            (d*-0.81649658, d*0.47140452, d*-0.33333333)
        ]
    ),
    "D2d***": Atoms(#MODIFICAR TODO
        "XXXX",
        positions=[
            (d*-0.17022212, d*0.09827779, d*0.98049269),
            (d*0.17022212, d*-0.89165810, d*-0.41948808),
            (d*0.85730963, d*0.29841237, d*-0.41948808),
            (d*-0.85730963, d*0.49496795, d*-0.14151652),
        ]
    ),
    "Oh": Atoms(
        "XXXXXX",
        positions=[
            (0, 0, d),
            (0, d, 0),
            (d, 0, 0),
            (0, 0, -d),
            (0, -d, 0),
            (-d, 0, 0)
        ]
    )
}


class SBU:
    """Container class for a building unit information."""

    def __init__(self, name, atoms=None):
        """Initialize a building unit, from an ASE Atoms."""
        logger.debug("New instance of SBU {0}".format(name))
        self.name = name
        self.atoms = atoms
        self.mmtypes = []
        self.bonds = []
        self.shape = []
        self.pg = None
        if self.atoms is not None:
            self._analyze()
        logger.debug("{self}".format(self=self.__str__()))

    def get_atoms(self):
        """Return a copy of the topology as ASE Atoms."""
        logger.debug("SBU {0}: returning atoms.".format(self.name))
        return self.atoms.copy()

    def _analyze(self):
        """Guess the mmtypes, bonds and pointgroup."""
        logger.debug("SBU {}: analyze bonding and symmetry.".format(self.name))
        # Detects geometry from dummy atoms
        dummies = Atoms([x for x in self.atoms if x.symbol == "X"])
        if len(dummies) > 0:
            pg = symmetry.PointGroup(mol=dummies.copy(), tol=0.1)
            max_order = min(8, len(dummies))
            shape = symmetry.get_symmetry_elements(
                mol=dummies.copy(),
                max_order=max_order
            )
            self.shape = shape
            self.pg = pg.schoenflies

        ####REVISAR TODO
        ## bonds, mmtypes = analyze_mm(self.get_atoms())
        ####>>>> Testing a particular SBU
        ####if self.name == "dicarboxylate_oxalate":
        ####    print(bonds)
        ####    print(self.atoms.get_chemical_symbols())
        ####    exit()

        ##self.bonds = bonds
        ##self.mmtypes = mmtypes

    def set_atoms(self, atoms, analyze=False):
        """Set new Atoms object and reanalyze."""
        logger.debug("Resetting Atoms in SBU {0}".format(self.name))
        self.atoms = atoms
        if analyze:
            logger.debug("\tAnalysis required.")
            self._analyze()

    def copy(self):
        """Return a copy of the object."""
        logger.debug("SBU {0}: creating copy.".format(self.name))
        new = SBU(name=str(self.name), atoms=None)
        new.set_atoms(atoms=self.get_atoms(), analyze=False)
        new.mmtypes = np.copy(self.mmtypes)
        new.bonds = np.copy(self.bonds)
        new.shape = list(self.shape)
        new.pg = self.pg

        return new

    def transfer_tags(self, fragment):
        """Transfer tags between an aligned fragment and the SBU."""
        logger.debug("\tTagging dummies in SBU {n}.".format(n=self.name))
        # we keep a record of used tags.
        unused = [x.index for x in self.atoms if x.symbol == "X"]
        for atom in fragment:
            ids = [s.index for s in self.atoms if s.index in unused]
            pf = atom.position
            ps = self.atoms.positions[unused]
            d = np.linalg.norm(ps-pf, axis=1)
            si = ids[np.argmin(d)]
            self.atoms[si].tag = atom.tag
            unused.remove(si)

    def define_geometry(self, element, schoenflies):
        """
        Generate SBU with a geometry from an element.

        It will read an element and an ASE Atoms object will be created with the dummy atoms
        forming the indicated geometry.
        """
        struc = skeleton_X[schoenflies]
        if element != "":
            struc += ase.Atom(element, (0, 0, 0))
        self.atoms = struc
        self._analyze()


def read_sbu_database(path=None):
    """
    Return a dictionnary of ASE Atoms as SBUs.

    Parameters:
    -----------
    path : str
        Folder or file where the SBUs are contained.
    """
    logger.debug("Reading the database.")

    sbu = {}
    if path is not None:
        logger.info("Loading the building units from {0}".format(path))
        sbu_tmp = read_sbu(path=path)
        sbu.update(sbu_tmp)
    else:
        logger.info("Loading the building units from default library")
        sbu_tmp = read_sbu(path=None)
        sbu.update(sbu_tmp)
    sbu_len = len(sbu)

    logger.info("{0:<5} sbu loaded".format(sbu_len))
    return sbu
