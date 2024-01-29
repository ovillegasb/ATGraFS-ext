"""."""

import os
import logging
import pickle

from ase import Atoms
import numpy as np

from atgrafsE.utils import __data__
from atgrafsE.utils.io import read_sbu

from atgrafsE.utils import symmetry
from atgrafsE.utils.mmanalysis import analyze_mm

logger = logging.getLogger(__name__)


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
        logger.debug("SBU {0}: analyze bonding and symmetry.".format(self.name))

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
        bonds, mmtypes = analyze_mm(self.get_atoms())
        self.bonds = bonds
        self.mmtypes = mmtypes

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


def read_sbu_database(update=False, path=None, use_defaults=True):
    """Return a dictionnary of ASE Atoms as SBUs."""
    db_file = os.path.join(__data__, "sbu/sbu.pkl")
    user_db = (path is not None)
    no_dflt = (not os.path.isfile(db_file))
    logger.debug("db_file: %s" % db_file)
    logger.debug("user_db: %s" % user_db)
    logger.debug("no_dflt: %s" % no_dflt)
    if (user_db or update or no_dflt):
        sbu = {}
        if use_defaults:
            logger.info("Loading the building units from default library")
            sbu_tmp = read_sbu(path=None)
            sbu.update(sbu_tmp)

        if path is not None:
            logger.info("Loading the building units from {0}".format(path))
            sbu_tmp = read_sbu(path=path)
            sbu.update(sbu_tmp)

        sbu_len = len(sbu)
        logger.info("{0:<5} sbu loaded.".format(sbu_len))
        with open(db_file, "wb") as pkl:
            pickle.dump(obj=sbu, file=pkl)

        return sbu

    else:
        logger.info("Using saved sbu")
        with open(db_file, "rb") as pkl:
            sbu = pickle.load(file=pkl)
            sbu_len = len(sbu)
            logger.info("{0:<5} sbu loaded".format(sbu_len))
            return sbu
