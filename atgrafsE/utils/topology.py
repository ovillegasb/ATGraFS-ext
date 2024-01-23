"""Functions that manage the system topology database."""

import os
import logging
import requests
import shutil
import pickle
import copy

import numpy as np

from ase.neighborlist import NeighborList
from ase import Atoms
from ase.spacegroup import crystal, Spacegroup

from atgrafsE.utils import __data__
from atgrafsE.utils.io import read_cgd
from atgrafsE.utils import symmetry

logger = logging.getLogger(__name__)


class Topology:
    """Contener class for the topology information."""

    def __init__(self, name, atoms, analyze=True):
        """Construct for a topology, from an ASE Atoms."""
        logger.debug("Creating Topology {0}".format(name))
        self.name = name
        self.atoms = atoms 
        # initialize empty fragments
        # shapes and symmops will be used to find
        # corresponding SBUs.
        self.fragments = {}
        self.shapes = {}
        self.pointgroups = {}
        self.equivalent_sites = []
        # fill it in
        if analyze:
            self._analyze()

    def copy(self):
        """Return a copy of itself as a new instance."""
        new = self.__class__(
            name=str(self.name),
            atoms=self.atoms.copy(),
            analyze=False
        )
        new.fragments = copy.deepcopy(self.fragments)
        new.shapes = copy.deepcopy(self.shapes)
        return new

    def has_compatible_slots(self, sbu, coercion=False):
        """Return [shapes...] for the slots compatible with the SBU"""
        slots = []
        complist = [(ai, self.shapes[ai],self.pointgroups[ai]) 
                    for ai in self.fragments.keys()]
        seen_idx = []
        for idx, shape, pg in complist:
            if idx in seen_idx:
                continue
            eq_sites = [s for s in self.equivalent_sites if idx in s][0]
            eq_sites = [s for s in eq_sites if self.shapes[s][-1]==shape[-1]]
            seen_idx += eq_sites
            # test for compatible multiplicity  
            mult = (sbu.shape[-1] ==  shape[-1])
            if not mult:
                continue
            # pointgroups are more powerful identifiers
            if pg==sbu.pg:
                slots+=[tuple(c[1]) for c in complist if c[0] in eq_sites]
                continue
            # the sbu has at least as many symmetry axes
            symm = (sbu.shape[:-1]-shape[:-1]>=0).all()
            if symm:
                slots+=[tuple(c[1]) for c in complist if c[0] in eq_sites]
                continue
            if coercion:
                # takes objects of corresponding
                # multiplicity as compatible.
                slots+=[tuple(c[1]) for c in complist if c[0] in eq_sites]
                continue

        return slots

    def _get_cutoffs(self, Xis, Ais):
        """Return the cutoffs leading to the desired connectivity"""
        # initialize cutoffs to small non-zero skin partameter
        logger.debug("Generating cutoffs")
        skin=5e-3
        cutoffs=np.zeros(len(self.atoms)) + skin
        # we iterate over non-dummies
        for other_index in Ais:
            # we get the distances to all dummies and cluster accordingly
            dists = self.atoms.get_distances(other_index,Xis,mic=True)
            coord = self.atoms[other_index].number
            if coord < len(dists):
                # keep only the closest ones up to coordination
                these_dummies = np.argpartition(dists, coord)
                these_dummies = these_dummies[:coord]
                cutoff = dists[these_dummies].max()
            else:
                cutoff = dists.max()
            cutoffs[other_index] = cutoff
        return cutoffs

    def _analyze(self):
        """Analyze the topology to cut the fragments out."""
        # separate the dummies from the rest
        logger.debug("Analyzing fragments of topology {0}.".format(self.name))
        numbers = np.asarray(self.atoms.get_atomic_numbers())
        Xis = np.where(numbers == 0)[0]
        Ais = np.where(numbers > 0)[0]
        # setup the tags
        tags = np.zeros(len(self.atoms))
        tags[Xis] = Xis + 1
        self.atoms.set_tags(tags)
        tags = self.atoms.get_tags()
        # analyze
        # first build the neighborlist
        cutoffs = self._get_cutoffs(Xis=Xis, Ais=Ais)
        neighborlist = NeighborList(cutoffs=cutoffs, bothways=True, self_interaction=False, skin=0.0)
        neighborlist.update(self.atoms)
        # iterate over non-dummies to find dummy neighbors
        for ai in Ais:
            # get indices and offsets of dummies only!
            ni, no = neighborlist.get_neighbors(ai)
            ni, no = zip(*[(idx, off) for idx, off in list(zip(ni, no)) if idx in Xis])
            ni = np.asarray(ni)
            no = np.asarray(no)

            # get absolute positions, no offsets
            positions = self.atoms.positions[ni] + no.dot(self.atoms.cell)

            # create the Atoms object
            fragment = Atoms("X"*len(ni), positions, tags=tags[ni])

            # calculate the point group properties
            max_order = len(ni)
            shape = symmetry.get_symmetry_elements(
                mol=fragment.copy(),
                max_order=max_order
            )
            logger.debug("Symmetry elements")

            pg = symmetry.PointGroup(mol=fragment.copy(), tol=0.1)
            logger.debug("Point Group")
            # save that info
            self.fragments[ai] = fragment
            self.shapes[ai] = shape
            self.pointgroups[ai] = pg.schoenflies
        # now getting the equivalent sites using the Spacegroup object
        logger.debug("Getting the equivalent sites using the Spacegroup object")
        sg = self.atoms.info["spacegroup"]
        if not isinstance(sg, Spacegroup):
            sg = Spacegroup(sg)
        scaled_positions = self.atoms.get_scaled_positions()
        seen_indices = []
        symbols = np.array(self.atoms.get_chemical_symbols())
        for ai in Ais:
            if ai in seen_indices:
                continue
            sites, _ = sg.equivalent_sites(scaled_positions[ai])
            these_indices = []
            for site in sites:
                norms = np.linalg.norm(scaled_positions-site, axis=1)
                if norms.min() < 1e-6:
                    these_indices.append(norms.argmin())
                # take pbc into account
                norms = np.abs(norms - 1.0)
                if norms.min() < 1e-6:
                    these_indices.append(norms.argmin())

            these_indices = [idx for idx in these_indices if idx in Ais]
            seen_indices += these_indices
            self.equivalent_sites.append(these_indices)

        logger.debug("{es} equivalent sites kinds.".format(es=len(self.equivalent_sites)))


def download_topologies():
    """Downloads the topology file from the RCSR website"""
    url  = "http://rcsr.anu.edu.au/downloads/RCSRnets.cgd"
    root = os.path.join(__data__, "topologies")
    path = os.path.join(root, "nets.cgd")
    resp = requests.get(url, stream=True)
    if resp.status_code == 200:
        logger.info("Successfully downloaded the nets from RCSR.")
        resp.raw.decode_content = True
        with open(path,"wb") as outpt:
            shutil.copyfileobj(resp.raw, outpt)


def read_topologies_database(update=False, path=None, use_defaults=True):
    """Return a dictionary of topologies as ASE Atoms."""
    root     = os.path.join(__data__,"topologies")
    db_file  = os.path.join(root,"topologies.pkl")
    # http://rcsr.anu.edu.au/downloads/RCSRnets-2019-06-01.cgd
    cgd_file = os.path.join(root,"nets.cgd")
    logger.debug("root: %s" % root)
    logger.debug("db_file: %s" % db_file)
    logger.debug("cgd_file: %s" % cgd_file)
    topologies = {}
    # Tests if the database has been loaded.
    if ((not os.path.isfile(db_file)) or (update)):
        # Tests if the local database exists
        if (not os.path.isfile(cgd_file)) and use_defaults:
            # If it does not exist, it will download
            logger.info("Downloading the topologies from RCSR.")
            download_topologies()
        if use_defaults:
            logger.info("Loading the topologies from RCSR default library")
            topologies_tmp = read_cgd(path=None)
            topologies.update(topologies_tmp)
        if path is not None:
            logger.info("Loading the topologies from {0}".format(path))
            topologies_tmp = read_cgd(path=path)
            topologies.update(topologies_tmp)
        topologies_len = len(topologies)
        logger.info("{0:<5} topologies saved".format(topologies_len))

        with open(db_file,"wb") as pkl:
            pickle.dump(obj=topologies,file=pkl)

        return topologies

    else:
        logger.info("Using saved topologies")
        with open(db_file, "rb") as pkl:
            topologies = pickle.load(file=pkl)
            topologies_len = len(topologies)
            logger.info("{0:<5} topologies loaded".format(topologies_len))
            
            return topologies




