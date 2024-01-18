"""
Module that registers the main classes and functions of the ATGraFS-ext.
"""

import logging

import ase
import numpy as np

from collections import defaultdict
from scipy.linalg import orthogonal_procrustes

from atgrafsE.utils.topology import read_topologies_database
from atgrafsE.utils.topology import Topology
from atgrafsE.utils.sbu import read_sbu_database

from atgrafsE.framework import Framework

from atgrafsE.utils.sbu import SBU

logger = logging.getLogger(__name__) 

class Autografs:
    """Framework maker class to generate ASE Atoms objects from topologies."""

    def __init__(self, topology_path=None, sbu_path=None, use_defaults=True, update=False):
        """Constructor for the Autografs framework maker."""
        logger.info("{0:*^80}".format("*"))
        logger.info("* {0:^76} *".format("AuToGraFS"))
        logger.info("* {0:^76} *".format("Automatic Topological Generator for Framework Structures Extended"))
        logger.info("{0:*^80}".format("*"))
        logger.info("")
        logger.info("Reading the topology database.")
        self.topologies = read_topologies_database(
            path=topology_path,
            use_defaults=use_defaults,
            update=update
        )
        logger.info("Reading the building units database.")
        self.sbu = read_sbu_database(
            path=sbu_path,
            use_defaults=use_defaults,
            update=update
        )
        
        # container for current topology
        self.topology = None
        #container for current sbu mapping
        self.sbu_dict    = None
        logger.info("")

    def set_topology(self, topology_name, supercell=(1,1,1)):
        """Create and store the topology object."""
        logger.info("Topology set to --> {topo}".format(topo=topology_name.upper()))

        # make the supercell prior to alignment
        if isinstance(supercell,int):
            supercell = (supercell, supercell, supercell)

        topology_atoms  = self.topologies[topology_name]
        if supercell != (1,1,1):
            logger.info("{0}x{1}x{2} supercell of the topology is used.".format(*supercell))
            topology_atoms *= supercell

        # make the Topology object
        logger.info("Analysis of the topology.")
        topology = Topology(name=topology_name, atoms=topology_atoms)

        self.topology = topology

    def make(self, topology_name=None, sbu_names=None, sbu_dict=None, supercell=(1,1,1), coercion=False):
        """
        Create a framework using given topology and sbu.

        Main funtion of Autografs. The sbu names and topology's
        are to be taken from the compiled databases. The sbu_dict
        can also be passed for multiple components frameworks.
        If the sbu_names is a list of tuples in the shape 
        (name,n), the number n will be used as a drawing probability
        when multiple options are available for the same shape.

        Parameters:
        -----------
        topology_name : str
            Name of the topology to use.

        sbu_names : list of str
            List of names of the sbu to use.

        sbu_dict : dict
            (optional) One to one sbu to slot correspondance in the shape
            {index of slot : 'name of sbu'}.

        supercell : tuple(x3) of int
            (optional) Creates a supercell pre-treatment.

        coercion : bool
            (optional) Force the compatibility to only consider the multiplicity of SBU.

        """
        logger.info("{0:-^50}".format(" Starting Framework Generation "))
        logger.info("")

        # only set the topology if not already done
        if topology_name is not None:
            self.set_topology(topology_name=topology_name, supercell=supercell)

        # container for the aligned SBUs
        aligned = Framework()
        aligned.set_topology(self.topology)

        # identify the corresponding SBU
        try:
            if sbu_dict is None and sbu_names is not None:
                logger.info("Scheduling the SBU to slot alignment.")
                self.sbu_dict = self.get_sbu_dict(
                    sbu_names=sbu_names,
                    coercion=coercion
                )
            elif sbu_dict is not None:
                logger.info("SBU to slot alignment is user defined.")
                # the sbu_dict has been passed. if not SBU object, create them
                for k, v in sbu_dict.items():
                    if not isinstance(v, SBU):
                        if not isinstance(v, ase.Atoms):
                            name = str(v)
                            v = self.sbu[name].copy()
                        elif "name" in v.info.keys():
                            name = v.info["name"]
                        else:
                            name = str(k)
                        sbu_dict[k] = SBU(name=name, atoms=v)
                self.sbu_dict = sbu_dict
            else:
                raise RuntimeError("Either supply sbu_names or sbu_dict.")
        except RuntimeError as exc:
            logger.error("Slot to SBU mappping interrupted.")
            logger.error("{exc}".format(exc=exc))
            logger.info("No valid framework was generated. Please check your input.")
            logger.info("You can coerce sbu assignment by directly passing a slot to sbu dictionary.")
            return

        # some logging
        self.log_sbu_dict(sbu_dict=self.sbu_dict, topology=self.topology)

        # carry on
        alpha = 0.0
        for idx, sbu in self.sbu_dict.items():
            logger.debug("Treating slot number {idx}".format(idx=idx))
            logger.debug("\t|--> Aligning SBU {name}".format(name=sbu.name))
            # now align and get the scaling factor
            sbu,f = self.align(
                fragment=self.topology.fragments[idx],
                sbu=sbu
            )
            alpha += f
            aligned.append(index=idx, sbu=sbu)

        logger.info("")
        aligned.refine(alpha0=alpha)
        logger.info("")
        logger.info("Finished framework generation.")
        logger.info("")
        logger.info("{0:-^50}".format(" Post-treatment "))
        logger.info("")

        return aligned

    def log_sbu_dict(self, topology, sbu_dict=None):
        """Does some logging on the chosen SBU mapping."""
        for idx, sbu in sbu_dict.items():
            logging.info("\tSlot {sl}".format(sl=idx))
            logging.info("\t   |--> SBU {sbn}".format(sbn=sbu.name))

    def get_vector_space(self, X):
        """Returns a vector space as four points."""
        # initialize
        x0 = X[0]
        # find the point most orthogonal
        dots = [x.dot(x0)for x in X]
        i1 = np.argmin(dots)
        x1 = X[i1]
        # the second point maximizes the same with x1
        dots = [x.dot(x1) for x in X[1:]]
        i2 = np.argmin(dots)+1
        x2 = X[i2]
        # we find a third point
        dots = [x.dot(x1)+x.dot(x0)+x.dot(x2) for x in X]
        i3 = np.argmin(dots)
        x3 = X[i3]

        return np.asarray([x0,x1,x2,x3])

    def get_sbu_dict(self, sbu_names, coercion=False):
        """Return a dictionary of SBU by corresponding fragment.

        This stage get a one to one correspondance between
        each topology slot and an available SBU from the list of names.
        topology  -- the Topology object
        sbu_names -- the list of SBU names as strings
        coercion -- wether to force compatibility by coordination alone
        """
        logger.debug("Generating slot to SBU map.")
        assert self.topology is not None
        weights  = defaultdict(list)
        by_shape = defaultdict(list)
        for name in sbu_names:
            # check if probabilities included
            if isinstance(name,tuple):
                name,p = name
                p    = float(p)
                name = str(name)
            else:
                p = 1.0
            # create the SBU object
            sbu = SBU(name=name,atoms=self.sbu[name])
            slots = self.topology.has_compatible_slots(sbu=sbu,coercion=coercion)
            if not slots:
                logger.debug("SBU {s} has no compatible slot in topology {t}".format(s=name,t=self.topology.name))
                continue
            for slot in slots:
                weights[slot].append(p)
                by_shape[slot].append(sbu)
        # now fill the choices
        sbu_dict = {}
        for index,shape in self.topology.shapes.items():       
            # here, should accept weights also
            shape = tuple(shape)
            if shape not in by_shape.keys():
                logger.info("Unfilled slot at index {idx}".format(idx=index))
            p = weights[shape]
            # no weights means same proba
            p /= np.sum(p)
            sbu_chosen = np.random.choice(by_shape[shape], p=p).copy()
            logger.debug("Slot {sl}: {sb} chosen with p={p}.".format(
                sl=index,
                sb=sbu_chosen.name,
                p=p)
            )
            sbu_dict[index] = sbu_chosen
        return sbu_dict

    def align(self, fragment, sbu):
        """Return an aligned SBU.

        The SBU is rotated on top of the fragment
        using the procrustes library within scipy.
        a scaling factor is also calculated for all three
        cell vectors.
        fragment -- the slot in the topology, ASE Atoms
        sbu      -- object to align, ASE Atoms
        """
        # first, we work with copies
        fragment = fragment.copy()
        # normalize and center
        fragment_cop = fragment.positions.mean(axis=0)
        fragment.positions -= fragment_cop
        sbu.atoms.positions -= sbu.atoms.positions.mean(axis=0)

        # identify dummies in sbu
        sbu_Xis = [x.index for x in sbu.atoms if x.symbol=="X"]
        # get the scaling factor
        size_sbu = np.linalg.norm(sbu.atoms[sbu_Xis].positions, axis=1)
        size_fragment = np.linalg.norm(fragment.positions, axis=1)
        alpha = size_sbu.mean() / size_fragment.mean()

        # TODO check initial scaling: it goes up too much with unit cell
        ncop = np.linalg.norm(fragment_cop)
        if ncop<1e-6:
            direction  = np.ones(3,dtype=np.float32)
            direction /= np.linalg.norm(direction)
        else:
            direction = fragment_cop / ncop

        # scaling for better alignment
        fragment.positions = fragment.positions.dot(np.eye(3)*alpha)
        alpha *= direction / 2.0
        # getting the rotation matrix
        X0 = sbu.atoms[sbu_Xis].get_positions()
        X1 = fragment.get_positions()
        if X0.shape[0] > 5:
            X0 = self.get_vector_space(X0)
            X1 = self.get_vector_space(X1)
        R, s = orthogonal_procrustes(X0, X1)
        sbu.atoms.positions = sbu.atoms.positions.dot(R) + fragment_cop
        fragment.positions += fragment_cop
        # TEST
        # ase.visualize.view(sbu.atoms+fragment)
        res_d = ase.geometry.distance(sbu.atoms[sbu_Xis], fragment)
        logger.debug("Residual distance: {d}".format(d=res_d))
        # tag the atoms
        sbu.transfer_tags(fragment)

        return sbu, alpha

