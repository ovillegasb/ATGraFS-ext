"""
Module that registers the main classes and functions of the ATGraFS-ext.
"""

import logging

from atgrafsE.utils.topology import read_topologies_database
from atgrafsE.utils.sbu import read_sbu_database

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

    def make(self, topology_name=None, sbu_names=None, sbu_dict=None, supercell=(1,1,1), coercion=False):
        """
        Create a framework using given topology and sbu.

        Main funtion of Autografs. The sbu names and topology's
        are to be taken from the compiled databases. The sbu_dict
        can also be passed for multiple components frameworks.
        If the sbu_names is a list of tuples in the shape 
        (name,n), the number n will be used as a drawing probability
        when multiple options are available for the same shape.
        topology_name -- name of the topology to use
        sbu_names     -- list of names of the sbu to use
        sbu_dict -- (optional) one to one sbu to slot correspondance
                    in the shape {index of slot : 'name of sbu'}
        supercell -- (optional) creates a supercell pre-treatment
        coercion -- (optional) force the compatibility to only consider
                    the multiplicity of SBU
        """
        logger.info("{0:-^50}".format(" Starting Framework Generation "))
        logger.info("")
