"""
Module that registers the main classes and functions of the ATGraFS-ext.
"""

import logging

logger = logging.getLogger(__name__) 

class Autografs:
    """Framework maker class to generate ASE Atoms objects from topologies."""

    def __init__(self):
        """Constructor for the Autografs framework maker."""
        logger.info("{0:*^80}".format("*"))
        logger.info("* {0:^76} *".format("AuToGraFS"))