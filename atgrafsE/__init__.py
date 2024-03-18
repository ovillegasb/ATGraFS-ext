#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ATGraFS-ext - *Automatic Topological Generator for Framework Structure Extended*.

ATGraFS-ext was released as a module to extend the original AuToGraFS capabilities.


For more information you can access: https://github.com/ovillegasb/ATGraFS-ext

References:
-----------
AuToGraFS: Automatic Topological Generator for Framework Structures.
Addicoat, M. a, Coupry, D. E., & Heine, T. (2014).
The Journal of Physical Chemistry. A, 118(40), 9607–14.

"""

import logging
from atgrafsE.core import Autografs


__all__ = ["core"]
__version__ = "0.0.1"
__author__ = "Orlando Villegas"
__email__ = "ovillegas.bello0317@gmail.com"
__credits__ = ["Prof. Matthew Addicoat", "Demien Coupry"]
__license__ = "GPLv3"
__maintainer__ = "Orlando Villegas"
__status__ = "Production"
__copyright__ = "Copyright 2024"


logging.basicConfig(
    format="{} - {} | {}: {}".format(
        "%(asctime)s",
        "%(name)s",
        "%(levelname)s",
        "%(message)s"
    ),
    encoding='utf-8',
    datefmt='%I:%M:%S'
)  # filename='atgrafsE.log'


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
