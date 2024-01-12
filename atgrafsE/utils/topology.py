"""Functions that manage the system topology database."""

import os
import logging
import requests
import shutil
import pickle

from atgrafsE.utils import __data__
from atgrafsE.utils.io import read_cgd

logger = logging.getLogger(__name__)


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




