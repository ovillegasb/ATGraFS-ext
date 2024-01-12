"""."""

import os
import logging
import pickle

from atgrafsE.utils import __data__
from atgrafsE.utils.io import read_sbu

logger = logging.getLogger(__name__)

def read_sbu_database(update=False, path=None, use_defaults=True):
    """Return a dictionnary of ASE Atoms as SBUs."""
    db_file = os.path.join(__data__,"sbu/sbu.pkl")
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
        with open(db_file,"wb") as pkl:
            pickle.dump(obj=sbu,file=pkl)

        return sbu

    else:
        logger.info("Using saved sbu")
        with open(db_file,"rb") as pkl:
            sbu = pickle.load(file=pkl)
            sbu_len = len(sbu)
            logger.info("{0:<5} sbu loaded".format(sbu_len))
            return sbu