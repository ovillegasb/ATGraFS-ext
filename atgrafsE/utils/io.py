"""."""

import os
import logging

import numpy as np
import ase
from ase.data import chemical_symbols
from ase.spacegroup import crystal

from atgrafsE.utils import __data__

logger = logging.getLogger(__name__)

coffee = """
    (  )   (   )  )
     ) (   )  (  (
     ( )  (    ) )
     _____________
    <_____________> ___
    |             |/ _ \\
    |               | | |
    |               |_| |
 ___|             |\\___/
/    \\___________/    \\
\\_____________________/
"""

def read_cgd(path=None):
    """
    Return a dictionary of topologies as ASE Atoms objects

    The format CGD is used mainly by the Systre software and by Autografs. All
    details can be read on the website: http://rcsr.anu.edu.au/systre

    The program SYSTRE is designed to analyze periodic nets as they arise in
    the study of extended crystal structures (as opposed to molecular crystals).
    The acronym stands for Symmetry, Structure and Refinement. SYSTRE uses a
    method called barycentric placement to determine the ideal (i.e., maximal
    embeddable) symmetry of a crystal net and to analyze its topological
    structure. It then generates a unique "fingerprint" for the topological
    type of the given net and uses that to look up the structure in its built-in
    database

    Currently, SYSTRE's built-in database contains all feasible nets from the
    RCSR online resource as of May 22, 2013.

    Reticular Chemistry Structure Resource (RCSR) constructed especially to be
    useful in the design of new structures of crystalline materials and in the
    analysis of old ones (cf. the discussion of reticular chemistry by Yaghi,
    O. M. et al. Nature 2003, 423, 705-714). 

    """
    root = os.path.join(__data__,"topologies")
    topologies = {}
    # we need the names of the groups and their correspondance in ASE spacegroup data
    # this was compiled using Levenshtein distances and regular expressions
    groups_file = os.path.join(root,"HermannMauguin.dat")
    logger.debug("groups_file: %s" % groups_file)
    grpf = open(groups_file,"rb")
    groups = {l.split()[0]:l.split()[1] for l in grpf.read().decode("utf8").splitlines()}
    grpf.close()

    # read the rcsr topology data
    if path is None:
        topology_file = os.path.join(root,"nets.cgd")
    else:
        topology_file = os.path.abspath(path)

    # the script as such starts here
    error_counter = 0
    with open(topology_file,"rb") as tpf:
        text = tpf.read().decode("utf8")
        # split the file by topology
        topologies_raw = [t.strip().strip("CRYSTAL") for t in text.split("END")]
        topologies_len = len(topologies_raw)
        logger.info("{0:<5} topologies before treatment".format(topologies_len))
        # long operation
        logger.info("This might take a few minutes. Time for coffee!")
        ####logger.info("(")
        ####logger.info(" )")
        ####logger.info("[_])")
        logger.info(coffee)
        #### TODO
        #### Paralelizar este paso
        for topology_raw in topologies_raw:
            # read from the template.
            # the edges are easier to comprehend by edge center
            try:
                lines = topology_raw.splitlines()
                lines = [l.split() for l in lines if len(l)>2]
                name = None
                group = None
                cell = []
                symbols = []
                nodes = []
                for l in lines:
                    if l[0].startswith("NAME"):
                        name = l[1].strip()
                    elif l[0].startswith("GROUP"):
                        group = l[1]
                    elif l[0].startswith("CELL"):
                        cell = np.array(l[1:], dtype=np.float32)
                    elif l[0].startswith("NODE"):
                        this_symbol = chemical_symbols[int(l[2])]
                        this_node = np.array(l[3:], dtype=np.float32)
                        nodes.append(this_node)
                        symbols.append(this_symbol)
                    elif (l[0].startswith("#") and l[1].startswith("EDGE_CENTER")):
                        # linear connector
                        this_node = np.array(l[2:], dtype=np.float32)
                        nodes.append(this_node)
                        symbols.append("He")
                    elif l[0].startswith("EDGE"):
                        # now we append some dummies
                        s    = int((len(l)-1)/2)
                        midl = int((len(l)+1)/2)
                        x0  = np.array(l[1:midl],dtype=np.float32).reshape(-1,1)
                        x1  = np.array(l[midl:] ,dtype=np.float32).reshape(-1,1)
                        xx  = np.concatenate([x0,x1],axis=1).T
                        com = xx.mean(axis=0)
                        xx -= com
                        xx  = xx.dot(np.eye(s)*0.5)
                        xx += com
                        nodes   += [xx[0],xx[1]]
                        symbols += ["X","X"]
                nodes = np.array(nodes)
                if len(cell)==3:
                    # 2D net, only one angle and two vectors.
                    # need to be completed up to 6 parameters
                    pbc  = [True,True,False] 
                    cell = np.array(list(cell[0:2])+[10.0,90.0,90.0]+list(cell[2:]), dtype=np.float32)
                    # node coordinates also need to be padded
                    nodes = np.pad(nodes, ((0,0),(0,1)), 'constant', constant_values=0.0)
                elif len(cell)<3:
                    error_counter += 1
                    continue
                else:
                    pbc = True
                # now some postprocessing for the space groups
                setting = 1
                if ":" in group:
                    # setting might be 2
                    group, setting = group.split(":")
                    try: 
                        setting = int(setting.strip())
                    except ValueError:
                        setting = 1
                # ASE does not have all the spacegroups implemented yet
                if group not in groups.keys():
                    error_counter += 1
                    continue
                else:
                    # generate the crystal
                    group     = int(groups[group])
                    topology  = crystal(symbols=symbols,
                                        basis=nodes,
                                        spacegroup=group,
                                        setting=setting,
                                        cellpar=cell,
                                        pbc=pbc,
                                        primitive_cell=False,
                                        onduplicates="keep")
                    # TODO !!! find a way to use equivalent positions for 
                    # the multiple components frameworks !!!
                    # use the info keyword to store it
                    topologies[name] = topology
            except Exception:
                error_counter += 1
                continue
    logger.info("Topologies read with {err} errors.".format(err=error_counter))

    return topologies


def read_sbu(path=None, formats=["xyz"]):
    """
    Return a dictionary of Atoms objects.

    If the path is not specified, use the default library.
    TODO: Should use a chained iterable of path soon.
    path    -- where to find the sbu
    formats -- what molecular file format to read
    """
    if path is not None:
        path = os.path.abspath(path)
    else:
        path = os.path.join(__data__,"sbu")

    logger.debug("path: {}".format(path))

    SBUs = {}
    for sbu_file in os.listdir(path):
        ext = sbu_file.split(".")[-1]
        if ext in formats:
            for sbu in ase.io.iread(os.path.join(path,sbu_file)):
                try:
                    name  = sbu.info["name"]
                    SBUs[name] = sbu
                except Exception as e:
                    continue
    return SBUs

