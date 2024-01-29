"""Submodule that records information input and output functions."""

import os
import re
import logging

import numpy as np
import ase
from ase.data import chemical_symbols
from ase.spacegroup import crystal
from ase.spacegroup.spacegroup import get_datafile

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


def read_spacegroup_data():
    """Read the names of the spacegroups contained in the ASE database."""
    grps = {}
    spacegroup_data = get_datafile()
    logger.debug("The ASE database file will be searched: %s" % spacegroup_data)
    regex_grps = re.compile(r"""
        ^(?P<num>\d+)\s+
        """, re.X)

    with open(spacegroup_data, "r", encoding="utf8") as DATA:
        for line in DATA:
            if regex_grps.match(line):
                line = line.split()
                grps["".join(line[1:]).upper()] = int(line[0])
    return grps


def read_cgd(path=None):
    """
    Return a dictionary of topologies as ASE Atoms objects.

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
    root = os.path.join(__data__, "topologies")
    topologies = {}
    # we need the names of the groups and their correspondance in ASE spacegroup data
    # this was compiled using Levenshtein distances and regular expressions

    # The ASE's own database is loaded.
    spacegroups = read_spacegroup_data()

    # read the rcsr topology data
    if path is None:
        topology_file = os.path.join(root, "nets.cgd")
    else:
        topology_file = os.path.abspath(path)

    # the script as such starts here
    error_counter = 0
    with open(topology_file, "r") as tpf:
        text = tpf.read()
        # split the file by topology
        topologies_raw = [t.strip().strip("CRYSTAL") for t in text.split("END")]
        # Remove lines that may be blank
        topologies_raw = [t for t in topologies_raw if len(t.splitlines()) > 0]

        topologies_len = len(topologies_raw)
        logger.info("{0:<5} topologies before treatment".format(topologies_len))
        # long operation
        logger.info("This might take a few minutes. Time for coffee!")
        logger.info(coffee)
        ####TODO
        #### Paralelizar este paso
        for topology_raw in topologies_raw:
            # read from the template.
            # the edges are easier to comprehend by edge center
            lines = topology_raw.splitlines()
            lines = [line.split() for line in lines if len(line) > 2]
            name = None
            group = None
            cell = []
            symbols = []
            nodes = []
            for line in lines:
                if line[0].startswith("NAME"):
                    name = line[1].strip()
                elif line[0].startswith("GROUP"):
                    group = line[1].upper()
                elif line[0].startswith("CELL"):
                    cell = np.array(line[1:], dtype=np.float64)
                elif line[0].startswith("NODE"):
                    this_symbol = chemical_symbols[int(line[2])]
                    this_node = np.array(line[3:], dtype=np.float64)
                    nodes.append(this_node)
                    symbols.append(this_symbol)
                elif (line[0].startswith("#") and line[1].startswith("EDGE_CENTER")):
                    # linear connector
                    this_node = np.array(line[2:], dtype=np.float64)
                    nodes.append(this_node)
                    symbols.append("He")
                elif line[0].startswith("EDGE"):
                    # now we append some dummies
                    s = int((len(line)-1)/2)
                    midl = int((len(line)+1)/2)
                    x0 = np.array(line[1:midl], dtype=np.float64).reshape(-1, 1)
                    x1 = np.array(line[midl:], dtype=np.float64).reshape(-1, 1)
                    xx = np.concatenate([x0, x1], axis=1).T
                    com = xx.mean(axis=0)
                    xx -= com
                    xx = xx.dot(np.eye(s)*0.5)
                    xx += com
                    nodes += [xx[0], xx[1]]
                    symbols += ["X", "X"]

            if group is None:
                logger.debug("The group not detected, lines:")
                print(lines)
                error_counter += 1
                continue

            nodes = np.array(nodes)

            # Only cells with 3 dimensions or more will be considered.
            # The lengths are considered to be angstroms.
            if len(cell) == 3:
                # 2D net, only one angle and two vectors.
                # need to be completed up to 6 parameters
                # FORMAT: CELL a b gamma
                ##TODO
                pbc = [True, True, False]
                c = 10.0  # WHY?, test
                alpha = 90.0
                beta = 90.0
                cell = np.array(
                    list(cell[0:2]) + [c, alpha, beta] + list(cell[2:]), dtype=np.float64
                )
                # node coordinates also need to be padded
                nodes = np.pad(nodes, ((0, 0), (0, 1)), 'constant', constant_values=0.0)
            elif len(cell) < 3:
                logger.debug("The next group contains a cell with less than 3 dimensions: %s" % group)
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
            try:
                spg_num = spacegroups[group]
            except KeyError:
                logger.debug("The following group has not been detected in the ASE database: %s" % group)
                error_counter += 1
                continue

            # print(symbols, nodes, group, spg_num, setting, cell, pbc)
            try:
                topology = crystal(
                    symbols=symbols,
                    basis=nodes,
                    spacegroup=spg_num,
                    setting=setting,
                    cellpar=cell,
                    pbc=pbc,
                    primitive_cell=False,
                    onduplicates="keep"
                )
            except AssertionError:
                logger.debug("An error has been found inside the crital function, check defined parameters c, alpha and beta, group: %s" % group)
                error_counter += 1
                continue
            # TODO !!! find a way to use equivalent positions for
            # the multiple components frameworks !!!
            # use the info keyword to store it
            topologies[name] = topology

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
        path = os.path.join(__data__, "sbu")

    logger.debug("path: {}".format(path))

    SBUs = {}
    # if this is a file
    if path.endswith("xyz"):
        sbu_file = path
        ext = sbu_file.split(".")[-1]
        for sbu in ase.io.iread(sbu_file):
            try:
                name = sbu.info["name"]
                SBUs[name] = sbu
            except Exception as e:
                continue

    else:
        for sbu_file in os.listdir(path):
            ext = sbu_file.split(".")[-1]
            if ext in formats:
                for sbu in ase.io.iread(os.path.join(path, sbu_file)):
                    try:
                        name = sbu.info["name"]
                        SBUs[name] = sbu
                    except Exception as e:
                        continue
    return SBUs


def write_gin(path, atoms, bonds, mmtypes):
    """Write an GULP input file to disc."""
    with open(path, "w") as fileobj:
        fileobj.write('opti conp molmec noautobond conjugate cartesian unit positive unfix\n')
        fileobj.write('maxcyc 500\n')
        fileobj.write('switch bfgs gnorm 1.0\n')
        pbc = atoms.get_pbc()
        if pbc.any():
            cell = atoms.get_cell().tolist()
            if not pbc[2]:
                fileobj.write('{0}\n'.format('svectors'))
                fileobj.write('{0:.3f} {1:.3f} {2:.3f}\n'.format(*cell[0]))
                fileobj.write('{0:.3f} {1:.3f} {2:.3f}\n'.format(*cell[1]))
            else:
                fileobj.write('{0}\n'.format('vectors'))
                fileobj.write('{0:.3f} {1:.3f} {2:.3f}\n'.format(*cell[0]))
                fileobj.write('{0:.3f} {1:.3f} {2:.3f}\n'.format(*cell[1]))
                fileobj.write('{0:.3f} {1:.3f} {2:.3f}\n'.format(*cell[2]))
        fileobj.write('{0}\n'.format('cartesian'))
        symbols = atoms.get_chemical_symbols()
        #We need to map MMtypes to numbers. We'll do it via a dictionary
        symb_types=[]
        mmdic={}
        types_seen=1
        for m, s in zip(mmtypes,symbols):
            if m not in mmdic:
                mmdic[m]="{0}{1}".format(s,types_seen)
                types_seen+=1
                symb_types.append(mmdic[m])
            else:
                symb_types.append(mmdic[m])
        # write it
        for s, (x, y, z), in zip(symb_types, atoms.get_positions()):
            fileobj.write('{0:<4} {1:<7} {2:<15.8f} {3:<15.8f} {4:<15.8f}\n'.format(s,'core',x,y,z))
        fileobj.write('\n')
        bondstring = {
            4:'quadruple',
            3:'triple',
            2:'double',
            1.5:'resonant',
            1.0:'',
            0.5:'half',
            0.25:'quarter'
        }
        #write the bonding
        for (i0, i1), b in np.ndenumerate(bonds):
            if i0 < i1 and b > 0.0:
                fileobj.write('{0} {1:<4} {2:<4} {3:<10}\n'.format('connect',i0+1,i1+1,bondstring[b]))
        fileobj.write('\n')
        fileobj.write('{0}\n'.format('species'))
        for  k,v in  mmdic.items():
            fileobj.write('{0:<5} {1:<5}\n'.format(v,k))
        fileobj.write('\n')
        fileobj.write('library uff4mof\n')
        fileobj.write('\n')
        name = ".".join(path.split("/")[-1].split(".")[:-1])
        fileobj.write('output movie xyz {0}.xyz\n'.format(name))
        fileobj.write('output gen {0}.gen\n'.format(name))
        if sum(pbc)==3:
            fileobj.write('output cif {0}.cif\n'.format(name))
        return None

