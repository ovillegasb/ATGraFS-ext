"""TODO"""

import logging
import copy
import os

import ase
import scipy.optimize
import scipy.linalg
import numpy as np

from collections import defaultdict

from atgrafsE.utils.io import write_gin

logger = logging.getLogger(__name__)


class Framework:
    """
    The Framework object contain the results of an Autografs run.

    Designed to store and post-process the aligned fragments and the
    corresponding topology of an Autografs-made structure. 
    Methods available are setters and getters for the topology, individual SBU,
    rescaling of the total structure, and finally the connecting and returning
    of a clean ASE Atoms object.
    The SBUs are assumed to already be aligned and tagged by the 
    Framework generator, and both bonds and mmtypes are assumed to be ordered.
    The bond matrices are also assumed to be block symmetrical.
    """
    def __init__(self, topology=None, building_units=None, mmtypes=None, bonds=None):
        """Constructor for the Framework class.

        Can be initialized empty and filled consecutively.
        topology -- Topology object.
        SBU      -- list of ASE Atoms objects.
        mmtypes  -- array of type names. e.g: 'C_R','O_3'...
        bonds    -- block-symmetric matrix of bond orders.
        """
        logger.debug("Creating Framework instance.")
        self.topology = topology
        if building_units is not None:
            self.SBU = building_units
        else:
            self.SBU = {}
        if mmtypes is not None:
            self.mmtypes = np.asarray(mmtypes)
        else:
            self.mmtypes = []
        if bonds is not None:
            self.bonds = np.asarray(bonds)
        else:
            self.bonds = []
        # keep a dict of elements to delete in 
        # each SBU at connection time. Necessary for example
        # during iterative functionalization.
        self._todel = defaultdict(list)

    def __contains__(self, obj):
        """Iterable intrinsic"""
        r = False
        if hasattr(obj, 'atoms'):
            r = any([obj.atoms==sbu.atoms for sbu in self.SBU.values()])
        return r

    def __delitem__(self, key):
        """Indexable intrinsic"""
        logger.info("Deleting {0} {1}".format(self.SBU[key].name,key))
        del self.SBU[key]

    def __setitem__(self, key, obj):
        """Indexable intrinsic"""
        if hasattr(obj, 'atoms'):
            self.SBU[key] = object

    def __getitem__(self, key):
        """Indexable intrinsic"""
        return self.SBU[key]

    def __len__(self):
        """Sizeable intrinsic"""
        return len(self.SBU)

    def __iter__(self):
        """Iterable intrinsic"""
        return iter(self.SBU.items())

    def copy(self):
        """Return a copy of itself as a new instance"""
        new = self.__class__(
            topology = self.get_topology(),
            building_units = copy.deepcopy(self.SBU),
            mmtypes = self.get_mmtypes(),
            bonds = self.get_bonds()
        )
        new._todel = copy.deepcopy(self._todel)
        return new

    def get_atoms(self, dummies=False):
        """Return the concatenated Atoms objects.

        The concatenation can either remove the dummies and
        connect the corresponding atoms or leave hem in place
        clean -- remove the dummies if True
        """
        logger.debug("Creating ASE Atoms from framework.")
        if dummies:
            logger.debug("\tDummies will be kept.")
            logger.debug("\tNo connection between SBU will occur.")
        else:
            logger.debug("\tDummies will be removed during connection.")
        # concatenate every sbu into one Atoms object
        framework = self.copy()
        cell = framework.topology.atoms.get_cell()
        pbc  = framework.topology.atoms.get_pbc()
        structure = ase.Atoms(cell=cell,pbc=pbc)
        for idx,sbu in framework:
            atoms = sbu.atoms.copy()
            todel = framework._todel[idx]
            if len(todel)>0:
                del atoms[todel]
            framework[idx].set_atoms(atoms,analyze=True)
            structure += atoms
        bonds   = framework.get_bonds()
        mmtypes = framework.get_mmtypes()
        symbols = np.asarray(structure.get_chemical_symbols())
        if not dummies:
            # keep track of dummies
            xis   = [x.index for x in structure if x.symbol=="X"]
            tags  = structure.get_tags()
            pairs = [np.argwhere(tags==tag) for tag in set(tags[xis])]
            for pair in pairs:
                # if lone dummy, cap with hydrogen
                if len(pair)==1:
                    xi0 = pair[0]
                    xis.remove(xi0)
                    symbols[xi0] = "H"
                    mmtypes[xi0] = "H_" 
                else:
                    xi0,xi1 = pair
                    bonds0  = np.where(bonds[:,xi0]>0.0)[0]
                    bonds1  = np.where(bonds[:,xi1]>0.0)[0]
                    # dangling bit, mayhaps from defect
                    if len(bonds0)==0 and len(bonds1)!=0:
                        xis.remove(xi1)
                        symbols[xi1] = "H"
                        mmtypes[xi1] = "H_" 
                    elif len(bonds1)==0 and len(bonds0)!=0:
                        xis.remove(xi0)
                        symbols[xi0] = "H"
                        mmtypes[xi0] = "H_"                         
                    else:
                        # the bond order will be the maximum one
                        bo      = max(np.amax(bonds[xi0,:]),
                                      np.amax(bonds[xi1,:]))
                        # change the bonds
                        ix        = np.ix_(bonds0,bonds1)
                        bonds[ix] = bo
                        ix        = np.ix_(bonds1,bonds0)
                        bonds[ix] = bo
            # book keeping on what has disappeared
            structure.set_chemical_symbols(symbols)
            bonds   = np.delete(bonds,xis,axis=0)
            bonds   = np.delete(bonds,xis,axis=1)
            mmtypes = np.delete(mmtypes,xis)
            del structure[xis]

        return structure, bonds, mmtypes

    def get_mmtypes(self):
        """Return and update the current bond matrix"""
        logger.debug("Updating framework MM types.")
        mmtypes = []
        for _,sbu in self:
            mmtypes.append(sbu.mmtypes)
        mmtypes = np.hstack(mmtypes)
        self.mmtypes = mmtypes
        return np.copy(self.mmtypes)

    def get_bonds(self):
        """Return and update the current bond matrix"""
        logger.debug("Updating framework bond matrix.")
        bonds = []
        for _,sbu in self:
            bonds.append(sbu.bonds)
        bonds = scipy.linalg.block_diag(*bonds)
        self.bonds = bonds
        return bonds

    def set_topology(self, topology):
        """Set the topology attribute with an ASE Atoms object."""
        logger.debug("Setting topology.")
        self.topology = topology.copy()

    def get_topology(self):
        """Return a copy of the topology attribute as an ASE Atoms object."""
        return self.topology.copy()

    def append(self, index, sbu, update=False):
        """Append all data releted to a building unit in the framework.

        This includes the ASE Atoms object, the bonding matrix, and the 
        molecular mechanics atomic types as numpy arrays. These three objects 
        are related through indexing: sbu[i] has a MM type mmtypes[i] and 
        a bonding array of bonds[i,:] or bonds[:,i]
        sbu     -- the Atoms object
        bonds   -- the bonds numpy array, of size len(sbu) by len(sbu).
        mmtypes -- the MM atomic types.
        """
        # first append the atoms object to the list of sbu
        logger.debug("Appending SBU {n} to framework.".format(n=sbu.name))

        self.SBU[index] = sbu
        if update:
            # make the bonds matrix with a new block
            self.bonds   = self.get_bonds()
            # append the atom types
            self.mmtypes = self.get_mmtypes()

    def scale(self, cellpar):
        """Scale the building units positions by a factor alpha.

        This uses the correspondance between the atoms in the topology
        and the building units in the SBU list. Indeed, SBU[i] is centered on 
        topology[i]. By scaling the topology, we obtain a new center for the 
        sbu.
        alpha -- scaling factor
        """
        if len(cellpar)==3:
            # 2D case
            cellpar = [cellpar[0],cellpar[1],0.0,90.0,90.0,cellpar[2]]
        # scale using topology as template
        cell = ase.geometry.cellpar_to_cell(cellpar)
        self.topology.atoms.set_cell(cell,scale_atoms=True)
        # then center the SBUs on this position
        for i, sbu in self:
            center = self.topology.atoms[i]
            cop = sbu.atoms.positions.mean(axis=0)
            sbu.atoms.positions += center.position - cop

    def refine(self, alpha0=[1.0, 1.0, 1.0]):
        """Refine cell scaling to minimize distances between dummies.

        We already have tagged the corresponding dummies during alignment,
        so we just need to calculate the MSE of the distances between 
        identical tags in the complete structure
        alpha0 -- starting point of the scaling search algorithm
        """
        logger.info("Refining unit cell.")
        # get the scaled cell, normalized
        I     = np.eye(3)*alpha0
        cell0 = self.topology.atoms.get_cell()
        pbc = sum(self.topology.atoms.get_pbc())
        cell0 = cell0.dot(I/np.linalg.norm(cell0,axis=0))
        cellpar0 = ase.geometry.cell_to_cellpar(cell0, radians=False)
        if pbc==2:
            cellpar0 = [cellpar0[0],cellpar0[1],cellpar0[5]]
            cellpar0 = np.array(cellpar0)
        # compile a list of mutual pairs
        atoms,_,_    = self.get_atoms(dummies=True)
        tags         = atoms.get_tags()
        # find the pairs...
        pairs = [np.argwhere(tags==tag) for tag in set(tags) if tag>0]
        pairs = [p for p in pairs if len(p)==2]
        pairs =  np.asarray(pairs).reshape(-1,2)
        # define the cost function
        def MSE(x):
            """Return cost of scaling as MSE of distances"""
            # scale with this parameter
            self.scale(cellpar=x)
            atoms,_,_    = self.get_atoms(dummies=True)
            # reinitialize stuff
            self.scale(cellpar=cellpar0)
            # find the distances
            d = [atoms.get_distance(i0,i1,mic=True) for i0,i1 in pairs]
            d = np.asarray(d)
            mse = np.mean(d**2)
            logger.info("\t|--> Scaling error = {e:>5.3f}".format(e=mse))
            return mse
        # first get an idea of the bounds.
        bounds = list(zip(0.1*cellpar0, 2.0*cellpar0))
        result = scipy.optimize.minimize(
            fun = MSE, 
            x0 = cellpar0, 
            method = "L-BFGS-B", 
            bounds = bounds, 
            tol=0.05, 
            options={"eps":0.1}
        )

        self.scale(cellpar=result.x)
        logger.info("Best cell parameters found:")
        logger.info("\ta = {a:<5.1f} Angstroms".format(a=result.x[0]))
        logger.info("\tb = {b:<5.1f} Angstroms".format(b=result.x[1]))
        if pbc == 2:
            logger.info("\tgamma = {gamma:<3.1f} degrees".format(gamma=result.x[2]))
        else:
            logger.info("\tc = {c:<5.1f} Angstroms".format(c=result.x[2]))
            logger.info("\talpha = {alpha:<3.1f} degrees".format(alpha=result.x[3]))
            logger.info("\tbeta  = {beta:<3.1f} degrees".format(beta=result.x[4]))
            logger.info("\tgamma = {gamma:<3.1f} degrees".format(gamma=result.x[5]))

    def write(self, f="./mof", ext="gin"):
        """Write a chemical information file to disk in selected format"""
        atoms, bonds, mmtypes = self.get_atoms(dummies=False)

        # add a numerical artifact for optimization in z direction
        if sum(atoms.pbc)==2:
            atoms.positions[:,2] += numpy.random.rand(len(atoms))*0.01

        path = os.path.abspath("{path}.{ext}".format(path=f,ext=ext))
        logger.info("Framework saved to disk at {p}".format(p=path))

        if ext=="gin":
            write_gin(path,atoms,bonds,mmtypes)
        else:
            ase.io.write(path,atoms)
