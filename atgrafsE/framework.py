"""Submodule that contains the functions and classes to work with metal-organic Framework."""

import logging
import copy
import os

import ase
import scipy.optimize
import scipy.linalg
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.transform import Rotation

import numpy as np
import random
import itertools as it

from collections import defaultdict

from atgrafsE.utils.topology import Topology
from atgrafsE.utils.operations import calculate_angle


logger = logging.getLogger(__name__)


class Framework:
    """The Framework object contain the results of an Autografs run.

    Designed to store and post-process the aligned fragments and the corresponding topology of an
    Autografs-made structure.  Methods available are setters and getters for the topology,
    individual SBU, rescaling of the total structure, and finally the connecting and returning of a
    clean ASE Atoms object. The SBUs are assumed to already be aligned and tagged by the Framework
    generator, and both bonds and mmtypes are assumed to be ordered. The bond matrices are also
    assumed to be block symmetrical.
    """

    def __init__(self, topology=None, building_units=None, mmtypes=None, bonds=None):
        """Initialize the Framework class.

        Can be initialized empty and filled consecutively.

        Parameters:
        -----------
        topology : Topology class.
            Class with system topology.
        SBU : list of ase.Atoms.
            A list of ASE Atomos objects.
        mmtypes : numpy.array of str.
            Array of type names. e.g: 'C_R','O_3'...
        bonds : numpy.array.
            Block-symmetric matrix of bond orders.
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

        self.complete_alignment = False

    def __contains__(self, obj):
        """Iterate intrinsic."""
        r = False
        if hasattr(obj, 'atoms'):
            r = any([obj.atoms == sbu.atoms for sbu in self.SBU.values()])
        return r

    def __delitem__(self, key):
        """Indexable intrinsic."""
        logger.info("Deleting {0} {1}".format(self.SBU[key].name, key))
        del self.SBU[key]

    def __setitem__(self, key, obj):
        """Indexable intrinsic."""
        if hasattr(obj, 'atoms'):
            self.SBU[key] = object

    def __getitem__(self, key):
        """Indexable intrinsic."""
        return self.SBU[key]

    def __len__(self):
        """Sizeable intrinsic."""
        return len(self.SBU)

    def __iter__(self):
        """Iterate intrinsic."""
        return iter(self.SBU.items())

    def copy(self):
        """Return a copy of itself as a new instance."""
        new = self.__class__(
            topology=self.get_topology(),
            building_units=copy.deepcopy(self.SBU),
            mmtypes=self.get_mmtypes(),
            bonds=self.get_bonds()
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
        pbc = framework.topology.atoms.get_pbc()
        structure = ase.Atoms(cell=cell, pbc=pbc)
        for idx, sbu in framework:
            atoms = sbu.atoms.copy()
            todel = framework._todel[idx]
            if len(todel) > 0:
                del atoms[todel]
            framework[idx].set_atoms(atoms, analyze=True)
            structure += atoms
        bonds = framework.get_bonds()
        mmtypes = framework.get_mmtypes()
        symbols = np.asarray(structure.get_chemical_symbols())
        if not dummies:
            # keep track of dummies
            xis = [x.index for x in structure if x.symbol == "X"]
            tags = structure.get_tags()
            pairs = [np.argwhere(tags == tag) for tag in set(tags[xis])]
            for pair in pairs:
                # if lone dummy, cap with hydrogen
                if len(pair) == 1:
                    xi0 = pair[0]
                    xis.remove(xi0)
                    symbols[xi0] = "H"
                    mmtypes[xi0] = "H_"
                else:
                    xi0, xi1 = pair
                    bonds0 = np.where(bonds[:, xi0] > 0.0)[0]
                    bonds1 = np.where(bonds[:, xi1] > 0.0)[0]
                    #####exit()
                    # dangling bit, mayhaps from defect
                    if len(bonds0) == 0 and len(bonds1) != 0:
                        xis.remove(xi1)
                        symbols[xi1] = "H"
                        mmtypes[xi1] = "H_"
                    elif len(bonds1) == 0 and len(bonds0) != 0:
                        xis.remove(xi0)
                        symbols[xi0] = "H"
                        mmtypes[xi0] = "H_"
                    else:
                        # the bond order will be the maximum one
                        bo = max(np.amax(bonds[xi0, :]),
                                 np.amax(bonds[xi1, :]))
                        # change the bonds
                        ix = np.ix_(bonds0, bonds1)
                        bonds[ix] = bo
                        ix = np.ix_(bonds1, bonds0)
                        bonds[ix] = bo
            # book keeping on what has disappeared
            structure.set_chemical_symbols(symbols)
            bonds = np.delete(bonds, xis, axis=0)
            bonds = np.delete(bonds, xis, axis=1)
            mmtypes = np.delete(mmtypes, xis)
            del structure[xis]

        return structure, bonds, mmtypes

    def get_mmtypes(self):
        """Return and update the current mmtypes."""
        logger.debug("Updating framework MM types.")
        mmtypes = []
        for _, sbu in self:
            mmtypes.append(sbu.mmtypes)
        mmtypes = np.hstack(mmtypes)
        self.mmtypes = mmtypes
        return np.copy(self.mmtypes)

    def get_bonds(self):
        """Return and update the current bond matrix."""
        logger.debug("Updating framework bond matrix.")
        bonds = []
        for _, sbu in self:
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

    def get_supercell(self, m=(2, 2, 2)):
        """Return a framework supercell using m as multiplier.

        Setup this way to keep the whole modifications that could
        have been made to the framework.
        """
        if isinstance(m, int):
            m = (m, m, m)
        logger.info("Creating supercell {0}x{1}x{2}.".format(*m))
        # get the offset direction ranges
        x = list(range(0, m[0], 1))
        y = list(range(0, m[1], 1))
        z = list(range(0, m[2], 1))
        # new framework object
        supercell = self.copy()
        ocell = supercell.topology.atoms.get_cell()
        ocellpar = ase.geometry.cell_to_cellpar(ocell)
        newcellpar = ocellpar.copy()
        newcellpar[0] *= m[0]
        newcellpar[1] *= m[1]
        newcellpar[2] *= m[2]
        newcell = ase.geometry.cellpar_to_cell(newcellpar)
        otopo = supercell.topology.copy()
        supercell.topology.atoms.set_cell(newcell, scale_atoms=False)
        L = len(otopo.atoms)
        # for the correct tagging analysis
        superatoms = otopo.atoms.copy().repeat(rep=m)
        # ERROR PRONE! the scaling modifies the shape
        # in the topology, resulting in weird behaviour in
        # very deformed cells.
        supertopo = Topology(name="supertopo", atoms=superatoms, analyze=True)
        # iterate over offsets and add the corresponding objects
        for offset in it.product(x, y, z):
            # central cell, ignore
            if offset == (0, 0, 0):
                for atom in otopo.atoms.copy():
                    if atom.symbol == "X":
                        continue
                    if atom.index not in supercell.SBU.keys():
                        continue
                    # directly tranfer new tags
                    sbu = supercell[atom.index]
                    sbu.transfer_tags(supertopo.fragments[atom.index])
            else:
                offset = np.asarray(offset)
                coffset = offset.dot(ocell)
                for atom in otopo.atoms.copy():
                    atom.position += coffset
                    supercell.topology.atoms.append(atom)
                    if atom.symbol == "X":
                        continue
                    # check the new idx of the SBU
                    newidx = len(supercell.topology.atoms)-1
                    # check that the SBU was not deleted before
                    if atom.index not in supercell.SBU.keys():
                        continue
                    sbu = supercell[atom.index].copy()
                    sbu.atoms.positions += coffset
                    sbu.transfer_tags(supertopo.fragments[newidx])
                    supercell.append(
                        index=newidx,
                        sbu=sbu,
                        update=False
                    )
                    supercell._todel[newidx] = list(supercell._todel[atom.index])
        return supercell

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
            self.bonds = self.get_bonds()
            # append the atom types
            self.mmtypes = self.get_mmtypes()

    def scale(self, cellpar):
        """
        Scale the building units positions by a factor alpha.

        This uses the correspondance between the atoms in the topology
        and the building units in the SBU list. Indeed, SBU[i] is centered on
        topology[i]. By scaling the topology, we obtain a new center for the
        sbu.
        alpha -- scaling factor
        """
        if len(cellpar) == 3:
            # 2D case
            cellpar = [cellpar[0], cellpar[1], 0.0, 90.0, 90.0, cellpar[2]]
        # scale using topology as template
        cell = ase.geometry.cellpar_to_cell(cellpar)
        self.topology.atoms.set_cell(cell, scale_atoms=True)
        # then center the SBUs on this position
        for i, sbu in self:
            center = self.topology.atoms[i]
            cop = sbu.atoms.positions.mean(axis=0)
            sbu.atoms.positions += center.position - cop

    def refine(self, alpha0=[1.0, 1.0, 1.0]):
        """
        Refine cell scaling to minimize distances between dummies.

        We already have tagged the corresponding dummies during alignment,
        so we just need to calculate the MSE of the distances between
        identical tags in the complete structure
        alpha0 -- starting point of the scaling search algorithm
        """
        logger.info("Refining unit cell.")
        # get the scaled cell, normalized
        I_m = np.eye(3)*alpha0
        cell0 = self.topology.atoms.get_cell()
        pbc = sum(self.topology.atoms.get_pbc())
        cell0 = cell0.dot(I_m/np.linalg.norm(cell0, axis=0))
        cellpar0 = ase.geometry.cell_to_cellpar(cell0, radians=False)
        if pbc == 2:
            cellpar0 = [cellpar0[0], cellpar0[1], cellpar0[5]]
            cellpar0 = np.array(cellpar0)

        # compile a list of mutual pairs
        atoms, _, _ = self.get_atoms(dummies=True)
        tags = atoms.get_tags()
        # find the pairs...
        pairs = [np.argwhere(tags == tag) for tag in set(tags) if tag > 0]
        pairs = [p for p in pairs if len(p) == 2]
        pairs = np.asarray(pairs).reshape(-1, 2)

        class StopOptimization(Exception):
            pass

        def MSE(x):
            """Return cost of scaling as MSE of distances."""
            # scale with this parameter
            self.scale(cellpar=x)
            atoms, _, _ = self.get_atoms(dummies=True)
            # reinitialize stuff
            self.scale(cellpar=cellpar0)
            # find the distances
            d = [atoms.get_distance(i0, i1, mic=True) for i0, i1 in pairs]
            d = np.asarray(d)
            mse = np.mean(d**2)
            logger.info("\t|--> Scaling error = {e:>5.4f}".format(e=mse))
            max_scale_tol = 600
            if mse > max_scale_tol:
                logger.info("Scale error is too large to be minimized: {max_scale} < {e:>5.4f}".format(max_scale=max_scale_tol, e=mse))
                raise StopOptimization("Scale criteria exceeded")

            return mse

        # first get an idea of the bounds.
        bounds = list(zip(0.01*cellpar0, 2.0*cellpar0))
        result = None
        try:
            result = scipy.optimize.minimize(
                fun=MSE,
                x0=cellpar0,
                method="L-BFGS-B",
                bounds=bounds,
                tol=0.01
            )
        except StopOptimization:
            logger.info("Stopped optimization by criterion")

        if result is None:
            logger.info("Sorry no convergence, scale exceeded")
            return False

        if result.success:
            logger.info("The system has been minimized correctly")
        else:
            logger.info("Sorry no convergence in the minimization was found")
            return False

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

        return True

    def alignment_sbu(self, sbu_dict=None):
        """Return the MOF with the sbu aligned."""
        if sbu_dict is None:
            raise Exception("Sorry, no sbu was found")

        alpha = 0.0
        for idx, sbu in sbu_dict.items():
            logger.debug("Treating slot number {idx}".format(idx=idx))
            logger.debug("\t|--> Aligning SBU {name}".format(name=sbu.name))
            # now align and get the scaling factor
            sbu, f = self.align(
                fragment=self.topology.fragments[idx],
                sbu=sbu
            )
            alpha += f
            self.append(index=idx, sbu=sbu)

        logger.info("")
        self.complete_alignment = self.refine(alpha0=alpha)

    @property
    def MOF(self):
        """MOF generated by adding SBUs to the structure."""
        mof = ase.Atoms()
        if not self.complete_alignment:
            logger.info("SBUs are not aligned")

        for i, sbu in self:
            atoms = sbu.atoms.copy()
            if sbu.pg == "D*h":
                # Random rotation
                pos = atoms.positions
                dummies = []
                for atom in atoms:
                    if atom.symbol == "X":
                        dummies.append(atom.position)
                ax1, ax2 = dummies
                com = atoms.get_center_of_mass()
                pos -= com

                rot_vec = ax2 - ax1
                rot_vec = rot_vec / np.linalg.norm(rot_vec)

                angles = np.arange(5, 95, 5)
                # ang_deg = random.uniform(-15, 15)
                ang_deg = random.choice([-1, 1]) * random.choice(angles)
                rotacion = Rotation.from_rotvec(np.deg2rad(ang_deg) * rot_vec)
                news_pos = rotacion.apply(pos)
                news_pos += com
                atoms.set_positions(news_pos)

            elif sbu.pg == "Oh":
                # Random rotation
                pos = atoms.positions
                dummies = []
                for atom in atoms:
                    if atom.symbol == "X":
                        dummies.append(atom.position)

                # centroid
                centroid = np.sum(dummies, axis=0) / len(dummies)

                main_axis_Oh = []
                for (v1, v2) in it.combinations(dummies, 2):
                    angle_v12 = calculate_angle(v1, v2, centroid)
                    is_axis = np.isclose(angle_v12, 180.0, atol=2.0)
                    if is_axis:
                        main_axis_Oh.append((v1, v2))

                ax1, ax2 = main_axis_Oh[np.random.choice(len(main_axis_Oh))]
                com = atoms.get_center_of_mass()
                pos -= com
                ang_deg = random.choice([-1, 1]) * random.choice([0, 90, 180, 270])

                rot_vec = ax2 - ax1
                rot_vec = rot_vec / np.linalg.norm(rot_vec)

                rotacion = Rotation.from_rotvec(np.deg2rad(ang_deg) * rot_vec)
                news_pos = rotacion.apply(pos)
                news_pos += com
                atoms.set_positions(news_pos)

            for atom in atoms:
                if atom.symbol == "X":
                    continue
                else:
                    mof += atom

        cell = self.topology.atoms.get_cell()
        if self.topology.atoms.cell.rank < 3:
            cell[2] = np.array([0.0, 0.0, 15.0])

        pbc = self.topology.atoms.get_pbc()

        mof.set_cell(cell)
        mof.set_pbc(pbc)
        # add a numerical artifact for optimization in z direction
        if sum(mof.pbc) == 2:
            mof.positions[:, 2] += np.random.rand(len(mof))*0.001

        # If there is only one atom in the box it will expand by one unit in the xy plane.
        if len(mof) == 1:
            mof *= (3, 3, 1)

        return mof

    def align(self, fragment, sbu):
        """Return an aligned SBU.

        The SBU is rotated on top of the fragment using the procrustes library within scipy.
        A scaling factor is also calculated for all three cell vectors.

        Parameters:
        -----------
        fragment : ase.Atoms
            The slot in the topology, ASE Atoms.

        sbu : atgrafsE.utils.sbu.SBU
            Object to align, ASE Atoms.
        """
        logger.debug("{0:-^50}".format(" Starting the alignment of the sbu "))
        logger.debug("SBU: {}".format(sbu.name))
        # first, we work with copies
        fragment = fragment.copy()
        # normalize and center
        # The sbu is moved to the center of the fragment.
        fragment_cop = fragment.positions.mean(axis=0)
        fragment.positions -= fragment_cop
        sbu.atoms.positions -= sbu.atoms.positions.mean(axis=0)

        # identify dummies in sbu
        sbu_Xis = [x.index for x in sbu.atoms if x.symbol == "X"]
        # get the scaling factor
        size_sbu = np.linalg.norm(sbu.atoms[sbu_Xis].positions, axis=1)
        size_fragment = np.linalg.norm(fragment.positions, axis=1)
        alpha = size_sbu.mean() / size_fragment.mean()

        ncop = np.linalg.norm(fragment_cop)
        if ncop < 1e-6:
            direction = np.ones(3, dtype=np.float64)
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

        # find an orthogonal matrix which most closely maps X0 to X1
        R, s = orthogonal_procrustes(X0, X1)
        sbu.atoms.positions = sbu.atoms.positions.dot(R) + fragment_cop
        fragment.positions += fragment_cop
        res_d = ase.geometry.distance(sbu.atoms[sbu_Xis], fragment)
        logger.debug("Residual distance: {d}".format(d=res_d))
        # tag the atoms
        sbu.transfer_tags(fragment)
        logger.debug("{0:-^50}".format(" End of alignment "))

        return sbu, alpha

    def get_vector_space(self, X):
        """Return a vector space as four points."""
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

        return np.asarray([x0, x1, x2, x3])

    def write(self, ext, f="./mof", **kwargs):
        """
        Write a chemical information file to disk in selected format.

        Parameters:
        -----------
        f : str
            File name.

        ext : str
            File extension.
        """
        path = os.path.abspath("{path}".format(path=f))
        logger.info("Framework saved to disk at {p}".format(p=path))

        if not path.endswith(ext):
            path += f".{ext}"

        if ext in ["vasp", "xyz"]:
            ase.io.write(path, self.MOF, format=ext, **kwargs)
        else:
            raise ValueError(f"Format: {ext} not found in ATGraFS-ext")
