"""Submodule containing functions and classes to study the symmetry and operations of Atoms."""

import logging

import itertools as it
import numpy as np

from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from scipy.cluster.hierarchy import fclusterdata

from atgrafsE.utils.operations import rotation, reflection, is_valid_op, inertia

logger = logging.getLogger(__name__)


class PointGroup:
    """A class to analyze the point group of a molecule.

    The general outline of the algorithm is as follows:

    1. Center the molecule around its center of mass.
    2. Compute the inertia tensor and the eigenvalues and eigenvectors.
    3. Handle the symmetry detection based on eigenvalues.
        a. Linear molecules have one zero eigenvalue. Possible symmetry
           operations are C*v or D*v
        b. Asymetric top molecules have all different eigenvalues. The
           maximum rotational symmetry in such molecules is 2
        c. Symmetric top molecules have 1 unique eigenvalue, which gives a
           unique rotation axis.  All axial point groups are possible
           except the cubic groups (T & O) and I.
        d. Spherical top molecules have all three eigenvalues equal. They
           have the rare T, O or I point groups.
    """

    def __init__(self, mol, tol=0.1):
        """Initialize the class.

        The default settings are usually sufficient.

        -- mol : Molecule to determine point group for.
        """
        self.tol = tol
        self.etol = self.tol / 3.0
        self.mol = mol

        # center molecule
        self.mol.positions -= self.mol.positions.mean(axis=0)

        # normalize
        norm_factor = np.amax(np.linalg.norm(self.mol.positions, axis=1, keepdims=True))
        if norm_factor < 1e-6:
            norm_factor = 1.0

        self.mol.positions /= norm_factor
        self.symmops = {"C": [],
                        "S": [],
                        "sigma": []}
        # identity
        self.symmops["I"] = np.eye(3)

        # inversion
        inversion = - np.eye(3)

        if is_valid_op(self.mol, inversion):
            self.symmops["-I"] = inversion
            logger.debug("Inversion center present")
        else:
            self.symmops["-I"] = None

        self.nrot = 0
        self.nref = 0
        self.analyze()

        if self.schoenflies in ["C1v", "C1h"]:
            self.schoenflies = "Cs"

    def analyze(self):
        """Analyze the symmetry of the system."""
        if len(self.mol) == 1:
            self.schoenflies = "Kh"
        else:
            # Get the inertia tensor
            xyz = self.mol.get_positions()
            W = self.mol.get_masses()
            Inertia = inertia(xyz=xyz, W=W)
            Inertia /= np.sum(np.linalg.norm(xyz, axis=1)*W)

            # calculate the eigenvalues
            eigvals, eigvecs = np.linalg.eigh(Inertia)
            self.eigvecs = eigvecs.T
            self.eigvals = eigvals
            e0, e1, e2 = self.eigvals
            eig_zero = np.abs(np.product(self.eigvals)) < self.etol**3
            eig_same = (abs(e0-e1) < self.etol) & (abs(e0-e2) < self.etol)
            eig_diff = (abs(e0-e1) > self.etol) & (abs(e0-e2) > self.etol) & (abs(e1-e2) > self.etol)
            logger.debug("Inertia tensor eigenvalues = {0:3.2f} {1:3.2f} {2:3.2f}".format(*self.eigvals))
            # determine which process to go through
            if eig_zero:
                logger.debug("Linear molecule detected")
                self.analyze_linear()
            elif eig_same:
                logger.debug("Spherical top molecule detected")
                self.analyze_spherical_top()
            elif eig_diff:
                logger.debug("Asymmetric top molecule detected")
                self.analyze_asymmetric_top()
            else:
                logger.debug("Symmetric top molecule detected")
                self.analyze_symmetric_top()

    def analyze_cyclic_groups(self):
        """Handle cyclic group molecules."""
        order, axis, op = max(self.symmops["C"], key=lambda v: v[0])
        self.schoenflies = "C{0}".format(order)
        mirror_type = self.find_reflection_plane(axis)
        if mirror_type == "h":
            self.schoenflies += "h"
        elif mirror_type == "v":
            self.schoenflies += "v"
        elif mirror_type is None:
            rotoref = reflection(axis).dot(rotation(axis=axis, order=2*order))
            if is_valid_op(self.mol, rotoref):
                self.schoenflies = "S{0}".format(2*order)
                self.symmops["S"] += [(order, axis, rotoref),]

    def analyze_linear(self):
        """TODO"""
        inversion = - np.eye(3)
        if self.symmops["-I"] is not None:
            self.schoenflies = "D*h"
        else:
            self.schoenflies = "C*v"

    def analyze_spherical_top(self):
        """Handles Sperhical Top Molecules, which belongs to the T, O or I point
        groups.
        """
        self.find_spherical_axes()
        if self.nrot == 0:
            logger.debug("Accidental spherical top!")
            self.analyze_symmetric_top()
        # get the main axis and order
        order,axis,op = max(self.symmops["C"], key=lambda v: v[0])
        if order < 3:
            logger.debug("Accidental speherical top!")
            self.analyze_symmetric_top()
        elif order == 3:
            mirror_type = self.find_reflection_plane(axis)
            if mirror_type != "":
                if self.symmops["-I"] is not None:
                    self.schoenflies = "Th"
                else:
                    self.schoenflies = "Td"
            else:
                self.schoenflies = "T"
        elif order == 4:
            if self.symmops["-I"] is not None:
                self.schoenflies = "Oh"
            else:
                self.schoenflies = "O"
        elif order == 5:
            if self.symmops["-I"] is not None:
                self.schoenflies = "Ih"
            else:
                self.schoenflies = "I"

    def analyze_asymmetric_top(self):
        """Handles assymetric top molecules, which cannot contain rotational
        symmetry larger than 2.
        """
        for axis in self.eigvecs:
            op = rotation(axis=axis,order=2)
            if is_valid_op(self.mol,op):
                self.symmops["C"] += [(2,axis,op),]
                self.nrot += 1
        if   self.nrot == 0:
            logger.debug("No rotation symmetries detected.")
            self.analyze_nonrotational_groups()
        elif self.nrot == 3:
            logger.debug("Dihedral group detected.")
            self.analyze_dihedral_groups()
        else:
            logger.debug("Cyclic group detected.")
            self.analyze_cyclic_groups()

    def analyze_symmetric_top(self):
        """Handles symetric top molecules which has one unique eigenvalue whose
        corresponding principal axis is a unique rotational axis.  More complex
        handling required to look for R2 axes perpendicular to this unique
        axis.
        """
        # identify unique axis
        if abs(self.eigvals[0] - self.eigvals[1]) < self.etol:
            unique_axis = self.eigvecs[2]
        elif abs(self.eigvals[1] - self.eigvals[2]) < self.etol:
            unique_axis = self.eigvecs[0]
        else:
            unique_axis = self.eigvecs[1]
        order = self.detect_rotational_symmetry(unique_axis)
        logger.debug("Rotation symmetries = {0}".format(order))
        if self.nrot > 0:
            self.has_perpendicular_C2(unique_axis)
        if self.nrot > 1:
            self.analyze_dihedral_groups()
        elif self.nrot == 1:
            self.analyze_cyclic_groups()
        else:
            self.analyze_nonrotational_groups()

    def analyze_nonrotational_groups(self):
        """Handle molecules with no rotational symmetry.

        Only possible point groups are C1, Cs and Ci.
        """
        self.schoenflies = "C1"
        if self.symmops["-I"] is not None:
            self.schoenflies = "Ci"
        else:
            for v in self.eigvecs:
                mirror_type = self.find_reflection_plane(v)
                if mirror_type is not None:
                    self.schoenflies = "Cs"
                    break

    def analyze_dihedral_groups(self):
        """Handle dihedral group molecules.

        i.e those with intersecting R2 axes and a main axis.
        """
        order, axis, op = max(self.symmops["C"], key=lambda v: v[0])
        self.schoenflies = "D{0}".format(order)
        mirror_type = self.find_reflection_plane(axis)
        if mirror_type == "h":
            self.schoenflies += "h"
        elif not mirror_type == "":
            self.schoenflies += "d"

    def find_spherical_axes(self):
        """Looks for R5, R4, R3 and R2 axes in spherical top molecules.  Point
        group T molecules have only one unique 3-fold and one unique 2-fold
        axis. O molecules have one unique 4, 3 and 2-fold axes. I molecules
        have a unique 5-fold axis.
        """
        rot_present = {2:False, 3:False, 4:False, 5:False}
        test_set = self.find_possible_equivalent_positions()
        for c1, c2, c3 in it.combinations(test_set, 3):
            for cc1, cc2 in it.combinations([c1, c2, c3], 2):
                if not rot_present[2]:
                    test_axis = cc1 + cc2
                    if np.linalg.norm(test_axis) > self.tol:
                        op = rotation(axis=test_axis, order=2)
                        if is_valid_op(self.mol, op):
                            logger.debug("Found axis with order {0}".format(2))
                            rot_present[2] = True
                            self.symmops["C"] += [(2, test_axis, op),]
                            self.nrot += 1

            test_axis = np.cross(c2 - c1, c3 - c1)
            if np.linalg.norm(test_axis) > self.tol:
                for order in (3, 4, 5):
                    if not rot_present[order]:
                        op = rotation(axis=test_axis, order=order)
                        if is_valid_op(self.mol, op):
                            logger.debug("Found axis with order {0}".format(order))
                            rot_present[order] = True
                            self.symmops["C"] += [(order, test_axis, op), ]
                            self.nrot += 1
                            break
            if (rot_present[2] & rot_present[3] & (rot_present[4] | rot_present[5])):
                break

    def find_possible_equivalent_positions(self, axis=None):
        """Return the smallest list of atoms with the same species and
        distance from origin AND does not lie on the specified axis.  This
        maximal set limits the possible rotational symmetry operations,
        since atoms lying on a test axis is irrelevant in testing rotational
        symmetryOperations.
        """
        def not_on_axis(index):
            v = np.cross(self.mol.positions[index], axis)
            return np.linalg.norm(v) > self.tol

        valid_sets = []
        numbers = self.mol.get_atomic_numbers()
        dists = np.linalg.norm(self.mol.get_positions(), axis=1, keepdims=True)
        clusters = fclusterdata(dists, self.tol, criterion='distance')
        for cval in set(clusters):
            indices = np.where(clusters == cval)[0]
            #most common specie of set only
            counts = np.bincount(numbers[indices])
            indices = indices[np.where(numbers[indices] == np.argmax(counts))[0]]
            if axis is not None:
                indices = list(filter(not_on_axis, indices))
            if len(indices) > 1:
                # no symmetry for a set of 1...
                valid_sets.append(self.mol.positions[indices])

        return min(valid_sets, key=lambda s: len(s))

    def find_reflection_plane(self, axis):
        """Look for mirror symmetry of specified type about axis.

        Possible types are "h" or "vd".  Horizontal (h) mirrors are perpendicular to
        the axis while vertical (v) or diagonal (d) mirrors are parallel.  v
        mirrors has atoms lying on the mirror plane while d mirrors do not.
        """
        symmop = None
        # First test whether the axis itself is the normal to a mirror plane.
        op = reflection(axis)
        if is_valid_op(self.mol, op):
            symmop = ("h", axis, op)
        else:
            # Iterate through all pairs of atoms to find mirror
            for s1, s2 in it.combinations(self.mol, 2):
                if s1.symbol == s2.symbol:
                    normal = s1.position - s2.position
                    if normal.dot(axis) < self.tol:
                        op = reflection(normal)
                        if is_valid_op(self.mol, op):
                            if self.nrot > 1:
                                symmop = ("d", normal, op)
                                for prev_order, prev_axis, prev_op in self.symmops["C"]:
                                    if not np.linalg.norm(prev_axis - axis) < self.tol:
                                        if np.dot(prev_axis, normal) < self.tol:
                                            symmop = ("v", normal, op)
                                            break
                            else:
                                symmop = ("v", normal, op)
                            break
        if symmop is not None:
            self.symmops["sigma"] += [symmop,]
            return symmop[0]
        else:
            return symmop

    def detect_rotational_symmetry(self, axis):
        """Determine the rotational symmetry about supplied axis.

        Used only for symmetric top molecules which has possible rotational symmetry
        operations > 2.
        """
        min_set = self.find_possible_equivalent_positions(axis=axis)
        max_sym = len(min_set)
        for order in range(max_sym, 0, -1):
            if max_sym % order != 0:
                continue
            op = rotation(axis=axis, order=order)
            if is_valid_op(self.mol, op):
                logger.debug("Found axis with order {0}".format(order))
                self.symmops["C"] += [(order, axis, op),]
                self.nrot += 1
                return order
        return 1

    def has_perpendicular_C2(self, axis):
        """Check for R2 axes perpendicular to unique axis.

        For handling symmetric top molecules.
        """
        min_set = self.find_possible_equivalent_positions(axis=axis)
        found = False
        for s1, s2 in it.combinations(min_set, 2):
            test_axis = np.cross(s1 - s2, axis)
            if np.linalg.norm(test_axis) > self.tol:
                op = rotation(axis=test_axis, order=2)
                if is_valid_op(self.mol, op):
                    self.symmops["C"] += [(2, test_axis, op),]
                    self.nrot += 1
                    found = True
        return found


def unique_axes(potential_axes, epsilon=0.1):
    """Return non colinear potential_axes only."""
    axes = [potential_axes[0], ]
    for axis in potential_axes[1:]:
        colinear = False
        for seen_axis in axes:
            dot = abs(seen_axis.dot(axis))
            if dot > 1.0 - epsilon:
                colinear = True
                break
        if not colinear:
            axes.append(axis)

    return np.asarray(axes)


def get_potential_axes(mol):
    """Return all potential symmetry axes, faces, nodes, midway points, etc."""
    potential_axes = []
    try:
        # full analysis needed
        qhull = ConvexHull(mol.get_positions())
        logger.debug("Using Convex Hull algorithm for axis detection.")
        # forming the hyperplane equation of the facet
        for equation in qhull.equations:
            # normal of simplices
            axis = equation[0:3]
            potential_axes.append(axis)

        # Indices of points forming the vertices of the convex hull. For 2-D convex hulls, the
        # vertices are in counterclockwise order. For other dimensions, they are in input order.
        for node in qhull.vertices:
            # exterior points
            axis = qhull.points[node]
            norm = np.linalg.norm(axis)
            potential_axes.append(axis)

        # Indices of points forming the simplical facets of the convex hull.
        for simplex in qhull.simplices:
            for side in it.combinations(list(simplex), 2):
                # middle of side
                side = np.array(side)
                axis = qhull.points[side].mean(axis=0)
                potential_axes.append(axis)
                # perpendicular to side
                a0, a1 = qhull.points[side]
                axis = np.cross(a0, a1)
                potential_axes.append(axis)

    except QhullError:
        logger.debug("Planar connectivity detected.")
        # coplanarity detected
        positions = mol.get_positions()
        potential_axes += [axis for axis in positions]

        # since coplanar, any two simplices describe the plane
        for side in it.combinations(list(positions), 2):
            # middle of side
            axis = np.array(side).mean(axis=0)
            potential_axes.append(axis)
            axis = np.cross(side[0], side[1])
            potential_axes.append(axis)

        for a0, a1 in it.combinations(list(potential_axes), 2):
            axis = np.cross(a0, a1)
            potential_axes.append(axis)

    potential_axes = np.array(potential_axes)
    norm = np.linalg.norm(potential_axes, axis=1)
    norm[norm < 1e-3] = 1.0
    potential_axes /= norm[:, None]
    # remove zero norms
    mask = np.isnan(potential_axes).any(axis=1)
    potential_axes = potential_axes[~mask]
    axes = unique_axes(potential_axes)

    return axes


def get_symmetry_elements(mol, max_order=8, epsilon=0.1):
    """Return an array counting the found symmetries of the object, up to axes of order.

    Enables fast comparison of compatibility between objects: if array1 - array2 > 0, that means
    object1 fits the slot2.
    """
    logger.debug("{0:-^80}".format(" The search for symmetry of elements has been started "))

    if len(mol) == 1:
        logger.debug("Point-symmetry detected.")
        symmetries = np.array([0, 0, 0, 0, 0, 1])
        logger.debug("{0:-^80}".format(" End of symmetry search "))
        return symmetries

    if len(mol) == 2:
        logger.debug("Linear connectivity detected.")
        # simplest, linear case
        symmetries = np.array([1, 1, 0, 1, 1, 2])
        logger.debug("{0:-^80}".format(" End of symmetry search "))
        return symmetries

    mol.center(about=0)
    # ensure there is a sufficient distance between connections
    dist = mol.get_all_distances().mean()
    #TODO? why 10.0?
    if dist < 10.0:
        alpha = 10.0/dist
        mol.positions = mol.positions.dot(np.eye(3)*alpha)

    logger.debug("DIST {}".format(dist))
    axes = get_potential_axes(mol)
    # array for the operations:
    # size = (1inversion+rotationorders+rotinvorders+2planes+1multiplicity)
    symmetries = np.zeros(4+2*max_order)
    # 3D Inversion Matrix from origin system
    # or origin reflexion
    inv = -1.0*np.eye(3)
    has_inv = is_valid_op(mol, inv)
    symmetries[0] += int(has_inv)

    # rotations:
    # ----------
    principal_order = 1
    principal_axes = []
    for axis in axes:
        for order in range(2, max_order + 1):
            rot = rotation(axis, order)
            has_rot = is_valid_op(mol, rot)
            symmetries[order - 1] += int(has_rot)
            if has_rot:
                logger.debug("Detected: C{order}".format(order=order))
            # bookkeeping for the planes
            if has_rot and order > principal_order:
                principal_order = order
                principal_axes = [axis,]
            elif has_rot and order == principal_order:
                principal_axes.append(axis)

    # planes
    # ------
    for axis in axes:
        ref = reflection(axis)
        has_ref = is_valid_op(mol, ref)
        if has_ref:
            dots = [abs(axis.dot(max_axis)) for max_axis in principal_axes]
            if not dots:
                continue
            mindot = np.amin(dots)
            maxdot = np.amax(dots)
            if maxdot > 1.0 - epsilon:
                # sigma h
                symmetries[-2] += 1
                logger.debug("Detected: sigma h")
            elif mindot < epsilon:
                # sigma vd
                symmetries[-3] += 1
                logger.debug("Detected: sigma vd")

    # rotoreflections
    # ---------------
    for axis in axes:
        ref = reflection(axis)
        for order in range(2, max_order+1):
            rot = rotation(axis, order)
            rr = rot.dot(ref)
            has_rr = is_valid_op(mol, rr)
            symmetries[order+max_order-2] += int(has_rr)
            if has_rr:
                logger.debug("Detected: S{order}".format(order=order))
    # multiplicity
    symmetries[-1] = len([x for x in mol if x.symbol == "X"])

    # symmetries : [inv, ax1:nR, ax2:nR, ax3:nR, ..., pl1:nRf, pl2:nRf, pl3:nRf]
    logger.debug("{0:-^80}".format(" End of symmetry search "))

    return np.array(symmetries, dtype=int)
