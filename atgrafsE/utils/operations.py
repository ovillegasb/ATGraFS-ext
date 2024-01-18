"""."""

import numpy as np


def inertia(xyz, W):
    """Return the inertia matrix of coordinates, weighted"""
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    x2 = x*x
    y2 = y*y
    z2 = z*z
    Ixx =  np.sum((y2 + z2) * W)
    Iyy =  np.sum((x2 + z2) * W)
    Izz =  np.sum((x2 + y2) * W)
    Ixy = -np.sum((x  * y ) * W)
    Iyz = -np.sum((y  * z ) * W)
    Ixz = -np.sum((x  * z ) * W)
    I = np.array([[Ixx,Ixy,Ixz],
                  [Ixy,Iyy,Iyz],
                  [Ixz,Iyz,Izz]],dtype=np.float64)
    return I

def rotation(axis, order):
    """Return a rotation matrix around the axis"""
    M = np.eye(3)
    norm = np.linalg.norm(axis)
    if norm<1e-3:
        norm = 1.0
    axis /= norm
    v0      = axis[0]
    v1      = axis[1]
    v2      = axis[2]
    theta   = 2.0*np.pi/order
    costh   = np.cos(theta)
    sinth   = np.sin(theta)
    M[0,0] = costh + (1.0-costh)*v0**2
    M[1,1] = costh + (1.0-costh)*v1**2
    M[2,2] = costh + (1.0-costh)*v2**2
    M[1,0] = (1.0-costh)*v0*v1 + v2*sinth
    M[0,1] = (1.0-costh)*v0*v1 - v2*sinth
    M[2,0] = (1.0-costh)*v0*v2 - v1*sinth
    M[0,2] = (1.0-costh)*v0*v2 + v1*sinth
    M[2,1] = (1.0-costh)*v1*v2 + v0*sinth
    M[1,2] = (1.0-costh)*v1*v2 - v0*sinth   
    return M


def reflection(axis):
    """Return a reflection matrix around the axis"""
    M = np.eye(3)    
    norm = np.linalg.norm(axis)
    if norm<1e-3:
        norm = 1.0
    axis /= norm
    v0      = axis[0]
    v1      = axis[1]
    v2      = axis[2]
    M[0,0] = 1.0-2.0*v0*v0
    M[1,1] = 1.0-2.0*v1*v1
    M[2,2] = 1.0-2.0*v2*v2
    M[1,0] =    -2.0*v0*v1 
    M[0,1] =    -2.0*v0*v1 
    M[2,0] =    -2.0*v0*v2 
    M[0,2] =    -2.0*v0*v2 
    M[2,1] =    -2.0*v1*v2 
    M[1,2] =    -2.0*v1*v2
    return M


def is_valid_op(mol, symmop, epsilon = 0.1):
    """Check if a particular symmetry operation is a valid symmetry operation
    for a molecule, i.e., the operation maps all atoms to another
    equivalent atom.
        -- mol : ASE Atoms object. subject of symmop
        -- symmop: Symmetry operation to test.
        -- epsilon : numerical tolerance of the
    """
    distances = []
    mol0 = mol.copy()
    mol1 = mol.copy()
    mol1.positions = mol1.positions.dot(symmop)
    workmol = mol0 + mol1
    other_indices = list(range(len(mol0), len(workmol), 1))
    for atom_index in range(0, len(mol0), 1):
        dist = workmol.get_distances(atom_index, other_indices, mic=False)
        distances.append(np.amin(dist))
    distances = np.array(distances)

    return (distances<epsilon).all()

