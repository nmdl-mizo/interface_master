"""
interface_generator.py

"""
import os
from numpy.linalg import det, norm, inv
from numpy import (cross, cos, sin, array, column_stack,
                   eye, arccos, around, sqrt, dot)
from pymatgen.core.structure import Structure
import numpy as np
from interfacemaster.cellcalc import (
    MID, DSCcalc, get_primitive_hkl, get_right_hand, get_pri_vec_inplane,
    get_ortho_two_v, ang, get_normal_from_MI)


def get_disorientation(L1, L2, v1, hkl1, v2, hkl2):
    """
    produce a rotation matrix
    so that the hkl1 plane overlap with the hkl2 plane
    and the v1 colinear with v2

    Parameters
    ----------
    L1, L2 : numpy array
        lattice basis sets
    v1, v2 : numpy array
        vectors
    hkl1, hkl2 : numpy array
        Miller indices

    Returns
    ----------
    rot_mat : numpy array
        a rotation matrix
    """

    # normal vector
    n1 = get_normal_from_MI(L1, hkl1)
    n2 = get_normal_from_MI(L2, hkl2)

    # auxiliary lattice
    Av1 = cross(np.dot(L1, v1), n1)
    Av2 = cross(np.dot(L2, v2), n2)

    # get the auxiliary lattices
    AL1 = column_stack((np.dot(L1, v1), n1, Av1))
    AL2 = column_stack((np.dot(L2, v2), n2, Av2))

    # unit mtx
    AL1 = get_unit_mtx(AL1)
    AL2 = get_unit_mtx(AL2)

    return np.dot(AL1, inv(AL2))


def get_unit_mtx(lattice):
    """
    return a unit lattice so that the length of every column vectors is 1

    Parameters
    ----------
    lattice : numpy array
        lattice basis set

    Returns
    ----------
    lattice_return : numpy array
        basis set of a unit lattice
    """
    lattice_return = np.array([v / norm(v) for v in lattice.T]).T
    return lattice_return


def rot(a, theta):
    """
    produce a rotation matrix

    Parameters
    ----------
    a : numpy array
        rotation axis
    Theta: float
        rotation angle

    Returns
    ----------
    rot_mat : numpy array
        a rotation matrix
    """
    c = float(cos(theta))
    s = float(sin(theta))
    a = a / norm(a)
    ax, ay, az = a
    return np.array([[c + ax * ax * (1 - c), ax * ay * (1 - c) - az * s,
                      ax * az * (1 - c) + ay * s],
                    [ay * ax * (1 - c) + az * s, c + ay * ay * (1 - c),
                        ay * az * (1 - c) - ax * s],
                     [az * ax * (1 - c) - ay * s, az * ay * (1 - c) + ax * s,
                      c + az * az * (1 - c)]], dtype=np.float64)


def three_dot(M1, M2, M3):
    """
    compute the three continuous dot product

    Parameters
    ----------
    M1, M2, M3 : numpy array
        matrices

    Returns
    ----------
    P : numpy array
        dot product
    """
    return np.dot(np.dot(M1, M2), M3)


def get_ang_list(m1, n):
    """
    return a list of ang cos between one list of vecor and one vector

    Parameters
    ----------
    m1 : numpy array
        list of vectors
    n : numpy array
        a vector

    Returns
    ----------
    c : numpy array
        list of cos
    """
    return 1 / norm(n) * abs(np.dot(m1, n)) / norm(m1, axis=1)


def cross_plane(lattice, n, lim, orthogonal, tol, inclination_tol=sqrt(2) / 2):
    """
    get a primitive lattice vector cross a plane

    Parameters
    ----------
    lattice : numpy array
        lattice matrix
    n : numpy array
        a normal vector
    lim : int
        control how many vectors to be generated
    tol : float
        tolerance judging orthogonal

    Returns
    ----------
    n_p : numpy array
        a primitve vector normal to the plane
    """
    x = np.arange(-lim, lim, 1)
    y = x
    z = x
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)[0]]
    indice_0 = indice_0[np.where(np.gcd.reduce(indice_0, axis=1) == 1)[0]]
    ltc_p = np.dot(indice_0, lattice.T)
    ltc_p = ltc_p[np.argsort(norm(ltc_p, axis=1))]
    dot_list = get_ang_list(ltc_p, n)
    if not orthogonal:
        normal_v = ltc_p[np.where(dot_list >= inclination_tol)[0]]
        normal_v = normal_v[np.argsort(norm(normal_v, axis=1))]
        normal_v = normal_v[0]
    else:
        try:
            normal_v = ltc_p[np.where(abs(dot_list - 1) < tol)[0]][0]
        except Exception as exc:
            raise RuntimeError(
                'failed to find a vector cross the plane. '
                'try larger lim '
                'or smaller tol or use non-orthogonal cell') from exc
    return normal_v


def get_sites_elements(structure):
    """
    get the coordinates of atoms and the elements

    Parameters
    ----------
    structure : pymatgen structure class
        input structure

    Returns
    ----------
    atoms : numpy array
        fractional coordinates of atoms in the primitive cell
    elements : numpy aray
        list of element name of the atoms
    return:
    """
    atoms = np.array([0, 0, 0])
    elements = []
    for i in structure.sites:
        atoms = np.vstack((atoms, i.frac_coords))
        elements.append(i.species_string)
    atoms = np.delete(atoms, 0, axis=0)
    return atoms, np.array(elements)


def POSCAR_to_cif(Poscar_name, Cif_name):
    """
    generate a cif file for the structure in a POSCAR file

    Parameters
    ----------
    Poscar_name : str
        name of a POSCAR file
    Cif_name : str
        name of a cif file
    """
    structure = Structure.from_file(Poscar_name)
    structure.to(filename=Cif_name)
    del structure


def write_LAMMPS(
        lattice,
        atoms,
        elements,
        filename='lmp_atoms_file',
        orthogonal=False):
    """
    write LAMMPS input atom file file of a supercell

    Parameters
    ----------
    lattice : numpy array
        lattice matrix
    atoms : numpy array
        fractional atoms coordinates
    elements : numpy array
        list of element name of the atoms
    filename : str
        filename of LAMMPS input atom file to write, default "lmp_atoms_file"
    orthogonal : bool
        whether write orthogonal cell, default False
    """

    # list of elements
    element_species = np.unique(elements)

    # to Cartesian
    atoms = np.dot(lattice, atoms.T).T

    # get the speicie identifiers (e.g. 1 for Cu, 2 for O...)
    element_indices = np.arange(len(element_species)) + 1
    species_identifiers = np.arange(len(atoms))

    for es, ei in zip(element_species, element_indices):
        indices_this_element = np.where(elements == es)[0]
        species_identifiers[indices_this_element] = ei

    species_identifiers = np.array([species_identifiers])

    # the atom ID
    IDs = np.arange(len(atoms)).reshape(1, -1) + 1

    # get the final format
    Final_format = np.concatenate((IDs.T, species_identifiers.T), axis=1)
    Final_format = np.concatenate((Final_format, atoms), axis=1)

    # define the box
    # xlo, xhi
    xhi, yhi, zhi = lattice[0][0], lattice[1][1], lattice[2][2]
    xlo, ylo, zlo = 0, 0, 0
    xy = lattice[:, 1][0]
    xz = lattice[:, 2][0]
    yz = lattice[:, 2][1]

    with open(filename, 'w', encoding="utf-8") as f:
        f.write(
            '#LAMMPS input file of atoms generated by interface_master. '
            'The elements are: ')
        for ei, es in zip(element_indices, element_species):
            f.write(f'{ei} {es} ')
        f.write(f'\n {len(atoms)} atoms \n \n')
        f.write(f'{len(element_species)} atom types \n \n')
        f.write(f'{xlo:.8f} {xhi:.8f} xlo xhi \n')
        f.write(f'{ylo:.8f} {yhi:.8f} ylo yhi \n')
        f.write(f'{zlo:.8f} {zhi:.8f} zlo zhi \n\n')
        if not orthogonal:
            f.write(f'{xy:.8f} {xz:.8f} {yz:.8f} xy xz yz \n\n')
        f.write('Atoms \n \n')
        np.savetxt(f, Final_format, fmt='%i %i %.16f %.16f %.16f')
    f.close()


def write_POSCAR(lattice, atoms, elements, filename='POSCAR'):
    """
    write Poscar file of a supercell

    Parameters
    ----------
    lattice : numpy array
        lattice matrix
    atoms : numpy array
        fractional atoms coordinates
    elements : numpy array
        list of element name of the atoms
    filename : str
        filename of POSCAR filename to wirte, default "POSCAR"
    """
    element_species = np.unique(elements)
    atoms_list = []
    num_list = []
    for es in element_species:
        atoms_this_element = atoms[np.where(elements == es)[0]]
        atoms_list.append(atoms_this_element)
        num_list.append(len(atoms_this_element))

    if len(element_species) > 1:
        atoms = np.zeros(3)
        for i in atoms_list:
            atoms = np.vstack((atoms, i))
        atoms = np.delete(atoms, 0, axis=0)

    with open(filename, 'w', encoding="utf-8") as f:
        f.write('#POSCAR generated by IF_master \n')

        # matrix
        f.write("1\n")
        for ai in lattice.T:
            f.write(' '.join(list(map(lambda x: f'{x:.16f}', ai))) + ' \n')

        # elements
        for i in element_species:
            f.write(str(i) + ' ')
        f.write('\n')

        # num of atoms
        for i in num_list:
            f.write(str(i) + ' ')
        f.write('\n')
        f.write("Direct\n")
        np.savetxt(f, atoms, fmt='%.16f %.16f %.16f')
    f.close()


def cell_expands(lattice, atoms, elements, xyz):
    """
    expand certain supercell

    Parameters
    ----------
    lattice : numpy array
        lattice matrix
    atoms : numpy array
        list of atom fractional coordinates
    elements : numpy array
        list of element name of the atoms
    xyz : list of int
        list of expansion factor for the x, y, z directions

    Returns
    ----------
    lattice : numpy array
        lattice matrix
    atoms : numpy array
        atom coordinates
    elements : numpy array
        list of element name of the atoms
    """
    mtx = lattice.copy()
    dimX, dimY, dimZ = xyz
    x_shifts = np.arange(dimX)
    y_shifts = np.arange(dimY)
    z_shifts = np.arange(dimZ)

    g1_shifts = np.array(np.meshgrid(
        x_shifts, y_shifts, z_shifts)).T.reshape(-1, 3)
    atoms_expand = atoms.repeat(
        len(g1_shifts), axis=0) + np.tile(g1_shifts, (len(atoms), 1))
    elements = elements.repeat(len(g1_shifts))
    for i in range(3):
        atoms_expand[:, i] = atoms_expand[:, i] / xyz[i]
        mtx[:, i] = mtx[:, i] * xyz[i]
    return mtx, atoms_expand, elements


def get_array_bounds(U):
    """
    get the meshgrid formed by three sets of lower & upper bounds.

    the bounds cooresponds to the 8 vertices of the cell
    consisting of the three column vectors of a matrix

    Parameters
    ----------
    U : numpy array
        integer matrix

    Returns
    ----------
    indice : numpy array
        meshgrid made by the lower and upper bounds of the indices
    """
    Mo = U.copy()
    # get the coordinates of 8 vertices
    P1 = [0, 0, 0]
    P2 = Mo[:, 0]
    P3 = Mo[:, 1]
    P4 = Mo[:, 2]
    P5 = P2 + P3
    P6 = P2 + P4
    P7 = P3 + P4
    P8 = P2 + P3 + P4
    Points = np.vstack((P1, P2, P3, P4, P5, P6, P7, P8))
    # enclose the 8 verticies
    min1 = np.round(min(Points[:, 0] - 1), 0)
    max1 = np.round(max(Points[:, 0] + 1), 0)

    min2 = np.round(min(Points[:, 1] - 1), 0)
    max2 = np.round(max(Points[:, 1] + 1), 0)

    min3 = np.round(min(Points[:, 2] - 1), 0)
    max3 = np.round(max(Points[:, 2] + 1), 0)

    x = np.arange(min1 - 2, max1 + 2, 1)
    y = np.arange(min2 - 2, max2 + 2, 1)
    z = np.arange(min3 - 2, max3 + 2, 1)

    indice = (
        np.stack(
            np.meshgrid(
                x,
                y,
                z)
        ).T).reshape(len(x) * len(y) * len(z), 3)
    return indice


def super_cell(U, lattice, Atoms, elements):
    """
    make supercell

    Parameters
    ----------
    U : numpy array
        coefficients of the LC of three vectors from the basis
    atoms : numpy array
        fractional coordinates of atoms
    elements : numpy array
        list of the element names of these atoms

    Returns
    ----------
    lattice : numpy array
        lattice matrix
    atoms : numpy array
        atom coordinates
    elements : numpy array
        list of element name of the atoms
    """
    indice = get_array_bounds(U)

    # 1.get atoms
    Atoms = Atoms.repeat(len(indice), axis=0) + \
        np.tile(indice, (len(Atoms), 1))
    elements = elements.repeat(len(indice))
    tol = 1e-10

    # 3.delete atoms dropping outside
    Atoms = np.dot(inv(U), Atoms.T).T
    Atoms_try = Atoms.copy() + [tol, tol, tol]
    con = (Atoms_try[:, 0] < 1) & (Atoms_try[:, 0] >= 0) \
        & (Atoms_try[:, 1] < 1) & (Atoms_try[:, 1] >= 0) \
        & (Atoms_try[:, 2] < 1) & (Atoms_try[:, 2] >= 0)
    indices = np.where(con)[0]
    Atoms = Atoms[indices]
    elements = elements[indices]
    lattice = np.dot(lattice, U)
    return Atoms, elements, lattice


def shift_termi_left(lattice, dp, atoms, elements):
    """
    changing terminate involves requiring to cut or extend the cell
    for identical interfaces

    Parameters
    ----------
    lattice : numpy array
        a matrix with column lattice vectors
    dp : float
        height of termination shift
    atoms : numpy array
        fractional coordinates of the atoms
    elements : list
        list of element names

    Returns
    ----------
    atoms, elements : tuple of numpy array and list
        shifted atom coordinates, corresponding list of elements
    """
    n = cross(lattice[:, 1], lattice[:, 2])
    position_shift = dp / ang(lattice[:, 0], n) / norm(lattice[:, 0])
    atoms[:, 0] = atoms[:, 0] + position_shift
    if dp > 0:
        inner = (atoms[:, 0] < 1) & (atoms[:, 0] > 2 * position_shift)
        elements = elements[inner]
        atoms = atoms[inner]
        # shift to origin
        atoms[:, 0] = atoms[:, 0] - 2 * position_shift
        # to cartesian
        atoms = np.dot(lattice, atoms.T).T
        # cut
        lattice[:, 0] = lattice[:, 0] * (1 - 2 * position_shift)
        # back
        atoms = np.dot(inv(lattice), atoms.T).T
    else:
        atoms_c_1 = atoms.copy()
        atoms_c_2 = atoms.copy()
        elements_c = elements.copy()
        atoms_c_1[:, 0] += 1
        atoms_c_2[:, 0] += -1
        atoms = np.vstack((atoms, atoms_c_1, atoms_c_2))
        elements = np.append(elements, elements_c)
        elements = np.append(elements, elements_c)
        inner = (atoms[:, 0] < 1) & (atoms[:, 0] > 2 * position_shift)
        elements = elements[inner]
        atoms = atoms[inner]
        # shift to origin
        atoms[:, 0] = atoms[:, 0] - 2 * position_shift
        # to cartesian
        atoms = np.dot(lattice, atoms.T).T
        # cut
        lattice[:, 0] = lattice[:, 0] * (1 - 2 * position_shift)
        # back
        atoms = np.dot(inv(lattice), atoms.T).T

    return atoms, elements


def shift_none_copy(lattice, dp, atoms):
    """
    changing terminate involves without requiring to cut or extend the cell
    for identical interfaces

    Parameters
    ----------
    lattice : numpy array
        a matrix with column lattice vectors
    dp : foat
        height of termination shift
    atoms : numpy array
        fractional coordinates of the atoms

    Returns
    ----------
    atoms : numpy array
        shifted atom coordinates
    """
    n = cross(lattice[:, 1], lattice[:, 2])
    position_shift = dp / ang(lattice[:, 0], n) / norm(lattice[:, 0])
    atoms[:, 0] = atoms[:, 0] + position_shift
    atoms[:, 0] = atoms[:, 0] - np.floor(atoms[:, 0])
    return atoms


def shift_termi_right(lattice, dp, atoms, elements):
    """
    changing terminate involves requiring to cut or extend the cell
    for identical interfaces

    Parameters
    ----------
    lattice : numpy array
        a matrix with column lattice vectors
    dp : float
        height of termination shift
    atoms : numpy array
        fractional coordinates of the atoms
    elements : list
        list of element names

    Returns
    ----------
    atoms, elements : numpy array
        shifted atom coordinates, corresponding list of elements
    """
    n = cross(lattice[:, 1], lattice[:, 2])
    position_shift = dp / ang(lattice[:, 0], n) / norm(lattice[:, 0])
    atoms[:, 0] = atoms[:, 0] + position_shift
    if dp < 0:
        inner = (atoms[:, 0] > 0) & (atoms[:, 0] < 1 + 2 * position_shift)
        elements = elements[inner]
        atoms = atoms[inner]
        # to cartesian
        atoms = np.dot(lattice, atoms.T).T
        # cut
        lattice[:, 0] = lattice[:, 0] * (1 + 2 * position_shift)
        # back
        atoms = np.dot(inv(lattice), atoms.T).T
    else:
        atoms_c_1 = atoms.copy()
        atoms_c_2 = atoms.copy()
        elements_c = elements.copy()
        atoms_c_1[:, 0] += 1
        atoms_c_2[:, 0] += -1
        atoms = np.vstack((atoms, atoms_c_1, atoms_c_2))
        elements = np.append(elements, elements_c)
        elements = np.append(elements, elements_c)
        inner = (atoms[:, 0] > 0) & (atoms[:, 0] < 1 + 2 * position_shift)
        elements = elements[inner]
        atoms = atoms[inner]
        # to cartesian
        atoms = np.dot(lattice, atoms.T).T
        # cut
        lattice[:, 0] = lattice[:, 0] * (1 + 2 * position_shift)
        # back
        atoms = np.dot(inv(lattice), atoms.T).T

    return atoms, elements


def excess_volume(lattice_1, lattice_bi, atoms_1, atoms_2, dx):
    """
    introduce vacuum between the interfaces

    Parameters
    ----------
    lattice_1 : numpy array
        lattice matrix of the first slab
    lattice_bi : numpy array
        lattice matrix of the bicrystal
    atoms_1, atoms_2 : numpy array
        atom fractional coordinates of slab 1, slab 2
    dx : float
        length of expands normal to the interface
        with the same units as lattice para

    Returns
    ----------
    lattice_bi, atoms_1, atoms2: numpyarray
        lattice matrix of bicrystal supercell,
        atom coordinates of left slab,
        and atom coordinates of right slab
    """
    n = cross(lattice_1[:, 1], lattice_1[:, 2])
    normal_shift = dx / ang(lattice_1[:, 0], n) / norm(lattice_bi[:, 0].copy())
    normal_shift_cart = normal_shift * lattice_bi[:, 0]
    atoms_2 = np.dot(lattice_bi.copy(), atoms_2.copy().T).T
    atoms_2 = atoms_2.copy() + normal_shift_cart
    lattice_bi[:, 0] = (2 * normal_shift + 1) * lattice_bi[:, 0]
    atoms_1[:, 0] = 1 / (2 * normal_shift + 1) * atoms_1[:, 0]
    atoms_2 = np.dot(inv(lattice_bi), atoms_2.T).T
    return lattice_bi, atoms_1, atoms_2


def surface_vacuum(lattice_1, lattice_bi, atoms_bi, vx):
    """
    introduce vacuum at one of the tails of the bicrystal cell

    Parameters
    ----------
    lattice_1 : numpy array
        lattice matrix of the first slab
    lattice_bi : numpy array
        lattice matrix of the bicrystal
    atoms_bi : numpy array
        atom fractional coordinates of the bicrystal
    vx : float
        length of the vacuum bulk with units as lattice para
    """
    #n = cross(lattice_1[:, 1], lattice_1[:, 2])
    #normal_shift = vx / ang(lattice_1[:, 0], n) / norm(lattice_1[:, 0])
    atoms_cart = dot(lattice_bi, atoms_bi.T).T
    lattice_bi[:, 0] = lattice_bi[:, 0] * (1 + vx / norm(lattice_bi[:, 0]))
    atoms_bi = dot(inv(lattice_bi), atoms_cart.T).T
    #atoms_bi[:, 0] = 1 / (1 + normal_shift) * atoms_bi[:, 0]
    return atoms_bi, lattice_bi

def unit_cell_axis(axis):
    """
    get an unit orthogonal cell with the x-axis collinear with certain axis
    """
    v1 = axis / norm(axis)
    v2 = np.zeros(3)
    if v1[0] == 0 and v1[1] == 0:
        v2[1] = 1
    else:
        v2[0], v2[1] = -v1[1], v1[0]
    v2 = v2 / norm(v2)
    v3 = cross(v1, v2)
    v3 = v3 / norm(v3)
    B = np.column_stack((v1, v2, v3))
    B = get_right_hand(B)
    return B


def unit_v(vector):
    """
    get the unit vector of a vector
    """
    return vector / norm(vector)


def adjust_orientation(lattice):
    """
    adjust the orientation of a lattice so that its first axis is along
    x-direction and the second axis is in the x-y plane

    Parameters
    ----------
    lattice : numpy array
        a matrix with column lattice vectors

    Returns
    ----------
    lattice, R: numpy array
        rotated lattice, rotation matrix
    """
    lattice_0 = lattice.copy()
    v1 = lattice[:, 0]
    v3 = cross(lattice[:, 0], lattice[:, 1])
    v2 = cross(v3, v1)

    v1, v2, v3 = unit_v(v1), unit_v(v2), unit_v(v3)
    this_orientation = np.column_stack((v1, v2, v3))
    desti_orientation = np.eye(3)
    R = np.dot(desti_orientation, inv(this_orientation))
    lattice = np.dot(R, lattice)
    # check that a1 and a2 points to positive:

    R = np.dot(lattice, inv(lattice_0))

    return lattice, R


def convert_vector_index(lattice_0, lattice_f, v_0):
    """
    convert the index of a vector into a different basis
    """
    v_0 = np.dot(lattice_0, v_0)
    v_f = np.dot(inv(lattice_f), v_0)
    return v_f


def get_height(lattice):
    """
    get the distance of the two surfaces (crossing the first vector) of a cell
    """
    n = cross(lattice[:, 1], lattice[:, 2])
    height = abs(np.dot(lattice[:, 0], n) / norm(n))
    return height


def get_plane_vectors(lattice, n):
    """
    a function get the two vectors normal to a vector

    Parameters
    ----------
    lattice : numpy array
        lattice matrix with column lattice vecotrs
    n : numpy array
        a vector
    Returns
    ----------
    B, indices : numpy array
        B two plane vectors
    """
    tol = 1e-8
    B = np.eye(3, 2)
    count = 0
    indices = []
    for i in range(3):
        if norm(lattice[:, i]) > 0 and abs(np.dot(lattice[:, i], n)) < tol:
            B[:, count] = lattice[:, i]
            indices.append(i)
            count += 1
    if count != 2:
        raise RuntimeError(
            'error: the CSL does not include two vectors in the interface')
    return B, indices


def reciprocal_lattice(B):
    """
    return the reciprocal lattice of B
    """
    v1, v2, v3 = B.T
    V = np.dot(v1, cross(v2, v3))
    b1 = cross(v2, v3) / V
    b2 = cross(v3, v1) / V
    b3 = cross(v1, v2) / V
    return np.column_stack((b1, b2, b3))


def d_hkl(lattice, hkl):
    """
    return the lattice plane spacing of (hkl) of a lattice
    """
    rep_L = reciprocal_lattice(lattice)
    d = 1 / norm(np.dot(rep_L, hkl))
    return d


def terminates_scanner_left(slab, atoms, elements, d, round_n=5):
    """
    find all different atomic planes
    within 1 lattice plane displacing (in the interface), for the left slab

    Parameters
    ----------
    slab : numpy array
        basic vectors of the slab
    atoms : numpy array
        fractional coordinates of atoms
    elements : list
        list of name of the atoms
    d : float
        1 lattice plane displacing
    round_n : int
        num of bits to round the fraction coordinates to judge identical planes

    Returns
    ----------
    plane_list : list
        list of planes of atom fraction coordinates
    element_list : list
        list of elements in each plane
    indices_list : list
        list of indices of atoms in each plane
    dp_list : list
        list of dp parameters as input to select corresponding termination
    """
    plane_list = []
    element_list = []
    indices_list = []
    dp_list = []
    height = get_height(slab)
    normal = cross(slab[:, 1], slab[:, 2])
    normal = normal / norm(normal)
    atoms_cart = np.dot(slab, atoms.T).T
    projections = abs(np.dot(atoms_cart, normal))
    atoms_round = np.ceil(projections.copy() * 10**round_n) / 10**round_n
    x_coords = np.unique(atoms_round)
    plane_index = 1
    position = x_coords[-plane_index]
    while position >= height - d:
        indices_here = np.where(atoms_round == x_coords[-plane_index])[0]
        dp_here = height - position
        dp_list.append(abs(dp_here))
        indices_list.append(indices_here)
        plane_list.append(atoms[indices_here])
        element_list.append(elements[indices_here])
        plane_index += 1
        try:
            position = x_coords[-plane_index]
        except IndexError:
            position = -np.inf
    return plane_list, element_list, indices_list, dp_list


def get_R_to_screen(lattice):
    """
    get a rotation matrix
    to make the interface plane of the slab located in the screen
    """
    v2 = lattice[:, 1]
    v2 = v2 / norm(v2)
    v1 = cross(lattice[:, 1], lattice[:, 2])
    v1 = v1 / norm(v1)
    v3 = cross(v1, v2)
    v3 = v3 / norm(v3)
    here = np.column_stack((v2, v3, v1))
    there = np.eye(3)
    return np.dot(there, inv(here))  # R here = there


def terminates_scanner_right(slab, atoms, elements, d, round_n=5):
    """
    find all different atomic planes
    within 1 lattice plane displacing (in the interface), for the right slab

    Parameters
    ----------
    slab : numpy array
        basic vectors of the slab
    atoms : numpy array
        fractional coordinates of atoms
    elements : list
        list of name of the atoms
    d : float
        1 lattice plane displacing
    round_n : int
        num of bits to round the fraction coordinates to judge identical planes

    Returns
    ----------
    plane_list : list
        list of planes of atom fraction coordinates
    element_list : list
        list of elements in each plane
    indices_list : list
        list of indices of atoms in each plane
    """
    plane_list = []
    element_list = []
    indices_list = []
    dp_list = []
    normal = cross(slab[:, 1], slab[:, 2])
    normal = normal / norm(normal)
    atoms_cart = np.dot(slab, atoms.T).T
    projections = abs(np.dot(atoms_cart, normal))
    atoms_round = np.ceil(projections.copy() * 10**round_n) / 10**round_n
    x_coords = np.unique(atoms_round)
    plane_index = 0
    position = 0
    while position <= d:
        indices_here = np.where(atoms_round == x_coords[plane_index])[0]
        dp_here = x_coords[plane_index]
        dp_list.append(abs(dp_here))
        indices_list.append(indices_here)
        plane_list.append(atoms[indices_here])
        element_list.append(elements[indices_here])
        plane_index += 1

        try:
            position = x_coords[plane_index]
        except IndexError:
            position = np.inf
    return plane_list, element_list, indices_list, dp_list

# """
# from here functions to draw atomic planes
# """


def draw_cell(subfig, xs, ys, alpha=0.3, color='k', width=0.5):
    """
    draw the two-dimensional cell
    """
    for i in range(4):
        subfig.plot(xs[i], ys[i], c=color, linewidth=width, alpha=alpha)


def clean_fig(num1, num2, axes):
    """
    hide the frames
    """
    for a in range(num1 * num2):
        for b in range(3):
            axes[a][b].spines['top'].set_visible(False)
            axes[a][b].spines['right'].set_visible(False)
            axes[a][b].spines['bottom'].set_visible(False)
            axes[a][b].spines['left'].set_visible(False)
            axes[a][b].get_yaxis().set_visible(False)
            axes[a][b].get_xaxis().set_visible(False)
            axes[a][b].set(facecolor="w")


def Xs_Ys_cell(lattice):
    """
    get the Xs and Ys arrays to draw a cell for a given lattice
    """
    # four verticies
    P2 = lattice[:, 1]
    P3 = lattice[:, 1] + lattice[:, 2]
    P4 = lattice[:, 2]

    P1 = [0, 0]
    P2 = [P2[0], P2[1]]
    P3 = [P3[0], P3[1]]
    P4 = [P4[0], P4[1]]

    x1 = [P1[0], P2[0]]
    y1 = [P1[1], P2[1]]

    x2 = [P2[0], P3[0]]
    y2 = [P2[1], P3[1]]

    x3 = [P3[0], P4[0]]
    y3 = [P3[1], P4[1]]

    x4 = [P4[0], P1[0]]
    y4 = [P4[1], P1[1]]
    xs = [x1, x2, x3, x4]
    ys = [y1, y2, y3, y4]
    return xs, ys


def draw_slab(xs, ys, axes, num, plane_list, lattice_to_screen,
              elements_list, column, colors, all_elements,
              l_r, titlesize, legendsize):
    """
    draw the terminating planes in the plane list for a slab

    Parameters
    ----------
    xs, ys : numpy array
        x,y arrays to draw the two-dimensional cell
    axes : list
        list of figures
    num : int
        number of terminations
    plane_list : list
        list of planes of atoms
    elements_list : list
        list of names of elments for these atoms
    lattice_to_screen : numpy array
        rotated lattice with the interface lying in the screen
    l_r : str
        left or right slab
    titlesize : int
        fontsize of title
    legendsize : int
        fontsize of legend
    """
    for i in range(num):
        # get atoms in this plane
        plane_atoms = plane_list[i]
        plane_atoms = np.dot(lattice_to_screen, plane_atoms.T).T
        # how many different elements in this plane
        plane_elements = elements_list[i]
        element_names = np.unique(plane_elements)

        # looping for different elements
        for en in element_names:
            # get the atoms of j elment
            single_element_atoms = plane_atoms[
                np.where(plane_elements == en)[0]]

            # draw atoms of this element
            Xs = single_element_atoms[:, 0]
            Ys = single_element_atoms[:, 1]
            element_index_here = np.where(all_elements == en)[0][0]
            draw_cell(axes[i][column], xs, ys)
            axes[i][column].scatter(
                Xs, Ys,
                c=colors[element_index_here],
                s=100, label=en)
            axes[i][column].set_title(
                f'Plane {i + 1} of {l_r} Cryst.',
                fontsize=titlesize)
            axes[i][column].axis('scaled')
            axes[i][column].legend(fontsize=legendsize, borderpad=0.5, loc=0)


def write_trans_file(v1, v2, n1, n2):
    """
    write a file including translation information for LAMMPS

    Parameters
    ----------
    v1, v2 : numpy array
        CNID vectors
    n1, n2 : int
        num of grids for v1 & v2
    """
    with open('paras', 'w', encoding="utf-8") as f:
        f.write(f'variable cnidv1x equal {v1[0]/n1} \n')
        f.write(f'variable cnidv2x equal {v1[0]/n2} \n')

        f.write(f'variable cnidv1y equal {v1[1]/n1} \n')
        f.write(f'variable cnidv2y equal {v1[1]/n2} \n')

        f.write(f'variable cnidv1z equal {v1[2]/n1} \n')
        f.write(f'variable cnidv2z equal {v1[2]/n2} \n')

        f.write(f'variable na equal {n1} \n')
        f.write(f'variable nb equal {n2} \n')


def draw_slab_dich(
        xs,
        ys,
        c_xs,
        c_ys,
        axes,
        num1,
        plane_list_1,
        lattice_to_screen_1,
        elements_list_1,
        colors,
        all_elements,
        num2,
        plane_list_2,
        lattice_to_screen_2,
        elements_list_2,
        titlesize):
    """
    draw the dichromatic patterns

    Parameters
    ----------
    xs, ys : numpy array
        x,y arrays to draw the interface cell
    c_xs, c_ys : numpy array
        x,y arrays to draw the CNID cell
    axes : list
        list of figures
    num1, num2 : int
        number of terminations
    plane_list_1, plane_list_2 : numpy array
        list of atom coordinates of the terminating planes
    elements_list_1, elements_list_2 : list
        list of corresponding element names
    lattice_to_screen_1, lattice_to_screen_2 : numpy array
        rotated lattices facing its interface orientation to the screen
    colors : str
        colors to classify different elements
    all_elements : list
        list of all the elements for all the atoms
    titlesize : int
        fontsize of the title
    """
    for i in range(num1):
        for j in range(num2):
            # get atoms in this plane
            plane_atoms_1 = plane_list_1[i]
            plane_atoms_1 = np.dot(lattice_to_screen_1, plane_atoms_1.T).T
            plane_atoms_2 = plane_list_2[j]
            plane_atoms_2 = np.dot(lattice_to_screen_2, plane_atoms_2.T).T
            # how many different elements in this plane
            plane_elements_1 = elements_list_1[i]
            element_names_1 = np.unique(plane_elements_1)
            plane_elements_2 = elements_list_2[j]
            element_names_2 = np.unique(plane_elements_2)
            for en1 in element_names_1:
                # get the atoms of j elment
                single_element_atoms_1 = plane_atoms_1[
                    np.where(plane_elements_1 == en1)[0]]

                # draw atoms of this element
                Xs = single_element_atoms_1[:, 0]
                Ys = single_element_atoms_1[:, 1]
                element_index_here_1 = np.where(all_elements == en1)[0][0]
                axes[i * num2 + j][2].scatter(
                    Xs, Ys,
                    c=colors[element_index_here_1],
                    s=200, label=en1 + " (L)", alpha=0.5)
                axes[i * num2 + j][2].set_title(
                    f'{i} left {j} right.', fontsize=titlesize)
                axes[i * num2 + j][2].axis('scaled')
                for en2 in element_names_2:
                    # get the atoms of j elment
                    single_element_atoms_2 = plane_atoms_2[
                        np.where(plane_elements_2 == en2)[0]]

                    # draw atoms of this element
                    Xs = single_element_atoms_2[:, 0]
                    Ys = single_element_atoms_2[:, 1]
                    element_index_here_2 = np.where(all_elements == en2)[0][0]
                    draw_cell(axes[i * num2 + j][2], xs, ys)
                    draw_cell(axes[i * num2 + j][2], c_xs,
                              c_ys, color='b', alpha=1, width=2)
                    axes[i * num2 + j][2].scatter(
                        Xs, Ys,
                        c=colors[-element_index_here_2 - 1],
                        s=50, label=en2 + " (R)", alpha=0.5)
                    axes[i * num2 + j][2].axis('scaled')

# """
# Below is some sampling functions
# """


def get_nearest_pair(lattice, atoms, indices):
    """
    a function return the indices of two nearest atoms in a periodic block
    inspired from https://github.com/oekosheri/GB_code

    Parameters
    ----------
    lattice : numpy array
        lattice matrix
    atoms : numpy array
        fractional coordinates
    indices : numpy array
        indices of atoms
    """

    # get Cartesian
    pos_1 = np.dot(lattice, atoms.copy().T).T
    pos_2 = pos_1
    # get 9 reps of atoms (PBC)
    reps = np.array([-1, 0, 1])
    x_shifts = [0]
    y_shifts = reps
    z_shifts = reps
    planar_shifts = np.array(
        np.meshgrid(x_shifts, y_shifts, z_shifts),
        dtype=float).T.reshape(-1, 3)
    planar_shifts = np.dot(lattice, planar_shifts.T).T

    # make images for one set
    n_images = len(planar_shifts)
    n_1 = len(atoms)
    n_2 = len(atoms)
    n_1_images = n_images * n_1

    # repeate to the images
    pos_1_images = pos_1.repeat(n_images, axis=0) + \
        np.tile(planar_shifts, (n_1, 1))
    pos_1_image_index_map = np.arange(n_1).repeat(n_images)

    # repeat to match num of set 2
    pos_1_rep = pos_1_images.repeat(n_2, axis=0)
    pos_1_index_map = pos_1_image_index_map.repeat(n_2)

    # repeat to match num of set 1
    pos_2_rep = np.tile(pos_2, (n_1_images, 1))
    pos_2_index_map = np.tile(np.arange(n_2), n_1_images)

    # all the distances
    distances = norm(pos_1_rep - pos_2_rep, axis=1)
    # none zero distances
    distances_none_zero = distances[distances > 0]
    distances_none_zero = np.unique(distances_none_zero)

    # not a distance to itself image
    for i in distances_none_zero:
        distances_id = np.where(distances == i)[0][0]
        if pos_1_index_map[distances_id] != pos_2_index_map[distances_id]:
            break

    return (indices[pos_1_index_map[distances_id]],
            indices[pos_2_index_map[distances_id]],
            pos_1_rep[distances_id],
            pos_2_rep[distances_id])


class core:
    """
    Core class for dealing with an interface
    """

    def __init__(self, file_1, file_2, prim_1=True, prim_2=True, verbose=True):
        self.file_1 = file_1  # cif file name of lattice 1
        self.file_2 = file_2  # cif file name of lattice 2
        self.structure_1 = Structure.from_file(
            file_1, primitive=prim_1, sort=False, merge_tol=0.0)
        self.structure_2 = Structure.from_file(
            file_2, primitive=prim_2, sort=False, merge_tol=0.0)

        self.conv_lattice_1 = Structure.from_file(
            file_1, primitive=False, sort=False, merge_tol=0.0
        ).lattice.matrix.T

        self.conv_lattice_2 = Structure.from_file(
            file_2, primitive=False, sort=False, merge_tol=0.0
        ).lattice.matrix.T

        self.lattice_1 = self.structure_1.lattice.matrix.T
        self.lattice_2 = self.structure_2.lattice.matrix.T
        self.lattice_2_TD = self.structure_2.lattice.matrix.T.copy()
        self.CSL = np.eye(3)  # CSL cell in cartesian
        self.du = 0.005
        self.S = 0.005
        self.D = np.eye(3)
        self.sgm1 = 100  # sigma 1
        self.sgm2 = 100  # sigma 2
        self.R = np.eye(3)  # rotation matrix
        self.axis = np.zeros(3)  # rotation axis
        self.theta = 0.0  # rotation angle
        self.U1 = np.eye(3)
        self.U2 = np.eye(3)
        self.bicrystal_U1 = np.eye(3)  # indices of the slab of lattice 1
        self.bicrystal_U2 = np.eye(3)  # indices of the slab of lattice 2
        self.CNID = np.eye(3, 2)
        self.cell_calc = DSCcalc()
        self.name = str
        self.dd = 0.005
        self.min_perp_length = 0.0
        self.orientation = np.eye(3)  # initial disorientation
        self.a1 = np.eye(3)
        self.a2 = np.eye(3)
        self.orient = np.eye(3)  # adjusted orientation for better visulaizing
        self.d1 = float  # lattice plane distance
        self.d2 = float  # lattice plane distance
        self.plane_list_1 = []  # list of terminating plane atoms
        self.plane_list_2 = []
        self.elements_1 = []  # list of terminating plane atom elements
        self.elements_2 = []
        self.elements_list_1 = []
        self.elements_list_2 = []
        self.indices_list_1 = []  # termination plane atoms's indices
        self.indices_list_2 = []
        self.dp_list_1 = []  # list of dp parameter to select termination
        self.dp_list_2 = []
        # rotate the two slabs so that the screen is crossing the interface
        self.R_see_plane = np.eye
        # lattice matrix of the final slabs forming the interface
        self.slab_lattice_1 = np.eye(3)
        self.slab_lattice_2 = np.eye(3)
        self.a2_transform = np.eye(3)
        # get the atoms in the primitive cell
        self.atoms_1, self.elements_1 = get_sites_elements(self.structure_1)
        self.atoms_2, self.elements_2 = get_sites_elements(self.structure_2)
        # save the information of the bicrystal box
        self.lattice_bi = np.eye(3)
        self.atoms_bi = np.zeros(3)
        self.elements_bi = []
        # whether the bicrystal supercell is orthogonal
        self.bicrystal_ortho = False
        self.slab_structure_1 = Structure.from_file(
            file_1, primitive=True, sort=False, merge_tol=0.0)
        self.slab_structure_2 = Structure.from_file(
            file_1, primitive=True, sort=False, merge_tol=0.0)
        self.bicrystal_structure = Structure.from_file(
            file_1, primitive=True, sort=False, merge_tol=0.0)
        self.verbose = verbose
        if self.verbose:
            print(
                'Warning!, '
                'this programme will rewrite the POSCAR file in this dir!')

    def scale(self, factor_1, factor_2):
        """
        scale the lattice 1 and 2 with specific factors
        (both the lattice and conventional lattice)

        Parameters
        ----------
        factor_1 : float
            scale factor for lattice 1
        factor_2 : float
            scale factor for lattice 2
        """
        self.lattice_1 = self.lattice_1 * factor_1
        self.lattice_2 = self.lattice_2 * factor_2
        self.conv_lattice_1 = self.conv_lattice_1 * factor_1
        self.conv_lattice_2 = self.conv_lattice_2 * factor_2

    def parse_limit(self, du, S, sgm1, sgm2, dd):
        """
        set the limitation to accept an appx CSL

        Parameters
        ----------
        du, S, sgm1, sgm2, dd : parameters
            see the paper
        """
        self.du = du
        self.S = S
        self.sgm1 = sgm1
        self.sgm2 = sgm2
        self.dd = dd

    def _str_found_csl(self, sigma1, sigma2, D, axis=None, theta=None):
        str_found_csl = (
            f'U1 = \n{str(self.U1)}; sigma_1 = {str(sigma1)}\n\n'
            f'U2 = \n{str(self.U2)}; sigma_2 = {str(sigma2)}\n\n'
            f'D = \n{str(np.round(D,8))}\n\n'
        )
        if axis is not None and theta is not None:
            str_found_csl += (
                f'axis = {str(axis)} ; '
                f'theta = {str(np.round(theta / np.pi * 180,8))}\n'
            )
        return str_found_csl

    def _print_found_csl(self, sigma1, sigma2, D, axis=None, theta=None):
        print('Congrates, we found an appx CSL!\n')
        print(self._str_found_csl(sigma1, sigma2, D, axis=axis, theta=theta))

    def _write_found_csl(self, file, sigma1, sigma2, D, axis=None, theta=None):
        file.write(
            self._str_found_csl(
                sigma1,
                sigma2,
                D,
                axis=axis,
                theta=theta))

    def search_one_position(
            self,
            axis,
            theta,
            theta_range,
            dtheta,
            two_D=False):
        """
        main loop finding the appx CSL

        Parameters
        ----------
        axis : numpy array
            rotation axis
        theta : float
            initial rotation angle, in degree
        theta_range : float
            range varying theta, in degree
        dtheta : float
            step varying theta, in degree
        """
        axis = np.dot(self.lattice_1, axis)
        if self.verbose:
            print(axis)
        theta = theta / 180 * np.pi
        n = np.ceil(theta_range / dtheta)
        dtheta = theta_range / n / 180 * np.pi
        found = None

        if not two_D:
            a1 = self.lattice_1.copy()
            a2_0 = self.lattice_2.copy()

        else:
            # find the two primitive plane bases
            # miller indices
            hkl_1 = MID(self.lattice_1, axis)
            hkl_2 = MID(np.dot(self.orientation, self.lattice_2), axis)
            # plane bases
            plane_B_1 = get_pri_vec_inplane(hkl_1, self.lattice_1)
            plane_B_2 = get_pri_vec_inplane(
                hkl_2, np.dot(self.orientation, self.lattice_2))
            v_3 = cross(plane_B_1[:, 0], plane_B_1[:, 1])
            a1 = np.column_stack((plane_B_1, v_3))
            a2_0 = np.column_stack((plane_B_2, v_3))
            self.a1 = a1.copy()
            self.a2 = a2_0.copy()
            # a2_0 back to the initial orientation
            a2_0 = np.dot(inv(self.orientation), a2_0)
        # rotation loop
        file = open('log.one_position', 'w', encoding="utf-8")
        file.write('---Searching starts---\n')
        file.write('axis theta dtheta n S du sigma1_max sigma2_max\n')
        file.write(
            f'{axis} {theta} {dtheta} {n} {self.S} '
            f'{self.du} {self.sgm1} {self.sgm2}\n')
        file.write('-----------for theta-----------\n')
        for _ in range(int(n)):
            R = np.dot(self.orientation, rot(axis, theta))
            U = three_dot(inv(a1), R, a2_0)
            file.write('theta = ' + str(theta / np.pi * 180) + '\n')
            file.write('    -----for N-----\n')
            for N in range(1, self.sgm2 + 1):
                U_p = np.round(N * U) / N
                if not np.all((abs(U_p - U)) < self.du):
                    continue
                file.write('    N= ' + str(N) + " accepted" + '\n')
                R_p = three_dot(a1, U_p, inv(a2_0))
                D = np.dot(inv(R), R_p)
                if not ((abs(det(D) - 1) <= self.S)
                        and np.all(abs(D - np.eye(3)) < self.dd)):
                    continue
                file.write('    --D accepted--\n')
                file.write(f"    D, det(D) = {det(D)} \n")
                ax2 = three_dot(R, D, a2_0)
                calc = DSCcalc()
                try:
                    calc.parse_int_U(a1, ax2, self.sgm2)
                    calc.compute_CSL()
                except BaseException:
                    file.write('    failed to find CSL here \n')
                    continue
                if not abs(det(calc.U1)) <= self.sgm1:
                    file.write('    sigma too large \n')
                    continue
                found = True
                file.write('--------------------------------\n')
                file.write('Congrates, we found an appx CSL!\n')
                sigma1 = int(abs(np.round(det(calc.U1))))
                sigma2 = int(abs(np.round(det(calc.U2))))
                self.D = D
                self.U1 = np.array(np.round(calc.U1), dtype=int)
                self.U2 = np.array(np.round(calc.U2), dtype=int)
                self.lattice_2_TD = three_dot(R, D, a2_0)
                self.CSL = np.dot(a1, self.U1)
                self.R = R
                self.theta = theta
                self.axis = axis
                self.cell_calc = calc
                if two_D:
                    self.d1 = d_hkl(self.lattice_1, hkl_1)
                    self.d2 = d_hkl(
                        three_dot(R, D, self.lattice_2), hkl_2)
                    calc.compute_CNID([0, 0, 1])
                    self.CNID = np.dot(a1, calc.CNID)
                self._write_found_csl(
                    file, sigma1, sigma2, D, axis, theta)
                if self.verbose:
                    self._print_found_csl(
                        sigma1, sigma2, D, axis, theta)
                break
            if found:
                break
            theta += dtheta
        if not found:
            print(
                'failed to find a satisfying appx CSL. '
                'Try to adjust the limits according'
                'to the log file generated; or try another orientation.')

    def search_fixed(self, R, exact=False, tol=1e-8):
        """
        main loop finding the appx CSL

        Parameters
        ----------
        axis : numpy array
            rotation axis
        theta : float
            initial rotation angle, in degree
        theta_range : float
            range varying theta, in degree
        dtheta : float
            step varying theta, in degree
        """
        found = None
        a1 = self.lattice_1.copy()
        a2_0 = self.lattice_2.copy()
        # rotation loop
        file = open('log.fixed_search', 'w', encoding="utf-8")
        file.write('---Searching starts---\n')
        file.write('axis theta dtheta n S du sigma1_max sigma2_max\n')
        file.write(f'{self.S} {self.du} {self.sgm1} {self.sgm2}\n')
        file.write('-----------for theta-----------\n')
        U = three_dot(inv(a1), R, a2_0)
        file.write('    -----for N-----\n')
        for N in range(1, self.sgm2 + 1):
            U_p = np.round(N * U) / N
            if not np.all((abs(U_p - U)) < self.du):
                continue
            file.write('N= ' + str(N) + " accepted" + '\n')
            R_p = three_dot(a1, U_p, inv(a2_0))
            D = np.dot(inv(R), R_p)
            if not (((abs(det(D) - 1) <= self.S)
                     and np.all(abs(D - np.eye(3)) < self.dd))):
                continue
            if exact:
                D = eye(3, 3)
                R = R_p
            file.write('--D accepted--\n')
            file.write(f"D, det(D) = {det(D)} \n")
            ax2 = three_dot(R, D, a2_0)
            calc = DSCcalc()
            calc.parse_int_U(a1, ax2, self.sgm2, tol)
            calc.compute_CSL(tol)
            if not abs(det(calc.U1)) <= self.sgm1:
                file.write('sigma too large \n')
                continue
            file.write('--------------------------------\n')
            file.write('Congrates, we found an appx CSL!\n')
            sigma1 = int(abs(np.round(det(calc.U1))))
            sigma2 = int(abs(np.round(det(calc.U2))))
            self.D = D
            self.U1 = np.array(np.round(calc.U1), dtype=int)
            self.U2 = np.array(np.round(calc.U2), dtype=int)
            self.lattice_2_TD = three_dot(R, D, a2_0)
            self.CSL = np.dot(a1, self.U1)
            self.R = R
            self.cell_calc = calc
            self._write_found_csl(file, sigma1, sigma2, D)
            if self.verbose:
                self._print_found_csl(sigma1, sigma2, D)
            break
        else:
            print(
                'failed to find a satisfying appx CSL. '
                'Try to adjust the limits according '
                'to the log file generated; or try another orientation.')

    def search_one_position_2D(
            self,
            hkl_1,
            hkl_2,
            theta_range,
            dtheta,
            pre_dt=False,
            pre_R=eye(
                3,
                3),
            integer_tol=1e-8,
            start=0,
            exact=False):
        """
        main loop finding the appx CSL

        Parameters
        ----------
        axis : numpy array
            rotation axis
        theta : float
            initial rotation angle, in degree
        theta_range : float
            range varying theta, in degree
        dtheta : float
            step varying theta, in degree
        """
        # get the normal of the two slabs
        n1 = get_normal_from_MI(self.lattice_1, hkl_1)
        n2 = get_normal_from_MI(self.lattice_2, hkl_2)
        # get the two plane bases
        b1 = get_pri_vec_inplane(hkl_1, self.lattice_1)
        b2_0 = get_pri_vec_inplane(hkl_2, self.lattice_2)
        # rotate the second crystal so that the two slabs connect
        self.set_orientation_axis(
            np.dot(inv(self.lattice_1), n1), np.dot(inv(self.lattice_2), n2))
        if pre_dt:
            # auxiliary vector
            self.orientation = pre_R
        # auxilary vector
        av_1 = cross(b1[:, 0], b1[:, 1])
        av_2_0 = cross(b2_0[:, 0], b2_0[:, 1])
        a1 = column_stack((b1, av_1))
        a2_0 = column_stack((b2_0, av_2_0 / norm(av_2_0) * norm(av_1)))
        if three_dot(av_1, self.orientation, av_2_0) < 0:
            v1, v2, v3 = b2_0[:, 1], b2_0[:, 0], - \
                av_2_0 / norm(av_2_0) * norm(av_1)
            a2_0 = column_stack((v1, v2, v3))
        # indices of the planal bases
        a2_0 = np.dot(self.orientation, a2_0)
        # starting point of rotation angle
        theta = start / 180 * np.pi
        # searching mesh
        n = np.ceil(theta_range / dtheta)
        # shifting angle each time
        dtheta = theta_range / n / 180 * np.pi

        found = None
        axis = n1
        # rotation loop
        file = open('log.one_position', 'w', encoding="utf-8")
        file.write('---Searching starts---\n')
        file.write('axis theta dtheta n S du sigma1_max sigma2_max\n')
        file.write(
            f'{axis} {theta} {dtheta} {n} {self.S} '
            f'{self.du} {self.sgm1} {self.sgm2}\n')
        file.write('-----------for theta-----------\n')
        for _ in range(int(n)):
            R = rot(n1, theta)
            U = three_dot(inv(a1), R, a2_0)
            file.write('theta = ' + str(theta / np.pi * 180) + '\n')
            file.write('    -----for N-----\n')
            for N in range(1, self.sgm2 + 1):
                tol = integer_tol
                U_p = np.round(N * U) / N
                if not (np.all((abs(U_p - U)) < self.du)
                        and np.all(abs(U_p[:, 2] - array([0, 0, 1])) < 1e-6)):
                    continue
                file.write('    N= ' + str(N) + " accepted" + '\n')
                R_p = three_dot(a1, U_p, inv(a2_0))
                D = np.dot(inv(R), R_p)
                if exact:
                    D = eye(3, 3)
                    R = R_p
                if not ((abs(det(D) - 1) <= self.S)
                        and np.all(abs(D - np.eye(3)) < self.dd)):
                    continue
                self.a2_transform = three_dot(R, D, self.orientation)
                file.write('    --D accepted--\n')
                file.write(f"    D, det(D) = {det(D)} \n")
                ax2 = three_dot(R, D, a2_0)
                calc = DSCcalc()
                try:
                    calc.parse_int_U(a1, ax2, self.sgm2)
                    calc.compute_CSL()
                except BaseException:
                    file.write('    failed to find CSL here \n')
                    continue
                if not abs(det(calc.U1)) <= self.sgm1:
                    file.write('    sigma too large \n')
                    continue
                found = True
                file.write('--------------------------------\n')
                file.write('Congrates, we found an appx CSL!\n')
                self.D = D
                self.U1 = np.array(np.round(calc.U1), dtype=int)
                CSL_here = np.dot(a1, self.U1)
                Pv_1_indices = get_plane_vectors(CSL_here, av_1)[1]
                CSL_vs_plane = CSL_here.T[Pv_1_indices].T
                self.U1 = np.dot(inv(self.lattice_1), CSL_vs_plane)
                self.U1 = np.array(np.round(self.U1), dtype=int)
                a2 = np.dot(self.a2_transform, self.lattice_2)
                self.U2 = np.dot(inv(a2), CSL_vs_plane)
                self.U2 = np.array(np.round(self.U2), dtype=int)
                sigma1 = int(
                    abs(np.round(norm(cross(
                        self.U1[:, 0],
                        self.U1[:, 1])))))
                sigma2 = int(
                    abs(np.round(norm(cross(
                        self.U2[:, 0],
                        self.U2[:, 1])))))
                self.lattice_2_TD = three_dot(
                    R, D, self.orientation)
                self.lattice_2_TD = np.dot(
                    self.lattice_2_TD, self.lattice_2)
                self.CSL = np.dot(a1, self.U1)
                self.cell_calc = calc
                self.cell_calc.compute_CNID([0, 0, 1], tol)
                CNID = self.cell_calc.CNID
                self.CNID = np.dot(a1, CNID)
                self.R = R
                self.theta = theta
                self.axis = n1
                self._write_found_csl(
                    file, sigma1, sigma2, D, axis, theta)
                if self.verbose:
                    self._print_found_csl(
                        sigma1, sigma2, D, axis, theta)
                break
            if found:
                break
            theta += dtheta
        if not found:
            print(
                'failed to find a satisfying appx CSL. '
                'Try to adjust the limits according'
                'to the log file generated; or try another orientation.')

    def search_all_position(
            self,
            axis,
            theta,
            theta_range,
            dtheta,
            two_D=False):
        """
        main loop finding all the CSL lattices satisfying the limit

        Parameters
        ----------
        axis : numpy array
            rotation axis
        theta : float
            initial rotation angle, in degree
        theta_range : list
            range varying theta, in degree
        dtheta : float
            step varying theta, in degree

        Notes
        ----------
        log.all_position :
            all the searching information
        results :
            information of the found approximate CSL
        """
        axis = np.dot(self.lattice_1, axis)
        if self.verbose:
            print(axis)
        theta = theta / 180 * np.pi
        n = np.ceil(theta_range / dtheta)
        dtheta = theta_range / n / 180 * np.pi

        if not two_D:
            a1 = self.lattice_1.copy()
            a2_0 = np.dot(self.orientation, self.lattice_2).copy()

        else:
            # find the two primitive plane bases
            # miller indices
            hkl_1 = MID(self.lattice_1, axis)
            if self.verbose:
                print(axis)
            hkl_2 = MID(np.dot(self.orientation, self.lattice_2), axis)
            # plane bases
            plane_B_1 = get_pri_vec_inplane(hkl_1, self.lattice_1)
            plane_B_2 = get_pri_vec_inplane(
                hkl_2, np.dot(self.orientation, self.lattice_2))
            v_3 = cross(plane_B_1[:, 0], plane_B_1[:, 1])
            a1 = np.column_stack((plane_B_1, v_3))
            a2_0 = np.column_stack((plane_B_2, v_3))
            self.a1 = a1.copy()
            self.a2 = a2_0.copy()
            # a2_0 back to the initial orientation
            a2_0 = np.dot(inv(self.orientation), a2_0)
        # rotation loop
        file = open('log.all_position', 'w', encoding="utf-8")
        file_r = open('results', 'w', encoding="utf-8")
        file.write('---Searching starts---\n')
        file.write('axis theta dtheta n S du sigma1_max sigma2_max\n')
        file.write(
            f'{axis} {theta} {dtheta} {n} {self.S} '
            f'{self.du} {self.sgm1} {self.sgm2}\n')
        file.write('-----------for theta-----------\n')
        for _ in range(int(n)):
            R = np.dot(self.orientation, rot(axis, theta))
            U = three_dot(inv(a1), R, a2_0)
            file.write('theta = ' + str(theta / np.pi * 180) + '\n')
            file.write('    -----for N-----\n')
            for N in range(1, self.sgm2 + 1):
                U_p = np.round(N * U) / N
                if not np.all((abs(U_p - U)) < self.du):
                    continue
                file.write('    N= ' + str(N) + " accepted" + '\n')
                R_p = three_dot(a1, U_p, inv(a2_0))
                D = np.dot(inv(R), R_p)
                if not ((abs(det(D) - 1) <= self.S)
                        and np.all(abs(D - np.eye(3)) < self.dd)):
                    continue
                file.write('    --D accepted--\n')
                file.write(f"    D, det(D) = {det(D)} \n")
                ax2 = three_dot(R, D, a2_0)
                calc = DSCcalc()
                try:
                    calc.parse_int_U(a1, ax2, self.sgm2)
                    calc.compute_CSL()
                except BaseException:
                    file.write('    failed to find CSL here \n')
                    continue
                if not abs(det(calc.U1)) <= self.sgm1:
                    file.write('    sigma too large \n')
                    continue
                file.write('--------------------------------\n')
                file_r.write('--------------------------------\n')
                file.write('Congrates, we found an appx CSL!\n')
                sigma1 = int(abs(np.round(det(calc.U1))))
                sigma2 = int(abs(np.round(det(calc.U2))))

                self._write_found_csl(
                    file, sigma1, sigma2, D, axis, theta)
                self._write_found_csl(
                    file_r, sigma1, sigma2, D, axis, theta)
                if two_D:
                    calc.compute_CNID([0, 0, 1])
                    self.CNID = np.dot(a1, calc.CNID)
                    file_r.write(
                        'CNID = \n' + str(self.CNID) + '\n')
                if self.verbose:
                    self._print_found_csl(
                        sigma1, sigma2, D, axis, theta)
            theta += dtheta

    def get_bicrystal(self, dydz=None, dx=0, dp1=0, dp2=0,
                      xyz_1=None, xyz_2=None, vx=0, filename='POSCAR',
                      two_D=False, filetype='VASP', mirror=False, KTI=False):
        """
        generate a cif file for the bicrystal structure

        Parameters
        ----------
        dydz : numpy array
            translation vector in the interface
        dx : float
            translation normal to the interface
        dp1, dp2 : float
            termination of slab 1, 2
        xyz1, xyz2 : list
            expansion of slab 1, 2
        vx : float
            vacuum spacing, default 0
        filename : str
            filename, default 'POSCAR'
        two_D : bool
            whether a two CSL
        filetype : str
            filetype, 'VASP' or 'LAMMPS', default 'VASP'
        mirror : bool
            mirror, default False
        KTI : bool
            KTI, default False
        """
        if dydz is None:
            dydz = np.zeros(3)
        if xyz_1 is None:
            xyz_1 = np.ones(3)
        if xyz_2 is None:
            xyz_2 = np.ones(3)
        # get the atoms in the primitive cell
        lattice_1, atoms_1, elements_1 = (
            self.lattice_1.copy(), self.atoms_1.copy(), self.elements_1.copy())
        lattice_2, atoms_2, elements_2 = (
            self.lattice_2.copy(), self.atoms_2.copy(), self.elements_2.copy())

        # deform & rotate lattice_2
        if two_D:
            lattice_2 = np.dot(self.a2_transform, lattice_2)
        else:
            lattice_2 = three_dot(self.R, self.D, lattice_2)
        # make supercells of the two slabs
        atoms_1, elements_1, lattice_1 = super_cell(
            self.bicrystal_U1, lattice_1, atoms_1, elements_1)
        if not mirror:
            atoms_2, elements_2, lattice_2 = super_cell(
                self.bicrystal_U2, lattice_2, atoms_2, elements_2)
        else:
            eps = 0.000001
            atoms_2 = atoms_1.copy()
            elements_2 = elements_1.copy()
            lattice_2 = lattice_1.copy()
            atoms_2[:, 0] = - atoms_2[:, 0] + 1
            atoms_c, elements_c = atoms_2.copy(), elements_2.copy()
            elements_c = elements_c[atoms_c[:, 0] + eps > 1]
            atoms_c = atoms_c[atoms_c[:, 0] + eps > 1]
            atoms_c[:, 0] = atoms_c[:, 0] - 1
            atoms_2 = np.vstack((atoms_2, atoms_c))
            elements_2 = np.append(elements_2, elements_c)

            # delete overwrapping atoms
            elements_2 = elements_2[np.where(atoms_2[:, 0] + eps < 1)]
            atoms_2 = atoms_2[atoms_2[:, 0] + eps < 1]

        # expansion
        if not (np.all(xyz_1 == 1) and np.all(xyz_2 == 1)):
            if not np.all([xyz_1[1], xyz_1[2]] == [xyz_2[1], xyz_2[2]]):
                raise RuntimeError(
                    'the two slabs must expand '
                    'to the same dimension in the interface plane')
            lattice_1, atoms_1, elements_1 = cell_expands(lattice_1, atoms_1,
                                                          elements_1, xyz_1)
            lattice_2, atoms_2, elements_2 = cell_expands(lattice_2, atoms_2,
                                                          elements_2, xyz_2)

        # termination
        if dp1 != 0:
            if KTI:
                atoms_1, elements_1 = shift_termi_left(
                    lattice_1, dp1, atoms_1, elements_1)
            else:
                atoms_1 = shift_none_copy(lattice_1, dp1, atoms_1)
        if dp2 != 0:
            if KTI:
                atoms_2, elements_2 = shift_termi_right(
                    lattice_2, dp2, atoms_2, elements_2)
            else:
                atoms_2 = shift_none_copy(lattice_2, dp2, atoms_2)

        # adjust the orientation
        lattice_1, self.orient = adjust_orientation(lattice_1)
        lattice_2 = np.dot(self.orient, lattice_2)
        write_POSCAR(lattice_1, atoms_1, elements_1, 'POSCAR')
        POSCAR_to_cif('POSCAR', 'cell_1.cif')
        self.slab_structure_1 = Structure.from_file(
            'POSCAR', sort=False, merge_tol=0.0)
        write_POSCAR(lattice_2, atoms_2, elements_2, 'POSCAR')
        POSCAR_to_cif('POSCAR', 'cell_2.cif')
        self.slab_structure_2 = Structure.from_file(
            'POSCAR', sort=False, merge_tol=0.0)
        os.remove('POSCAR')

        self.R_see_plane = get_R_to_screen(lattice_1)

        self.plane_list_1, self.elements_list_1, \
            self.indices_list_1, self.dp_list_1 \
            = terminates_scanner_left(lattice_1, atoms_1, elements_1, self.d1)

        self.plane_list_2, self.elements_list_2, \
            self.indices_list_2, self.dp_list_2 \
            = terminates_scanner_right(lattice_2, atoms_2, elements_2, self.d2)

        self.slab_lattice_1 = lattice_1.copy()
        self.slab_lattice_2 = lattice_2.copy()

        # combine the two lattices and translate atoms
        lattice_bi = lattice_1.copy()
        if two_D:
            height_1 = get_height(lattice_1)
            height_2 = get_height(lattice_2)
            lattice_bi[:, 0] = lattice_bi[:, 0] * (1 + height_2 / height_1)
            # convert to cartesian
            atoms_1 = np.dot(lattice_1, atoms_1.T).T
            atoms_2 = np.dot(lattice_2, atoms_2.T).T
            # translate
            atoms_2 = atoms_2 + lattice_1[:, 0]
            # back to the fractional coordinates
            atoms_1 = np.dot(inv(lattice_bi), atoms_1.T).T
            atoms_2 = np.dot(inv(lattice_bi), atoms_2.T).T
        else:
            lth1, lth2 = norm(lattice_1[:, 0]), norm(lattice_2[:, 0])
            lattice_bi[:, 0] = lattice_1[:, 0] + lattice_2[:,0]
            atoms_1[:, 0] = atoms_1[:, 0] * lth1 / (lth1 + lth2)
            atoms_2[:, 0] = (atoms_2[:, 0] * lth2 + lth1) / (lth1 + lth2)

        # excess volume
        if dx != 0:
            lattice_bi, atoms_1, atoms_2 = excess_volume(
                lattice_1, lattice_bi, atoms_1, atoms_2, dx)

        # in-plane translation
        if norm(dydz) > 0:
            dydz = np.dot(self.orient, dydz)
            plane_shift = np.dot(inv(lattice_bi), dydz)
            atoms_2 = atoms_2 + plane_shift

        # combine the two slabs
        elements_bi = np.append(elements_1, elements_2)
        atoms_bi = np.vstack((atoms_1, atoms_2))

        # wrap the periodic boundary condition
        atoms_bi[:, 1] = atoms_bi[:, 1] - np.floor(atoms_bi[:, 1])
        atoms_bi[:, 2] = atoms_bi[:, 2] - np.floor(atoms_bi[:, 2])
        # vacummn
        if vx > 0:
            atoms_bi, lattice_bi = surface_vacuum(lattice_1, lattice_bi, atoms_bi, vx)

        # save
        self.lattice_bi = lattice_bi
        self.atoms_bi = atoms_bi
        self.elements_bi = elements_bi
        write_POSCAR(lattice_bi, atoms_bi, elements_bi, 'POSCAR')
        self.bicrystal_structure = Structure.from_file(
            'POSCAR', sort=False, merge_tol=0.0)
        os.remove('POSCAR')
        if filetype == 'VASP':
            write_POSCAR(lattice_bi, atoms_bi, elements_bi, filename)
        elif filetype == 'LAMMPS':
            write_LAMMPS(
                lattice_bi,
                atoms_bi,
                elements_bi,
                filename,
                self.bicrystal_ortho)
        else:
            raise RuntimeError(
                "Sorry, we only support for 'VASP' or 'LAMMPS' output")

    def sample_CNID(
            self, grid, dx=0, dp1=0, dp2=0,
            xyz_1=None, xyz_2=None, vx=0, two_D=False,
            filename='POSCAR', filetype='VASP'):
        """
        sampling the CNID and generate POSCARs

        Parameters
        ----------
        grid : numpy array
            2D grid of sampling
        """
        if xyz_1 is None:
            xyz_1 = [1, 1, 1]
        if xyz_2 is None:
            xyz_2 = [1, 1, 1]
        os.mkdir('CNID_inputs')
        if self.verbose:
            print('CNID')
            print(np.round(np.dot(inv(self.lattice_1), self.CNID), 8))
            print(f'making {grid[0] * grid[1]} files...')
        n1 = grid[0]
        n2 = grid[1]
        v1, v2 = self.CNID.T
        for i in range(n1):
            for j in range(n2):
                dydz = v1 / n1 * i + v2 / n2 * j
                self.get_bicrystal(
                    dydz=dydz, dx=dx, dp1=dp1, dp2=dp2,
                    xyz_1=xyz_1, xyz_2=xyz_2, vx=vx, two_D=two_D,
                    filename=f'CNID_inputs/{filename}_{i}_{j}',
                    filetype=filetype)
        if self.verbose:
            print('completed')

    def sample_lattice_planes(self, dx=0, xyz_1=None, xyz_2=None, vx=0,
                              two_D=False, filename='POSCAR', filetype='VASP'):
        """
        sampling non-identical lattice planes terminating at the GB
        """
        if xyz_1 is None:
            xyz_1 = [1, 1, 1]
        if xyz_2 is None:
            xyz_2 = [1, 1, 1]
        os.mkdir('terminating_shift_inputs')
        if self.verbose:
            print('terminating_sampling...')
        position_here = 0
        count = 1
        while abs(position_here) < 1 / 2 * self.min_perp_length:
            self.get_bicrystal(
                dx=dx, dp2=position_here,
                xyz_1=xyz_1, xyz_2=xyz_2, vx=vx, two_D=two_D,
                filename=f'terminating_shift_inputs/{filename}_{count}',
                filetype=filetype)
            position_here -= self.d2
            count += 1
        if self.verbose:
            print('completed')

    def set_orientation_axis(self, axis_1, axis_2, tol=1e-10):
        """
        rotate lattice_2
        so that its axis_2 coincident with the axis_1 of lattice_1
        """
        axis_1 = np.dot(self.lattice_1, axis_1.T).T
        axis_1 = axis_1 / norm(axis_1)
        axis_2 = np.dot(self.lattice_2, axis_2.T).T
        axis_2 = axis_2 / norm(axis_2)
        c = cross(axis_2, axis_1)
        if norm(c) < tol:
            R = eye(3, 3)
        else:
            angle = arccos(np.dot(axis_1, axis_2))
            R = rot(c, angle)
        self.orientation = R

    def compute_bicrystal(
            self,
            hkl,
            lim=20,
            normal_ortho=False,
            plane_ortho=False,
            tol_ortho=1e-10,
            tol_integer=1e-8,
            align_rotation_axis=False,
            rotation_axis=None,
            inclination_tol=sqrt(2) / 2):
        """
        compute the transformation
        to obtain the supercell of the two slabs forming a interface

        Parameters
        ----------
        hkl : numpy array
            miller indices of the plane expressed in lattice_1
        lim : int
            the limit searching for a CSL vector cross the plane, default 20
        normal_ortho : bool
            whether limit the vector crossing the GB to be normal to the GB,
            default False
        plane_ortho : bool
            whether limit the two vectors in the GB plane to be orthogonal,
            default False
        tol_ortho : float
            tolerance judging whether orthogonal, default 1e-10
        tol_integer : float
            tolerance judging integer, default 1e-8
        align_rotation_axis : bool
            whether to align to rotation axis, default False
        rotation_axis : list
            rotation axis, defalt [1, 1, 1]
        inclination_tol : float
            control the angle between the interface and the cross vector,
            default sqrt(2)/2
        """
        if rotation_axis is None:
            rotation_axis = [1, 1, 1]
        if normal_ortho and plane_ortho:
            self.bicrystal_ortho = True
        self.d1 = d_hkl(self.lattice_1, hkl)
        lattice_2 = three_dot(self.R, self.D, self.lattice_2)
        hkl_2 = get_primitive_hkl(hkl, self.lattice_1, lattice_2, tol_integer)
        self.d2 = d_hkl(lattice_2, hkl_2)
        hkl_c = get_primitive_hkl(
            hkl, self.lattice_1, self.CSL, tol_integer
        )  # miller indices of the plane in CSL
        hkl_c = np.array(hkl_c)
        if self.verbose:
            print('hkl in CSL:', hkl_c)
        # plane bases of the CSL lattice plane
        plane_B = get_pri_vec_inplane(hkl_c, self.CSL)
        if plane_ortho and (
                abs(np.dot(plane_B[:, 0], plane_B[:, 1])) > tol_ortho):
            plane_B = get_ortho_two_v(
                plane_B,
                lim,
                tol_ortho,
                align_rotation_axis,
                rotation_axis)
        if align_rotation_axis:
            if norm(cross(plane_B[:, 0], rotation_axis) > 1e-3):
                change_v = plane_B[:, 1].copy()
                plane_B[:, 1] = plane_B[:, 0]
                plane_B[:, 0] = change_v
        plane_n = cross(plane_B[:, 0], plane_B[:, 1])  # plane normal
        v3 = cross_plane(
            self.CSL, plane_n, lim, normal_ortho,
            tol_ortho, inclination_tol)  # a CSL basic vector cross the plane
        supercell = np.column_stack(
            (v3, plane_B))  # supercell of the bicrystal
        supercell = get_right_hand(supercell)  # check right-handed
        if normal_ortho:
            self.min_perp_length = norm(v3)
        self.bicrystal_U1 = np.array(
            np.round(
                np.dot(inv(self.lattice_1), supercell), 8), dtype=int)
        self.bicrystal_U2 = np.array(
            np.round(
                np.dot(inv(self.lattice_2_TD), supercell), 8), dtype=int)
        self.cell_calc.compute_CNID(hkl, tol_integer)
        CNID = self.cell_calc.CNID
        self.CNID = np.dot(self.lattice_1, CNID)
        if self.verbose:
            print('cell 1:')
            print(array(np.round(self.bicrystal_U1, 8), dtype=int))
            print('cell 2:')
            print(array(np.round(self.bicrystal_U2, 8), dtype=int))

    def compute_bicrystal_two_D(
            self,
            hkl_1,
            hkl_2,
            lim=20,
            normal_ortho=False,
            tol_ortho=1e-10,
            tol_integer=1e-8,
            inclination_tol=sqrt(2) / 2):
        """
        compute the transformation
        to obtain the supercell of the two slabs
        forming a interface (only two_D periodicity)

        Parameters
        ----------
        hkl1, hkl2 : numpy array
            miller indices of the plane expressed in lattice_1 and lattice_2
        lim : int
            the limit searching for a CSL vector cross the plane, default 20
        normal_ortho : bool
            whether limit the vector crossing the GB to be normal to the GB,
            default False
        tol_ortho : float
            tolerance judging whether orthogonal, default 1e-10
        tol_integer : float
            tolerance judging integer, default 1e-8
        inclination_tol : float
            control the angle between the interface and the cross vector,
            default sqrt(2)/2
        """
        self.d1 = d_hkl(self.lattice_1, hkl_1)
        lattice_2 = np.dot(self.a2_transform, self.lattice_2)
        normal_1 = get_normal_from_MI(self.lattice_1, hkl_1)
        hkl_2 = MID(lattice_2, normal_1, tol_integer)
        self.d2 = d_hkl(lattice_2, hkl_2)
        # the two slabs with auxilary vector
        plane_1 = np.dot(self.lattice_1, self.U1)

        # the transformed lattice_2
        a2 = np.dot(self.a2_transform, self.lattice_2)
        plane_2 = np.dot(a2, self.U2)

        v3_1 = cross_plane(
            self.lattice_1,
            normal_1,
            lim,
            normal_ortho,
            tol_ortho,
            inclination_tol)
        v3_2 = cross_plane(
            a2,
            normal_1,
            lim,
            normal_ortho,
            tol_ortho,
            inclination_tol)
        if np.dot(v3_1, v3_2) < 0:
            v3_2 = - v3_2

        # unit slabs
        cell_1 = np.column_stack((v3_1, plane_1))
        cell_2 = np.column_stack((v3_2, plane_2))

        # right_handed
        cell_1 = get_right_hand(cell_1)
        cell_2 = get_right_hand(cell_2)

        # supercell index
        self.bicrystal_U1 = np.dot(inv(self.lattice_1), cell_1)
        self.bicrystal_U2 = np.dot(inv(a2), cell_2)
        if self.verbose:
            print('cell 1:')
            print(array(np.round(self.bicrystal_U1, 8), dtype=int))
            print('cell 2:')
            print(array(np.round(self.bicrystal_U2, 8), dtype=int))

    def define_lammps_regions(
            self,
            region_names,
            region_los,
            region_his,
            ortho=False):
        """
        generate a file defining some regions
        in the LAMMPS and define the atoms
        inside these regions into some groups.

        Parameters
        ----------
        region_names : list
            list of name of regions
        region_los : list
            list of the low bounds
        region_his : list
            list of the hi bounds
        ortho : bool
            whether the cell is orthogonal
        """

        if (len(region_los) != len(region_names)) or (
                len(region_los) != len(region_his)):
            raise RuntimeError(
                "the region_names, low & high bounds "
                "must have same num of elements.")

        xy = self.lattice_bi[:, 1][0]
        xz = self.lattice_bi[:, 2][0]
        yz = self.lattice_bi[:, 2][1]

        with open('blockfile', 'w', encoding="utf-8") as fb:
            for rn, rl, rh in zip(region_names, region_los, region_his):
                if not ortho:
                    fb.write(
                        f'region {rn} prism {rl} {rh} EDGE EDGE EDGE EDGE '
                        f'{xy} {xz} {yz} units box \n')
                else:
                    fb.write(
                        f'region {rn} block {rl} {rh} EDGE EDGE EDGE EDGE '
                        'units box \n')
                fb.write(f'group {rn} region {rn} \n')


def terminates_scanner_slab_structure(structure, hkl):
    atoms, elements = get_sites_elements(structure)
    lattice = structure.lattice.matrix.T
    plane_B = get_pri_vec_inplane(hkl, lattice)
    d = d_hkl(lattice, hkl)
    v3 = cross_plane(lattice, cross(
        plane_B[:, 0], plane_B[:, 1]), 10, False, 0.001, 0.001)
    supercell = np.column_stack((v3, plane_B))  # supercell of the bicrystal
    supercell = get_right_hand(supercell)
    U = array(around(np.dot(inv(lattice), supercell)), dtype=int)
    print(f'cross vector: \n{str(array([U[:,0]]).T)}\nlength: {str(norm(v3))}')
    print(
        f'plane basis: \n{str(U[:,[1,2]])}\n'
        f'length: {str(norm(supercell[:,1]))} {str(norm(supercell[:,2]))}')
    atoms, elements, lattice = super_cell(U, lattice, atoms, elements)
    write_POSCAR(lattice, atoms, elements, 'POSCAR_primitive')
    lattice, _ = adjust_orientation(lattice)
    plane_list, element_list, _, dp_list = terminates_scanner_left(
        supercell, atoms, elements, d, round_n=5)
    return plane_list, element_list, dp_list


def get_surface_slab(
        structure,
        hkl,
        replica=None,
        inclination_tol=sqrt(2) / 2,
        termi_shift=0,
        vacuum_height=0,
        plane_normal=False,
        normal_perp=False,
        normal_tol=1e-3,
        lim=20,
        filename='POSCAR',
        filetype='VASP'):
    """
    get a superlattice of a slab containing a desired surface
    as crystal plane hkl

    Parameters
    ----------
    structure : Structure object
        Structure object of the unit cell structure
    hkl : numpy array
        miller indices of the surface plane
    replica : list
        expansion of the primitive slab cell, default [1, 1, 1]
    inclination_tol : float
        required minimum cos value of the angle
        between the basic crossing vector and the surface, default sqrt(2)/2
    termi_shift : float
        shift the termination of the surface, default 0
    vacuum_height : float
        height of vaccum
    plane_normal : bool
        whether requiring the two vectors
        in the surface plane to be perpendicular, default False
    normal_perp : bool
        whether requiring the crossing vector to be perpendicular to the plane,
        default False
    normal_tol :
        tolerance to judge whether perpendicular, default 1e-3
    lim : int
        control the number of generated vectors
        to search for the crossing vectors and perpendicular vectors,
        default 20
    filename : str
        name of the generated atom files
    filetype : str, "VASP" or "LAMMPS"
        type of the generated atom files (for VASP or LAMMPS)

    Returns
    ----------
        slab_structure : structure object
            slab structure
    """
    if replica is None:
        replica = [1, 1, 1]
    atoms, elements = get_sites_elements(structure)
    lattice = structure.lattice.matrix.T
    plane_B = get_pri_vec_inplane(hkl, lattice)
    if plane_normal and (
            abs(np.dot(plane_B[:, 0], plane_B[:, 1])) > normal_tol):
        plane_B = get_ortho_two_v(plane_B, lim, normal_tol, False)
    v3 = cross_plane(
        lattice, cross(plane_B[:, 0], plane_B[:, 1]),
        lim, normal_perp, normal_tol, inclination_tol)
    supercell = np.column_stack((v3, plane_B))  # supercell of the bicrystal
    supercell = get_right_hand(supercell)
    U = array(around(np.dot(inv(lattice), supercell)), dtype=int)
    print(f'cross vector: \n{str(array([U[:,0]]).T)}\nlength: {str(norm(v3))}')
    print(
        f'plane basis: \n{str(U[:,[1,2]])}\n'
        f'length: {str(norm(supercell[:,1]))} {str(norm(supercell[:,2]))}')
    atoms, elements, lattice = super_cell(U, lattice, atoms, elements)
    lattice, _ = adjust_orientation(lattice)
    if not np.all(replica == 1):
        lattice, atoms, elements = cell_expands(
            lattice, atoms, elements, replica)

    if termi_shift != 0:
        atoms = shift_none_copy(lattice, termi_shift, atoms)

    if vacuum_height > 0:
        atoms, lattice = surface_vacuum(lattice, lattice, atoms, vacuum_height)

    write_POSCAR(lattice, atoms, elements, 'POSCAR')
    slab_structure = Structure.from_file('POSCAR', sort=False, merge_tol=0.0)
    os.remove('POSCAR')
    if filetype == 'VASP':
        write_POSCAR(lattice, atoms, elements, filename)
    elif filetype == 'LAMMPS':
        orthogonal = False
        if plane_normal and normal_perp:
            orthogonal = True
        write_LAMMPS(lattice, atoms, elements, filename, orthogonal)
    else:
        raise RuntimeError(
            "Sorry, we only support for 'VASP' or 'LAMMPS' output")
    return slab_structure
