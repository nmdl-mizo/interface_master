from numpy.linalg import det, norm, inv
from numpy import dot, cross, ceil, floor, cos, sin, tile, array, arange, meshgrid, delete
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp.inputs import Poscar
import numpy as np
from interfacemaster.cellcalc import MID, DSCcalc, get_primitive_hkl, get_right_hand, find_integer_vectors, get_pri_vec_inplane, get_ortho_two_v, ang
import os
import matplotlib.pyplot as plt

def rot(a, Theta):
    """
    produce a rotation matrix
    arguments:
    a --- rotation axis
    Theta --- rotation angle
    return:
    a rotation matrix
    """
    c = float(cos(Theta))
    s = float(sin(Theta))
    a = a / norm(a)
    ax, ay, az = a
    return np.array([[c + ax * ax * (1 - c), ax * ay * (1 - c) - az * s,
                      ax * az * (1 - c) + ay * s],
                    [ay * ax * (1 - c) + az * s, c + ay * ay * (1 - c),
                        ay * az * (1 - c) - ax * s],
                     [az * ax * (1 - c) - ay * s, az * ay * (1 - c) + ax * s,
                      c + az * az * (1 - c)]], dtype = np.float64)

def rational_mtx(M, N):
    """
    find a rational matrix close to M
    arguments:
    M --- original matrix
    N --- denominator
    return:
    a rational matrix
    """
    B = np.eye(3)
    for i in range(3):
        for j in range(3):
            B[i][j] = round(N * M[i][j])
    return B, N

def three_dot(M1, M2, M3):
    """
    compute the three continuous dot product
    """
    return dot(dot(M1,M2),M3)

def get_ang_list(m1, n):
    """
    compute a list of ang cos between one list of vecor and one vector
    """
    return 1 / norm(n) * abs(dot(m1, n)) / norm(m1, axis = 1)

def cross_plane(lattice, n, lim, orthogonal, tol):
    """
    get a primitive lattice vector cross a plane
    argument:
    lattice --- lattice matrix
    n --- a normal vector
    lim --- control how many vectors to be generated
    tol --- tolerance judging orthogonal
    return:
    a primitve vector normal to the plane
    """
    x = np.arange(-lim, lim, 1)
    y = x
    z = x
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)[0]]
    indice_0 = indice_0[np.where(np.gcd.reduce(indice_0, axis=1) == 1)[0]]
    ltc_p = dot(indice_0, lattice.T)
    ltc_p = ltc_p[np.argsort(norm(ltc_p, axis=1))]
    dot_list = get_ang_list(ltc_p, n)
    if orthogonal == False:
        normal_v = ltc_p[np.where(dot_list >= 0.75)[0]]
        normal_v = normal_v[np.argsort(norm(normal_v, axis=1))]
        normal_v = normal_v[0]
    else:
        try:
            normal_v = ltc_p[np.where(abs(dot_list - 1) < tol)[0]][0]
        except:
            raise RuntimeError('failed to find a vector cross the plane. try larger lim or smaller tol or use non-orthogonal cell')
    return normal_v

def get_sites_elements(structure):
    """
    get the coordinates of atoms and the elements
    arguments:
    structure --- pymatgen structure class
    return:
    atoms --- atom coordinates
    elements --- list of element name of the atoms
    return:
    atoms --- fractional coordinates of atoms in the primitive cell
    elements --- list of element names of the atom
    """
    atoms = np.array([0, 0, 0])
    elements = []
    for i in structure.sites:
        atoms = np.vstack((atoms, i.frac_coords))
        elements.append(i.species_string)
    atoms = np.delete(atoms, 0, axis = 0)
    return atoms, np.array(elements)

def POSCAR_to_cif(Poscar_name, Cif_name):
    """
    generate a cif file for the structure in a POSCAR file
    """
    structure = Structure.from_file(Poscar_name)
    structure.to(filename=Cif_name)
    del structure

def write_LAMMPS(lattice, atoms, elements, filename = 'lmp_atoms_file', orthogonal = False):
    """
    write LAMMPS input atom file file of a supercell
    argument:
    lattice --- lattice matrix
    atoms --- fractional atoms coordinates
    elements --- list of element name of the atoms
    orthogonal --- whether write orthogonal cell
    """

    #list of elements
    element_species = np.unique(elements)

    #to Cartesian
    atoms = dot(lattice, atoms.T).T

    #get the speicie identifiers (e.g. 1 for Cu, 2 for O...)
    element_indices = np.arange(len(element_species)) + 1
    species_identifiers = np.arange(len(atoms))

    for i in range(len(element_species)):
        indices_this_element = np.where(elements == element_species[i])[0]
        species_identifiers[indices_this_element] = element_indices[i]

    species_identifiers = np.array([species_identifiers])

    #the atom ID
    IDs = np.arange(len(atoms)).reshape(1, -1) + 1

    #get the final format
    Final_format = np.concatenate((IDs.T, species_identifiers.T), axis=1)
    Final_format = np.concatenate((Final_format, atoms), axis = 1)

    #define the box
    #xlo, xhi
    xhi, yhi, zhi = lattice[0][0], lattice[1][1], lattice[2][2]
    xlo, ylo, zlo = 0, 0, 0
    xy = lattice[:,1][0]
    xz = lattice[:,2][0]
    yz = lattice[:,2][1]

    with open(filename, 'w') as f:
        f.write('#LAMMPS input file of atoms generated by interface_master. The elements are: ')
        for i in range(len(element_indices)):
            f.write('{0} {1} '.format(element_indices[i], element_species[i]))
        f.write('\n {} atoms \n \n'.format(len(atoms)))
        f.write('{} atom types \n \n'.format(len(element_species)))
        f.write('{0:.8f} {1:.8f} xlo xhi \n'.format(xlo, xhi))
        f.write('{0:.8f} {1:.8f} ylo yhi \n'.format(ylo, yhi))
        f.write('{0:.8f} {1:.8f} zlo zhi \n\n'.format(zlo, zhi))
        if orthogonal == False:
            f.write('{0:.8f} {1:.8f} {2:.8f} xy xz yz \n\n'.format(xy, xz, yz))
        f.write('Atoms \n \n')
        np.savetxt(f, Final_format, fmt='%i %i %.16f %.16f %.16f')
    f.close()

def write_POSCAR(lattice, atoms, elements, filename = 'POSCAR'):
    """
    write Poscar file of a supercell
    argument:
    lattice --- lattice matrix
    atoms --- fractional atoms coordinates
    elements --- list of element name of the atoms
    """
    element_species = np.unique(elements)
    atoms_list = []
    num_list = []
    for i in range(len(element_species)):
        atoms_this_element = atoms[np.where(elements == element_species[i])[0]]
        atoms_list.append(atoms_this_element)
        num_list.append(len(atoms_this_element))

    if len(element_species) > 1:
        atoms = np.array([[0,0,0]],dtype = float)
        for i in atoms_list:
            atoms = np.vstack((atoms,i))
        atoms = np.delete(atoms,0,axis = 0)

    with open(filename, 'w') as f:
        f.write('#POSCAR generated by IF_master \n')

        #matrix
        f.write("1\n")
        f.write('{0:.16f} {1:.16f} {2:.16f} \n'.format(lattice[:,0][0],lattice[:,0][1],lattice[:,0][2]))
        f.write('{0:.16f} {1:.16f} {2:.16f} \n'.format(lattice[:,1][0],lattice[:,1][1],lattice[:,1][2]))
        f.write('{0:.16f} {1:.16f} {2:.16f} \n'.format(lattice[:,2][0],lattice[:,2][1],lattice[:,2][2]))

        #elements
        for i in element_species:
            f.write(str(i) + ' ')
        f.write('\n')

        #num of atoms
        for i in num_list:
            f.write(str(i) + ' ')
        f.write('\n')
        f.write("Direct\n")
        np.savetxt(f, atoms, fmt='%.16f %.16f %.16f')
    f.close()

def cell_expands(lattice, atoms, elements, xyz):
    """
    expand certain supercell
    arguments:
    lattice --- lattice matrix
    atoms --- list of atom fractional coordinates
    elements --- list of element name of the atoms
    xyz --- expansions
    return:
    lattice --- lattice matrix
    atoms --- atom coordinates
    elements --- list of element name of the atoms
    """
    mtx = lattice.copy()
    dimX, dimY, dimZ = xyz
    x_shifts = np.arange(dimX)
    y_shifts = np.arange(dimY)
    z_shifts = np.arange(dimZ)

    dim_array = np.array([dimX,dimY,dimZ])
    g1_shifts = np.array(np.meshgrid(x_shifts, y_shifts, z_shifts)).T.reshape(-1, 3)
    atoms_expand = atoms.repeat(len(g1_shifts), axis=0) + np.tile(g1_shifts, (len(atoms), 1))
    elements = elements.repeat(len(g1_shifts))
    for i in range(3):
        atoms_expand[:,i] = atoms_expand[:,i]/xyz[i]
        mtx[:,i] = mtx[:,i] * xyz[i]
    return mtx, atoms_expand, elements

def get_array_bounds(U):
    """
    get the meshgrid formed by three sets of lower & upper bounds.
    the bounds cooresponds to the 8 vertices of the cell consisting of the three column vectors of a matrix
    argument:
    U --- integer matrix
    return:
    indice --- meshgrid made by the lower and upper bounds of the indices
    """
    Mo = U.copy()
    # get the coordinates of 8 vertices
    P1 = [0,0,0]
    P2 = Mo[:,0]
    P3 = Mo[:,1]
    P4 = Mo[:,2]
    P5 = P2 + P3
    P6 = P2 + P4
    P7 = P3 + P4
    P8 = P2 + P3 + P4
    Points = np.vstack((P1,P2,P3,P4,P5,P6,P7,P8))
    # enclose the 8 verticies
    min1 = np.round(min(Points[:,0]-1),0)
    max1 = np.round(max(Points[:,0]+1),0)

    min2 = np.round(min(Points[:,1]-1),0)
    max2 = np.round(max(Points[:,1]+1),0)

    min3 = np.round(min(Points[:,2]-1),0)
    max3 = np.round(max(Points[:,2]+1),0)
    a = np.array([[min1,max1], [min2,max2], [min3,max3]])

    x = np.arange(min1 - 2, max1 + 2, 1)
    y = np.arange(min2 - 2, max2 + 2, 1)
    z = np.arange(min3 - 2, max3 + 2, 1)

    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) * len(y) * len(z), 3)
    return indice

def super_cell(U, lattice, Atoms, elements):
    """
    make supercell
    argument:
    U --- coefficients of the LC of three vectors from the basis
    atoms --- fractional coordinates of atoms
    elements --- list of the element names of these atoms
    return:
    lattice --- lattice matrix
    atoms --- atom coordinates
    elements --- list of element name of the atoms
    """
    indice = get_array_bounds(U)

    #1.get atoms
    Atoms = Atoms.repeat(len(indice),axis=0) + np.tile(indice,(len(Atoms),1))
    elements = elements.repeat(len(indice))
    tol = 1e-10

    #3.delete atoms dropping outside
    Atoms = dot(inv(U),Atoms.T).T
    Atoms_try = Atoms.copy()
    Atoms_try = Atoms + [tol, tol, tol]
    con = (Atoms_try[:,0] < 1) & (Atoms_try[:,0] >= 0) \
        & (Atoms_try[:,1] < 1) & (Atoms_try[:,1] >= 0) \
        & (Atoms_try[:,2] < 1) & (Atoms_try[:,2] >= 0)
    indices = np.where(con)[0]
    Atoms = Atoms[indices]
    elements = elements[indices]
    lattice = dot(lattice, U)
    return Atoms, elements, lattice

def shift_termi_left(lattice, dp, atoms, elements):
    """
    changing terminate involves requiring to cut the cell
    """
    n = cross(lattice[:,1],lattice[:,2])
    position_shift = dp / ang(lattice[:,0], n) / norm(lattice[:,0])
    atoms[:,0] = atoms[:,0] + position_shift
    if dp > 0:
        inner = (atoms[:,0] < 1) & (atoms[:,0] > 2 * position_shift)
        elements = elements[inner]
        atoms = atoms[inner]
        #shift to origin
        atoms[:,0] = atoms[:,0] - 2 * position_shift
        #to cartesian
        atoms = dot(lattice, atoms.T).T
        #cut
        lattice[:,0] = lattice[:,0] * (1 - 2 * position_shift)
        #back
        atoms = dot(inv(lattice), atoms.T).T
    else:
        atoms_c_1 = atoms.copy()
        atoms_c_2 = atoms.copy()
        elements_c = elements.copy()
        atoms_c_1[:,0] += 1
        atoms_c_2[:,0] += -1
        atoms = np.vstack((atoms, atoms_c_1, atoms_c_2))
        elements = np.append(elements, elements_c)
        elements = np.append(elements, elements_c)
        inner = (atoms[:,0] < 1) & (atoms[:,0] > 2 * position_shift)
        elements = elements[inner]
        atoms = atoms[inner]
        #shift to origin
        atoms[:,0] = atoms[:,0] - 2 * position_shift
        #to cartesian
        atoms = dot(lattice, atoms.T).T
        #cut
        lattice[:,0] = lattice[:,0] * (1 - 2 * position_shift)
        #back
        atoms = dot(inv(lattice), atoms.T).T
        
    return atoms, elements

def shift_termi_right(lattice, dp, atoms, elements):
    """
    changing terminate involves requiring to cut the cell
    """
    n = cross(lattice[:,1],lattice[:,2])
    position_shift = dp / ang(lattice[:,0], n) / norm(lattice[:,0])
    atoms[:,0] = atoms[:,0] + position_shift
    if dp < 0:
        inner = (atoms[:,0] > 0) & (atoms[:,0] < 1 + 2 * position_shift)
        elements = elements[inner]
        atoms = atoms[inner]
        #to cartesian
        atoms = dot(lattice, atoms.T).T
        #cut
        lattice[:,0] = lattice[:,0] * (1 + 2 * position_shift)
        #back
        atoms = dot(inv(lattice), atoms.T).T
    else:
        atoms_c_1 = atoms.copy()
        atoms_c_2 = atoms.copy()
        elements_c = elements.copy()
        atoms_c_1[:,0] += 1
        atoms_c_2[:,0] += -1
        atoms = np.vstack((atoms, atoms_c_1, atoms_c_2))
        elements = np.append(elements, elements_c)
        elements = np.append(elements, elements_c)
        inner = (atoms[:,0] > 0) & (atoms[:,0] < 1 + 2 * position_shift)
        elements = elements[inner]
        atoms = atoms[inner]
        #to cartesian
        atoms = dot(lattice, atoms.T).T
        #cut
        lattice[:,0] = lattice[:,0] * (1 + 2 * position_shift)
        #back
        atoms = dot(inv(lattice), atoms.T).T
        
    return atoms, elements
    
def excess_volume(lattice_1, lattice_bi, atoms_1, atoms_2, dx):
    """
    introduce vacuum between the interfaces
    argument:
    lattice_1 --- lattice matrix of the first slab
    lattice_bi --- lattice matrix of the bicrystal
    atoms_1, atoms_2 --- atom fractional coordinates of slab 1, slab 2
    dx --- length of expands normal to the interface with the same units as lattice para
    """
    n = cross(lattice_1[:,1],lattice_1[:,2])
    normal_shift = dx / ang(lattice_1[:,0], n) / norm(lattice_bi[:,0].copy())
    normal_shift_cart = normal_shift * lattice_bi[:,0]
    lattice_bi[:,0] = (normal_shift + 1) * lattice_bi[:,0]
    atoms_1[:,0] = 1 / (normal_shift + 1) * atoms_1[:,0]
    atoms_2 = dot(lattice_bi, atoms_2.T).T
    atoms_2 = atoms_2 + normal_shift_cart
    atoms_2 = dot(inv(lattice_bi), atoms_2.T).T

def surface_vacuum(lattice_1, lattice_bi, atoms_bi, vx):
    """
    introduce vacuum at one of the tails of the bicrystal cell
    argument:
    lattice_1 --- lattice matrix of the first slab
    lattice_bi --- lattice matrix of the bicrystal
    atoms_bi --- atom fractional coordinates of the bicrystal
    vx --- length of the vacuum bulk with units as lattice para
    """
    n = cross(lattice_1[:,1],lattice_1[:,2])
    normal_shift = vx / ang(lattice_1[:,0], n) / norm(lattice_1[:,0])
    lattice_bi[:,0] = lattice_bi[:,0] * (1 + normal_shift)
    atoms_bi[:,0] = 1 / (1 + normal_shift) * atoms_bi[:,0]

def unit_cell_axis(axis):
    """
    get an unit orthogonal cell with the x-axis collinear with certain axis
    """
    v1 = axis / norm(axis)
    v2 = np.array([0,0,0],dtype = float)
    if v1[0] == 0 and v1[1] == 0:
        v2[1] = 1
    else:
        v2[0], v2[1] = -v1[1], v1[0]
    v2 = v2 / norm(v2)
    v3 = cross(v1, v2)
    v3 = v3 / norm(v3)
    B = np.column_stack((v1,v2,v3))
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
    """
    lattice_0 = lattice.copy()
    v1 = lattice[:,0]
    v3 = cross(lattice[:,0],lattice[:,1])
    v2 = cross(v3,v1)

    v1, v2, v3 = unit_v(v1), unit_v(v2), unit_v(v3)
    this_orientation = np.column_stack((v1,v2,v3))
    desti_orientation = np.eye(3)
    R = dot(desti_orientation, inv(this_orientation))     
    lattice = dot(R, lattice)
    #check that a1 and a2 points to positive:
    
    R = dot(lattice, inv(lattice_0))
    
    return lattice, R

def convert_vector_index(lattice_0, lattice_f, v_0):
    """
    convert the index of a vector into a different basis
    """
    v_0 = dot(lattice_0, v_0)
    v_f = dot(inv(lattice_f), v_0)
    return v_f

def print_near_axis(dv, lattice_1, lattice_2, lim=5):
    """
    searching for near coincident lattice vectors
    """
    x = np.arange(-lim, lim, 1)
    y = x
    z = x
    tol = 1e-10
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)[0]]
    num = len(indice_0)
    ltc_p_1 = dot(indice_0, lattice_1.T)
    ltc_p_2 = dot(indice_0, lattice_2.T)

    v_index_1 = np.arange(num).repeat(num)
    v_index_2 = np.tile(np.arange(num), num)
    ltc_p_1_rep = ltc_p_1.repeat(num, axis = 0)
    ltc_p_2_rep = np.tile(ltc_p_2, (num,1))

    distances = abs(norm(ltc_p_1_rep, axis = 1) - norm(ltc_p_2_rep, axis=1))
    close_ids = np.where(distances <= dv)[0]

    close_vecs_1 = ltc_p_1_rep[close_ids]
    close_vecs_2 = ltc_p_2_rep[close_ids]
    close_vecs_2 = close_vecs_2[np.argsort(norm(close_vecs_1,axis = 1))]
    close_vecs_1 = close_vecs_1[np.argsort(norm(close_vecs_1,axis = 1))]

    print('       e1        e2       dv      length_v1    length_v2')
    for i in range(len(close_vecs_1)):
        e1 = np.round(dot(inv(lattice_1),close_vecs_1[i]),8)
        e2 = np.round(dot(inv(lattice_2),close_vecs_2[i]),8)
        dv = abs(norm(close_vecs_1[i]) - norm(close_vecs_2[i]))
        norm_1 = norm(close_vecs_1[i])
        norm_2 = norm(close_vecs_2[i])
        print(e1, e2, dv, norm_1, norm_2)

def get_height(lattice):
    """
    get the distance of the two surfaces (crossing the first vector) of a cell
    """
    n = cross(lattice[:,1], lattice[:,2])
    height = abs(dot(lattice[:,0], n) / norm(n))
    return height

def get_plane_vectors(lattice, n):
    """
    a function get the two vectors normal to a vector
    arguments:
    lattice - lattice matrix
    n - a vector
    return - B two plane vectors
    """
    tol = 1e-8
    B = np.eye(3,2)
    count = 0
    for i in range(3):
        if abs(dot(lattice[:,i], n)) < tol:
            B[:,count] = lattice[:,i]
            count += 1
    if count != 2:
        raise RuntimeError('error: the CSL does not include two vectors in the interface')
    return B

def reciprocal_lattice(B):
    """
    return the reciprocal lattice of B
    """
    v1, v2, v3 = B.T
    V = dot(v1, cross(v2, v3))
    b1 = cross(v2,v3) / V
    b2 = cross(v3,v1) / V
    b3 = cross(v1,v2) / V
    return np.column_stack((b1, b2, b3))

def d_hkl(lattice, hkl):
    """
    return the lattice plane spacing of (hkl) of a lattice
    """
    rep_L = reciprocal_lattice(lattice)
    d = 1 / norm(dot(rep_L, hkl))
    return d

def terminates_scanner_left(slab, atoms, elements, d, round_n = 5):
    """
    find all different atomic planes within 1 lattice plane displacing (in the interface), for the left slab
    arguments:
    slab --- basic vectors of the slab
    atoms --- fractional coordinates of atoms
    elements --- list of name of the atoms
    d --- 1 lattice plane displacing
    round_n -- num of bits to round the fraction coordinates to judge identical planes
    return:
    plane_list --- list of planes of atom fraction coordinates
    element_list --- list of elements in each plane
    indices_list --- list of indices of atoms in each plane
    dp_list --- list of dp parameters as input to select corresponding termination
    tol
    """
    plane_list = []
    element_list = []
    indices_list = []
    dp_list = []
    height = get_height(slab)
    normal = cross(slab[:,1], slab[:,2])
    normal = normal / norm(normal)
    atoms_cart = dot(slab,atoms.T).T
    projections = abs(dot(atoms_cart, normal))
    atoms_round = ceil(projections.copy() * 10000) / 10000
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
        except:
            position = -np.inf
    return plane_list, element_list, indices_list, dp_list

def get_R_to_screen(lattice):
    """
    get a rotation matrix to make the interface plane of the slab located in the screen
    """
    v2 = lattice[:,1]
    v2 = v2 / norm(v2)
    v1 = cross(lattice[:,1], lattice[:,2])
    v1 = v1 / norm(v1)
    v3 = cross(v1, v2)
    v3 = v3 / norm(v3)
    here = np.column_stack((v2,v3,v1))
    there = np.eye(3)
    return dot(there, inv(here)) # R here = there

def terminates_scanner_right(slab, atoms, elements, d, round_n = 5):
    """
    find all different atomic planes within 1 lattice plane displacing (in the interface), for the right slab
    arguments:
    slab --- basic vectors of the slab
    atoms --- fractional coordinates of atoms
    elements --- list of name of the atoms
    d --- 1 lattice plane displacing
    round_n -- num of bits to round the fraction coordinates to judge identical planes
    return:
    plane_list --- list of planes of atom fraction coordinates
    element_list --- list of elements in each plane
    indices_list -- list of indices of atoms in each plane
    tol
    """
    plane_list = []
    element_list = []
    indices_list = []
    dp_list = []
    normal = cross(slab[:,1], slab[:,2])
    normal = normal / norm(normal)
    atoms_cart = dot(slab,atoms.T).T
    projections = abs(dot(atoms_cart, normal))
    atoms_round = floor(projections.copy() * 10000) / 10000
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
        except:
            position = np.inf
    return plane_list, element_list, indices_list, dp_list

"""
from here functions to draw atomic planes
"""

def draw_cell(subfig,xs,ys,alpha = 0.3, color = 'k', width = 0.5):
    for i in range(4):
        subfig.plot(xs[i],ys[i],c=color,linewidth=width,alpha = alpha)

def clean_fig(num1,num2,axes):
    for a in range(num1 * num2):
        for b in range(3):
            axes[a][b].spines['top'].set_visible(False)
            axes[a][b].spines['right'].set_visible(False)
            axes[a][b].spines['bottom'].set_visible(False)
            axes[a][b].spines['left'].set_visible(False)
            axes[a][b].get_yaxis().set_visible(False)
            axes[a][b].get_xaxis().set_visible(False)
            axes[a][b].set(facecolor = "w")

def Xs_Ys_cell(lattice):
    #four verticies
    P1 = np.array([0,0,0])
    P2 = lattice[:,1]
    P3 = lattice[:,1] + lattice[:,2]
    P4 = lattice[:,2]

    P1 = [0,0]
    P2 = [P2[0],P2[1]]
    P3 = [P3[0],P3[1]]
    P4 = [P4[0],P4[1]]

    x1 = [P1[0],P2[0]]
    y1 = [P1[1],P2[1]]

    x2 = [P2[0],P3[0]]
    y2 = [P2[1],P3[1]]

    x3 = [P3[0],P4[0]]
    y3 = [P3[1],P4[1]]

    x4 = [P4[0],P1[0]]
    y4 = [P4[1],P1[1]]
    xs = [x1, x2, x3, x4]
    ys = [y1, y2, y3, y4]
    return xs, ys

def draw_slab(xs, ys, axes, num, plane_list, lattice_to_screen, \
              elements_list, column, colors, all_elements, \
              elements_indices, l_r, titlesize, legendsize):
    for i in range(num):
        #get atoms in this plane
        plane_atoms = plane_list[i]
        plane_atoms = dot(lattice_to_screen, plane_atoms.T).T
        #how many different elements in this plane
        plane_elements = elements_list[i]
        element_names = np.unique(plane_elements)

        #looping for different elements
        for j in range(len(element_names)):
            #get the atoms of j elment
            single_element_atoms = plane_atoms[np.where(plane_elements \
                                                        == element_names[j])[0]]

            #draw atoms of this element
            Xs = single_element_atoms[:,0]
            Ys = single_element_atoms[:,1]
            element_index_here = np.where(all_elements == element_names[j])[0][0]
            draw_cell(axes[i][column],xs,ys)
            axes[i][column].scatter(Xs, Ys, c = colors[element_index_here],\
                               s = 100, label = element_names[j])
            axes[i][column].set_title('Plane {0} of {1} Cryst.'.\
                                      format(i + 1, l_r),fontsize = titlesize)
            axes[i][column].axis('scaled')
            axes[i][column].legend(fontsize = legendsize, borderpad=0.5, loc = 0)

def write_trans_file(v1, v2, n1, n2):
    """
    write a file including translation information for LAMMPS
    """
    with open('paras', 'w') as f:
        f.write('variable cnidv1x equal {} \n'.format(v1[0]/n1))
        f.write('variable cnidv2x equal {} \n'.format(v2[0]/n2))

        f.write('variable cnidv1y equal {} \n'.format(v1[1]/n1))
        f.write('variable cnidv2y equal {} \n'.format(v2[1]/n2))

        f.write('variable cnidv1z equal {} \n'.format(v1[2]/n1))
        f.write('variable cnidv2z equal {} \n'.format(v2[2]/n2))

        f.write('variable na equal {} \n'.format(n1))
        f.write('variable nb equal {} \n'.format(n2))

def draw_slab_dich(xs, ys, c_xs, c_ys,axes,num1, plane_list_1, lattice_to_screen_1, \
              elements_list_1, colors, all_elements, elements_indices,\
                  num2, plane_list_2, lattice_to_screen_2,\
                  elements_list_2, titlesize):
    for i in range(num1):
        for j in range(num2):
            #get atoms in this plane
            plane_atoms_1 = plane_list_1[i]
            plane_atoms_1 = dot(lattice_to_screen_1, plane_atoms_1.T).T
            plane_atoms_2 = plane_list_2[j]
            plane_atoms_2 = dot(lattice_to_screen_2, plane_atoms_2.T).T
            #how many different elements in this plane
            plane_elements_1 = elements_list_1[i]
            element_names_1 = np.unique(plane_elements_1)
            plane_elements_2 = elements_list_2[j]
            element_names_2 = np.unique(plane_elements_2)
            for l in range(len(element_names_1)):
                #get the atoms of j elment
                single_element_atoms_1 = plane_atoms_1[np.where(plane_elements_1 \
                                                            == element_names_1[l])[0]]

                #draw atoms of this element
                Xs = single_element_atoms_1[:,0]
                Ys = single_element_atoms_1[:,1]
                element_index_here_1 = np.where(all_elements == element_names_1[l])[0][0]
                axes[i*num2+j][2].scatter(Xs, Ys, c = colors[element_index_here_1],\
                                   s = 200, label = element_names_1[l] + " (L)", alpha = 0.5)
                axes[i*num2+j][2].set_title('{0} left {1} right.'.\
                                          format(i,j),fontsize = titlesize)
                axes[i*num2+j][2].axis('scaled')
                for n in range(len(element_names_2)):
                    #get the atoms of j elment
                    single_element_atoms_2 = plane_atoms_2[np.where(plane_elements_2 \
                                                                == element_names_2[n])[0]]

                    #draw atoms of this element
                    Xs = single_element_atoms_2[:,0]
                    Ys = single_element_atoms_2[:,1]
                    element_index_here_2 = np.where(all_elements == element_names_2[n])[0][0]
                    draw_cell(axes[i*num2+j][2],xs,ys)
                    draw_cell(axes[i*num2+j][2],c_xs,c_ys,color = 'b', alpha = 1, width = 2)
                    axes[i*num2+j][2].scatter(Xs, Ys, c = colors[-element_index_here_2-1],\
                                       s = 50, label = element_names_2[n] +  " (R)",alpha = 0.5)
                    axes[i*num2+j][2].axis('scaled')

"""
Below is some sampling functions
"""
 
def get_nearest_pair(lattice, atoms, indices):
    """
    a function return the indices of two nearest atoms in a periodic block
    from https://github.com/oekosheri/GB_code
    """

    #get Cartesian
    pos_1 = dot(lattice, atoms.copy().T).T
    pos_2 = pos_1
    #get 9 reps of atoms (PBC)
    reps = np.array([-1, 0, 1])
    x_shifts = [0]
    y_shifts = reps
    z_shifts = reps
    planar_shifts = np.array(np.meshgrid(x_shifts, y_shifts, \
                                         z_shifts),dtype = float).T.reshape(-1, 3)
    planar_shifts = dot(lattice, planar_shifts.T).T

    #make images for one set
    n_images = len(planar_shifts)
    n_1 = len(atoms)
    n_2 = len(atoms)
    n_1_images = n_images * n_1

    #repeate to the images
    pos_1_images = pos_1.repeat(n_images, axis=0) + np.tile(planar_shifts, (n_1, 1))
    pos_1_image_index_map = np.arange(n_1).repeat(n_images)

    #repeat to match num of set 2
    pos_1_rep = pos_1_images.repeat(n_2, axis=0)
    pos_1_index_map = pos_1_image_index_map.repeat(n_2)

    #repeat to match num of set 1
    pos_2_rep = np.tile(pos_2, (n_1_images, 1))
    pos_2_index_map = np.tile(np.arange(n_2), n_1_images)

    #all the distances
    distances = norm(pos_1_rep - pos_2_rep, axis=1)
    #none zero distances
    distances_none_zero = distances[distances > 0]
    distances_none_zero = np.unique(distances_none_zero)

    #not a distance to itself image
    for i in distances_none_zero:
        distances_id = np.where(distances == i)[0][0]
        if pos_1_index_map[distances_id] != pos_2_index_map[distances_id]:
            break

    return indices[pos_1_index_map[distances_id]], indices[pos_2_index_map[distances_id]], \
     pos_1_rep[distances_id], pos_2_rep[distances_id]
 
def searching_indices(atoms, coordinates):
    """
    get the indices of the coordinates in the atoms
    """
    return np.where( norm((atoms - coordinates), axis = 1) < 1e-8)[0]
 
def get_two_IDs_and_new_original_atoms(coordinates_1, coordinates_2, atoms, displacement):
    """
    get the two IDs of the two coordinates in atoms, and move
    the coordinates_1 by displacement
    """
    ID_1 = searching_indices(atoms, coordinates_1)
    ID_2 = searching_indices(atoms, coordinates_2)
    atoms[ID_1] = atoms[ID_1] + displacement
    return ID_1, ID_2, atoms
 
def delete_insert(lattice, atoms, elements, xlo, xhi, original_atoms):
    """
    a function delete two nearest atoms and insert one at the middle of them
    """
    xlo = xlo / norm(lattice[:,0])
    xhi = xhi / norm(lattice[:,0])

    #select atoms near the interface
    selected_indices = np.where((atoms[:,0] > xlo) & (atoms[:,0] < xhi))[0]
    selected_atoms = atoms[selected_indices]
    id1, id2, start, end = get_nearest_pair(lattice, selected_atoms, selected_indices)

    #middle & displacement
    middle_atom = (start + end) / 2
    middle_element = elements[id1]
    close_atom_1 = dot(lattice, atoms[id1])
    close_atom_2 = dot(lattice, atoms[id2])
    displace = middle_atom - close_atom_1
    displace_frac = dot(inv(lattice), displace)
    
    #get the original IDs of the pair of atoms and displace the first in the original atoms
    ogn_ID_1, ogn_ID_2, original_atoms = get_two_IDs_and_new_original_atoms(atoms[id1], atoms[id2], \
                                                                     original_atoms, displace_frac)
    
    #delete
    atoms = np.delete(atoms, [id1, id2], axis = 0)
    elements = np.delete(elements, [id1, id2])

    #insert
    middle_atom = dot(inv(lattice), middle_atom)
    atoms = np.vstack((atoms, middle_atom))
    elements = np.append(elements, middle_element)

    #check distance now
    #select atoms near the interface
    selected_indices = np.where((atoms[:,0] > xlo) & (atoms[:,0] < xhi))[0]
    selected_atoms = atoms[selected_indices]
    id1, id2, start, end = get_nearest_pair(lattice, selected_atoms, selected_indices)
    if id1 == id2:
        d_nearest_now = np.inf
    else:
        d_nearest_now = norm(start - end)

    return atoms, elements, d_nearest_now, len(atoms), ogn_ID_1, ogn_ID_2, displace, original_atoms
 
def sampling_deletion(lattice, atoms, elements, xlo, xhi, nearest_d, trans_name):
    """
    looping deletion of atoms until no atoms are nearer than
    one atom distance
    """
    data = np.array([-1, -1, -1, -1, -1],dtype = float)
    original_atoms = atoms.copy()
    c_atoms = atoms.copy()
    c_atoms_2 = atoms.copy()
    c_elements = elements.copy()
    nearest_now = delete_insert(lattice, c_atoms, c_elements, xlo, xhi, c_atoms_2)[2]
    del c_atoms
    del c_elements
    del c_atoms_2
    count = 1
    num = len(atoms)
    #write_LAMMPS(lattice, atoms, elements, filename = trans_name + '_0', orthogonal = True)
    while nearest_now < nearest_d and num > 1:
        atoms, elements, nearest_now, num, trans_ID, del_ID, displacement, original_atoms = delete_insert(lattice, atoms, elements, xlo, xhi, original_atoms)
        #print(nearest_now)
        #write_LAMMPS(lattice, atoms, elements, filename = trans_name + '_{}'.format(count), orthogonal = True)
        data = np.append(data, [trans_ID[0]+1, del_ID[0]+1])
        for i in range(3):
            data = np.append(data, displacement[i])
        count += 1
    with open(trans_name, 'w') as f:
        np.savetxt(f, data, fmt='%s')
    return count
    
 
def RBT_deletion_one_by_one(lattice, atoms, elements, CNID_frac, grid, bound, d_nearest, xlo, xhi):
    """
    a function generate atom files sampling RBT & deleting atoms
    """
    original_atoms = atoms.copy()
    original_elements = elements.copy()
    v1 = CNID_frac[:,0] / grid[0]
    v2 = CNID_frac[:,1] / grid[1]
    delete_nums_per_trans_file = open('del_nums', 'w')
    for i in range(grid[0]):
        for j in range(grid[1]):
            #copy atoms
            atoms_here = original_atoms.copy()
            #move right crystal
            right_indices = np.where(atoms_here[:,0] > bound / norm(lattice[:,0]))[0]
            atoms_here[right_indices] = atoms_here[right_indices] + v1 * i + v2 * j
            elements_here = original_elements.copy()
            #generate files
            dele_count = sampling_deletion(lattice, atoms_here, \
                  elements_here, xlo, xhi, d_nearest*0.99, '{0}_{1}'.format(i+1,j+1))
            delete_nums_per_trans_file.write('{}\n'.format(dele_count))
    delete_nums_per_trans_file.close()
            
class core:
    def __init__(self, file_1, file_2):
        self.afile_1 = file_1 # cif file name of lattice 1
        self.file_2 = file_2 # cif file name of lattice 2
        self.structure_1 = Structure.from_file(file_1, primitive=True, sort=False, merge_tol=0.0)
        self.structure_2 = Structure.from_file(file_2, primitive=True, sort=False, merge_tol=0.0)

        self.conv_lattice_1 = Structure.from_file(file_1, primitive=False, sort=False, merge_tol=0.0) \
                                .lattice.matrix.T

        self.conv_lattice_2 = Structure.from_file(file_2, primitive=False, sort=False, merge_tol=0.0) \
                                .lattice.matrix.T

        self.lattice_1 = self.structure_1.lattice.matrix.T
        self.lattice_2 = self.structure_2.lattice.matrix.T
        self.lattice_2_TD = self.structure_2.lattice.matrix.T.copy()
        self.CSL = np.eye(3) # CSL cell in cartesian
        self.du = 0.005
        self.S = 0.005
        self.D = np.eye(3)
        self.sgm1 = 100 # sigma 1
        self.sgm2 = 100 # sigma 2
        self.R = np.eye(3) # rotation matrix
        self.axis = np.array([0.0,0.0,0.0]) # rotation axis
        self.theta = 0.0 # rotation angle
        self.U1 = np.eye(3)
        self.U2 = np.eye(3)
        self.bicrystal_U1 = np.eye(3) # indices of the slab of lattice 1
        self.bicrystal_U2 = np.eye(3) # indices of the slab of lattice 2
        self.CNID = np.eye(3,2)
        self.cell_calc = DSCcalc()
        self.name = str
        self.dd = float
        self.orientation = np.eye(3) # initial disorientation
        self.a1 = np.eye(3)
        self.a2_0 = np.eye(3)
        self.orient = np.eye(3) #adjusted orientation for better visulaizing
        self.d1 = float #lattice plane distance
        self.d2 = float #lattice plane distance
        self.plane_list_1 = [] #list of terminating plane atoms
        self.plane_list_2 = []
        self.elements_1 = [] #list of terminating plane atom elements
        self.elements_2 = []
        self.indices_list_1 = [] #termination plane atoms's indices
        self.indices_list_2 = []
        self.dp_list_1 = [] #list of dp parameter to select termination
        self.dp_list_2 = []
        self.R_see_plane = np.eye #rotate the two slabs so that the screen is crossing the interface
        self.slab_lattice_1 = np.eye(3) #lattice matrix of the final slabs forming the interface
        self.slab_lattice_2 = np.eye(3)
        #get the atoms in the primitive cell
        self.atoms_1, self.elements_1 = get_sites_elements(self.structure_1)
        self.atoms_2, self.elements_2 = get_sites_elements(self.structure_2)

        #save the information of the bicrystal box
        self.lattice_bi = np.eye(3)
        self.atoms_bi = np.array([0.0,0.0,0.0])
        self.elements_bi = []
        print('Warning!, this programme will rewrite the POSCAR file in this dir!')

    def parse_limit(self, du, S, sgm1, sgm2, dd):
        """
        set the limitation to accept an appx CSL
        arguments -- see the paper
        """
        self.du = du
        self.S = S
        self.sgm1 = sgm1
        self.sgm2 = sgm2
        self.dd = dd

    def search_one_position(self, axis, theta, theta_range, dtheta, two_D = False):
        """
        main loop finding the appx CSL
        arguments:
        axis -- rotation axis
        theta -- initial rotation angle, in degree
        theta_range -- range varying theta, in degree
        dtheta -- step varying theta, in degree
        """
        axis = dot(self.lattice_1, axis)
        print(axis)
        file = open('log.one_position','w')
        theta = theta / 180 * np.pi
        n = ceil(theta_range/dtheta)
        dtheta = theta_range / n / 180 * np.pi
        x = np.arange(n)
        Ns = np.arange(1, self.sgm2 + 1)
        found = None

        if two_D == False:
            a1 = self.lattice_1.copy()
            a2_0 = self.lattice_2.copy()

        else:
            #find the two primitive plane bases
            #miller indices
            hkl_1 = MID(self.lattice_1, axis)
            hkl_2 = MID(dot(self.orientation, self.lattice_2), axis)
            #plane bases
            plane_B_1 = get_pri_vec_inplane(hkl_1, self.lattice_1)
            plane_B_2 = get_pri_vec_inplane(hkl_2, dot(self.orientation, self.lattice_2))
            v_3 = cross(plane_B_1[:,0], plane_B_1[:,1])
            print(100000)
            a1 = np.column_stack((plane_B_1, v_3))
            a2_0 = np.column_stack((plane_B_2, v_3))
            self.a1 = a1.copy()
            self.a2 = a2_0.copy()
            #a2_0 back to the initial orientation
            a2_0 = dot(inv(self.orientation), a2_0)
            self.a2_0 = a2_0.copy()
        # rotation loop
        file.write('---Searching starts---\n')
        file.write('axis theta dtheta n S du sigma1_max sigma2_max\n')
        file.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.\
              format(axis, theta, dtheta, n, self.S, self.du, self.sgm1, self.sgm2))
        file.write('-----------for theta-----------\n')
        for i in x:
            N = 1
            R = dot(self.orientation, rot(axis, theta))
            U = three_dot(inv(a1), R, a2_0)
            file.write('theta = ' + str(theta / np.pi * 180) + '\n')
            file.write('    -----for N-----\n')
            while N <= self.sgm2:
                tol = 1e-10
                Uij, N = rational_mtx(U,N)
                U_p = 1 / N * Uij
                if np.all((abs(U_p-U)) < self.du):
                    file.write('    N= ' + str(N) + " accepted" + '\n')
                    R_p = three_dot(a1, U_p, inv(a2_0))
                    D = dot(inv(R),R_p)
                    if (abs(det(D)-1) <= self.S) and \
                    np.all(abs(D-np.eye(3)) < self.dd):
                        here_found = True
                        file.write('    --D accepted--\n')
                        file.write("    D, det(D) = {0} \n".format(det(D)))
                        ax2 = three_dot(R,D,a2_0)
                        calc = DSCcalc()
                        try:
                            calc.parse_int_U(a1, ax2, self.sgm2)
                            calc.compute_CSL()
                        except:
                            file.write('    failed to find CSL here \n')
                            here_found = False
                        if here_found and abs(det(calc.U1)) <= self.sgm1:
                            found = True
                            file.write('--------------------------------\n')
                            file.write('Congrates, we found an appx CSL!\n')
                            sigma1 = int(abs(np.round(det(calc.U1))))
                            sigma2 = int(abs(np.round(det(calc.U2))))
                            self.D = D
                            self.U1 = np.array(np.round(calc.U1),dtype = int)
                            self.U2 = np.array(np.round(calc.U2),dtype = int)
                            self.lattice_2_TD = three_dot(R, D, a2_0)
                            self.CSL = dot(a1, self.U1)
                            self.R = R
                            self.theta = theta
                            self.axis = axis
                            self.cell_calc = calc
                            if two_D:
                                self.d1 = d_hkl(self.lattice_1, hkl_1)
                                self.d2 = d_hkl(three_dot(R, D, self.lattice_2), hkl_2)
                                calc.compute_CNID([0,0,1])
                                self.CNID = dot(a1, calc.CNID)
                            file.write('U1 = \n' + \
                                       str(self.U1) + '; sigma_1 = ' + \
                                       str(sigma1) + '\n')

                            file.write('U2 = \n' + str(self.U2) + '; sigma_2 = ' \
                                       + str(sigma1) + '\n')

                            file.write('D = \n' + str(np.round(D,8)) + '\n')

                            file.write('axis = ' + str(axis) + ' ; theta = ' \
                                       + str(np.round(theta / np.pi * 180,8)) \
                                       + '\n')

                            print('Congrates, we found an appx CSL!\n')
                            print('U1 = \n' + \
                                       str(self.U1) + '; sigma_1 = ' + \
                                       str(sigma1) + '\n')

                            print('U2 = \n' + str(self.U2) + '; sigma_2 = ' \
                                       + str(sigma1) + '\n')

                            print('D = \n' + str(np.round(D,8)) + '\n')

                            print('axis = ' + str(axis) + ' ; theta = ' \
                                       + str(np.round(theta / np.pi * 180,8)) \
                                       + '\n')

                            break
                        else:
                            file.write('    sigma too large \n')
                N += 1
            if found:
                break
            theta += dtheta
        if not found:
            print('failed to find a satisfying appx CSL. Try to adjust the limits according \
                  to the log file generated; or try another orientation.')


    def search_all_position(self, axis, theta, theta_range, dtheta, two_D = False):
        """
        main loop finding all the CSL lattices satisfying the limit
        arguments:
        axis -- rotation axis
        theta -- initial rotation angle, in degree
        theta_range -- range varying theta, in degree
        dtheta -- step varying theta, in degree
        output:
        log.all_position --- all the searching information
        results --- information of the found approximate CSL
        """
        axis = dot(self.lattice_1, axis)
        print(axis)
        file = open('log.all_position','w')
        file_r = open('results','w')
        theta = theta / 180 * np.pi
        n = ceil(theta_range/dtheta)
        dtheta = theta_range / n / 180 * np.pi
        x = np.arange(n)
        Ns = np.arange(1, self.sgm2 + 1)

        if two_D == False:
            a1 = self.lattice_1.copy()
            a2_0 = dot(self.orientation, self.lattice_2).copy()

        else:
            #find the two primitive plane bases
            #miller indices
            hkl_1 = MID(self.lattice_1, axis)
            print(axis)
            hkl_2 = MID(dot(self.orientation, self.lattice_2), axis)
            #plane bases
            plane_B_1 = get_pri_vec_inplane(hkl_1, self.lattice_1)
            plane_B_2 = get_pri_vec_inplane(hkl_2, dot(self.orientation, self.lattice_2))
            v_3 = cross(plane_B_1[:,0], plane_B_1[:,1])
            a1 = np.column_stack((plane_B_1, v_3))
            a2_0 = np.column_stack((plane_B_2, v_3))
            self.a1 = a1.copy()
            self.a2 = a2_0.copy()
            #a2_0 back to the initial orientation
            a2_0 = dot(inv(self.orientation), a2_0)
            self.a2_0 = a2_0.copy()
        # rotation loop
        file.write('---Searching starts---\n')
        file.write('axis theta dtheta n S du sigma1_max sigma2_max\n')
        file.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.\
              format(axis, theta, dtheta, n, self.S, self.du, self.sgm1, self.sgm2))
        file.write('-----------for theta-----------\n')
        for i in x:
            N = 1
            R = dot(self.orientation, rot(axis, theta))
            U = three_dot(inv(a1), R, a2_0)
            file.write('theta = ' + str(theta / np.pi * 180) + '\n')
            file.write('    -----for N-----\n')
            while N <= self.sgm2:
                tol = 1e-10
                Uij, N = rational_mtx(U,N)
                U_p = 1 / N * Uij
                if np.all((abs(U_p-U)) < self.du):
                    file.write('    N= ' + str(N) + " accepted" + '\n')
                    R_p = three_dot(a1, U_p, inv(a2_0))
                    D = dot(inv(R),R_p)
                    if (abs(det(D)-1) <= self.S) and \
                    np.all(abs(D-np.eye(3)) < self.dd):
                        here_found = True
                        file.write('    --D accepted--\n')
                        file.write("    D, det(D) = {0} \n".format(det(D)))
                        ax2 = three_dot(R,D,a2_0)
                        calc = DSCcalc()
                        try:
                            calc.parse_int_U(a1, ax2, self.sgm2)
                            calc.compute_CSL()
                        except:
                            file.write('    failed to find CSL here \n')
                            here_found = False
                        if here_found and abs(det(calc.U1)) <= self.sgm1:
                            file.write('--------------------------------\n')
                            file_r.write('--------------------------------\n')
                            file.write('Congrates, we found an appx CSL!\n')
                            sigma1 = int(abs(np.round(det(calc.U1))))
                            sigma2 = int(abs(np.round(det(calc.U2))))

                            if two_D:
                                calc.compute_CNID([0,0,1])
                                self.CNID = dot(a1, calc.CNID)
                            file.write('U1 = \n' + \
                                       str(self.U1) + '; sigma_1 = ' + \
                                       str(sigma1) + '\n')

                            file.write('U2 = \n' + str(self.U2) + '; sigma_2 = ' \
                                       + str(sigma1) + '\n')

                            file.write('D = \n' + str(np.round(D,8)) + '\n')

                            file.write('axis = ' + str(axis) + ' ; theta = ' \
                                       + str(np.round(theta / np.pi * 180,8)) \
                                       + '\n')

                            file_r.write('U1 = \n' + \
                                       str(self.U1) + '; sigma_1 = ' + \
                                       str(sigma1) + '\n')

                            file_r.write('U2 = \n' + str(self.U2) + '; sigma_2 = ' \
                                       + str(sigma1) + '\n')

                            file_r.write('D = \n' + str(np.round(D,8)) + '\n')

                            file_r.write('axis = ' + str(axis) + ' ; theta = ' \
                                       + str(np.round(theta / np.pi * 180,8)) \
                                       + '\n')
                            if two_D:
                                file_r.write('CNID = \n' + str(self.CNID) + '\n')

                            print('Congrates, we found an appx CSL!\n')
                            print('U1 = \n' + \
                                       str(self.U1) + '; sigma_1 = ' + \
                                       str(sigma1) + '\n')

                            print('U2 = \n' + str(self.U2) + '; sigma_2 = ' \
                                       + str(sigma1) + '\n')

                            print('D = \n' + str(np.round(D,8)) + '\n')

                            print('axis = ' + str(axis) + ' ; theta = ' \
                                       + str(np.round(theta / np.pi * 180,8)) \
                                       + '\n')
                        else:
                            file.write('    sigma too large \n')
                N += 1
            theta += dtheta

    def get_bicrystal(self, dydz = np.array([0.0,0.0,0.0]), dx = 0, dp1 = 0, dp2 = 0, \
                      xyz_1 = [1,1,1], xyz_2 = [1,1,1], vx = 0, filename = 'POSCAR', \
                      two_D = False, filetype = 'VASP', LAMMPS_file_ortho = False, mirror = False):
        """
        generate a cif file for the bicrystal structure
        argument:
        dydz --- translation vector in the interface
        dx --- translation normal to the interface
        dp1 --- termination of slab 1
        dp2 --- termination of slab 2
        xyz --- expansion
        two_D --- whether a two CSL
        LAMMPS_file_ortho --- whether the output LAMMPS has orthogonal cell
        """
        #get the atoms in the primitive cell
        lattice_1, atoms_1, elements_1 = self.lattice_1.copy(), self.atoms_1.copy(), self.elements_1.copy()
        lattice_2, atoms_2, elements_2 = self.lattice_2.copy(), self.atoms_2.copy(), self.elements_2.copy()

        # deform & rotate lattice_2
        lattice_2 = three_dot(self.R, self.D, lattice_2)
        #make supercells of the two slabs
        atoms_1, elements_1, lattice_1 = super_cell(self.bicrystal_U1, \
                                                    lattice_1, atoms_1, elements_1)
        if mirror == False:
            atoms_2, elements_2, lattice_2 = super_cell(self.bicrystal_U2, \
                                                lattice_2, atoms_2, elements_2)
        else:
            atoms_2, elements_2, lattice_2 = atoms_1.copy(), elements_1.copy(), lattice_1.copy()
            atoms_2[:,0] = - atoms_2[:,0] + 1
            atoms_c, elements_c = atoms_2.copy(), elements_2.copy()
            elements_c = elements_c[atoms_c[:,0] + 0.000001 > 1]
            atoms_c = atoms_c[atoms_c[:,0] + 0.000001 > 1]
            atoms_c[:,0] = atoms_c[:,0] - 1
            atoms_2 = np.vstack((atoms_2, atoms_c))
            elements_2 = np.append(elements_2, elements_c)
            
        
        #expansion
        if not (np.all(xyz_1 == 1) and np.all(xyz_2 == 1)):
            if not np.all([xyz_1[1], xyz_1[2]] == [xyz_2[1], xyz_2[2]]):
                raise RuntimeError('error: the two slabs must expand to the same dimension in the interface plane')
            lattice_1, atoms_1, elements_1 = cell_expands(lattice_1, atoms_1, \
                                                          elements_1, xyz_1)
            lattice_2, atoms_2, elements_2 = cell_expands(lattice_2, atoms_2, \
                                                          elements_2, xyz_2)

        #termination
        if dp1 != 0:
            atoms_1, elements_1 = shift_termi_left(lattice_1, dp1, atoms_1, elements_1)
        if dp2 != 0:
            atoms_2, elements_2 = shift_termi_left(lattice_2, dp2, atoms_2, elements_2)


        #adjust the orientation
        lattice_1, self.orient = adjust_orientation(lattice_1)
        lattice_2 = dot(self.orient, lattice_2)
        
        write_POSCAR(lattice_1, atoms_1, elements_1, 'POSCAR')
        POSCAR_to_cif('POSCAR','cell_1.cif')
        write_POSCAR(lattice_2, atoms_2, elements_2, 'POSCAR')
        POSCAR_to_cif('POSCAR','cell_2.cif')
        os.remove('POSCAR')

        self.R_see_plane = get_R_to_screen(lattice_1)

        self.plane_list_1, self.elements_list_1, self.indices_list_1, self.dp_list_1 \
         = terminates_scanner_left(lattice_1, atoms_1, elements_1, self.d1)

        self.plane_list_2, self.elements_list_2, self.indices_list_2, self.dp_list_2 \
         = terminates_scanner_right(lattice_2, atoms_2, elements_2, self.d2)

        self.slab_lattice_1 = lattice_1.copy()
        self.slab_lattice_2 = lattice_2.copy()
        """
        write_POSCAR(lattice_1, atoms_1, elements_1, 'POSCAR')
        POSCAR_to_cif('POSCAR','cell_1.cif')
        write_POSCAR(lattice_2, atoms_2, elements_2, 'POSCAR')
        POSCAR_to_cif('POSCAR','cell_2.cif')
        os.remove('POSCAR')
        """
        #combine the two lattices and translate atoms
        lattice_bi = lattice_1.copy()
        print(lattice_bi)
        if two_D:
            height_1 = get_height(lattice_1)
            height_2 = get_height(lattice_2)
            lattice_bi[:,0] = lattice_bi[:,0] * (1 + height_2 / height_1)
            #convert to cartesian
            atoms_1 = dot(lattice_1, atoms_1.T).T
            atoms_2 = dot(lattice_2, atoms_2.T).T
            #translate
            atoms_2 = atoms_2 + lattice_1[:,0]
            #back to the fractional coordinates
            atoms_1 = dot(inv(lattice_bi), atoms_1.T).T
            atoms_2 = dot(inv(lattice_bi), atoms_2.T).T
        else:
            lattice_bi[:,0] = 2 * lattice_bi[:,0]
            atoms_1[:,0] = atoms_1[:,0] / 2
            atoms_2[:,0] = (atoms_2[:,0] + 1) / 2

        #excess volume
        if dx != 0:
            excess_volume(lattice_1, lattice_bi, atoms_1, atoms_2, dx)

        #in-plane translation
        if norm(dydz) > 0:
            dydz = dot(self.orient, dydz)
            plane_shift = dot(inv(lattice_bi), dydz)
            atoms_2 = atoms_2 + plane_shift

        #combine the two slabs
        elements_bi = np.append(elements_1, elements_2)
        atoms_bi = np.vstack((atoms_1, atoms_2))

        #wrap the periodic boundary condition
        atoms_bi[:,1] = atoms_bi[:,1] - floor(atoms_bi[:,1])
        atoms_bi[:,2] = atoms_bi[:,2] - floor(atoms_bi[:,2])
        #vacummn
        if vx > 0:
            surface_vacuum(lattice_1, lattice_bi, atoms_bi, vx)

        #save
        self.lattice_bi = lattice_bi
        self.atoms_bi = atoms_bi
        self.elements_bi = elements_bi

        if filetype == 'VASP':
            write_POSCAR(lattice_bi, atoms_bi, elements_bi, filename)
        elif filetype == 'LAMMPS':
            write_LAMMPS(lattice_bi, atoms_bi, elements_bi, filename, LAMMPS_file_ortho)
        else:
            raise RuntimeError("Sorry, we only support for 'VASP' or 'LAMMPS' output")

    def sample_CNID(self, grid, dx = 0, dp1 = 0, dp2 = 0, \
                      xyz_1 = [1,1,1], xyz_2 = [1,1,1], vx = 0, two_D = False, filename = 'POSCAR', filetype = 'VASP'):
        """
        sampling the CNID and generate POSCARs
        argument:
        grid --- 2D grid of sampling
        """
        print('CNID')
        print(np.round(dot(inv(self.lattice_1),self.CNID),8))
        print('making {} files'.format(grid[0] * grid[1]) + '...')
        n1 = grid[0]
        n2 = grid[1]
        v1, v2 = self.CNID.T
        for i in range(n1):
            for j in range(n2):
                dydz = v1 / n1 * i + v2 / n2 * j
                self.get_bicrystal(dydz = dydz, dx = dx, dp1 = dp1, dp2 = dp2, \
                      xyz_1 = xyz_1, xyz_2 = xyz_2, vx = vx, two_D = two_D, filename = filename + '.' + str(i) + '.' + str(j), filetype = filetype)
        print('completed')

    def set_orientation_axis(self, axis_1, axis_2):
        """
        rotate lattice_2 so that its axis_2 coincident with the axis_1 of lattice_1
        """
        axis_1 = dot(self.lattice_1, axis_1)
        axis_1 = axis_1 / norm(axis_1)
        axis_2 = dot(self.lattice_2, axis_2)
        axis_2 = axis_2 / norm(axis_2)
        c = cross(axis_1, axis_2)
        c = c / norm(c)
        b_1 = cross(c, axis_1)
        b_1 = b_1 / norm(b_1)
        b_2 = cross(c, axis_2)
        b_2 = b_2 / norm(b_2)
        
        cell_1 = np.column_stack((axis_1, b_1, c))
        cell_2 = np.column_stack((axis_2, b_2, c))

        R = dot(cell_1, inv(cell_2))
        self.orientation = R

    def compute_bicrystal(self, hkl, lim = 20, normal_ortho = False, plane_ortho = False, tol = 1e-10):
        """
        compute the transformation to obtain the supercell of the two slabs forming a interface
        argument:
        hkl --- miller indices of the plane expressed in lattice_1
        lim --- the limit searching for a CSL vector cross the plane
        normal_ortho --- whether limit the vector crossing the GB to be normal to the GB
        plane_ortho --- whether limit the two vectors in the GB plane to be orthogonal
        tol --- tolerance judging whether orthogonal
        """
        self.d1 = d_hkl(self.lattice_1, hkl)
        lattice_2 = three_dot(self.R, self.D, self.lattice_2)
        hkl_2 = get_primitive_hkl(hkl, self.lattice_1, lattice_2)
        self.d2 = d_hkl(lattice_2, hkl_2)
        hkl_c = get_primitive_hkl(hkl, self.lattice_1, self.CSL) # miller indices of the plane in CSL
        hkl_c = np.array(hkl_c)
        plane_B = get_pri_vec_inplane(hkl_c, self.CSL) # plane bases of the CSL lattice plane
        if (plane_ortho == True) and (abs(dot(plane_B[:,0], plane_B[:,1])) > tol):
            plane_B = get_ortho_two_v(plane_B, lim, tol)
        plane_n = cross(plane_B[:,0], plane_B[:,1]) # plane normal
        v3 = cross_plane(self.CSL, plane_n, lim, normal_ortho, tol) # a CSL basic vector cross the plane
        supercell = np.column_stack((v3, plane_B)) # supercell of the bicrystal
        supercell = get_right_hand(supercell) # check right-handed
        self.bicrystal_U1 = np.array(np.round(dot(inv(self.lattice_1), supercell),8),dtype = int)
        self.bicrystal_U2 = np.array(np.round(dot(inv(self.lattice_2_TD), supercell),8),dtype = int)
        self.cell_calc.compute_CNID(hkl)
        CNID = self.cell_calc.CNID
        self.CNID = dot(self.lattice_1, CNID)
        print('cell 1:')
        print(self.bicrystal_U1)
        print('cell 2:')
        print(self.bicrystal_U1)

    def compute_bicrystal_two_D(self, lim = 20, normal_ortho = False, plane_ortho = False, tol = 1e-10):
        """
        compute the transformation to obtain the supercell of the two slabs forming a interface (only two_D periodicity)
        argument:
        lim --- the limit searching for a CSL vector cross the plane
        normal_ortho --- whether limit the vector crossing the GB to be normal to the GB
        plane_ortho --- whether limit the two vectors in the GB plane to be orthogonal
        tol --- tolerance judging whether orthogonal
        """
        #the two slabs with auxilary vector
        slab_1 = dot(self.a1, self.U1)
        slab_2 = dot(three_dot(self.R, self.D, self.a2_0), self.U2)
        #the transformed lattice_2
        a2 = three_dot(self.R, self.D, self.lattice_2)

        #two of the three vectors other than the auxilary vector
        B1 = get_plane_vectors(slab_1, self.axis)
        if (plane_ortho == True) and (abs(dot(B1[:,0], B1[:,1])) > tol):
            B1 = get_ortho_two_v(B1, lim, tol)
        #the third vector
        v3_1 = cross_plane(self.lattice_1, self.axis, lim, normal_ortho, tol)
        v3_2 = cross_plane(a2, self.axis, lim, normal_ortho, tol)
        if dot(v3_1, v3_2) < 0:
            v3_2 = - v3_2

        #unit slabs
        cell_1 = np.column_stack((v3_1, B1))
        cell_2 = np.column_stack((v3_2, B1))

        #right_handed
        cell_1 = get_right_hand(cell_1)
        cell_2 = get_right_hand(cell_2)

        #supercell index
        self.bicrystal_U1 = dot(inv(self.lattice_1), cell_1)
        self.bicrystal_U2 = dot(inv(a2), cell_2)

    def draw_terminations(self, titlesize = 50, legendsize = 50, single_element_size=100, \
                          left_element_size = 50, right_element_size = 100, figuresize = (30,30), figuredpi = 600):
        num1 = len(self.dp_list_1)
        num2 = len(self.dp_list_2)
        pat_num = num1 * num2

        #make figure
        fig, axes = plt.subplots(figsize = figuresize, nrows = pat_num, ncols = 3)

        #num of different elements
        colors = ['blue','orange','green','red','purple','brown',\
                  'pink','gray','olive','cyan']
        all_elements = np.unique(np.append(self.elements_1,\
                                           self.elements_2))
        elements_indices = np.arange(len(all_elements))

        #rotate the lattice to the screen
        lattice_rot_to_screen_1 = dot(self.R_see_plane, \
                                      self.slab_lattice_2)
        lattice_rot_to_screen_2 = dot(self.R_see_plane, \
                                      self.slab_lattice_2)
        CNID = dot(self.orient, self.CNID)
        CNID_rot_to_screen = dot(self.R_see_plane, \
                                      CNID)
        CNID_rot_to_screen = np.column_stack(([1,0,0],CNID_rot_to_screen))
        xs, ys = Xs_Ys_cell(lattice_rot_to_screen_1)
        c_xs, c_ys = Xs_Ys_cell(CNID_rot_to_screen)
        clean_fig(num1,num2,axes)

        draw_slab(xs, ys, axes,num1, self.plane_list_1, lattice_rot_to_screen_1, \
                  self.elements_list_1, 0, colors, all_elements,\
                  elements_indices, l_r = 'left',titlesize = titlesize, legendsize = legendsize)

        draw_slab(xs, ys, axes,num2, self.plane_list_2, lattice_rot_to_screen_2, \
                  self.elements_list_2, 1, colors, all_elements,\
                  elements_indices, l_r = 'right',titlesize = titlesize, legendsize = legendsize)

        draw_slab_dich(xs, ys, c_xs, c_ys, axes,num1, self.plane_list_1, lattice_rot_to_screen_1, \
                      self.elements_list_1, colors, all_elements, elements_indices,\
                          num2, self.plane_list_2, lattice_rot_to_screen_1,\
                          self.elements_list_2,titlesize)
        print('saving high resolution figure will take some time...please wait for a while :D')
        fig.savefig('atomic_planes', dpi = figuredpi)

    def define_lammps_regions(self, region_names, region_los, region_his, ortho = False):
        """
        generate a file defining some regions in the LAMMPS and define the atoms
        inside these regions into some groups.
        argument:
        region_names --- list of name of regions
        region_los --- list of the low bounds
        region_his --- list of the hi bounds
        ortho --- whether the cell is orthogonal
        """

        if (len(region_los) != len(region_names)) or (len(region_los) != len(region_his)):
            raise RuntimeError("the region_names, low & high bounds must have same num of elements.")

        xy = self.lattice_bi[:,1][0]
        xz = self.lattice_bi[:,2][0]
        yz = self.lattice_bi[:,2][1]

        with open('blockfile', 'w') as fb:
            for i in range(len(region_names)):
                if ortho == False:
                    fb.write('region {0} prism {1:.16f} {2:.16f} EDGE EDGE EDGE EDGE {3:.16f} {4:.16f} {5:.16f} units box \n'.\
                format(region_names[i], region_los[i], region_his[i], xy, xz, yz))
                else:
                    fb.write('region {0} block {1:.16f} {2:.16f} EDGE EDGE EDGE EDGE units box \n'.\
                format(region_names[i], region_los[i], region_his[i]))
                fb.write('group {0} region {1} \n'.format(region_names[i], region_names[i]))
