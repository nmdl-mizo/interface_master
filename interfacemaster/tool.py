from numpy import *
from numpy.linalg import *
from pymatgen.core.structure import Structure

def get_indices_from_cart(lattice, cart):
    return dot(inv(lattice), cart)
    
def get_indices_from_cart(lattice, cart):
    return dot(inv(lattice), cart)

def get_rational_mtx(M, lim=50):
    found = False
    for i in range(1, lim+1):
        here = M * i
        if int_mtx(here):
            found = True
            break
    if found:
        return array(around(here), dtype=int), i
    else:
        print('failed to found rational matrix')

#generating KPOINTS
def generate_KPOINTS(poscar_file, dens):
    pos_struc = Structure.from_file(poscar_file)
    a, b, c = norm(pos_struc.lattice.matrix, axis = 1)
    ka, kb, kc = int(ceil(dens/a)), int(ceil(dens/b)), int(ceil(dens/c))
    return ka, kb, kc

#write KPOINTS
def write_KPOINTS_gama(poscar_file, dens, filename = 'KPOINTS'):
    ka, kb, kc = generate_KPOINTS(poscar_file, dens)
    with open(filename, 'w') as f:
        f.write('KPOINTS by YJ\n')
        f.write('0\n')
        f.write('Gamma\n')
        f.write(f'{ka} {kb} {kc}\n')
        f.write('0 0 0')

#get the coordinate of the x% atoms at the ends
def get_bounds(xs, TR, fracture, Tlength):
    num = len(xs)
    if TR == 'L':
        print(f'left bound: {xs[argsort(xs)][int(ceil(num*fracture))]}')
        return xs[argsort(xs)][int(ceil(num*fracture))] + 0.4/Tlength
    elif TR == 'R':
        print(f'right bound: {xs[argsort(xs)][int(ceil(-num*fracture))]}')
        return xs[argsort(xs)][-int(ceil(num*fracture))] - 0.4/Tlength
    else:
        raise CustomeError('must be L or R')

#fix atom
def get_fix_atom_TFarray(original_pos_file, Llength, fraction, both=True):
    #get middle bound
    stct = Structure.from_file(original_pos_file, sort = 'False')
    Tlength = norm(stct.lattice.matrix[0])
    middle_bound = (Llength - 0.5)/Tlength
    
    with open(original_pos_file, 'r') as f:
        lines = f.readlines()
    try:
        skiprows = where(array(lines) == 'Direct\n')[0][0] + 1
    except:
        try:
            skiprows = where(array(lines) == 'direct\n')[0][0] + 1
        except:
            try:
                skiprows = where(array(lines) == 'Cartesian\n')[0][0] + 1
            except:
                skiprows = where(array(lines) == 'cartesian\n')[0][0] + 1
    
    #read data
    coords = loadtxt(original_pos_file, skiprows= skiprows, usecols=(0,1,2))
    #left & right atoms
    left_coords = coords[coords[:,0] < middle_bound]
    right_coords = coords[coords[:,0] > middle_bound]

    left_bound = get_bounds(left_coords[:,0], 'L', fraction, Tlength)
    right_bound = get_bounds(right_coords[:,0], 'R', fraction, Tlength)

    TF_array = repeat(array([['T', 'T', 'T']]), len(coords), axis = 0)
    
    atom_number = len(coords)
    indices = arange(atom_number)

    con_left = coords[:,0] < left_bound
    con_right = coords[:,0] > right_bound

    left_bound_ids = indices[con_left]
    right_bound_ids = indices[con_right]
    
    if both:
        fix_ids = union1d(left_bound_ids, right_bound_ids)
    else:
        fix_ids = left_bound_ids
    
    TF_array[fix_ids] = ['F','F','F']

    return TF_array, fix_ids, coords[fix_ids]

def combine_poscar_TFarray(poscar_file, TFarray, filename):
    with open(poscar_file, 'r') as f:
        lines = f.readlines()
    try:
        skiprows = where(array(lines) == 'Direct\n')[0][0] + 1
    except:
        try:
            skiprows = where(array(lines) == 'direct\n')[0][0] + 1
        except:
            try:
                skiprows = where(array(lines) == 'Cartesian\n')[0][0] + 1
            except:
                skiprows = where(array(lines) == 'cartesian\n')[0][0] + 1
    atoms = loadtxt(poscar_file, skiprows = skiprows, usecols=(0,1,2))
    atoms = char.mod("%.16f", atoms)

    front_contents = array(lines)[arange(skiprows-1)]
    back_contents = column_stack((atoms, TFarray))
    with open(filename, 'w') as f:
        for i in front_contents:
            f.write(i)
        f.write('Selective dynamics\n')
        f.write('direct\n')
        savetxt(f, back_contents, fmt = '%s %s %s %s %s %s')
