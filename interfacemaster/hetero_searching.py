from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, ZSLMatch, reduce_vectors
from pymatgen.core.surface import SlabGenerator
from interfacemaster.cellcalc import get_primitive_hkl, get_pri_vec_inplane, get_normal_index, get_normal_from_MI, rot
from interfacemaster.interface_generator import core, convert_vector_index, get_disorientation
from interfacemaster.tool import get_indices_from_cart
from numpy import *
from numpy.linalg import *
import os
import shutil
import json
from pymatgen.io.cif import CifWriter

def rational_to_float(x):
    if len(x.split('/')) == 1:
        return float(x)
    else:
        return float(x.split('/')[0])/float(x.split('/')[1])

class plane_set:
    def __init__(self, lattice = eye(3), hkl = [1,0,0], v1 = [1,0,0], v2 = [1,0,0]):
        self.lattice = lattice
        self.hkl = hkl
        self.v1 = v1
        self.v2 = v2
        try:
            self.cart_v1 = dot(self.lattice, self.v1)
            self.cart_v2 = dot(self.lattice, self.v2)
        except:
            self.cart_v1 = dot(self.lattice, apply_function_to_array(self.v1, rational_to_float))
            self.cart_v2 = dot(self.lattice, apply_function_to_array(self.v2, rational_to_float))
    def output(self):
        return self.v1, self.v2, self.hkl

def round_int(M):
    return array(around(M, 8), dtype = int)

def apply_function_to_array(arr, func):
    if isinstance(arr, list):
        return [apply_function_to_array(x, func) for x in arr]
    elif isinstance(arr, ndarray):
        return array([apply_function_to_array(x, func) for x in arr])
    else:
        return func(arr)

def float_to_rational(x, lim =50):
    found = False
    for i in range(1, lim+1):
        if abs(x*i - round(x*i)) < 1e-5:
            found = True
            break
    if found:
        if i == 1:
            return f'{int(round(x*i))}'
        else:
            return f'{int(round(x*i))}/{i}'
    else:
        raise RuntimeError('failed to found rational matrix')

def get_rational_mtx(M):
    return apply_function_to_array(M, float_to_rational)

def plane_set_transform(old_set, new_l, format = 'int'):
    hkl = get_primitive_hkl(old_set.hkl, old_set.lattice, new_l)
    v1 = dot(inv(new_l), old_set.cart_v1)
    v2 = dot(inv(new_l), old_set.cart_v2)
    if format == 'int':
        v1, v2 = round_int(v1), round_int(v2)
    elif format == 'rational':
        v1, v2 = apply_function_to_array(v1, float_to_rational), apply_function_to_array(v2, float_to_rational)
    else:
        raise RuntimeError('must be int or rational')
    return plane_set(new_l, hkl, v1, v2)

class Match_data:
    def __init__(self, index, interface_core, data):
        i = index
        self.index = index
        self.plane_set_substrate = plane_set(interface_core.lattice_1, data[i][9:12], data[i][12:15], data[i][15:18])
        self.plane_set_film = plane_set(interface_core.lattice_2, data[i][0:3], data[i][3:6], data[i][6:9])
        self.plane_set_substrate_conv = plane_set_transform(self.plane_set_substrate, interface_core.conv_lattice_1, 'rational')
        self.plane_set_film_conv = plane_set_transform(self.plane_set_film, interface_core.conv_lattice_2, 'rational')

def colinear(v1,v2):
    if norm(cross(v1,v2))/norm(v1)/norm(v2) < 1e-6:
        return True
    else:
        return False

def same_mtch(sub_v1, sub_v2, film_v1, film_v2, sub_v1s, sub_v2s, film_v1s, film_v2s):
    same = False
    for i in range(len(sub_v1s)):
        #11 22 11 22
        con1 = colinear(sub_v1, sub_v1s[i]) and colinear(sub_v2, sub_v2s[i]) and (colinear(film_v1, film_v1s[i]) and colinear(film_v2, film_v2s[i]))
        #12 21 11 22
        con2 = colinear(sub_v1, sub_v2s[i]) and colinear(sub_v2, sub_v1s[i]) and (colinear(film_v1, film_v1s[i]) and colinear(film_v2, film_v2s[i]))
        #11 22 12 21
        con3 = colinear(sub_v1, sub_v1s[i]) and colinear(sub_v2, sub_v2s[i]) and (colinear(film_v1, film_v2s[i]) and colinear(film_v2, film_v1s[i]))
        #12 21 12 21
        con4 = colinear(sub_v2, sub_v1s[i]) and colinear(sub_v2, sub_v1s[i]) and (colinear(film_v1, film_v2s[i]) and colinear(film_v2, film_v1s[i]))
        
        if con1 or con2 or con3 or con4:
            same = True
            #print('same!')
            #print(sub_v1, sub_v2, film_v1, film_v2)
            #print(sub_v1s[i], sub_v2s[i], film_v1s[i], film_v2s[i])
            break

    return same
    
def data_from_matches(matches, film, substrate):
    #extract matching data
    film = film.get_primitive_structure()
    substrate = substrate.get_primitive_structure()
    film_vecs = []
    sub_vecs = []
    film_millers = []
    sub_millers = []
    strains = []
    areas = []
    for i in matches:
        film_vecs.append(around(dot(inv(film.lattice.matrix.T), i.film_sl_vectors.T),8))
        sub_vecs.append(around(dot(inv(substrate.lattice.matrix.T), i.substrate_sl_vectors.T),8))
        film_millers.append(i.film_miller)
        sub_millers.append(i.substrate_miller)
        strains.append(i.von_mises_strain)
        areas.append(norm(cross(i.substrate_sl_vectors[0], i.substrate_sl_vectors[1])))
    
    areas = around(array(areas), 5)
    sorted_areas = unique(areas)
    ids = []
    for i in sorted_areas:
        ids.append(where(areas == i)[0][0])
        
    film_vecs = array(film_vecs)[ids]
    sub_vecs = array(sub_vecs)[ids]
    film_millers = array(film_millers)[ids]
    sub_millers = array(sub_millers)[ids]
    strains = array(strains)[ids]
    areas = array(areas)[ids]
    
    for i in range(len(film_vecs)):
        if i == 0:
            film_vecs_1s = film_vecs[i][:,0]
            film_vecs_2s = film_vecs[i][:,1]
            sub_vecs_1s = sub_vecs[i][:,0]
            sub_vecs_2s = sub_vecs[i][:,1]
        else:
            film_vecs_1s = vstack((film_vecs_1s,film_vecs[i][:,0]))
            film_vecs_2s = vstack((film_vecs_2s,film_vecs[i][:,1]))
            sub_vecs_1s = vstack((sub_vecs_1s, sub_vecs[i][:,0]))
            sub_vecs_2s = vstack((sub_vecs_2s, sub_vecs[i][:,1]))
    return column_stack((film_millers, film_vecs_1s, film_vecs_2s, sub_millers, sub_vecs_1s, sub_vecs_2s, strains, areas))

def remove_before(dict):
    try:
        shutil.rmtree(dict)
    except:
        print('no folder')

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, integer):
            return int(obj)
        elif isinstance(obj, floating):
            return float(obj)
        elif isinstance(obj, ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        else:
            return super(NpEncoder, self).default(obj)

from copy import deepcopy

class hetero_searcher:
    def __init__(self, substct_stct, film_stct, substct_name, film_name):
        self.substct_stct = substct_stct #substrate structure
        self.film_stct = film_stct #film structure
        self.substct_name = substct_name
        self.film_name = film_name
        
    def matching(self, film_max_miller=2, substrate_max_miller=2):
        #define pymatgen substrate analyzer class
        sub_analyzer = SubstrateAnalyzer(film_max_miller = film_max_miller, substrate_max_miller = substrate_max_miller)
        
        sub_analyzer.calculate(film = self.film_stct.get_primitive_structure(), \
                               substrate = self.substct_stct.get_primitive_structure())
                               
        matches = list(sub_analyzer.calculate(film = self.film_stct.get_primitive_structure(), substrate = self.substct_stct.get_primitive_structure()))
        self.data = data_from_matches(matches, self.film_stct, self.substct_stct)
        print(f'{len(self.data)} non-identical matchings found')
        
    def generating(self, max_anum = 150, min_slab_length = 15):
        #generate interfaces
        my_interface = core(self.substct_stct,\
                            self.film_stct, verbose=False)

        it_folder = f'{self.substct_name}_{self.film_name}_interfaces'
        
        remove_before(it_folder)
        os.mkdir(it_folder)
        results = {}
        count = 1
        sub_v1s, sub_v2s = [], []
        film_v1s, film_v2s = [], []
        data = self.data
        self.interfaces = {}
        for i in range(len(data)):
        #for i in [0]:
            mtch_data = Match_data(i, my_interface, data)
            R = get_disorientation(L1 = my_interface.lattice_1, L2 = my_interface.lattice_2, \
                               v1 = mtch_data.plane_set_substrate.v1, hkl1 = mtch_data.plane_set_substrate.hkl, \
                               v2 = mtch_data.plane_set_film.v1, hkl2 = mtch_data.plane_set_film.hkl)

            my_interface.parse_limit(du = 0.5, S  = 0.5, sgm1=100, sgm2=100, dd = 0.05)

            #Do searching!
            my_interface.search_one_position_2D(hkl_1 = mtch_data.plane_set_substrate.hkl, hkl_2 = mtch_data.plane_set_film.hkl, theta_range = 3, \
                                                dtheta = 0.01, pre_R=R, pre_dt= True, integer_tol = 1e-6)
            my_interface.compute_bicrystal_two_D(hkl_1 = mtch_data.plane_set_substrate.hkl, hkl_2=mtch_data.plane_set_film.hkl, \
                                                 normal_ortho = False, lim = 50, tol_ortho = 1e-2, tol_integer=1e-3, inclination_tol= 0.1)
            x1 = int(ceil(min_slab_length/my_interface.height_1))
            x2 = int(ceil(min_slab_length/my_interface.height_2))
            my_interface.get_bicrystal(two_D = True, xyz_1 = [x1,1,1], xyz_2 = [x2,1,1], vx=5, filename= 'test_poscar')
            
            if len(my_interface.atoms_bi) < max_anum:
                sub_v1, sub_v2, hkl_sub = mtch_data.plane_set_substrate.output()
                film_v1, film_v2, hkl_film = mtch_data.plane_set_film.output()
                if count == 1:
                    same = False
                else:
                    #same = same_mtch(sub_v1, sub_v2, film_v1, film_v2, sub_v1s, sub_v2s, film_v1s, film_v2s)
                    same = False
                if not same:
                    os.mkdir(f'{it_folder}/{count}')
                    my_interface.get_bicrystal(two_D = True, xyz_1 = [x1,1,1], xyz_2 = [x2,1,1], vx=5, dx=0.5, filename=f'{it_folder}/{count}/POSCAR')
                    shutil.copy('super_sub.cif', f'{it_folder}/{count}/slab_substrate.cif')
                    shutil.copy('super_film.cif', f'{it_folder}/{count}/slab_film.cif')
                    CifWriter(my_interface.bicrystal_structure).write_file(f'{it_folder}/{count}/interface.cif')
                    #get supercell indices for each lattice
                    cart_id = dot(my_interface.lattice_1,my_interface.bicrystal_U1)
                    cstl_id_1 = array((around(dot(inv(my_interface.lattice_1), cart_id), 8)), dtype = int)
                    cstl_id_1_conv = apply_function_to_array(dot(inv(my_interface.conv_lattice_1), cart_id), float_to_rational)
                    
                    cart_id = dot(my_interface.lattice_2,my_interface.bicrystal_U2)
                    cstl_id_2 = array((around(dot(inv(my_interface.lattice_2), cart_id), 8)), dtype = int)
                    cstl_id_2_conv = apply_function_to_array(dot(inv(my_interface.conv_lattice_2), cart_id), float_to_rational)

                    results[count] = {}
                    results[count]['film_prim_hkl'] = array(mtch_data.plane_set_film.hkl, dtype = int)
                    results[count]['film_prim_v1'] = array(mtch_data.plane_set_film.v1, dtype = int)
                    results[count]['film_prim_v2'] = array(mtch_data.plane_set_film.v2, dtype = int)
                    results[count]['film_conv_hkl'] = array(mtch_data.plane_set_film_conv.hkl, dtype = int)
                    results[count]['film_conv_v1'] = mtch_data.plane_set_film_conv.v1
                    results[count]['film_conv_v2'] = mtch_data.plane_set_film_conv.v2
            
                    results[count]['substrate_prim_hkl'] = array(mtch_data.plane_set_substrate.hkl, dtype = int)
                    results[count]['substrate_prim_v1'] = array(mtch_data.plane_set_substrate.v1, dtype = int)
                    results[count]['substrate_prim_v2'] = array(mtch_data.plane_set_substrate.v2, dtype = int)
                    results[count]['substrate_conv_hkl'] = array(mtch_data.plane_set_substrate_conv.hkl, dtype = int)
                    results[count]['substrate_conv_v1'] = mtch_data.plane_set_substrate_conv.v1
                    results[count]['substrate_conv_v2'] = mtch_data.plane_set_substrate_conv.v2
                    results[count]['CSL area'] = data[i][-1]
                    results[count]['strain'] = my_interface.D
                    results[count]['von_mises_strain'] = data[i][-2]
                    results[count]['atom_num'] = len(my_interface.atoms_bi)
                    
                    cnid_indices_1_prim = get_rational_mtx(get_indices_from_cart(my_interface.lattice_1, my_interface.CNID))
                    transL2 = dot(my_interface.a2_transform, my_interface.lattice_2)
                    cnid_indices_2_prim = get_rational_mtx(get_indices_from_cart(transL2, my_interface.CNID))
                    results[count]['substrate_prim_CNID_express'] = cnid_indices_1_prim
                    results[count]['film_prim_CNID_express'] = cnid_indices_2_prim
                    
                    cnid_indices_1_conv = get_rational_mtx(get_indices_from_cart(my_interface.conv_lattice_1, my_interface.CNID))
                    transL2 = dot(my_interface.a2_transform, my_interface.conv_lattice_2)
                    cnid_indices_2_conv = get_rational_mtx(get_indices_from_cart(transL2, my_interface.CNID))
                    results[count]['substrate_conv_CNID_express'] = cnid_indices_1_conv
                    results[count]['film_conv_CNID_express'] = cnid_indices_2_conv
                    c1, c2 = my_interface.CNID.T
                    results[count]['CNID area'] = norm(cross(c1, c2))
                    
                    with open(f'{it_folder}/{count}/interface.info','w') as f:
                        f.write(f"""substrate primitive miller indices:{array(mtch_data.plane_set_substrate.hkl, dtype = int)}
film primitive miller indices: {array(mtch_data.plane_set_film.hkl, dtype = int)}

substrate conventional miller indices:{array(mtch_data.plane_set_substrate_conv.hkl, dtype = int)}
film conventional miller indices:{array(mtch_data.plane_set_film_conv.hkl, dtype = int)}

strain:
{my_interface.D}\n
atom number: {len(my_interface.atoms_bi)}\n
substrate primitive indices in supercell (a, b, c):
{cstl_id_1}

film primitive indices in supercell (a, b, c):
{cstl_id_2}

substrate conventional indices in supercell (a, b, c):
{cstl_id_1_conv}

film conventional indices in supercell (a, b, c):
{cstl_id_2_conv}
"""
                             )
                    self.interfaces[count] = deepcopy(my_interface)
                    count += 1
                    sub_v1s.append(sub_v1)
                    sub_v2s.append(sub_v2)
                    film_v1s.append(film_v1)
                    film_v2s.append(film_v2)
        print(f'{len(sub_v1s)} interfaces generated, files saved in the dict {it_folder}')
        self.res_dict = results
        for r_i in ['log.one_position', 'min_film.cif', 'min_sub.cif', 'super_film.cif', 'super_sub.cif', 'test_poscar']:
            os.remove(r_i)
    
    def write_results(self, filename):
        results = self.res_dict
        json_obj = json.dumps(self.res_dict,  cls=MyEncoder)
        with open(f'{filename}.json', 'w') as f:
            f.write(json_obj)
        with open(f'{filename}_list.dat', 'w') as f:
            f.write('sub_hkl sub_v1 sub_v2 film_hkl film_v1 film_v2 strain area\n')
            for i in results:
                f.write(f"""{tuple(results[i]["substrate_conv_hkl"])} {results[i]["substrate_conv_v1"]} {results[i]["substrate_conv_v2"]} {tuple(results[i]["film_conv_hkl"])} {results[i]["film_conv_v1"]} {results[i]["film_conv_v2"]} {results[i]["von_mises_strain"]} {results[i]["CSL area"]}\n""")
                
        
