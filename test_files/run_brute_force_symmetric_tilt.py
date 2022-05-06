from interfacemaster.symmetric_tilt import *
from interfacemaster.cellcalc import get_primitive_hkl, rot
from interfacemaster.interface_generator import core, print_near_axis, convert_vector_index,\
                                                write_LAMMPS, write_trans_file
import argparse
import os
parser = argparse.ArgumentParser(description = 'manual to this script')
parser.add_argument("--axis", type = int, default = 0, 0, 1)
parser.add_argument("--lim", type = int, default = 3)
parser.add_argument("--max_sigma", type = int, default = 100)
parser.add_argument("--max_index", type = int, default = 10)
parser.add_argument("--core_num", type = int, default = 14)
parser.add_argument("--change_termi", type = str, default = 'n')
parser.add_argument("--rbt_grid", type = float, default = 0.5)
#mother path
mother_path = os.getcwd()
thetas, sigmas, hkls = sample_STGB([axis[0], axis[1], axis[2]], lim, max_sigma, max_index)

args = parser.parse_args()

copy_files = ['GB.in', 'Si.tersoff.modc']

print('-----detected GBs-----')
print('theta   sigma   hkl')
for i in range(len(hkls)):
    print(around(thetas[i]/pi*180,2), sigmas[i], hkls[i])

axis = [axis[0], axis[1], axis[2]]
for i in range(len(hkls)):
    os.mkdir(str(i+1))
    for j in copy_files:
        shutil.copy(j, os.path.join(str(i+1), j)
    R = rot(axis,thetas[i])
    my_interface = core('cif_files/Si_mp-149_conventional_standard.cif',\
                        'cif_files/Si_mp-149_conventional_standard.cif')
    my_interface.parse_limit(du = 1e-4, S  = 1e-4, sgm1=100000, sgm2=100000, dd = 1e-4)
    my_interface.search_fixed(R, exact=True, tol = 1e-4)
    hkl = get_primitive_hkl(hkls[i], my_interface.conv_lattice_1, my_interface.lattice_1, tol = 1e-3)
    my_interface.compute_bicrystal(hkl, normal_ortho =True, plane_ortho=True, tol_ortho = 1e-3, tol_integer = 1e-3, \
                                   align_rotation_axis = True, rotation_axis = axis)
    x_dimension = ceil(100/norm(dot(my_interface.lattice_1,my_interface.bicrystal_U1)[:,0]))
    y_dimension = ceil(40/norm(dot(my_interface.lattice_1,my_interface.bicrystal_U1)[:,1]))
    z_dimension = ceil(40/norm(dot(my_interface.lattice_1,my_interface.bicrystal_U1)[:,2]))
    my_interface.get_bicrystal(xyz_1 = [x_dimension,y_dimension,z_dimension], \
                               xyz_2 = [x_dimension,y_dimension,z_dimension], filetype='VASP',\
                               filename = 'POSCAR')

    my_run = GB_runner(my_interface)
    region_names = ['middle', 'low']
    region_los = [my_run.boundary - 40, 'EDGE']
    region_his = [my_run.boundary + 40, 40]
    my_run.interface.define_lammps_regions(region_names, region_los, region_his, True)
    if args.change_termi == 'y':
        my_run.get_terminations(True)
    else:
        my_run.get_terminations(False)
        
    my_run.get_RBT_list(args.rbt_grid)
    my_run.main_run()
    os.chdir(mother_path)