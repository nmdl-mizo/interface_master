from symmetric_tilt import *
from brute_force import *
from numpy import arange
from interfacemaster.cellcalc import get_primitive_hkl, rot
from interfacemaster.interface_generator import core
import argparse
import os
import shutil
parser = argparse.ArgumentParser(description = 'manual to this script')
parser.add_argument("--axis", type = int, default = [0,0,1])
parser.add_argument("--lim", type = int, default = 10)
parser.add_argument("--max_sigma", type = int, default = 100)
parser.add_argument("--max_index", type = int, default = 10)
parser.add_argument("--core_num", type = int, default = 14)
parser.add_argument("--change_termi", type = str, default = 'y')
parser.add_argument("--rbt_grid", type = float, default = 0.2)
parser.add_argument("--distribute", type = str, default = 'n')
parser.add_argument("--initial", type = int, default = 0)
parser.add_argument("--final", type = int, default = 0)
args = parser.parse_args()

#mother path
mother_path = os.getcwd()

thetas, sigmas, hkls = sample_STGB([args.axis[0], args.axis[1], args.axis[2]], args.lim, args.max_sigma, args.max_index)

copy_files = ['GB.in', 'Si.tersoff.modc']

print('-----detected GBs-----')
print('theta     sigma    hkl')
print(len(hkls))
for i in range(len(hkls)):
    print(around(thetas[i]/pi*180,2), sigmas[i], hkls[i])

axis = [args.axis[0], args.axis[1], args.axis[2]]
if args.distribute == 'n':
    tasks = hkls
else:
    tasks = hkls[arange(initial-1, final)[0]]
for i in range(tasks):
    os.mkdir(str(i+1))
    for j in copy_files:
        shutil.copy(j, os.path.join(str(i+1), j))
    R = rot(axis,thetas[i])
    my_interface = core('Si_mp-149_conventional_standard.cif',\
                        'Si_mp-149_conventional_standard.cif')
    os.chdir(str(i+1))
    my_interface.scale(5.43356/5.468728, 5.43356/5.468728)
    my_interface.parse_limit(du = 1e-4, S  = 1e-4, sgm1=100000, sgm2=100000, dd = 1e-4)
    my_interface.search_fixed(R, exact=True, tol = 1e-4)
    hkl = get_primitive_hkl(tasks[i], my_interface.conv_lattice_1, my_interface.lattice_1, tol = 1e-3)
    my_interface.compute_bicrystal(hkl, normal_ortho =True, plane_ortho=True, tol_ortho = 1e-3, tol_integer = 1e-3, \
                                   align_rotation_axis = True, rotation_axis = axis)
    x_dimension = ceil(100/norm(dot(my_interface.lattice_1,my_interface.bicrystal_U1)[:,0]))
    y_dimension = ceil(40/norm(dot(my_interface.lattice_1,my_interface.bicrystal_U1)[:,1]))
    z_dimension = ceil(40/norm(dot(my_interface.lattice_1,my_interface.bicrystal_U1)[:,2]))
    my_interface.get_bicrystal(xyz_1 = [x_dimension,y_dimension,z_dimension], \
                               xyz_2 = [x_dimension,y_dimension,z_dimension])
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
    my_run.main_run_terminations(args.core_num)
    os.chdir(mother_path)
