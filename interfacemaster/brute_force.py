from numpy import array, dot, round, var, average, pi, savetxt, repeat, tile, meshgrid, \
unique, sqrt, ceil, where, delete, vstack, loadtxt, arange, around
from numpy.linalg import inv, norm
from interfacemaster.interface_generator import write_LAMMPS
import os
import shutil
def find_pairs_with_closest_distances(atoms_here, bicrystal_lattice):
    """
    Find the pairs of atoms with closest distance
    arguments:
    atoms_here --- cartesian coordinates
    bicrystal_lattice --- lattice matrix of the bicrystal cell
    return:
    array_id_del --- ids of atoms to delete
    array_id_dsp --- ids of atoms to displace
    dsps --- displacements to be done
    distances_round[1] --- cutoff this turn
    distances_round[2] --- cloest distance after deletion
    """
    transL = bicrystal_lattice
    reps = array([-1, 0, 1])
    x_shifts = [0]
    y_shifts = reps
    z_shifts = reps
    planar_shifts = array(meshgrid(x_shifts, y_shifts, z_shifts),dtype = float).T.reshape(-1, 3)
    for i in range(len(planar_shifts)):
        planar_shifts[i,:] = dot(planar_shifts[i,:],transL.T)
    #expand by periodic boundary condition
    n_images = len(planar_shifts)
    n_1 = len(atoms_here)
    n_2 = len(atoms_here)
    n_1_images = n_images * n_1
    pos_1 = atoms_here
    pos_2 = atoms_here
    #building pairs
    #first pair expand for periodic boundary condition
    pos_1_images = pos_1.repeat(n_images, axis=0) + tile(planar_shifts, (n_1, 1))
    #ids of arrays
    pos_1_image_index_map = arange(n_1).repeat(n_images)
    
    #repeat first pair to build pairs with the second pair
    #atom coordinates
    pos_1_rep = pos_1_images.repeat(n_2, axis=0)
    #ids of arrays
    pos_1_index_map = pos_1_image_index_map.repeat(n_2)
    
    #repeat second pair to build pairs with the first pair
    pos_2_rep = tile(pos_2, (n_1_images, 1))
    #ids of arrays
    pos_2_index_map = tile(arange(n_2), n_1_images)
    
    #all the distances
    distances = norm(pos_1_rep - pos_2_rep, axis=1)
    #round, unique and sort
    distances_round = unique(around(distances, 5))
    
    #the distances shorter than cloest atomic distance in perfect crystal and larger than zero (not a self-pair)
    closest_pairs_id = where((distances<distances_round[1]+1e-3) & (distances>0))[0]
    
    #the array ids of del and dsp atoms
    #it is convenient to delete the PBC expanded atoms because in this case the displacement of the other pairs is to their center
    array_id_del = pos_1_index_map[closest_pairs_id]
    array_id_dsp = pos_2_index_map[closest_pairs_id]

    #screen out repeated pairs
    non_repeated_closest_pairs_id = screen_out_non_repeated_pairs(array_id_del, array_id_dsp)

    array_id_del = array_id_del[non_repeated_closest_pairs_id]
    array_id_dsp = array_id_dsp[non_repeated_closest_pairs_id]

    #the displacements of the dsp atoms
    dsps = 1/2 * (pos_1_rep[closest_pairs_id] + pos_2_rep[closest_pairs_id]) - pos_2_rep[closest_pairs_id]
    return array_id_del, array_id_dsp, dsps, distances_round[1], distances_round[2]

def screen_out_non_repeated_pairs(ids_1, ids_2):
    """
    input two pairs of ids with self-paring
    output the indices involving non-repeating pairs
    arguments:
    ids_1, ids_2 --- two pairs of ids
    return:
    screened_ids --- ids in ids_1 without repeating
    """
    arrays = arange(len(ids_1))
    screened_ids = []
    for i in range(len(arrays)):
        if i == 0:
                screened_ids.append(arrays[i])
        else:
            not_in = True
            for j in screened_ids:
                if ids_1[i] == ids_2[j]:
                    not_in = False
                    break
            if not_in:
                    screened_ids.append(arrays[i])
    return screened_ids

class GB_runner():
    """
    a class doing brute-foce searching for elemental GBs
    """
    def __init__(self, my_interface):
        """
        argument:
        my_interface --- interface core object
        """
        if len(unique(my_interface.elements_bi)) > 1:
            raise RuntimeError('error: only available for monoatomic systems')
        self.interface = my_interface
        self.boundary = 1/2 * norm(my_interface.lattice_bi[:,0]) - 0.00001
        self.clst_atmc_dstc = sqrt(3)/4 * norm(my_interface.conv_lattice_1[:,0])
        self.RBT_list = []
        self.terminations = []
        
    def get_terminations(self, changing_termination = False):
        """
        argument:
        changing_termination --- whether to sample different terminating planes
        """
        if changing_termination == False:
            self.terminations = [[0,0]]
        else:
            self.terminations = [[0, 0],\
                                 [0, - (self.interface.dp_list_1[0] - 0.0001)],\
                                 [self.interface.dp_list_1[0] + 0.0001, 0]]
        
    def get_RBT_list(self, grid_size):
        """
        argument:
        grid_size --- size of grid sampling RBT
        """
        CNID = dot(self.interface.orient,self.interface.CNID)
        v1 = CNID[:,0]
        v2 = CNID[:,1]
        n1 = int(ceil(norm(v1)/grid_size))
        n2 = int(ceil(norm(v2)/grid_size))
        print(n1,n2)
        for i in range(n1):
            for j in range(n2):
                self.RBT_list.append(v1*i/n1 + v2*j/n2)
        
    def divide_region(self):
        """
        divide simulation regions
        """
        all_atoms = dot(self.interface.lattice_bi.copy(), self.interface.atoms_bi.copy().T).T
        middle_ids = where((all_atoms[:,0] < self.boundary + self.clst_atmc_dstc)\
                           & (all_atoms[:,0] > self.boundary - self.clst_atmc_dstc))[0]
        self.middle_atoms = all_atoms[middle_ids]
        self.left_atoms = self.middle_atoms.copy()[where(self.middle_atoms[:,0] < self.boundary)[0]]
        self.right_atoms = self.middle_atoms.copy()[where(self.middle_atoms[:,0] >= self.boundary)[0]]
        self.bulk_atoms = all_atoms.copy()[where( (all_atoms[:,0] <= self.boundary - self.clst_atmc_dstc) \
                                     | (all_atoms[:,0] >= self.boundary + self.clst_atmc_dstc) )[0]]   
    def main_run(self, core_num):
        """
        main loop
        argument:
        core_num --- number of cores for simulation
        """
        count = 1
        os.mkdir('dump')
        for i in self.terminations:
            x_dimension = ceil(100/norm(dot(self.interface.lattice_1,self.interface.bicrystal_U1)[:,0]))
            y_dimension = ceil(40/norm(dot(self.interface.lattice_1,self.interface.bicrystal_U1)[:,1]))
            z_dimension = ceil(40/norm(dot(self.interface.lattice_1,self.interface.bicrystal_U1)[:,2]))
            self.interface.get_bicrystal(xyz_1 = [x_dimension,y_dimension,z_dimension], \
                                       xyz_2 = [x_dimension,y_dimension,z_dimension], filetype='LAMMPS',\
                                       filename = 'GB.dat', dp1 = i[0], dp2 = i[1])
            self.divide_region()
            for j in range(len(self.RBT_list)):
                bulk_atoms_here = self.bulk_atoms.copy()
                bulk_right_ids = where(bulk_atoms_here[:,0] > self.boundary)[0]
                bulk_atoms_here[bulk_right_ids] += self.RBT_list[j]
                displaced_atoms = self.right_atoms.copy() + self.RBT_list[j]
                GB_atoms = vstack((self.left_atoms, displaced_atoms))
                count = merge_operation(count, GB_atoms, self.interface.lattice_bi, \
                                        self.clst_atmc_dstc, self.RBT_list[j], bulk_atoms_here, \
                                        core_num, i[0], i[1])
        get_lowest()

def run_LAMMPS(core_num, count):
    """
    LAMMPS run command
    """
    os.system('mpirun -np {0} lmp_mpi -in GB.in -v count {1}'.format(core_num, count))

def read_energy():
    """
    LAMMPS run command
    """
    energy = loadtxt('energy_here.dat')
    return energy

def write_data_here(count, energy, dy, dz, dp1, dp2, delete_cutoff):
    """
    write data for each simulation
    argument:
    count --- index of simulation
    energy --- GB energy
    dy, dz --- RBT
    dp1, dp2 --- terminating shift
    delete_cutoff --- cutoff to merge atoms
    """
    with open('results.dat', 'a') as f:
        f.write('{0} {1} {2} {3} {4} {5} {6} \n'.format(count, energy, dy, dz, dp1, dp2, delete_cutoff))

def merge_operation(count_start, GB_atoms, bicrystal_lattice, clst_atmc_dstc, 
                    RBT, bulk_atoms, core_num, dp1, dp2):
    """
    merge loop
    """
    count = count_start
    #os.mkdir(foldername)
    cloest_distance_now = 0
    merge_operation_count = 0
    GB_atoms_here = GB_atoms.copy()
    while cloest_distance_now < 0.99 * clst_atmc_dstc:
        if merge_operation_count > 0:
            
            array_id_del, array_id_dsp, dsps, delete_cutoff, a = \
            find_pairs_with_closest_distances(GB_atoms_here, bicrystal_lattice)
            if len(array_id_del) > 0:
                GB_atoms_here[array_id_dsp] += dsps[0]
                GB_atoms_here = delete(GB_atoms_here, array_id_del, axis = 0)
                atoms = vstack((GB_atoms_here, bulk_atoms))
                array_id_del, array_id_dsp, dsps, cloest_distance_now, a = \
            find_pairs_with_closest_distances(GB_atoms_here, bicrystal_lattice)
                write_LAMMPS(bicrystal_lattice, dot(inv(bicrystal_lattice),atoms.T).T, repeat(['Si'],\
                         len(atoms)), filename = 'GB.dat', orthogonal = True)
                run_LAMMPS(core_num, count)
                energy_here = read_energy()
                write_data_here(count, energy_here, RBT[1], RBT[2], dp1, dp2, delete_cutoff)
                count += 1
        else:
            atoms = vstack((GB_atoms_here, bulk_atoms))
            write_LAMMPS(bicrystal_lattice, dot(inv(bicrystal_lattice),atoms.T).T, repeat(['Si'],\
                     len(atoms)), filename = 'GB.dat', orthogonal = True)
            run_LAMMPS(core_num, count)
            energy_here = read_energy()
            write_data_here(count, energy_here, RBT[1], RBT[2], dp1, dp2, 0)
            count += 1
        merge_operation_count += 1
    return count
    
def get_lowest():
    """
    get the lowest GB energy
    """
    count, energy, dy, dz, dp1, dp2, delete_cutoff = loadtxt('results.dat', unpack = True)
    lowest_count = count[where(energy == min(energy))[0][0]]
    lowest_energy = energy[where(energy == min(energy))[0][0]]
    with open('lowest_energy.dat', 'w') as f:
        f.write(str(lowest_energy))
    shutil.copy(os.path.join('dump',str(int(lowest_count))), 'global_min_structure.dat')
