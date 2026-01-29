import numpy as np
import os
import subprocess
from numpy.linalg import inv, norm, det
from pymatgen.core.structure import Structure
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.lammps.outputs import parse_lammps_dumps
from interfacemaster.interface_generator import core, registration_minimizer, get_height
from interfacemaster.cellcalc import rot, MID, get_normal_from_MI
from interfacemaster.twinning_search import calculate_approx_sigma
from jobflow import job, Maker
from dataclasses import dataclass
from typing import Dict, Any, List
from skopt.space import Real
from skopt import gp_minimize
from tqdm import tqdm

def get_structure_from_dump_file(dump_file_name):
    """从 LAMMPS dump 文件读取最后一帧结构"""
    dump = None
    for i in parse_lammps_dumps(dump_file_name):
        dump = i
    if dump is None:
        raise RuntimeError(f"无法从 {dump_file_name} 解析结构")
    lattice = dump.box.to_lattice()
    elements = dump.data['element'].to_numpy()
    x, y, z = dump.data['x'], dump.data['y'], dump.data['z']
    coords = np.column_stack((x, y, z))
    return Structure(lattice, elements, coords, coords_are_cartesian=True)

def extract_position(elements_all, xs, es, target_elements, bulk_energies, tol=0.1):
    """提取界面位置（基于能量异常）"""
    high_energy_xs = []
    for elem, bulk_e in zip(target_elements, bulk_energies):
        mask = (elements_all == elem)
        elem_xs = xs[mask]
        elem_es = es[mask]
        # 筛选能量显著高于 bulk 的原子
        high_energy_xs.extend(elem_xs[elem_es > bulk_e + tol])
    
    if not high_energy_xs:
        return np.average(xs) # 回退
    return np.average(high_energy_xs)

def get_double_gb_positions(dump_file_name, target_elements, bulk_energies, tol=0.1, shell=3.0):
    """获取两个等效界面的位置，用于定义退火区域"""
    dump = None
    for i in parse_lammps_dumps(dump_file_name):
        dump = i
    
    a = dump.box.to_lattice().a
    elements = dump.data['element'].to_numpy()
    x = dump.data['x']
    e = dump.data['c_energy']
    
    # 假设界面在大约 0 和 0.5 处（周期性边界）
    # 划分中间区域和侧边区域
    mid_mask = (x >= 0.25 * a) & (x < 0.75 * a)
    side_mask = ~mid_mask
    
    middle_elements = elements[mid_mask]
    middle_x = x[mid_mask]
    middle_e = e[mid_mask]

    side_elements = elements[side_mask]
    side_x = x[side_mask].copy()
    # 处理跨越边界的情况，将侧边原子平移到一起计算中心
    side_x[side_x >= 0.5 * a] -= a
    side_e = e[side_mask]
    
    middle_pos = extract_position(middle_elements, middle_x, middle_e, target_elements, bulk_energies, tol)
    side_pos = extract_position(side_elements, side_x, side_e, target_elements, bulk_energies, tol)
    
    # 转换回 [0, a] 范围
    if side_pos < 0: side_pos += a
    
    # 定义区域边界
    m_lo, m_hi = middle_pos - shell, middle_pos + shell
    
    if side_pos < 0.5 * a:
        s_lo = side_pos + shell
        s_hi = side_pos + a - shell
    else:
        s_lo = side_pos - a + shell
        s_hi = side_pos - shell
        
    return s_lo, s_hi, m_lo, m_hi

@dataclass
class TwinningJobflowMaker(Maker):
    name: str = "Twinning GB BO-Anneal-Relax"
    # 基础参数
    crystal_structure: Structure = None
    search_result: Dict[str, Any] = None # 包含 rotation_matrix, sigma, hkl 等
    ml_model_path: str = None # DPA 模型路径
    potential_type: str = "e3gnn" # 或 "deepmd" 等
    elements: List[str] = None
    bulk_energies: List[float] = None # 与 elements 对应
    
    # 优化参数
    trials: int = 40
    slab_length: float = 10.0
    z_range: List[float] = None # [min, max]
    random_state: int = 42
    
    # 退火参数
    temp: float = 2000.0
    anneal_steps: int = 2000
    cool_steps: int = 2000
    
    def __post_init__(self):
        if self.z_range is None:
            self.z_range = [-0.1, 0.1]
        if self.elements is None and self.crystal_structure:
            self.elements = [str(el) for el in self.crystal_structure.composition.elements]

    def get_lammps_static_input(self, model_path, elements):
        elem_str = " ".join(elements)
        return f"""
units           metal
atom_style      atomic
dimension       3
boundary        p p p
read_data       gb.data

pair_style      {self.potential_type}
pair_coeff      * * {model_path} {elem_str}

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

compute energy all pe/atom
compute total_energy all reduce sum c_energy
variable total_energy equal c_total_energy

thermo_style    custom step c_total_energy
run             0
print           ${{total_energy}} file sampled_energy.dat screen no
"""

    def get_lammps_relax_input(self, model_path, elements):
        elem_str = " ".join(elements)
        return f"""
units           metal
atom_style      atomic
dimension       3
boundary        p p p
read_data       gb.data

pair_style      {self.potential_type}
pair_coeff      * * {model_path} {elem_str}

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

compute energy all pe/atom
fix 1 all box/relax x 0.0

min_style       cg
minimize        0 1e-3 10000 10000

dump            1 all custom 1 relaxed_gb.dat id element type x y z c_energy
dump_modify     1 sort id element {elem_str}
run             0
"""

    def get_lammps_anneal_input(self, model_path, elements, s_lo, s_hi, m_lo, m_hi):
        elem_str = " ".join(elements)
        return f"""
units           metal
atom_style      atomic
dimension       3
boundary        p p p
read_data       gb.data

pair_style      {self.potential_type}
pair_coeff      * * {model_path} {elem_str}

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# 定义区域
region          side1 block EDGE {s_lo} EDGE EDGE EDGE EDGE
region          side2 block {s_hi} EDGE EDGE EDGE EDGE EDGE
region          side union 2 side1 side2
group           side region side

region          middle block {m_lo} {m_hi} EDGE EDGE EDGE EDGE
group           middle region middle

region          bulk1 block {s_lo} {m_lo} EDGE EDGE EDGE EDGE
group           bulk1 region bulk1
fix             bk1_rigid bulk1 rigid single

region          bulk2 block {m_hi} {s_hi} EDGE EDGE EDGE EDGE
group           bulk2 region bulk2
fix             bk2_rigid bulk2 rigid single

group           bulk union bulk1 bulk2
group           gb union side middle

timestep        0.001
compute         gb_temp gb temp/com
compute         energy all pe/atom

thermo          100
thermo_style    custom step c_gb_temp

fix             1 all box/relax x 0.0
fix             nve bulk nve

velocity        gb create {self.temp} {self.random_state} rot yes dist gaussian
fix             nvt_s side nvt temp {self.temp} {self.temp} 0.1
fix             nvt_m middle nvt temp {self.temp} {self.temp} 0.1

run             {self.anneal_steps}

unfix           nvt_s
unfix           nvt_m
fix             nvt_s side nvt temp {self.temp} 30.0 0.1
fix             nvt_m middle nvt temp {self.temp} 30.0 0.1

run             {self.cool_steps}

velocity        all zero linear
unfix           nvt_s
unfix           nvt_m
unfix           nve

min_style       cg
minimize        0 1e-3 10000 10000

dump            1 all custom 1 annealed_final.dat id element type x y z c_energy
dump_modify     1 sort id element {elem_str}
run             0
"""

    def run_lammps(self, input_str):
        with open('lammps.in', 'w') as f:
            f.write(input_str)
        cmd = "lmp -i lammps.in -log log.lammps"
        subprocess.run(cmd.split(), check=True, capture_output=True)

    def sample_energy(self, params):
        """BO 采样函数"""
        x, y, z, dp1, dp2, vx = params
        v1, v2 = self.my_interface.CNID.T
        dydz = x * v1 + y * v2
        
        gb = self.my_interface.get_bicrystal(
            xyz_1=[self.rep_k, 1, 1], xyz_2=[self.rep_k, 1, 1],
            dp1=dp1, dp2=dp2, dydz=dydz, dx=z, vx=vx, output=False
        )
        
        # 写入 LAMMPS data 并计算能量
        ld = LammpsData.from_structure(gb, atom_style='atomic')
        ld.write_file('gb.data')
        
        input_str = self.get_lammps_static_input(self.ml_model_path, self.elements)
        self.run_lammps(input_str)
        
        energy = float(np.loadtxt('sampled_energy.dat'))
        self.sampled_structures.append(gb)
        return energy

    @job
    def make(self):
        # 1. 初始化界面核心
        hkl = self.search_result['hkl']
        if isinstance(hkl, str) and "Cartesian" in hkl:
            L_p = self.crystal_structure.lattice.matrix.T
            hkl = MID(lattice=L_p, n=self.search_result['axis_cart'], tol=1e-2)
            
        self.my_interface = core(self.crystal_structure, self.crystal_structure, verbose=False)
        self.my_interface.parse_limit(du=5e-2, S=5e-2, sgm1=200, sgm2=200, dd=5e-2)
        self.my_interface.search_fixed(self.search_result['rotation_matrix'], exact=False)
        
        # 构建超胞
        try:
            self.my_interface.compute_bicrystal(hkl, lim=20, normal_ortho=True, tol_ortho=1e-2)
        except:
            self.my_interface.compute_bicrystal(hkl, lim=20, normal_ortho=False)
            
        v_cross_cart = np.dot(self.my_interface.lattice_1, self.my_interface.bicrystal_U1)
        h_single = get_height(v_cross_cart)
        self.rep_k = int(np.ceil(self.slab_length / h_single))
        
        # 2. 贝叶斯优化
        self.sampled_structures = []
        search_space = [
            Real(0, 1, name='x'), Real(0, 1, name='y'),
            Real(self.z_range[0], self.z_range[1], name='z'),
            Real(0, 1, name='dp1'), Real(0, 1, name='dp2'),
            Real(0, 2.0, name='vx')
        ]
        
        print(f"开始贝叶斯优化 (n_calls={self.trials})...")
        res_bo = gp_minimize(self.sample_energy, search_space, n_calls=self.trials, random_state=self.random_state)
        
        best_idx = np.argmin(res_bo.func_vals)
        best_gb = self.sampled_structures[best_idx]
        print(f"BO 完成。最优能量: {res_bo.fun:.4f} eV")
        
        # 3. 结构弛豫 (Static Relax)
        print("执行静态弛豫...")
        ld = LammpsData.from_structure(best_gb, atom_style='atomic')
        ld.write_file('gb.data')
        relax_input = self.get_lammps_relax_input(self.ml_model_path, self.elements)
        self.run_lammps(relax_input)
        relaxed_stct = get_structure_from_dump_file('relaxed_gb.dat')
        
        # 4. 退火 (Annealing)
        print("执行退火流程...")
        if self.bulk_energies:
            s_lo, s_hi, m_lo, m_hi = get_double_gb_positions('relaxed_gb.dat', self.elements, self.bulk_energies)
            ld = LammpsData.from_structure(relaxed_stct, atom_style='atomic')
            ld.write_file('gb.data')
            anneal_input = self.get_lammps_anneal_input(self.ml_model_path, self.elements, s_lo, s_hi, m_lo, m_hi)
            self.run_lammps(anneal_input)
            final_stct = get_structure_from_dump_file('annealed_final.dat')
        else:
            print("警告: 未提供 bulk_energies，跳过退火步骤。")
            final_stct = relaxed_stct
            
        print("所有流程已完成。")
        return {
            "final_structure": final_stct,
            "bo_results": {"x": res_bo.x_iters, "y": res_bo.func_vals.tolist()},
            "best_params": res_bo.x
        }
