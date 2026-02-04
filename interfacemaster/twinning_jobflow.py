import numpy as np
import os
from pymatgen.core.structure import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.core.interface import label_termination
from interfacemaster.interface_generator import core, get_height
from interfacemaster.cellcalc import MID, get_normal_from_MI
from pymatgen.io.vasp import Poscar
from jobflow import job, Maker
from dataclasses import dataclass
from typing import Dict, Any, List
from skopt.space import Real
from skopt import gp_minimize
 
# Try to import ASE and adapters
HAS_ASE = False
IMPORT_ERROR = None
try:
    from ase.optimize import LBFGS
    from ase.constraints import ExpCellFilter
    try:
        from pymatgen.io.ase import AseAtomsAdaptor as AseAdaptor
    except ImportError:
        from pymatgen.io.ase import AseAdaptor
    HAS_ASE = True
except Exception as e:
    IMPORT_ERROR = f"ASE import error: {e}"

# Optional calculators
try:
    from deepmd.calculator import DP as DeepmdDP
except Exception:
    DeepmdDP = None
try:
    from deepmd.calc import DeepMDPotential as DeepmdPotential
except Exception:
    DeepmdPotential = None
try:
    from sevenn.sevennet_calculator import SevenNetCalculator
except Exception:
    SevenNetCalculator = None
try:
    from orb_models.calculators import OrbCalculator
except Exception:
    OrbCalculator = None

def _termination_signature_from_slab(slab, ftol):
    """Return (top_label, bottom_label) using pymatgen label_termination."""
    top_label = label_termination(slab, ftol=ftol)
    frac = slab.frac_coords.copy()
    frac[:, 2] = (-frac[:, 2]) % 1.0
    bottom_struct = Structure(slab.lattice, slab.species, frac, coords_are_cartesian=False)
    bottom_label = label_termination(bottom_struct, ftol=ftol)
    return (top_label, bottom_label)

def _get_termination_pairs(
    structure,
    hkl,
    termination_ftol=0.25,
    termination_tol=1e-3,
    max_pairs=None,
):
    """Return list of (dp1, dp2, signature) for equivalent terminations."""
    slab_spacing = float(structure.lattice.d_hkl(hkl))
    sg = SlabGenerator(
        initial_structure=structure,
        miller_index=hkl,
        min_slab_size=slab_spacing,
        min_vacuum_size=0.0,
        center_slab=False,
        in_unit_planes=True,
    )
    slabs = sg.get_slabs(ftol=termination_ftol, filter_out_sym_slabs=False)
    if not slabs:
        return []

    groups = []
    for slab in slabs:
        shift = float(slab.shift) % 1.0
        sig = _termination_signature_from_slab(slab, termination_ftol)
        placed = False
        for grp in groups:
            if sig == grp[0]["signature"]:
                # 保留同一端面签名下的所有 shift（不在这里去重）
                grp.append({"shift": shift, "signature": sig})
                placed = True
                break
        if not placed:
            groups.append([{"shift": shift, "signature": sig}])

    term_pairs = []
    seen_pairs = set()
    for grp in groups:
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                s1, s2 = grp[i]["shift"], grp[j]["shift"]
                if abs(s1 - s2) > termination_tol:
                    a, b = (s1, s2) if s1 <= s2 else (s2, s1)
                    # 去重：同一端面签名下，(a,b) 近似相同的配对只保留一次
                    key = (grp[i]["signature"], round(a / termination_tol), round(b / termination_tol))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    term_pairs.append((a, b, grp[i]["signature"]))
                    if max_pairs is not None and len(term_pairs) >= max_pairs:
                        return term_pairs
    return term_pairs

def generate_equivalent_termination_structures(
    result,
    parent_stct,
    slab_length=10.0,
    termination_ftol=0.25,
    termination_tol=1e-3,
    tol_ortho=1e-2,
    return_slabs=False,
):
    """
    Given one search result, return bicrystal Structures for equivalent terminations.
    This only builds structures (no BO/relax).
    """
    hkl = result["hkl"]
    if isinstance(hkl, str) and "Cartesian" in hkl:
        L_p = parent_stct.lattice.matrix.T
        hkl = MID(lattice=L_p, n=result["axis_cart"], tol=1e-2)

    my_interface = core(parent_stct, parent_stct, verbose=False)
    my_interface.parse_limit(du=5e-2, S=5e-2, sgm1=200, sgm2=200, dd=5e-2)
    my_interface.search_fixed(result["rotation_matrix"], exact=False)

    try:
        my_interface.compute_bicrystal(hkl, lim=20, normal_ortho=True, tol_ortho=tol_ortho)
    except Exception:
        my_interface.compute_bicrystal(hkl, lim=20, normal_ortho=False)

    v_cross_cart = np.dot(my_interface.lattice_1, my_interface.bicrystal_U1)
    h_single = get_height(v_cross_cart)
    rep_k = int(np.ceil(slab_length / h_single))

    # Use the same supercell used for GB construction to judge terminations
    slab_structure = parent_stct.copy()
    slab_structure.make_supercell(np.array(my_interface.bicrystal_U1, dtype=int))
    # Convert hkl to the supercell basis for slab generation
    n_cart = get_normal_from_MI(parent_stct.lattice.matrix.T, hkl)
    hkl_sc = MID(lattice=slab_structure.lattice.matrix.T, n=n_cart, tol=1e-2)
    term_pairs = _get_termination_pairs(
        slab_structure, hkl_sc, termination_ftol=termination_ftol, termination_tol=termination_tol
    )
    structures = []
    for dp1, dp2, signature in term_pairs:
        slab_spacing = float(slab_structure.lattice.d_hkl(hkl_sc))
        sg = SlabGenerator(
            initial_structure=slab_structure,
            miller_index=hkl_sc,
            min_slab_size=slab_spacing,
            min_vacuum_size=0.0,
            center_slab=False,
            in_unit_planes=True,
        )
        slab1 = sg.get_slab(shift=dp1)
        slab2 = sg.get_slab(shift=dp2)

        stct = my_interface.get_bicrystal(
            xyz_1=[rep_k, 1, 1],
            xyz_2=[rep_k, 1, 1],
            dp1=dp1,
            dp2=dp2,
            dydz=np.zeros(3),
            dx=0.0,
            vx=0.0,
            output=False,
        )
        item = {"structure": stct, "dp1": dp1, "dp2": dp2, "signature": signature}
        if return_slabs:
            item["slab1"] = slab1
            item["slab2"] = slab2
        structures.append(item)
    return structures

def count_equivalent_termination_pairs(
    result,
    parent_stct,
    slab_length=10.0,
    termination_ftol=0.25,
    termination_tol=1e-3,
    tol_ortho=1e-2,
):
    """
    Return count of equivalent termination pairs for a given search result.
    This avoids building full GB structures.
    """
    hkl = result["hkl"]
    if isinstance(hkl, str) and "Cartesian" in hkl:
        L_p = parent_stct.lattice.matrix.T
        hkl = MID(lattice=L_p, n=result["axis_cart"], tol=1e-2)

    my_interface = core(parent_stct, parent_stct, verbose=False)
    my_interface.parse_limit(du=5e-2, S=5e-2, sgm1=200, sgm2=200, dd=5e-2)
    my_interface.search_fixed(result["rotation_matrix"], exact=False)

    try:
        my_interface.compute_bicrystal(hkl, lim=20, normal_ortho=True, tol_ortho=tol_ortho)
    except Exception:
        my_interface.compute_bicrystal(hkl, lim=20, normal_ortho=False)

    slab_structure = parent_stct.copy()
    slab_structure.make_supercell(np.array(my_interface.bicrystal_U1, dtype=int))
    n_cart = get_normal_from_MI(parent_stct.lattice.matrix.T, hkl)
    hkl_sc = MID(lattice=slab_structure.lattice.matrix.T, n=n_cart, tol=1e-2)

    term_pairs = _get_termination_pairs(
        slab_structure,
        hkl_sc,
        termination_ftol=termination_ftol,
        termination_tol=termination_tol,
        max_pairs=1,
    )
    return len(term_pairs)

@dataclass
class TwinningJobflowMaker(Maker):
    name: str = "Twinning GB BO-Relax (Symmetric Terminations)"
    # 基础参数
    crystal_structure: Structure = None
    search_result: Dict[str, Any] = None # 包含 rotation_matrix, sigma, hkl 等
    ml_model_path: str = None # 模型路径
    calc_type: str = "deepmd" # deepmd | sevennet | orb
    calc_kwargs: Dict[str, Any] = None
    bulk_energy_per_atom: float = None # eV/atom
    output_dir: str = "twinning_jobflow_results"
    
    # 优化参数
    trials: int = 40
    slab_length: float = 10.0
    z_range: List[float] = None # [min, max]
    random_state: int = 42
    termination_tol: float = 1e-3
    termination_ftol: float = 0.25
    interface_energy_window: float = 0.5 # J/m^2
    
    def __post_init__(self):
        if self.z_range is None:
            self.z_range = [-0.1, 0.1]
        if self.calc_kwargs is None:
            self.calc_kwargs = {}

    def _compute_interface_energy(self, total_energy, structure):
        """
        计算界面能 (J/m^2):
        gamma = (E_total - N * E_bulk) / (2 * A) * 16.0218
        其中 A 为单个界面面积 (Ang^2), E_total 单位 eV, E_bulk 为 eV/atom
        """
        if self.bulk_energy_per_atom is None:
            raise RuntimeError("必须提供 bulk_energy_per_atom (eV/atom) 才能计算界面能。")
        if self.bulk_energy_per_atom > 0:
            raise RuntimeError(
                f"bulk_energy_per_atom 为正值 ({self.bulk_energy_per_atom:.6f})，"
                "这通常是符号传反。请传入体相每原子的实际能量（通常为负）。"
            )
        n_atoms = len(structure)
        # 界面面积取结构的 b×c（假设 a 为法向）
        lat = structure.lattice.matrix
        area = np.linalg.norm(np.cross(lat[1], lat[2]))
        excess_e = total_energy - n_atoms * self.bulk_energy_per_atom
        gamma_eva2 = excess_e / (2.0 * area)
        gamma_jm2 = gamma_eva2 * 16.0217662
        return gamma_jm2, area

    def _build_calculator(self):
        if not HAS_ASE:
            raise RuntimeError(f"无法初始化 ASE: {IMPORT_ERROR}")
        if not self.ml_model_path and self.calc_type != "orb":
            raise RuntimeError("未提供 ml_model_path，无法初始化计算器。")

        calc_type = self.calc_type.lower()
        if calc_type == "deepmd":
            if DeepmdDP is not None:
                return DeepmdDP(model=self.ml_model_path, **self.calc_kwargs)
            if DeepmdPotential is not None:
                return DeepmdPotential(model=self.ml_model_path, **self.calc_kwargs)
            raise RuntimeError("DeepMD 计算器不可用。")
        if calc_type == "sevennet":
            if SevenNetCalculator is None:
                raise RuntimeError("SevenNetCalculator 不可用。")
            return SevenNetCalculator(model=self.ml_model_path, **self.calc_kwargs)
        if calc_type == "orb":
            if OrbCalculator is None:
                raise RuntimeError("OrbCalculator 不可用。")
            return OrbCalculator(model=self.ml_model_path, **self.calc_kwargs)
        raise RuntimeError(f"未知 calc_type: {self.calc_type}")

    def _hkl_spacing(self, hkl):
        """返回晶面间距 (Angstrom)。"""
        return float(self.crystal_structure.lattice.d_hkl(hkl))

    def _termination_signature(self, slab):
        """
        使用 pymatgen 的 label_termination 生成端面标签。
        返回 (top_label, bottom_label)，从而区分 A/B 与 B/A。
        """
        top_label = label_termination(slab, ftol=self.termination_ftol)
        # 通过翻转 z 分数坐标获得底面标签
        frac = slab.frac_coords.copy()
        frac[:, 2] = (-frac[:, 2]) % 1.0
        bottom_struct = Structure(slab.lattice, slab.species, frac, coords_are_cartesian=False)
        bottom_label = label_termination(bottom_struct, ftol=self.termination_ftol)
        return (top_label, bottom_label)

    def _get_termination_groups(self, hkl):
        """
        使用 SlabGenerator 生成不同 termination，并用 label_termination 分组。
        返回: List[List[dict]]，每个 dict 包含 shift、slab 和 signature。
        """
        try:
            slab_spacing = self._hkl_spacing(hkl)
            sg = SlabGenerator(
                initial_structure=self.crystal_structure,
                miller_index=hkl,
                min_slab_size=slab_spacing,
                min_vacuum_size=0.0,
                center_slab=False,
                in_unit_planes=True
            )
            slabs = sg.get_slabs(ftol=self.termination_ftol, filter_out_sym_slabs=False)
        except Exception:
            slabs = []

        if not slabs:
            return []

        groups = []
        for slab in slabs:
            shift = float(slab.shift) % 1.0
            sig = self._termination_signature(slab)
            placed = False
            for grp in groups:
                # 先要求端面标签一致（A/B 与 B/A 区分）
                if sig == grp[0]["signature"]:
                    grp.append({"shift": shift, "slab": slab, "signature": sig})
                    placed = True
                    break
            if not placed:
                groups.append([{"shift": shift, "slab": slab, "signature": sig}])
        return groups

    def sample_energy(self, params):
        """BO 采样函数（固定 dp1/dp2 来匹配等价端面）"""
        x, y, z, vx = params
        v1, v2 = self.my_interface.CNID.T
        dydz = x * v1 + y * v2
        
        gb = self.my_interface.get_bicrystal(
            xyz_1=[self.rep_k, 1, 1], xyz_2=[self.rep_k, 1, 1],
            dp1=self.current_dp1, dp2=self.current_dp2,
            dydz=dydz, dx=z, vx=vx, output=False
        )
        
        atoms = AseAdaptor.get_atoms(gb)
        atoms.calc = self.calc
        energy = atoms.get_potential_energy()
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

        self.calc = self._build_calculator()
        
        # 2. 分组等价端面（基于GB超胞），再对每对等价端面做 BO
        slab_structure = self.crystal_structure.copy()
        slab_structure.make_supercell(np.array(self.my_interface.bicrystal_U1, dtype=int))
        n_cart = get_normal_from_MI(self.crystal_structure.lattice.matrix.T, hkl)
        hkl_sc = MID(lattice=slab_structure.lattice.matrix.T, n=n_cart, tol=1e-2)
        term_pairs = _get_termination_pairs(
            slab_structure,
            hkl_sc,
            termination_ftol=self.termination_ftol,
            termination_tol=self.termination_tol,
        )
        print(f"共找到 {len(term_pairs)} 对等价端面，开始逐个优化...")
        if not term_pairs:
            raise RuntimeError("未找到等价端面组合（shift 不同）。请检查 termination_ftol/termination_tol。")

        term_results = []
        for dp1, dp2, signature in term_pairs:
            self.current_dp1 = dp1
            self.current_dp2 = dp2
            self.sampled_structures = []
            search_space = [
                Real(0, 1, name='x'), Real(0, 1, name='y'),
                Real(self.z_range[0], self.z_range[1], name='z'),
                Real(0, 2.0, name='vx')
            ]
            print(f"  - 优化 termination pair=({dp1:.4f}, {dp2:.4f}) (n_calls={self.trials})")
            res_bo = gp_minimize(self.sample_energy, search_space, n_calls=self.trials, random_state=self.random_state)
            local_best_idx = np.argmin(res_bo.func_vals)
            local_best_gb = self.sampled_structures[local_best_idx]
            local_best_energy = res_bo.fun
            gamma_jm2, area = self._compute_interface_energy(local_best_energy, local_best_gb)
            term_results.append({
                "dp1": dp1,
                "dp2": dp2,
                "signature": signature,
                "best_structure": local_best_gb,
                "best_energy": local_best_energy,
                "interface_energy": gamma_jm2,
                "area": area,
                "bo_result": res_bo,
            })

        # 3. 选择界面能接近最低的端面
        min_gamma = min(r["interface_energy"] for r in term_results)
        selected = [r for r in term_results if r["interface_energy"] <= min_gamma + self.interface_energy_window]
        print(f"最低界面能: {min_gamma:.4f} J/m^2，筛选 {len(selected)} 个端面进行结构优化...")

        # 4. 对筛选结果做结构优化并保存
        os.makedirs(self.output_dir, exist_ok=True)
        saved_files = []
        optimized_results = []
        for idx, r in enumerate(selected, start=1):
            print(f"  - 优化端面 {idx}/{len(selected)}: dp1={r['dp1']:.4f}, dp2={r['dp2']:.4f}")
            atoms = AseAdaptor.get_atoms(r["best_structure"])
            atoms.calc = self.calc
            ecf = ExpCellFilter(atoms, mask=[True, False, False, False, False, False])
            opt = LBFGS(ecf, logfile=None)
            opt.run(fmax=0.05)
            relaxed_stct = AseAdaptor.get_structure(atoms)
            filename = os.path.join(self.output_dir, f"GB_term_{idx}_dp1_{r['dp1']:.4f}_dp2_{r['dp2']:.4f}.vasp")
            Poscar(relaxed_stct).write_file(filename)
            saved_files.append(filename)
            optimized_results.append({
                "dp1": r["dp1"],
                "dp2": r["dp2"],
                "signature": r["signature"],
                "interface_energy": r["interface_energy"],
                "relaxed_structure": relaxed_stct,
                "file": filename,
            })

        print("所有流程已完成。")
        return {
            "min_interface_energy": min_gamma,
            "selected_count": len(selected),
            "saved_files": saved_files,
            "optimized_results": optimized_results,
        }
