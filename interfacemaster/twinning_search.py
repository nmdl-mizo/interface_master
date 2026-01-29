from interfacemaster.cellcalc import rot, MID, get_normal_from_MI, DSCcalc
import numpy as np
from numpy.linalg import inv, eig, norm, det
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os

# Try to import ASE and DeepMD for optimization
HAS_ASE_DEEPMD = False
IMPORT_ERROR = None
try:
    from deepmd.calculator import DP
    from ase import Atoms
    from ase.optimize import LBFGS
    from ase.constraints import ExpCellFilter
    try:
        from pymatgen.io.ase import AseAtomsAdaptor as AseAdaptor
    except ImportError:
        from pymatgen.io.ase import AseAdaptor
    HAS_ASE_DEEPMD = True
except Exception as e:
    IMPORT_ERROR = f"Primary error: {e}"
    try:
        from deepmd.calc import DeepMDPotential as DP
        from ase import Atoms
        from ase.optimize import LBFGS
        from ase.constraints import ExpCellFilter
        try:
            from pymatgen.io.ase import AseAtomsAdaptor as AseAdaptor
        except ImportError:
            from pymatgen.io.ase import AseAdaptor
        HAS_ASE_DEEPMD = True
        IMPORT_ERROR = None
    except Exception as e2:
        IMPORT_ERROR += f" | Secondary error: {e2}"

def get_variants(parent_stct, child_stct):
    """
    Generate lattice variants from parent and child structures.
    """
    sga_parent = SpacegroupAnalyzer(parent_stct)
    ops_parent = sga_parent.get_symmetry_operations()
    
    L_p = parent_stct.lattice.matrix.T
    L_c = child_stct.lattice.matrix.T
    
    F = np.dot(L_c, inv(L_p))
    C = np.dot(F.T, F)
    evals, evecs = eig(C)
    evals = np.real(evals)
    evals[evals < 0] = 0
    U = np.dot(evecs, np.dot(np.diag(np.sqrt(evals)), evecs.T))
    U = np.real(U)
    
    variants = []
    seen_U = []
    for op in ops_parent:
        R = op.rotation_matrix
        U_i = np.dot(R.T, np.dot(U, R))
        if not any(np.allclose(U_i, existing_U, atol=1e-4) for existing_U in seen_U):
            seen_U.append(U_i)
            variants.append(U_i)
            
    return variants, L_p

def get_low_index_planes(limit=3):
    """
    生成低指数 Miller 指数候选列表。
    """
    planes = []
    for h in range(-limit, limit + 1):
        for k in range(-limit, limit + 1):
            for l in range(-limit, limit + 1):
                if h == 0 and k == 0 and l == 0: continue
                if np.gcd.reduce([h, k, l]) == 1:
                    planes.append([h, k, l])
    return np.array(planes)

def calculate_compatibility_strain(Ui, Uj, hkl, L_p):
    """
    计算给定晶面 hkl 成为 Ui 和 Uj 之间完美孪晶面所需的最小应变。
    """
    n_hkl = get_normal_from_MI(L_p, hkl)
    n_hkl /= norm(n_hkl)
    
    Ui_inv = inv(Ui)
    C = np.dot(Ui_inv, np.dot(np.dot(Uj, Uj), Ui_inv))
    
    evals, evecs = eig(C)
    evals = np.real(evals)
    evecs = np.real(evecs)
    idx = evals.argsort()
    l1, l2, l3 = evals[idx]
    v1, v2, v3 = evecs[:, idx].T
    
    base_error = abs(l2 - 1.0)
    
    min_angle = 1e10
    for s1, s3 in [(1, 1), (1, -1)]:
        denom = np.sqrt(abs(l3 - l1))
        if denom < 1e-6: continue
        n_ideal = (s3 * np.sqrt(abs(l3 - l2)) * v3 + s1 * np.sqrt(abs(l2 - l1)) * v1) / denom
        n_ideal_ref = np.dot(Ui_inv, n_ideal)
        n_ideal_ref /= norm(n_ideal_ref)
        
        angle = np.arccos(np.clip(abs(np.dot(n_hkl, n_ideal_ref)), 0, 1))
        min_angle = min(min_angle, angle)
    
    angular_strain = np.sin(min_angle)
    total_strain = base_error + angular_strain
    return total_strain

def get_symmetry_operations(structure):
    """
    获取结构的旋转对称操作矩阵。
    """
    sga = SpacegroupAnalyzer(structure)
    ops = sga.get_symmetry_operations()
    return [op.rotation_matrix for op in ops]

def is_symmetry_equivalent(R1, R2, sym_ops, atol=1e-3):
    """
    判断旋转矩阵 R1 和 R2 是否在对称性 sym_ops 下等效。
    """
    for Oi in sym_ops:
        for Oj in sym_ops:
            R_test = np.dot(Oi, np.dot(R1, Oj))
            if np.allclose(R_test, R2, atol=atol):
                return True
    return False

def calculate_approx_sigma(L1, R, max_sigma, du=0.08):
    """
    计算给定旋转下的最小近似 Sigma 值。
    这里 Sigma 定义为使得 N*U 为整数矩阵的最小整数 N。
    """
    U = np.dot(inv(L1), np.dot(R, L1))
    for N in range(1, max_sigma + 1):
        U_p = N * U
        if np.all(abs(U_p - np.round(U_p)) < du):
            return N
    return None

def search_low_index_twinning(parent_file, child_file, max_strain=0.1, hkl_limit=4, max_sigma=100, ortho_only=False, tol_ortho=1e-2):
    """
    搜索低指数孪晶面，包含晶向、面法向及笛卡尔坐标轴，并进行对称性去重和单晶过滤。
    """
    from interfacemaster.interface_generator import core
    parent = Structure.from_file(parent_file)
    child = Structure.from_file(child_file)
    variants, L_p = get_variants(parent, child)
    hkl_candidates = get_low_index_planes(hkl_limit)
    sga_parent = SpacegroupAnalyzer(parent)
    sym_ops = [op.rotation_matrix for op in sga_parent.get_symmetry_operations()]
    
    cartesian_axes = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    # 增加更多常见角度
    angles_to_test = [np.pi/3, np.pi/2, 2*np.pi/3, np.pi, 4*np.pi/3, 3*np.pi/2, 5*np.pi/3]
    
    print(f"--- 增强深度搜索开始 (Max Sigma: {max_sigma}, HKL Limit: {hkl_limit}) ---")
    
    raw_results = []

    # 1. 传统的低指数轴/面法向旋转 (包含变体自旋)
    for hkl in hkl_candidates:
        axis_a = np.dot(L_p, hkl); axis_a /= (norm(axis_a) + 1e-12)
        axis_b = get_normal_from_MI(L_p, hkl); axis_b /= (norm(axis_b) + 1e-12)

        for test_axis in [axis_a, axis_b]:
            for angle in angles_to_test:
                R = rot(test_axis, angle)
                
                # 检查是否为单晶对称操作
                is_bulk = False
                for op_mat in sym_ops:
                    if np.allclose(R, op_mat, atol=1e-3):
                        is_bulk = True
                        break
                if is_bulk: continue

                sigma = calculate_approx_sigma(L_p, R, max_sigma, du=0.08)
                if sigma and sigma > 1:
                    raw_results.append({
                        'hkl': hkl, 'sigma': sigma, 'strain': 0.0,
                        'rotation_matrix': R, 'axis_cart': test_axis, 'type': 'Crystal Axis/Normal'
                    })

    # 2. 基于变体间关系的旋转 (针对 LNO 等菱方体系)
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            Ui = variants[i]
            Uj = variants[j]
            for hkl in hkl_candidates:
                strain = calculate_compatibility_strain(Ui, Uj, hkl, L_p)
                if strain < max_strain:
                    n_hkl = get_normal_from_MI(L_p, hkl)
                    for angle in angles_to_test:
                        R = rot(n_hkl, angle)
                        sigma = calculate_approx_sigma(L_p, R, max_sigma, du=0.08)
                        if sigma and sigma > 1:
                            raw_results.append({
                                'hkl': hkl, 'sigma': sigma, 'strain': strain,
                                'rotation_matrix': R, 'axis_cart': n_hkl, 'type': 'Variant-Related'
                            })

    # 3. 笛卡尔坐标轴旋转
    for c_axis in cartesian_axes:
        for angle in angles_to_test:
            R = rot(c_axis, angle)
            
            is_bulk = False
            for op_mat in sym_ops:
                if np.allclose(R, op_mat, atol=1e-3):
                    is_bulk = True
                    break
            if is_bulk: continue

            sigma = calculate_approx_sigma(L_p, R, max_sigma, du=0.05)
            if sigma and sigma > 1 and sigma <= max_sigma:
                raw_results.append({
                    'hkl': "Cartesian " + str(c_axis), 'sigma': sigma, 'strain': 0.0,
                    'rotation_matrix': R, 'axis_cart': c_axis, 'type': 'Cartesian Axis'
                })

    # --- 对称性去重 ---
    unique_results = []
    seen_keys = set()
    for candidate in raw_results:
        R_cand = candidate['rotation_matrix']
        sigma_cand = candidate['sigma']
        
        is_new = True
        for unique in unique_results:
            if sigma_cand == unique['sigma'] and is_symmetry_equivalent(R_cand, unique['rotation_matrix'], sym_ops):
                is_new = False
                if candidate['strain'] < unique['strain']:
                    unique['strain'] = candidate['strain']
                    unique['hkl'] = candidate['hkl']
                break
        if is_new:
            unique_results.append(candidate)

    # --- 正交性过滤 (保持不变) ---
    if ortho_only:
        from interfacemaster.interface_generator import core
        print(f"正在进行正交性过滤 (tol_ortho={tol_ortho})...")
        ortho_results = []
        my_interface = core(parent, parent, verbose=False)
        for res in unique_results:
            hkl_to_test = res['hkl']
            if isinstance(hkl_to_test, str) and "Cartesian" in hkl_to_test:
                try:
                    hkl_to_test = MID(lattice=L_p, n=res['axis_cart'], tol=1e-2)
                except: continue
            
            my_interface.parse_limit(du=5e-2, S=5e-2, sgm1=max(100, res['sigma']), sgm2=max(100, res['sigma']), dd=5e-2)
            try:
                # 寻找近似 CSL
                my_interface.search_fixed(res['rotation_matrix'], exact=False)
                # 尝试构建正交超胞
                my_interface.compute_bicrystal(hkl_to_test, lim=20, normal_ortho=True, tol_ortho=tol_ortho)
                v = my_interface.bicrystal_U1
                lattice_bi = np.dot(my_interface.lattice_1, v)
                dot12 = abs(np.dot(lattice_bi[:,0], lattice_bi[:,1])) / (norm(lattice_bi[:,0]) * norm(lattice_bi[:,1]))
                dot13 = abs(np.dot(lattice_bi[:,0], lattice_bi[:,2])) / (norm(lattice_bi[:,0]) * norm(lattice_bi[:,2]))
                if dot12 < tol_ortho and dot13 < tol_ortho:
                    ortho_results.append(res)
            except: continue
        unique_results = ortho_results

    unique_results.sort(key=lambda x: (x['sigma'], x['strain']))
    print(f"最终找到 {len(unique_results)} 个独特界面候选。")
    for res in unique_results[:15]:
        print(f"  轴/面: {res['hkl']} | Sigma: {res['sigma']} | 角度: {np.degrees(np.arccos((np.trace(res['rotation_matrix'])-1)/2)):.1f}° | 类型: {res['type']}")
        
    return unique_results

def generate_twinning_configs(result, parent_stct, thickness_limit=10.0, ml_model=None, rbt_calls=40, z_range=[-0.1, 0.1]):
    """
    通过 6 维贝叶斯优化，直接寻找能量最低的孪晶界面构型。
    """
    from interfacemaster.interface_generator import core, get_height, registration_minimizer
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    
    hkl = result['hkl']
    if isinstance(hkl, str) and "Cartesian" in hkl:
        L_p = parent_stct.lattice.matrix.T
        axis_cart = result['axis_cart']
        try:
            hkl = MID(lattice=L_p, n=axis_cart, tol=1e-2)
        except:
            print(f"警告: 无法为轴 {axis_cart} 找到对应的低指数 Miller 面。")
            return []
        
    R_twin = result['rotation_matrix']
    sigma = result['sigma']
    
    print(f"\n--- 为 (hkl): {hkl} [Sigma {sigma}] 进行全局最稳态搜索 ---")
    
    my_interface = core(parent_stct, parent_stct, verbose=False)
    my_interface.parse_limit(du=5e-2, S=5e-2, sgm1=max(100, sigma), sgm2=max(100, sigma), dd=5e-2)
    my_interface.search_fixed(R_twin, exact=False)
    
    try:
        my_interface.compute_bicrystal(hkl, lim=20, normal_ortho=True, tol_ortho=1e-2)
        v = my_interface.bicrystal_U1
        lattice_bi = np.dot(my_interface.lattice_1, v)
        dot12 = abs(np.dot(lattice_bi[:,0], lattice_bi[:,1])) / (norm(lattice_bi[:,0]) * norm(lattice_bi[:,1]))
        dot13 = abs(np.dot(lattice_bi[:,0], lattice_bi[:,2])) / (norm(lattice_bi[:,0]) * norm(lattice_bi[:,2]))
        if dot12 > 1e-2 or dot13 > 1e-2:
            raise RuntimeError("生成的超胞正交性不足")
    except Exception as e:
        my_interface.compute_bicrystal(hkl, lim=20, normal_ortho=False)
    
    v_cross_cart = np.dot(my_interface.lattice_1, my_interface.bicrystal_U1)
    h_single = get_height(v_cross_cart)
    rep_k = int(np.ceil(thickness_limit / h_single))
    
    folder_name = f"Twin_hkl_{hkl[0]}_{hkl[1]}_{hkl[2]}_Sigma_{sigma}_Global"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    calc = None
    if ml_model:
        if HAS_ASE_DEEPMD:
            print(f"正在加载 DPA 模型: {ml_model} ...")
            try:
                calc = DP(model=ml_model)
            except Exception as e:
                print(f"错误: 无法初始化 DPA 计算器: {e}")
        else:
            print(f"警告: 无法运行 DPA 优化。环境检查失败: {IMPORT_ERROR}")
        
    if calc:
        print(f"正在进行 6 维贝叶斯全局优化 (calls={rbt_calls})...")
        my_interface.set_energy_calculator(calc)
        # 初始化采样器尺寸和默认参数
        my_interface.parse_it_size([rep_k, 1, 1], [rep_k, 1, 1], 0)
        
        # 启动全局优化：优化参数为 [x, y, z, dp1, dp2, vx]
        res_bo = registration_minimizer(my_interface, n_calls=rbt_calls, z_range=z_range, full_opt=True)
        best_x, best_y, best_z, best_dp1, best_dp2, best_vx = res_bo.x
        
        v1, v2 = my_interface.CNID.T
        best_dydz = best_x * v1 + best_y * v2
        
        print(f"全局优化完成。最优能量: {res_bo.fun:.4f} eV")
        print(f"正在生成最稳态结构并进行最终松弛...")
        
        stct = my_interface.get_bicrystal(xyz_1=[rep_k, 1, 1], xyz_2=[rep_k, 1, 1], 
                                          dp1=best_dp1, dp2=best_dp2, dydz=best_dydz, dx=best_z, vx=best_vx, output=False)
        atoms = AseAdaptor.get_atoms(stct)
        atoms.calc = calc
        ecf = ExpCellFilter(atoms, mask=[True, False, False, False, False, False])
        opt = LBFGS(ecf, logfile=None)
        opt.run(fmax=0.05)
        
        filename = os.path.join(folder_name, "POSCAR_stable")
        from pymatgen.io.vasp import Poscar
        final_stct = AseAdaptor.get_structure(atoms)
        Poscar(final_stct).write_file(filename)
        print(f"结构已保存 -> {filename}")
        return [filename]
    else:
        print("警告: 未提供 ml_model，无法进行全局优化搜索。")
        return []

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        search_low_index_twinning(sys.argv[1], sys.argv[2])
