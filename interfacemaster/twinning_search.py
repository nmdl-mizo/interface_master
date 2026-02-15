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

def is_relative_symmetry_equivalent(R1, R2, sym_ops, atol=1e-3):
    """
    判断两个旋转矩阵是否只相差一个晶体对称操作：
    若 R_rel = R2 * inv(R1) (或其逆) 与某个对称操作一致，则认为等价。
    """
    R_rel = np.dot(R2, inv(R1))
    R_rel_inv = inv(R_rel)
    for O in sym_ops:
        if np.allclose(R_rel, O, atol=atol) or np.allclose(R_rel_inv, O, atol=atol):
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

def search_low_index_twinning(
    parent,
    child,
    max_strain=0.1,
    hkl_limit=4,
    max_sigma=100,
    ortho_only=False,
    tol_ortho=1e-2,
    max_atoms=None,
    max_results=None,
    prefilter_limit=None,
    require_equivalent_terminations=False,
    termination_ftol=0.25,
    termination_tol=1e-3,
    slab_length=10.0,
    debug_filters=False,
    termination_check_timeout_s=30,
):
    """
    搜索低指数孪晶面，包含晶向、面法向及笛卡尔坐标轴，并进行对称性去重和单晶过滤。
    parent/child: 可以是 Structure 实例或 CIF 文件路径。
    max_atoms: 过滤掉生成的最小双晶胞原子数大于该阈值的候选。
    max_results: 最多返回多少个候选（按 sigma/strain 排序后截断）。
    prefilter_limit: 仅用于加速的预裁剪数量（在重过滤前截断）。
    require_equivalent_terminations: 仅保留能生成等价端面配对的候选。
    debug_filters: 打印过滤细节。
    termination_check_timeout_s: 单个候选做端面可生成性检查的超时秒数（超时则跳过该候选）。
    """
    from interfacemaster.interface_generator import core
    parent = parent if isinstance(parent, Structure) else Structure.from_file(parent)
    child = child if isinstance(child, Structure) else Structure.from_file(child)
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
    for candidate in raw_results:
        R_cand = candidate['rotation_matrix']
        sigma_cand = candidate['sigma']
        hkl_cand = candidate['hkl']
        
        is_new = True
        for unique in unique_results:
            # 1. 检查旋转矩阵是否一致（严格）
            same_sigma = sigma_cand == unique['sigma']
            rot_equiv = np.allclose(R_cand, unique['rotation_matrix'], atol=1e-3)
            if same_sigma and rot_equiv:
                # 2. 如果旋转等效，进一步检查 Miller 指数 (hkl) 是否也等效
                # 注意：Cartesian 形式会是字符串，需要先转为 hkl
                hkl_equivalent = False
                hkl_cand_numeric = None
                hkl_unique_numeric = None
                # 处理 candidate 的 hkl
                if isinstance(hkl_cand, str) and "Cartesian" in hkl_cand:
                    try:
                        hkl_cand_numeric = MID(lattice=L_p, n=candidate['axis_cart'], tol=1e-2)
                    except Exception:
                        hkl_cand_numeric = None
                else:
                    hkl_cand_numeric = np.array(hkl_cand, dtype=float)
                # 处理 unique 的 hkl
                hkl_unique = unique['hkl']
                if isinstance(hkl_unique, str) and "Cartesian" in hkl_unique:
                    try:
                        hkl_unique_numeric = MID(lattice=L_p, n=unique['axis_cart'], tol=1e-2)
                    except Exception:
                        hkl_unique_numeric = None
                else:
                    hkl_unique_numeric = np.array(hkl_unique, dtype=float)

                if hkl_cand_numeric is None or hkl_unique_numeric is None:
                    # 无法比较则认为不等价，保留
                    hkl_equivalent = False
                else:
                    for op in sym_ops:
                        # 对称操作作用于面法向
                        op_hkl = np.dot(op, hkl_cand_numeric)
                        if np.allclose(abs(op_hkl), abs(hkl_unique_numeric), atol=1e-2):
                            hkl_equivalent = True
                            break
                
                if hkl_equivalent:
                    is_new = False
                    # 如果旋转和指数都等效，保留应变更小的
                    if candidate['strain'] < unique['strain']:
                        unique['strain'] = candidate['strain']
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
            
            # IMPORTANT: use the same CSL search limits as jobflow stage (sgm1/sgm2=200)
            # to avoid inconsistent bicrystal_U1 between search and jobflow.
            my_interface.parse_limit(du=5e-2, S=5e-2, sgm1=200, sgm2=200, dd=5e-2)
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

    # 先排序（用于后续预裁剪）
    unique_results.sort(key=lambda x: (x['sigma'], x['strain']))

    # 预裁剪：在重过滤前先限制数量以加速（避免丢掉可生成端面的候选）
    if prefilter_limit is not None:
        unique_results = unique_results[:prefilter_limit]

    # --- 原子数过滤 ---
    if max_atoms is not None:
        from interfacemaster.interface_generator import core
        from interfacemaster.interface_generator import get_height
        print(f"正在进行原子数过滤 (max_atoms={max_atoms})...")
        filtered_results = []
        my_interface = core(parent, parent, verbose=False)
        for res in unique_results:
            hkl_to_test = res['hkl']
            if isinstance(hkl_to_test, str) and "Cartesian" in hkl_to_test:
                try:
                    hkl_to_test = MID(lattice=L_p, n=res['axis_cart'], tol=1e-2)
                except:
                    continue
            # IMPORTANT: must match jobflow stage to avoid U1 mismatch (and atom count mismatch)
            my_interface.parse_limit(du=5e-2, S=5e-2, sgm1=200, sgm2=200, dd=5e-2)
            try:
                my_interface.search_fixed(res['rotation_matrix'], exact=False)
                try:
                    my_interface.compute_bicrystal(hkl_to_test, lim=20, normal_ortho=True, tol_ortho=tol_ortho)
                except Exception:
                    my_interface.compute_bicrystal(hkl_to_test, lim=20, normal_ortho=False)
                # 原子数过滤必须与后续 jobflow 的 GB 构建口径一致：
                # jobflow 会根据 slab_length 计算 rep_k，并用 xyz_1/xyz_2=[rep_k,1,1] 加厚双晶。
                v_cross_cart = np.dot(my_interface.lattice_1, my_interface.bicrystal_U1)
                h_single = get_height(v_cross_cart)
                rep_k = int(np.ceil(slab_length / h_single)) if h_single > 1e-12 else 1
                rep_k = max(1, rep_k)

                # 用实际构建的 GB 结构判断（与后续 BO/Relax 相同）
                stct = my_interface.get_bicrystal(
                    xyz_1=[rep_k, 1, 1],
                    xyz_2=[rep_k, 1, 1],
                    output=False,
                )
                est_atoms = len(stct)
                if debug_filters:
                    print(
                        f"  - atoms_est hkl {res['hkl']} sigma {res['sigma']} -> {est_atoms} "
                        f"(rep_k={rep_k}, h_single={h_single:.3f}Å, slab_length={slab_length})"
                    )
                if est_atoms <= max_atoms:
                    filtered_results.append(res)
            except Exception:
                continue
        unique_results = filtered_results

    # --- 端面可生成性过滤 ---
    if require_equivalent_terminations:
        from interfacemaster.twinning_jobflow import count_equivalent_termination_pairs
        import signal

        class _TerminationCheckTimeout(Exception):
            pass

        def _alarm_handler(signum, frame):
            raise _TerminationCheckTimeout()

        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _alarm_handler)

        print("正在进行端面可生成性过滤 (require_equivalent_terminations=True)...")
        filtered_results = []
        for idx, res in enumerate(unique_results, start=1):
            if max_results is not None and len(filtered_results) >= max_results:
                break
            try:
                if debug_filters:
                    print(f"  [term-check start {idx}/{len(unique_results)}] hkl {res['hkl']} sigma {res['sigma']}")
                if termination_check_timeout_s is not None and termination_check_timeout_s > 0:
                    signal.alarm(int(termination_check_timeout_s))
                n_pairs = count_equivalent_termination_pairs(
                    res,
                    parent,
                    slab_length=slab_length,
                    termination_ftol=termination_ftol,
                    termination_tol=termination_tol,
                    debug=debug_filters,
                )
                signal.alarm(0)
                if debug_filters:
                    print(f"  - hkl {res['hkl']} sigma {res['sigma']} -> term_pairs {n_pairs}")
                if n_pairs > 0:
                    filtered_results.append(res)
            except _TerminationCheckTimeout:
                if debug_filters:
                    print(f"  [term-check timeout] hkl {res['hkl']} sigma {res['sigma']} > {termination_check_timeout_s}s, skip")
                continue
            except Exception:
                signal.alarm(0)
                continue
        unique_results = filtered_results
        signal.signal(signal.SIGALRM, old_handler)

    # 最终排序并截断
    unique_results.sort(key=lambda x: (x['sigma'], x['strain']))
    if max_results is not None:
        unique_results = unique_results[:max_results]
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

def generate_single_gb_structure(
    result,
    parent_stct,
    slab_length=10.0,
    dp1=0.0,
    dp2=0.0,
    dx=0.0,
    vx=0.0,
    dydz=None,
    tol_ortho=1e-2,
    output_path="POSCAR_single_gb",
):
    """
    直接生成一个孪晶界结构（不做优化）。
    """
    from interfacemaster.interface_generator import core, get_height

    hkl = result["hkl"]
    if isinstance(hkl, str) and "Cartesian" in hkl:
        L_p = parent_stct.lattice.matrix.T
        axis_cart = result["axis_cart"]
        hkl = MID(lattice=L_p, n=axis_cart, tol=1e-2)

    my_interface = core(parent_stct, parent_stct, verbose=False)
    my_interface.parse_limit(du=5e-2, S=5e-2, sgm1=200, sgm2=200, dd=5e-2)
    my_interface.search_fixed(result["rotation_matrix"], exact=False)

    try:
        my_interface.compute_bicrystal(hkl, lim=20, normal_ortho=True, tol_ortho=tol_ortho)
    except Exception:
        my_interface.compute_bicrystal(hkl, lim=20, normal_ortho=False)

    v_cross_cart = np.dot(my_interface.lattice_1, my_interface.bicrystal_U1)
    h_single = get_height(v_cross_cart)
    v3_len = np.linalg.norm(v_cross_cart[:, 0])
    print(f"[GB] hkl={hkl}, |v3|={v3_len:.3f} Å, h_single={h_single:.3f} Å")
    rep_k = int(np.ceil(slab_length / h_single))

    if dydz is None:
        dydz = np.zeros(3)

    stct = my_interface.get_bicrystal(
        xyz_1=[rep_k, 1, 1],
        xyz_2=[rep_k, 1, 1],
        dp1=dp1,
        dp2=dp2,
        dydz=dydz,
        dx=dx,
        vx=vx,
        output=False,
    )

    from pymatgen.io.vasp import Poscar
    Poscar(stct).write_file(output_path)
    return stct

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        search_low_index_twinning(sys.argv[1], sys.argv[2])
