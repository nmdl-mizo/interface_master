from interfacemaster.cellcalc import rot, MID, get_normal_from_MI
import numpy as np
from numpy.linalg import inv, eig, norm, det
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def get_low_index_planes(limit=3):
    """
    生成低指数 Miller 指数候选列表。
    """
    planes = []
    for h in range(-limit, limit + 1):
        for k in range(-limit, limit + 1):
            for l in range(-limit, limit + 1):
                if h == 0 and k == 0 and l == 0: continue
                # 保持互质
                if np.gcd.reduce([h, k, l]) == 1:
                    planes.append([h, k, l])
    return np.array(planes)

def calculate_compatibility_strain(Ui, Uj, hkl, L_p):
    """
    计算给定晶面 hkl 成为 Ui 和 Uj 之间完美孪晶面所需的最小应变。
    """
    # 获取该面的法向量
    n_hkl = get_normal_from_MI(L_p, hkl)
    n_hkl /= norm(n_hkl)
    
    # Ui^-1 * Uj^2 * Ui^-1 的特征值分析
    Ui_inv = inv(Ui)
    C = np.dot(Ui_inv, np.dot(np.dot(Uj, Uj), Ui_inv))
    
    # 理论上的理想法向量 n_ideal
    evals, evecs = eig(C)
    idx = evals.argsort()
    l1, l2, l3 = evals[idx]
    v1, v2, v3 = evecs[:, idx].T
    
    # 如果 lambda_2 偏离 1 太远，基础晶格就不匹配
    base_error = abs(l2 - 1.0)
    
    # 计算 hkl 面与理论解之间的角度偏差
    # 理论上存在两个解
    min_angle = 1e10
    for s1, s3 in [(1, 1), (1, -1)]:
        denom = np.sqrt(abs(l3 - l1))
        if denom < 1e-6: continue
        n_ideal = (s3 * np.sqrt(abs(l3 - l2)) * v3 + s1 * np.sqrt(abs(l2 - l1)) * v1) / denom
        # 转换回参考系
        n_ideal_ref = np.dot(Ui_inv, n_ideal)
        n_ideal_ref /= norm(n_ideal_ref)
        
        angle = np.arccos(np.clip(abs(np.dot(n_hkl, n_ideal_ref)), 0, 1))
        min_angle = min(min_angle, angle)
    
    # 将角度偏差转化为等效形变 (粗略估算)
    angular_strain = np.sin(min_angle)
    total_strain = base_error + angular_strain
    return total_strain

def search_low_index_twinning(parent_file, child_file, max_strain=0.05, hkl_limit=3):
    """
    搜索低指数的孪晶面。
    max_strain: 允许的最大形变量 (默认 5%)
    hkl_limit: 最大 Miller 指数 (默认 3，即搜索到 333)
    """
    parent = Structure.from_file(parent_file)
    child = Structure.from_file(child_file)
    variants, L_p = get_variants(parent, child)
    hkl_candidates = get_low_index_planes(hkl_limit)
    
    print(f"--- 低指数搜索开始 (允许形变: {max_strain*100:.1f}%) ---")
    
    results = []
    seen_hkls = set()

    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            for hkl in hkl_candidates:
                strain = calculate_compatibility_strain(variants[i], variants[j], hkl, L_p)
                
                if strain < max_strain:
                    hkl_tuple = tuple(np.sort(np.abs(hkl)))
                    if hkl_tuple not in seen_hkls:
                        seen_hkls.add(hkl_tuple)
                        results.append({
                            'hkl': hkl,
                            'strain': strain,
                            'variants': (i, j)
                        })

    # 按形变量从小到大排序
    results.sort(key=lambda x: x['strain'])
    
    print(f"找到 {len(results)} 个符合条件的低指数孪晶面:")
    for res in results:
        print(f"  (hkl): {res['hkl']} | 需满足的形变: {res['strain']*100:.2f}%")
        
    return results
