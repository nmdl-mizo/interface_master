import numpy as np
import os
import time
from pymatgen.core.structure import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.operations import SymmOp
from interfacemaster.interface_generator import core, get_height
from interfacemaster.cellcalc import MID, get_normal_from_MI
from pymatgen.io.vasp import Poscar
from jobflow import job, Maker
from dataclasses import dataclass
from typing import Dict, Any, List
from skopt.space import Real
from skopt import gp_minimize
from itertools import combinations
 
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

# Optional fast symmetry equivalence check (InterOptimus-style)
try:
    from ase.utils.structure_comparator import SymmetryEquivalenceCheck
except Exception:
    SymmetryEquivalenceCheck = None

# Cache for expensive termination-pair checks during search stage
_TERM_PAIR_CACHE: dict[tuple, int] = {}

def _rot_cache_key(R: np.ndarray, ndigits: int = 6) -> tuple[int, ...]:
    """Quantize rotation matrix into a hashable key."""
    return tuple((np.round(np.asarray(R, dtype=float).reshape(-1), ndigits) * (10**ndigits)).astype(int).tolist())

def _reorient_slab_a_as_normal(slab):
    """
    Reorient slab so interface normal is along lattice a-axis.
    Original slab from SlabGenerator has normal along c-axis.
    """
    old_lat = slab.lattice.matrix
    # old: [a, b, c], new: [c, a, b]
    new_lat = np.array([old_lat[2], old_lat[0], old_lat[1]])
    return Structure(
        new_lat,
        slab.species,
        slab.cart_coords,
        coords_are_cartesian=True,
        site_properties=slab.site_properties,
    )

def _label_termination_fast(slab: Structure, ftol: float = 0.25, t_idx: int | None = None) -> str:
    """
    Fast, output-compatible equivalent of pymatgen.core.interface.label_termination.

    label_termination is often the dominant bottleneck because it builds an O(n^2) distance matrix
    and runs hierarchical clustering. For layering along c, this is equivalent to sorting z-heights
    and clustering by gaps > ftol (single-linkage cutoff).
    """
    frac_z = np.mod(slab.frac_coords[:, 2], 1.0)
    n = len(frac_z)

    if n == 1:
        form = slab.reduced_formula
        sp_symbol = SpacegroupAnalyzer(slab, symprec=0.1).get_space_group_symbol()
        out = f"{form}_{sp_symbol}_{len(slab)}"
        return out if t_idx is None else f"{t_idx}_{out}"

    h = float(slab.lattice.c)
    sort_idx = np.argsort(frac_z)
    z_sorted = frac_z[sort_idx] * h  # in [0, h)

    clusters: list[np.ndarray] = []
    start = 0
    gaps = np.diff(z_sorted)
    for k, gap in enumerate(gaps):
        if gap > ftol:
            clusters.append(sort_idx[start : k + 1])
            start = k + 1
    clusters.append(sort_idx[start:])

    # Periodic wrap-around merge if last and first are within ftol
    if len(clusters) > 1:
        wrap_gap = (z_sorted[0] + h) - z_sorted[-1]
        if wrap_gap <= ftol:
            clusters[0] = np.concatenate((clusters[-1], clusters[0]))
            clusters.pop()

    # Identify top plane by max mean frac_z (same criterion as pymatgen implementation)
    mean_fracs = [float(np.mean(frac_z[idxs])) for idxs in clusters]
    top_ids = clusters[int(np.argmax(mean_fracs))]

    top_plane_sites = [slab[int(i)] for i in top_ids]
    top_plane = Structure.from_sites(top_plane_sites)
    sp_symbol = SpacegroupAnalyzer(top_plane, symprec=0.1).get_space_group_symbol()
    form = top_plane.reduced_formula
    out = f"{form}_{sp_symbol}_{len(top_plane)}"
    return out if t_idx is None else f"{t_idx}_{out}"

def _is_colinear(a, b, directional: bool = False, tol: float = 1e-5) -> bool:
    """InterOptimus-style colinearity check for two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return False
    cos = float(np.dot(a, b) / (na * nb))
    if not directional:
        return (1 - abs(cos)) < tol
    return (1 - abs(cos)) < tol and cos > 0

def _get_rotation_from_match(lattice_by_row_vectors: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Convert StructureMatcher supercell matrix into a rotation matrix (InterOptimus)."""
    L = np.asarray(lattice_by_row_vectors, dtype=float)
    S = np.asarray(S, dtype=float)
    return np.dot((np.dot(S, L)).T, np.linalg.inv(L.T))

def _co_point_group_operations(sym_ops_s1, sym_ops_s2):
    """Union co-point group operations (InterOptimus)."""
    co_group = []
    for op1 in sym_ops_s1:
        for op2 in sym_ops_s2:
            for new_op in (op1 * op2, op2 * op1):
                if len(co_group) == 0:
                    co_group.append(new_op)
                    continue
                if not any(
                    np.allclose(new_op.rotation_matrix, existing_op.rotation_matrix)
                    and np.allclose(new_op.translation_vector, existing_op.translation_vector)
                    for existing_op in co_group
                ):
                    co_group.append(new_op)
    return co_group

def _pair_fit_interoptimus(
    film_slab_fit: Structure,
    sub_slab_fit: Structure,
    film_slab: Structure,
    sub_slab: Structure,
    matcher: StructureMatcher,
    c_periodic: bool,
) -> bool:
    """
    Determine whether two film/substrate slab pairs generate identical termination conditions.
    Directly adapted from InterOptimus/equi_term.py (with a minor bugfix for sub_map rotation).
    """
    # relative disorientation of slab and film compared with the existing one
    if film_slab_fit == film_slab:
        film_map = SymmOp.from_rotation_and_translation(np.eye(3), [0, 0, 0])
    else:
        film_transformation = matcher.get_transformation(film_slab, film_slab_fit)[0]
        film_rotation = _get_rotation_from_match(film_slab_fit.lattice.matrix, film_transformation)
        film_map = SymmOp.from_rotation_and_translation(film_rotation, [0, 0, 0])

    if sub_slab_fit == sub_slab:
        sub_map = SymmOp.from_rotation_and_translation(np.eye(3), [0, 0, 0])
    else:
        sub_transformation = matcher.get_transformation(sub_slab, sub_slab_fit)[0]
        sub_rotation = _get_rotation_from_match(sub_slab_fit.lattice.matrix, sub_transformation)
        sub_map = SymmOp.from_rotation_and_translation(sub_rotation, [0, 0, 0])

    film_over_sub = film_map * sub_map.inverse

    sym_ops_film = SpacegroupAnalyzer(film_slab_fit).get_point_group_operations(cartesian=True)
    sym_ops_sub = SpacegroupAnalyzer(sub_slab_fit).get_point_group_operations(cartesian=True)
    # co-group of film and slab to compare with
    co_group = _co_point_group_operations(sym_ops_film, sym_ops_sub)

    # condition 1: disorientation is a symmetry operation in co-group
    con1 = any(np.allclose(film_over_sub.rotation_matrix, symm_op.rotation_matrix, atol=1e-5) for symm_op in co_group)

    # condition 2: transformed plane normals are related back to represent the same plane
    rotated_film_normal = film_map.apply_rotation_only([0, 0, 1])
    rotated_sub_normal = sub_map.apply_rotation_only([0, 0, 1])

    if c_periodic:
        con2 = any(_is_colinear(sym_op.apply_rotation_only(rotated_film_normal), [0, 0, 1]) for sym_op in sym_ops_film) and any(
            _is_colinear(sym_op.apply_rotation_only(rotated_sub_normal), [0, 0, 1]) for sym_op in sym_ops_sub
        )
    else:
        con2 = False
        for op_film in sym_ops_film:
            for op_sub in sym_ops_sub:
                if _is_colinear(op_film.apply_rotation_only(rotated_film_normal), [0, 0, 1], directional=True) and _is_colinear(
                    op_sub.apply_rotation_only(rotated_sub_normal), [0, 0, 1], directional=True
                ):
                    if np.allclose(
                        op_film.apply_rotation_only(rotated_film_normal),
                        op_sub.apply_rotation_only(rotated_sub_normal),
                        rtol=1e-3,
                        atol=1e-5,
                    ):
                        con2 = True
                        break
            if con2:
                break

    return con1 and con2

def _dedupe_slab_pairs_interoptimus(
    film_slabs: list[Structure],
    sub_slabs: list[Structure],
    c_periodic: bool = True,
    max_unique: int | None = None,
    progress: bool = False,
):
    """
    InterOptimus-style clustering of slab pairs into unique termination conditions.

    Returns:
      reps: list of (film_idx, sub_idx) representative pairs (unique termination conditions)
    """
    matcher = StructureMatcher(primitive_cell=False, scale=False)
    ase_sec = SymmetryEquivalenceCheck() if SymmetryEquivalenceCheck is not None else None

    def _to_ase_atoms_safe(s: Structure):
        # InterOptimus uses Structure.to_ase_atoms(); in this codebase we rely on pymatgen's ASE adaptor.
        if hasattr(s, "to_ase_atoms"):
            return s.to_ase_atoms()
        if HAS_ASE:
            return AseAdaptor.get_atoms(s)
        raise RuntimeError("ASE 不可用，无法进行 SymmetryEquivalenceCheck。")

    reps: list[tuple[int, int, Structure, Structure]] = []
    items = len(film_slabs) * len(sub_slabs) if progress else None
    t_id = 0

    for i in range(len(film_slabs)):
        for j in range(len(sub_slabs)):
            film = film_slabs[i]
            sub = sub_slabs[j]
            identical_exist = False

            for (fi, sj, film_fit, sub_fit) in reps:
                if ase_sec is not None:
                    try:
                        if not (
                            ase_sec.compare(_to_ase_atoms_safe(film_fit), _to_ase_atoms_safe(film))
                            and ase_sec.compare(_to_ase_atoms_safe(sub_fit), _to_ase_atoms_safe(sub))
                        ):
                            continue
                    except Exception:
                        # fallback to StructureMatcher fit if to_ase_atoms is not present
                        if not (matcher.fit(film_fit, film) and matcher.fit(sub_fit, sub)):
                            continue
                else:
                    if not (matcher.fit(film_fit, film) and matcher.fit(sub_fit, sub)):
                        continue

                if _pair_fit_interoptimus(film_fit, sub_fit, film, sub, matcher, c_periodic):
                    identical_exist = True
                    break

            if not identical_exist:
                reps.append((i, j, film, sub))
                if max_unique is not None and len(reps) >= max_unique:
                    return [(x[0], x[1]) for x in reps]

            if progress and items is not None:
                percentage = int(round((t_id + 1) / items, 2) * 100)
                print("\r", end="")
                print(f"symmetry checking progress: {percentage}%: ", "▋" * (percentage // 2), end="")
                t_id += 1

    if progress and items is not None:
        print()
    return [(x[0], x[1]) for x in reps]
def _termination_signature_from_slab(slab, ftol):
    """Return an order-insensitive signature from top/bottom labels."""
    top_label = _label_termination_fast(slab, ftol=ftol)
    frac = slab.frac_coords.copy()
    frac[:, 2] = (-frac[:, 2]) % 1.0
    bottom_struct = Structure(slab.lattice, slab.species, frac, coords_are_cartesian=False)
    bottom_label = _label_termination_fast(bottom_struct, ftol=ftol)
    # Ignore order: (A, B) == (B, A)
    return tuple(sorted((top_label, bottom_label)))

def _get_termination_pairs(
    structure,
    hkl,
    termination_ftol=0.25,
    termination_tol=1e-3,
    max_pairs=None,
    return_label_records=False,
    profile: dict | None = None,
):
    """
    Return list of (dp1, dp2, signature) for bicrystal terminations.

    Equivalence rule (order-insensitive):
    - A bicrystal has two interfaces:
      I1 = combine(top(dp1), bottom(dp2))
      I2 = combine(top(dp2), bottom(dp1))
    - Two bicrystals are equivalent if the unordered pair {I1, I2} is identical.
    """
    t0 = time.perf_counter()
    slab_spacing = float(structure.lattice.d_hkl(hkl))
    sg = SlabGenerator(
        initial_structure=structure,
        miller_index=hkl,
        min_slab_size=slab_spacing,
        min_vacuum_size=0.0,
        center_slab=False,
        in_unit_planes=True,
        primitive=False,
        reorient_lattice=False,
    )
    slabs = sg.get_slabs(ftol=termination_ftol, filter_out_sym_slabs=False)
    t1 = time.perf_counter()
    if not slabs:
        if profile is not None:
            profile["slabgen_seconds"] = t1 - t0
            profile["label_seconds"] = 0.0
            profile["pair_seconds"] = 0.0
            profile["n_slabs"] = 0
        return []

    label_records = []
    records = []
    t_label0 = time.perf_counter()
    for slab in slabs:
        shift = float(slab.shift) % 1.0
        top_label = _label_termination_fast(slab, ftol=termination_ftol)
        frac = slab.frac_coords.copy()
        frac[:, 2] = (-frac[:, 2]) % 1.0
        bottom_struct = Structure(slab.lattice, slab.species, frac, coords_are_cartesian=False)
        bottom_label = _label_termination_fast(bottom_struct, ftol=termination_ftol)
        # Slab termination signature (order-insensitive), used for reporting and grouping.
        # NOTE: we still keep ordered top/bottom labels separately because interfaces are
        # formed by combining slab1_top with slab2_bottom, etc.
        slab_signature = tuple(sorted((top_label, bottom_label)))
        label_records.append(
            {
                "shift": shift,
                "top_label": top_label,
                "bottom_label": bottom_label,
                "signature": slab_signature,
                    "slab": slab,
            }
        )

        # 先做轻量去重：同一 top/bottom label 且 shift 非常接近
        is_dup = False
        for r in records:
            if (
                r["top_label"] == top_label
                and r["bottom_label"] == bottom_label
                and abs(r["shift"] - shift) < termination_tol
            ):
                is_dup = True
                break
        if not is_dup:
            records.append(
                {
                    "shift": shift,
                    "top_label": top_label,
                    "bottom_label": bottom_label,
                    "signature": slab_signature,
                    "slab": slab,
                }
            )
    t_label1 = time.perf_counter()

    # --- InterOptimus-style slab-pair equivalence (copy behavior) ---
    # We treat each (slab_i, slab_j) as a candidate termination condition and cluster
    # them under symmetry equivalence, returning one representative (dp1, dp2) per class.
    film_slabs = [r["slab"] for r in records]
    sub_slabs = [r["slab"] for r in records]

    # For our GBs we usually have no vacuum in termination slabs -> c_periodic=True
    # If max_pairs is set, we can stop after we have that many unique classes.
    rep_pairs = _dedupe_slab_pairs_interoptimus(
        film_slabs,
        sub_slabs,
        c_periodic=True,
        max_unique=max_pairs,
        progress=False,
    )

    term_pairs = []
    seen_bicrystal_signatures = set()
    t_pair0 = time.perf_counter()
    for (i, j) in rep_pairs:
        r1 = records[i]
        r2 = records[j]
        dp1 = float(r1["shift"])
        dp2 = float(r2["shift"])
        # Bicrystal signature (two-interface labels), for reporting only:
        # I1=(slab1_top, slab2_bottom), I2=(slab1_bottom, slab2_top), order-insensitive between I1/I2.
        iface1 = (r1["top_label"], r2["bottom_label"])
        iface2 = (r1["bottom_label"], r2["top_label"])
        bic_sig = tuple(sorted((iface1, iface2)))
        seen_bicrystal_signatures.add(bic_sig)
        term_pairs.append((dp1, dp2, bic_sig))
        if max_pairs is not None and len(term_pairs) >= max_pairs:
            break
    t_pair1 = time.perf_counter()

    if profile is not None:
        profile["slabgen_seconds"] = t1 - t0
        profile["label_seconds"] = t_label1 - t_label0
        profile["pair_seconds"] = t_pair1 - t_pair0
        profile["n_slabs"] = len(records)
        profile["n_bicrystal_signatures"] = len(seen_bicrystal_signatures)
    if return_label_records:
        return term_pairs, label_records
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
            primitive=False,
            reorient_lattice=False,
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
    debug=False,
    slow_seconds=2.0,
):
    """
    Return count of equivalent termination pairs for a given search result.
    This avoids building full GB structures.
    """
    hkl = result["hkl"]
    if isinstance(hkl, str) and "Cartesian" in hkl:
        L_p = parent_stct.lattice.matrix.T
        hkl = MID(lattice=L_p, n=result["axis_cart"], tol=1e-2)

    cache_key = (
        tuple(int(x) for x in np.array(hkl, dtype=int).tolist()),
        int(result.get("sigma", -1)),
        _rot_cache_key(result["rotation_matrix"]),
        float(termination_ftol),
        float(termination_tol),
        float(tol_ortho),
    )
    if cache_key in _TERM_PAIR_CACHE:
        if debug:
            print(f"    [cache hit] term_pairs={_TERM_PAIR_CACHE[cache_key]}")
        return _TERM_PAIR_CACHE[cache_key]

    t0 = time.perf_counter()
    my_interface = core(parent_stct, parent_stct, verbose=False)
    my_interface.parse_limit(du=5e-2, S=5e-2, sgm1=200, sgm2=200, dd=5e-2)
    if debug:
        print("    [stage] search_fixed ...")
    my_interface.search_fixed(result["rotation_matrix"], exact=False)
    t1 = time.perf_counter()

    try:
        if debug:
            print("    [stage] compute_bicrystal(normal_ortho=True) ...")
        my_interface.compute_bicrystal(hkl, lim=20, normal_ortho=True, tol_ortho=tol_ortho)
    except Exception:
        if debug:
            print("    [stage] compute_bicrystal(normal_ortho=False) ...")
        my_interface.compute_bicrystal(hkl, lim=20, normal_ortho=False)
    t2 = time.perf_counter()

    slab_structure = parent_stct.copy()
    if debug:
        print("    [stage] make_supercell(U1) + MID(hkl_sc) ...")
    slab_structure.make_supercell(np.array(my_interface.bicrystal_U1, dtype=int))
    n_cart = get_normal_from_MI(parent_stct.lattice.matrix.T, hkl)
    hkl_sc = MID(lattice=slab_structure.lattice.matrix.T, n=n_cart, tol=1e-2)
    t3 = time.perf_counter()

    prof: dict[str, float] = {}
    if debug:
        print("    [stage] _get_termination_pairs(max_pairs=1) ...")
    term_pairs = _get_termination_pairs(
        slab_structure,
        hkl_sc,
        termination_ftol=termination_ftol,
        termination_tol=termination_tol,
        max_pairs=1,
        profile=prof,
    )
    n_pairs = len(term_pairs)
    _TERM_PAIR_CACHE[cache_key] = n_pairs

    t4 = time.perf_counter()
    total = t4 - t0
    if debug and total >= slow_seconds:
        print(
            "    [timing] search_fixed={:.2f}s compute_bicrystal={:.2f}s supercell={:.2f}s "
            "slabgen={:.2f}s label={:.2f}s pair={:.2f}s total={:.2f}s".format(
                t1 - t0,
                t2 - t1,
                t3 - t2,
                prof.get("slabgen_seconds", 0.0),
                prof.get("label_seconds", 0.0),
                prof.get("pair_seconds", 0.0),
                total,
            )
        )
    return n_pairs

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
    relax_all_terminations: bool = True
    
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
        top_label = _label_termination_fast(slab, ftol=self.termination_ftol)
        # 通过翻转 z 分数坐标获得底面标签
        frac = slab.frac_coords.copy()
        frac[:, 2] = (-frac[:, 2]) % 1.0
        bottom_struct = Structure(slab.lattice, slab.species, frac, coords_are_cartesian=False)
        bottom_label = _label_termination_fast(bottom_struct, ftol=self.termination_ftol)
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
                in_unit_planes=True,
                primitive=False,
                reorient_lattice=False,
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
        term_pairs, label_records = _get_termination_pairs(
            slab_structure,
            hkl_sc,
            termination_ftol=self.termination_ftol,
            termination_tol=self.termination_tol,
            return_label_records=True,
        )
        print("所有端面的 label_termination:")
        for rec in label_records:
            print(
                f"  shift={rec['shift']:.4f} | top={rec['top_label']} | "
                f"bottom={rec['bottom_label']} | signature={rec['signature']}"
            )
        # Save all termination slabs before BO
        os.makedirs(self.output_dir, exist_ok=True)
        slab_dir = os.path.join(self.output_dir, "termination_slabs")
        os.makedirs(slab_dir, exist_ok=True)
        for idx, rec in enumerate(label_records, start=1):
            slab_file = os.path.join(
                slab_dir,
                f"slab_{idx}_shift_{rec['shift']:.4f}.vasp",
            )
            slab_a_normal = _reorient_slab_a_as_normal(rec["slab"])
            Poscar(slab_a_normal).write_file(slab_file)
        print(f"已保存 {len(label_records)} 个端面 slab 到: {slab_dir}")
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

        # 3. 选择需要优化的端面
        min_gamma = min(r["interface_energy"] for r in term_results)
        if self.relax_all_terminations:
            selected = term_results
            print(f"最低界面能: {min_gamma:.4f} J/m^2，优化所有 {len(selected)} 个端面...")
        else:
            selected = [r for r in term_results if r["interface_energy"] <= min_gamma + self.interface_energy_window]
            print(f"最低界面能: {min_gamma:.4f} J/m^2，筛选 {len(selected)} 个端面进行结构优化...")

        # 4. 对筛选结果做结构优化并保存
        os.makedirs(self.output_dir, exist_ok=True)
        saved_files = []
        optimized_results = []
        summary_lines = ["idx,dp1,dp2,interface_energy_Jm2,file"]
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
            summary_lines.append(f"{idx},{r['dp1']:.6f},{r['dp2']:.6f},{r['interface_energy']:.6f},{filename}")
            optimized_results.append({
                "dp1": r["dp1"],
                "dp2": r["dp2"],
                "signature": r["signature"],
                "interface_energy": r["interface_energy"],
                "relaxed_structure": relaxed_stct,
                "file": filename,
            })

        summary_path = os.path.join(self.output_dir, "interface_energy_summary.csv")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

        print("所有流程已完成。")
        return {
            "min_interface_energy": min_gamma,
            "selected_count": len(selected),
            "saved_files": saved_files,
            "summary_csv": summary_path,
            "optimized_results": optimized_results,
        }
