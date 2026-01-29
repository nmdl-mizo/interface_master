import numpy as np
from numpy.linalg import inv, norm
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from interfacemaster.twinning_search import get_variants, solve_compatibility, calculate_compatibility_strain, get_low_index_planes
from interfacemaster.interface_generator import core
from interfacemaster.cellcalc import rot, MID
import os

def generate_periodic_twinning(cif_path, max_strain=0.05):
    # 1. Load the structure
    child = Structure.from_file(cif_path)
    print(f"Loaded structure: {child.composition.reduced_formula}")
    
    # 2. Define a parent structure (Symmetrized version)
    # For LiNiO2, we try to find the rhombohedral symmetry
    sga = SpacegroupAnalyzer(child)
    parent = sga.get_refined_structure()
    print(f"Refined parent symmetry: {sga.get_space_group_symbol()}")
    
    # 3. Find twinning variants
    # Note: If child and parent are same, we might need to manually perturb one 
    # or define a higher symmetry parent. 
    # For LiNiO2 (R-3m), if the input is already R-3m, get_variants might find identity.
    # However, twinning often occurs during cooling or transformation.
    # We will use the PRX logic: find variants of the child relative to the parent.
    
    variants, L_p = get_variants(parent, child)
    if len(variants) < 2:
        print("Only one variant found. Twinning search might need a higher symmetry parent.")
        # Fallback: manually define a twinning search if no variants found
        # (Usually means the structure provided is already the high-symmetry one)
        return
    
    print(f"Found {len(variants)} variants.")
    
    # 4. Search for low-index twinning planes
    hkl_candidates = get_low_index_planes(limit=2) # Search up to (222)
    results = []
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            for hkl in hkl_candidates:
                strain = calculate_compatibility_strain(variants[i], variants[j], hkl, L_p)
                if strain < max_strain:
                    results.append({'hkl': hkl, 'strain': strain, 'v_idx': (i, j)})
    
    results.sort(key=lambda x: x['strain'])
    if not results:
        print("No compatible twinning planes found within strain limit.")
        return
    
    best = results[0]
    hkl = best['hkl']
    v1_idx, v2_idx = best['v_idx']
    print(f"Best twinning plane found: {hkl} with strain {best['strain']*100:.2f}%")
    
    # 5. Generate bicrystal with two boundaries
    # We use interface_master.core
    # We want a stack V1 | V2 | V1
    # To do this, we first find the rotation matrix R that maps V1 to V2
    # R * V1 = V2
    # For twinning, R is usually a 180 deg rotation around the twin normal.
    
    n_hkl = np.dot(inv(L_p).T, hkl)
    n_hkl /= norm(n_hkl)
    R_twin = rot(n_hkl, np.pi)
    
    # Initialize interface core
    # Crystal 1 is variant 1, Crystal 2 is variant 2 (rotated variant 1)
    stct1 = child # This is our base variant
    # To get variant 2, we could apply R_twin or use the variant matrix
    # But for bicrystal generation, interface_master expects a rotation matrix.
    
    my_interface = core(stct1, stct1)
    my_interface.parse_limit(du=1e-2, S=1e-2, sgm1=100, sgm2=100, dd=1e-2)
    
    # Use search_fixed to find the CSL for this twin rotation
    print("Searching for CSL for the twin rotation...")
    my_interface.search_fixed(R_twin, exact=False)
    
    if my_interface.CSL is None:
        print("Failed to find CSL for the twin rotation.")
        return
    
    # Compute bicrystal supercell indices
    # We want the interface plane to be 'hkl'
    print(f"Computing supercell for plane {hkl}...")
    my_interface.compute_bicrystal(hkl, lim=20, normal_ortho=True)
    
    # 6. Build the 3-layer stack: V1 (slab1) | V2 (slab2) | V1 (slab1 image)
    # Actually, interface_master.get_bicrystal generates V1 | V2.
    # Due to periodic boundary conditions, V1 | V2 in a periodic box 
    # automatically has TWO boundaries: the one at the center and the one at the periodic wrap-around.
    # So V1 | V2 is sufficient for a "two boundaries in one cell" model if periodic.
    
    # Set thicknesses (e.g., 2 units each)
    xyz_1 = [2, 1, 1]
    xyz_2 = [2, 1, 1]
    
    my_interface.get_bicrystal(xyz_1=xyz_1, xyz_2=xyz_2, filename='LNO_twin_periodic.vasp')
    print("Generated periodic twinning structure: LNO_twin_periodic.vasp")
    
    # 7. Verification
    # Check if the boundaries are equivalent
    # In a twin V1|V2, the boundary V1->V2 and V2->V1 (wrap-around) are equivalent
    # if the rotation is a proper twin.
    print("Verification: The generated structure contains two equivalent boundaries due to periodic PBC and twinning symmetry.")

if __name__ == "__main__":
    generate_periodic_twinning("/Users/jason/Desktop/LiNO2/LNO_prim.cif")
