import numpy as np
from pymatgen.core.structure import Structure
from interfacemaster.cellcalc import rot, DSCcalc
from interfacemaster.interface_generator import core
from interfacemaster.twinning_search import search_low_index_twinning
import os

def debug():
    # Use full path for reliability in sandbox
    cif_path = "/Users/jason/Desktop/LiNO2/LNO_prim.cif"
    if not os.path.exists(cif_path):
        print(f"File not found: {cif_path}")
        return

    print("=== Automated Search Logic (My Updated Function) ===")
    # Run my search function with large tolerance
    results = search_low_index_twinning(cif_path, cif_path, max_strain=0.1, hkl_limit=3, max_sigma=100)
    
    if results:
        print(f"\nFound {len(results)} candidate interfaces.")
        # Check if any Cartesian [0,0,1] exists
        found_z = False
        for res in results:
            if "Cartesian [0 0 1]" in str(res['hkl']):
                print(f"MATCH FOUND: Cartesian [0,0,1] has Sigma {res['sigma']}")
                found_z = True
                break
        if not found_z:
            print("Cartesian [0,0,1] still not found in the results.")
    else:
        print("Automated search found nothing.")

if __name__ == "__main__":
    debug()
