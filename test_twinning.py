import numpy as np
from interfacemaster.twinning_search import solve_compatibility, MID
from numpy.linalg import norm

def test_lmo_twinning():
    # Parent: Cubic a0 = 8.24
    a0 = 8.24
    L_p = np.eye(3) * a0
    
    # Child: Tetragonal a=8.0, c=9.3
    # Variant 1: c along z
    U1 = np.diag([8.0, 8.0, 9.3]) / a0
    # Variant 2: c along x
    U2 = np.diag([9.3, 8.0, 8.0]) / a0
    
    print("Testing Li2Mn2O4-like twinning...")
    solutions = solve_compatibility(U1, U2, L_p)
    
    if solutions:
        print(f"Found {len(solutions)} twinning planes.")
        for sol in solutions:
            print(f"  hkl: {sol['hkl']}")
    else:
        print("No compatible twinning planes found.")

if __name__ == "__main__":
    test_lmo_twinning()
