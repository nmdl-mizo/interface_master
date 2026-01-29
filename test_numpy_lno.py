import numpy as np
from numpy.linalg import inv, det, norm

def rot(a, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    a = a / norm(a)
    ax, ay, az = a
    return np.array([[c + ax * ax * (1 - c), ax * ay * (1 - c) - az * s,
                      ax * az * (1 - c) + ay * s],
                    [ay * ax * (1 - c) + az * s, c + ay * ay * (1 - c),
                        ay * az * (1 - c) - ax * s],
                     [az * ax * (1 - c) - ay * s, az * ay * (1 - c) + ax * s,
                      c + az * az * (1 - c)]], dtype=np.float64)

def get_lattice():
    # LiNiO2 primitive lattice from CIF
    a, b, c = 2.89962184, 2.89962184, 5.05868732
    alpha, beta, gamma = 73.34559279, 73.34559279, 60.00000000
    
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)
    
    # Fractional to Cartesian conversion matrix
    v1 = [a, 0, 0]
    v2 = [b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0]
    
    vz = c * np.sqrt(1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 + 2*np.cos(alpha_rad)*np.cos(beta_rad)*np.cos(gamma_rad)) / np.sin(gamma_rad)
    vx = c * (np.cos(beta_rad) - np.cos(alpha_rad)*np.cos(gamma_rad)) / np.sin(gamma_rad) # This is wrong, let's use a simpler way
    
    # Use standard lattice from params
    # From pymatgen Lattice.from_parameters
    def lattice_from_params(a, b, c, alpha, beta, gamma):
        alpha, beta, gamma = np.radians([alpha, beta, gamma])
        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)
        
        v1 = [a, 0, 0]
        v2 = [b * cos_gamma, b * sin_gamma, 0]
        v3_x = c * cos_beta
        v3_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        v3_z = np.sqrt(c**2 - v3_x**2 - v3_y**2)
        v3 = [v3_x, v3_y, v3_z]
        return np.array([v1, v2, v3]).T

    return lattice_from_params(a, b, c, 73.34559279, 73.34559279, 60.00000000)

def test_user_rotation():
    L1 = get_lattice()
    print("Lattice Matrix:\n", L1)
    
    axis = np.array([0,0,1])
    R = rot(axis, np.pi)
    
    U = np.dot(inv(L1), np.dot(R, L1))
    print("\nU matrix (inv(L1) * R * L1) for axis [0,0,1]:\n", U)
    
    # Try to find Sigma
    du = 0.05
    for N in range(1, 101):
        U_p = np.round(N * U) / N
        if np.all(abs(U_p - U) < du):
            sigma = int(abs(np.round(det(N * U_p))))
            # This is a bit simplified, but gives an idea
            print(f"Found approximate Sigma candidate at N={N}, det(N*U_p)={sigma}")
            break

if __name__ == "__main__":
    test_user_rotation()
