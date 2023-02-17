"""cellcalc.py"""
from numpy.linalg import norm, inv
from numpy import cross, square, arccos, pi, inf, cos, sin
import numpy as np


def rot(a, Theta):
    """
    produce a rotation matrix

    Parameters
    ----------
    a : numpy array
        rotation axis
    Theta : float
        rotation angle

    Returns
    ----------
    rot_mat : numpy array
        a rotation matrix
    """
    c = float(cos(Theta))
    s = float(sin(Theta))
    a = a / norm(a)
    ax, ay, az = a
    return np.array([[c + ax * ax * (1 - c), ax * ay * (1 - c) - az * s,
                      ax * az * (1 - c) + ay * s],
                    [ay * ax * (1 - c) + az * s, c + ay * ay * (1 - c),
                        ay * az * (1 - c) - ax * s],
                     [az * ax * (1 - c) - ay * s, az * ay * (1 - c) + ax * s,
                      c + az * az * (1 - c)]], dtype=np.float64)


def ang(v1, v2):
    """
    compute the cos of angle between v1 & v2

    Parameters
    ----------
    v1, v2 : numpy array
        vectors

    Returns
    ----------
    cos_12 : float
        the cos of angle between v1 and v2
    """
    return abs(np.dot(v1, v2) / norm(v1) / norm(v2))


def get_ortho_two_v(B, lim, tol,
                    align_rotation_axis=False, rotation_axis=None):
    """
    get orthogonal cell of a 2D basis

    Parameters
    ----------
    B : numpy array
        matrix of 2D basis vectors
    lim : int
        maximum coefficient
    tol : float
        tolerance
    align_rotation_axis : bool
        whether to specify a vector, default False
    rotation_axis :
        specified vector, default [1, 1, 1]

    Returns
    ----------
    cell_ortho : numpy array
        a perpendicular 2D basis
    """
    if rotation_axis is None:
        rotation_axis = [1, 1, 1]
    # meshes
    x = np.arange(-lim, lim + 1, 1)
    y = x

    indice = (np.stack(np.meshgrid(x, y)).T).reshape(len(x) ** 2, 2)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)]
    indice_0 = np.array(indice_0, dtype=np.float64)
    LP = np.dot(B, indice_0.T).T
    LP = LP[np.argsort(norm(LP, axis=1))]
    found = False
    count = 0
    if align_rotation_axis:
        for i in LP:
            if norm(cross(i, rotation_axis)) < tol:
                v1 = i
                break

    while not found and count < len(LP):
        if not align_rotation_axis:
            v1 = LP[count]
        for i in LP:
            if ang(v1, i) < tol:
                v2 = i
                found = True
                break
        if align_rotation_axis:
            break
        count += 1
    if not found:
        raise RuntimeError(
            'faild to find two orthogonal vectors in the GB plane, '
            'maybe you can try to increase the lim')
    return np.column_stack((v1, v2))


def dia_sym_mtx(U):
    """
    return the second-diagonal symmetry transformed matrix of U

    Parameters
    ----------
    U : numpy array
        matrix

    Returns
    ----------
    Ux : numpy array
        the second-diagonal symmetry transformed matrix of U

    """
    Ux = np.eye(3)
    for i in range(3):
        for j in range(3):
            Ux[i][j] = U[2 - j][2 - i]
    return Ux


def find_integer_vectors(v, sigma, tol=1e-8):
    """
    A function for finding the coefficients N
    so that Nv contains only integer elements and gcd(v1,v2,v3,N) = 1

    Parameters
    ----------
    v : numpy array
        a 3D vector
    sigma : int
        maximum limt of N for searching
    tol : float
        tolerancea to judge integer, default 1e-8

    Returns
    ----------
    Nv : numpy array
        vector of integer elements Nv
    N : numpy array
        integer coefficients N
    """
    found = False
    for i in range(1, sigma + 1):
        now = v * i
        if np.all(abs(now - np.round(now)) < tol):
            u, v, w = np.round(now)
            L = i
            found = True
            break
    if found:
        reduce = np.gcd.reduce([int(u), int(v), int(w)])
        now = np.array(np.round(now / reduce), dtype=int)
        return now, L
    else:
        raise RuntimeError(
            f'failed to find the rational vector of {v}\n'
            f' within lcd = {sigma}')


def find_integer_vectors_nn(v, sigma, tol=1e-8):
    """
    A function for finding the coefficients N
    so that Nv contains only integer elements, ignore gcd

    Parameters
    ----------
    v : numpy array
        a 3D array
    sigma : int
        maximum limt of N for searching
    tol : float
        tolerance, default 1e-8

    Returns
    ----------
    Nv : numpy array
        vector of integer elements Nv
    N : numpy array
        integer coefficients N
    """
    found = False
    for i in range(1, sigma + 1):
        now = v * i
        if np.all(abs(now - np.round(now)) < tol):
            _, v, _ = np.round(now)
            L = i
            found = True
            break
    if found:
        now = np.array(np.round(now), dtype=int)
        L = int(np.round(L))
        return now, L
    else:
        raise RuntimeError(
            f'failed to find the rational vector of {v}\n'
            f' within lcd = {sigma}')


def solve_DSC_equations(u, v, w, L, B):
    """
    a function for solving the integer equations for the DSC basis

    Parameters
    ----------
    u, v, w, L: int
        integer numbers
    B : numpy array
        basis set
    tol : float
        tolerance

    Returns
    ----------
    DSC_basis1, DSC_basis2 : numpy array
        DSC_basis sets
    """
    tol = 1e-8
    # get g_v, g_miu and g_lambda
    g_v = L / np.gcd(abs(w), L)
    g_lambda = np.gcd.reduce([abs(v), abs(w), L])
    g_miu = np.gcd(abs(w), L) / g_lambda

    g_v = int(round(g_v))
    g_lambda = int(round(g_lambda))
    g_miu = int(round(g_miu))

    # 0=<gama<g_v
    gamas = np.arange(0, g_v)
    for i in gamas:
        gama = i
        s = (g_miu * v - gama * w) / L
        # check whether s is a integer
        if abs(s - np.round(s)) < tol:
            break

    # 0=<alpha<g_miu, 0=<beta<g_v
    alphas = np.arange(0, g_miu)
    betas = np.arange(0, g_v)
    count = 0
    found = False
    while not found and count < len(alphas):
        alpha = alphas[count]
        for k in betas:
            beta = k
            r = (g_lambda * u - alpha * v - beta * w) / L
            # check whether r is a integer
            if abs(r - np.round(r)) < tol:
                found = True
                break
        count += 1
    if not found:
        raise RuntimeError(
            'failed to find integer solutions, something strange happens...')
    # DSC basis
    D1 = 1 / g_lambda * B[:, 0]
    D2 = alpha / (g_lambda * g_miu) * B[:, 0] + 1 / g_miu * B[:, 1]
    D3 = (alpha * gama + beta * g_miu) / (g_miu * g_v * g_lambda) * B[:, 0] \
        + gama / (g_miu * g_v) * B[:, 1] + 1 / g_v * B[:, 2]
    # print('alpha, beta, gama, lambda, miu, v')
    # print(alpha, beta, gama, g_lambda, g_miu, g_v)
    DSC_basis = np.column_stack((D1, D2, D3))
    return DSC_basis, np.dot(inv(B), DSC_basis)


def projection(u1, u2):
    """
    get the projection of u1 on u2

    Parameters
    ----------
    u1, u2 : numpy array
        vectors

    Returns
    ----------
    p12 : numpy array
        the projection of u1 on u2
    """
    return np.dot(u1, u2) / np.dot(u2, u2)


def Gram_Schmidt(B0):
    """
    Gram–Schmidt process

    Parameters
    ----------
    B0 : numpy array
        matrix

    Returns
    ----------
    Bstar : numpy array
        the orthogonalized basis for B0
    """
    Bstar = np.eye(3, len(B0.T))
    for i in range(len(B0.T)):
        if i == 0:
            Bstar[:, i] = B0[:, i]
        else:
            BHere = B0[:, i].copy()
            for j in range(i):
                BHere = BHere - projection(B0[:, i], B0[:, j]) * B0[:, j]
            Bstar[:, i] = BHere
    return Bstar


def LLL(B):
    """
    LLL lattice reduction algorithm

    https://en.wikipedia.org/wiki/Lenstra%E2%80%93Lenstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm

    Parameters
    ----------
    B : numpy array
        matrix

    Returns
    ----------
    Bhere : numpy array
        the basis obtained for B
    """
    Bhere = B.copy()
    Bstar = Gram_Schmidt(Bhere)
    delta = 3 / 4
    k = 1
    while k <= len(B.T) - 1:
        js = -np.sort(-np.arange(0, k))
        for j in js:
            ukj = projection(Bhere[:, k], Bstar[:, j])
            if abs(ukj) > 1 / 2:
                Bhere[:, k] = Bhere[:, k] - round(ukj) * Bhere[:, j]
                Bstar = Gram_Schmidt(Bhere)
        ukk_1 = projection(Bhere[:, k], Bstar[:, k - 1])
        if (np.dot(Bstar[:, k], Bstar[:, k])
                >= (delta - square(ukk_1))
                * np.dot(Bstar[:, k - 1], Bstar[:, k - 1])):
            k += 1
        else:
            m = Bhere[:, k].copy()
            Bhere[:, k] = Bhere[:, k - 1].copy()
            Bhere[:, k - 1] = m
            Bstar = Gram_Schmidt(Bhere)
            k = max(k - 1, 1)
    return Bhere


def get_normal_index(hkl, lattice):
    """
    get the coordinates in the lattice of a normal vector to the plane (hkl)

    Parameters
    ----------
    hkl : array_like
        input index
    lattice : array_like
        the basis of the lattice

    Returns
    ----------
    v_n : numpy array
        the coordinates in the lattice of a normal vector to the plane (hkl)
    """
    n, _ = get_plane(hkl, lattice)
    return np.dot(inv(lattice), n)


def get_primitive_hkl(hkl, C_lattice, P_lattice, tol=1e-8):
    """
    convert the miller indices from conventional cell to primitive cell

    Parameters
    ----------
    hkl : array_like
        input index
    C_lattice : numpy array
        conventional cell basis set
    P_lattice : numpy array
        primitive cell basis set
    tol : float
        tolerance to judge integer, default 1e-8

    Returns
    ----------
    hkl_p : numpy array
        the Miller indices in primitive cell
    """
    # 1. get normal
    n, Pc1 = get_plane(hkl, C_lattice)
    # print('the normal:' + str(n) + ' the point in the plane: ' + str(Pc1))
    # 2. get indices from normal
    hkl_p = get_indices_from_n_Pc1(n, P_lattice, Pc1)
    hkl_p = find_integer_vectors(hkl_p, 100000, tol)[0]
    return hkl_p


def get_plane(hkl, lattice):
    """
    get the normal vector and one in-plane point
    for the (hkl) plane of the lattice

    Parameters
    ----------
    hkl : array_like
        input index
    lattice : numpy array
        lattice basis set

    Returns
    ----------
    n : numpy array
        the normal vector
    p : numpy array
        an in-plane point in the (hkl) plane
    """
    points = np.eye(3)
    for i in range(3):
        if hkl[i] != 0:
            points[:, 0] = lattice[:, i] / hkl[i]
            count = i
            break
    count2 = 1
    for i in range(3):
        if i != count:
            if hkl[i] == 0:
                points[:, count2] = points[:, 0] + lattice[:, i]
            else:
                points[:, count2] = lattice[:, i] / hkl[i]
            count2 += 1
    n = cross((points[:, 0] - points[:, 1]), (points[:, 0] - points[:, 2]))
    return n, points[:, 0]


def get_indices_from_n_Pc1(n, lattice, Pc1):
    """
    get the Miller indices of certain plane
    with normal n and one in-plane point Pc1 for certain lattice

    Parameters
    ----------
    n : numpy array
        normal vector
    lattice : numpy array
        lattice basis set
    Pc1 : numpy array
        an in-plane point in the (hkl) plane

    Returns
    ----------
    hkl : numpy array
        the Miller indices of the desired plane
    """
    hkl = np.array([0, 0, 0], dtype=float)
    for i in range(3):
        hkl[i] = np.dot(lattice[:, i], n) / np.dot(Pc1, n)
    return hkl


def MID(lattice, n, tol=1e-8):
    """
    get the Miller indices with a normal n for the lattice

    Parameters
    ----------
    lattice : numpy array
        lattice basis set
    n : numpy array
        normal vector
    tol : float
        tolerance to judge integer, default 1e-8

    Returns
    ----------
    hkl : numpy array
        the Miller indices of desired plane
    """
    for i in range(3):
        if abs(np.dot(lattice[:, i], n)) > tol:
            Pc1 = lattice[:, i]
            break
    hkl = get_indices_from_n_Pc1(n, lattice, Pc1)
    hkl = find_integer_vectors(hkl, 10000, tol)[0]
    return hkl


def ext_euclid(a, b):
    """
    extended euclidean algorithm

    from https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm

    Parameters
    ----------
    a, b : int
        two input integers

    Returns
    ----------
    old_s, old_t, old_r : int
        the output integers
    """
    old_s, s = 1, 0
    old_t, t = 0, 1
    old_r, r = a, b
    if b == 0:
        return 1, 0, a
    else:
        while r != 0:
            q = old_r // r
            old_r, r = r, old_r - q * r
            old_s, s = s, old_s - q * s
            old_t, t = t, old_t - q * t
    return old_s, old_t, old_r


def get_pri_vec_inplane(hkl, lattice):
    """
    get two primitive lattice vector for the (hkl) plane from
    Banadaki A D, Patala S.
    An efficient algorithm for computing the primitive bases
    of a general lattice plane[J].
    Journal of Applied Crystallography, 2015, 48(2): 585-588.

    Parameters
    ----------
    hkl : numpy array
        the Miller indices
    lattice : numpy array
        lattice basis set

    Returns
    ----------
    v1, v2 : numpy array
        the two primitive lattice vectors
    """
    h, k, l = hkl
    if k == 0 and l == 0:
        return LLL(np.column_stack((lattice[:, 1], lattice[:, 2])))
    by, bz, c = ext_euclid(abs(k), abs(l))
    if h == 0:
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, -l / c, k / c])
    else:
        bx = -c / h
        if k != 0:
            by = k / abs(k) * by
        if l != 0:
            bz = l / abs(l) * bz
        v1 = np.array([0, -l / c, k / c])
        v2 = np.array([bx, by, bz])
    v2 = find_integer_vectors(v2, 1000)[0]
    return LLL(np.dot(lattice, np.column_stack((v1, v2))))


def get_right_hand(B):
    """
    get right handed lattice of B

    Parameters
    ----------
    B : numpy array
        lattice basis set

    Returns
    ----------
    B_r : numpy array
        the right_handed basis set of lattice B
    """
    if np.dot(B[:, 2], cross(B[:, 0], B[:, 1])) < 0:
        B[:, 2] = - B[:, 2]
    return B


def get_normal_from_MI(lattice, hkl):
    """
    get the normal vector of a lattice plane with known miller indices

    Parameters
    ----------
    lattice : numpy array
        lattice basis set
    hkl : numpy array
        the Miller indices

    Returns
    ----------
    n : numpy array
        a normal vector to the (hkl) plane
    """
    hkl = np.array(hkl)
    lattice = np.array(lattice, dtype=float)
    numzeros = len(np.where(hkl == 0)[0])
    if numzeros == 2:
        zero_v_index = np.where(hkl == 0)[0]
        return cross(lattice.T[zero_v_index[0]], lattice.T[zero_v_index[1]])
    elif numzeros == 1:
        non_zero_v_index = np.where(hkl != 0)[0]
        u1 = lattice.T[non_zero_v_index[0]] / hkl[non_zero_v_index[0]]
        u2 = lattice.T[non_zero_v_index[1]] / hkl[non_zero_v_index[1]]
        v1 = u1 - u2
        zero_v_index = np.where(hkl == 0)[0][0]
        v2 = lattice.T[zero_v_index]
        return cross(v1, v2)
    elif numzeros == 0:
        P1 = lattice[:, 0] / hkl[0]
        P2 = lattice[:, 1] / hkl[1]
        P3 = lattice[:, 2] / hkl[2]
        v1 = P1 - P2
        v2 = P1 - P3
        return cross(v1, v2)
    else:
        raise RuntimeError('hkl error')


def search_MI_n(lattice, n, tol, lim):
    """
    get the miller indices of planes
    whose normal vector is close to the input vector n

    Parameters
    ----------
    lattice : numpy array
        lattice basis set
    n : numpy array
        the normal vector
    tol : float
        tolerance to judge perpendicular
    lim : int
        upper limit of indices serching

    Returns
    ----------
    hkl : numpy array
        the Miller indices
    """
    x = np.arange(-lim, lim + 1, 1)
    y = x
    z = x
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)]
    indice_0 = indice_0[np.argsort(norm(indice_0, axis=1))]
    found = False
    for _ in indice_0:
        n_there = get_normal_from_MI(lattice, indice_0)
        dtheta = arccos(1 - ang(n_there, n))
        if dtheta < tol:
            print('plane found, dtheta = {}', format(dtheta))
            found = True
            break
    if not found:
        RuntimeError(
            'failed to find a close lattice plane for this normal vector, '
            'please increase the tol or adjust orientation')
    return MID(lattice, n_there, tol)


def match_rot(deft_rot, axis, tol, exact_rot, av_perpendicular):
    """
    find the intermediate rotation

    given an exact rotation matrix and a default rotation matrix,
    find the intermediate rotation matrix so that the operation combining
    the default rotation + the intermediate rotation
    is close to the excat rotation

    Parameters
    ----------
    deft_rot : numpy array
        initial orientation matrix
    axis : numpy array
        the rotation axix
    tol : float
        tolerance to judge coincidence
    exact_rot : numpy array
        target orientation matrix
    av_perpendicular : numpy array
        auxilliary vector

    Returns
    ----------
    inter_rot : numpy array
        intermediate rotation matrix
    """
    theta = 0
    found = False
    min_g = inf
    while theta < 2 * pi:
        inter_rot = rot(axis, theta)
        compare_rot = np.dot(deft_rot, inter_rot)
        compare_v = np.dot(compare_rot, av_perpendicular)
        compare_v = compare_v / norm(compare_v)
        exact_v = np.dot(exact_rot, av_perpendicular)
        exact_v = exact_v / norm(exact_v)
        if norm(cross(compare_v, exact_v)) < min_g:
            min_g = norm(cross(compare_v, exact_v))
        if norm(cross(compare_v, exact_v)) < tol:
            found = True
            if np.dot(exact_v, compare_v) < 0:
                inter_rot = rot(axis, theta + pi)
            break
        theta += 0.01 / 180 * pi
    if not found:
        raise RuntimeError(
            f'Disorientation match failed, '
            f'please try other planes or increase tolerance, min = {min_g}')
    return inter_rot


class DSCcalc:
    """
    core class computing DSC basis
    """

    def __init__(self):
        self.ai1 = np.eye(3)  # basis vectors of lattice 1
        self.ai2 = np.eye(3)  # basis vectors of lattice 2
        self.U = np.eye(3)  # ai1U = ai2
        self.sigma = int  # sigma2
        self.DSC = np.eye(3)
        self.CSL = np.eye(3)
        self.U1 = np.eye(3)  # ai1U1 = ai2U2
        self.U2 = np.eye(3)
        # describing ai2 by ai1 with int coe
        self.U_int = np.array(np.eye(3), dtype=int)
        # three cooresponding greatest common denominator
        self.Ls = np.arange(3)
        self.CNID = np.eye(3, 2)

    def parse_int_U(self, ai1, ai2, sigma, tol=1e-8):
        """
        initiate computation

        Parameters
        ----------
        ai1 : numpy array
            lattice 1 basis set
        ai2 : numpy array
            lattice 2 basis set
        sigma : int
            sigma value
        tol : float
            tolerance to judge coincidence, default 1e-8
        """
        self.ai1 = ai1
        self.ai2 = ai2
        self.sigma = sigma
        self.U = np.dot(inv(ai1), ai2)
        # get the integers uij
        for i in range(3):
            v = self.U[:, i]
            self.U_int[:, i], self.Ls[i] = find_integer_vectors(
                v, self.sigma, tol)

    def compute_DSC(self, to_LLL=True, tol=1e-8):
        """
        solve the DSC equation

        Parameters
        ----------
        to_LLL : bool
            whether do LLL algorithm for the found basis, default True
        tol : float
            tolerance to judge coincidence, default 1e-8
        """
        # integer LC of ai2_1
        # print('the U matrix')
        # print(str(self.U_int) + '\'' + str(self.sigma))
        # print('---------------------------------------')
        ks, L = find_integer_vectors(
            np.dot(inv(self.ai1), self.ai2[:, 0]), self.sigma, tol)
        # print('ai2_1 is expressed by ai1 as ' + str(ks) + '/' + str(L))
        k1, k2, k3 = ks
        # solve DSC Ei for the ai2_1 by ai1
        Ei = solve_DSC_equations(k1, k2, k3, L, self.ai1)[0]
        # print('Ei is \n' + str(Ei))
        # coefficients of ai2_2 expressed by DSC Ei
        es = np.dot(inv(Ei), self.ai2[:, 1])
        # print('ai2_2 is expressed by Ei as ' + str(es))
        # integer coefficients
        ls, M = find_integer_vectors(es, self.sigma, tol)
        # print('ls: ' + str(ls) + ' M: ' + str(M))
        if M == 1:
            # print('Ei satisfy ai2_2')
            # this DSC applies for ai2_2 now check ai2_3
            # coefficients of ai2_3 expressed by DSC Ei
            es = np.dot(inv(Ei), self.ai2[:, 2])
            # print('ai2_3 is expressed by Ei as ' + str(es))
            # integer coeficients
            ms, M_p = find_integer_vectors(es, self.sigma, tol)
            # print('ms: ' + str(ms) + ' M_p: ' + str(M_p))
            if M_p == 1:
                # print('Ei satisfy ai2_3')
                # this DSC also applies for ai2_3, output
                # print('-----------------------------')
                # print('sigma ' + str(L))
                DSC = Ei
            else:
                # this DSC does not apply for ai2_3
                m1, m2, m3 = ms
                # now compute the DSC Fi of ai2_3 by Ei, output
                Fi = solve_DSC_equations(m1, m2, m3, M_p, Ei)[0]
                # print('-----------------------------')
                # print('sigma ' + str(L * M_p))
                DSC = Fi
        else:
            # print('M > 1')
            # this DSC does not apply for ai2_2
            l1, l2, l3 = ls
            # now compute the DSC of ai2_2 by Ei
            Fi = solve_DSC_equations(l1, l2, l3, M, Ei)[0]
            # print('Fi is\n' + str(Fi))
            # check for ai2_3
            # coefficients of ai2_3 expressed by DSC Fi
            es = np.dot(inv(Fi), self.ai2[:, 2])
            # print('ai2_3 is expressed by Fi as ' + str(es))
            # integer coefficients
            ns, N = find_integer_vectors(es, self.sigma, tol)
            # print('ns: ' + str(ns) + ' N: ' + str(N))
            if N == 1:
                # print('N = 1, DSC got')
                # this DSC Fi applies for ai2_3
                # print('-----------------------------')
                # print('sigma ' + str(L * M))
                DSC = Fi
            else:
                # this DSC Fi does not apply for ai2_3
                # print('N > 1')
                n1, n2, n3 = ns
                Gi = solve_DSC_equations(n1, n2, n3, N, Fi)[0]
                # print('Gi is ' + str(Gi))
                # print('-----------------------------')
                # print('sigma ' + str(L * M * N))
                DSC = Gi
        DSC = get_right_hand(DSC)
        if to_LLL:
            self.DSC = LLL(DSC)
        else:
            self.DSC = DSC
        self.DSC = np.dot(inv(self.ai1), self.DSC)

    def compute_CSL(self, tol=1e-8):
        """
        compute CSL

        Parameters
        ----------
        tol : float
            tolerance to judge coincidence, default 1e-8
        """
        # symmetric matrix along the second diagonal
        Ux = dia_sym_mtx(self.U)
        a20 = np.dot(self.ai1, Ux)
        calc_csl = DSCcalc()
        calc_csl.parse_int_U(self.ai1, a20, self.sigma, tol)
        calc_csl.compute_DSC(to_LLL=True, tol=tol)
        DSC_ax = np.dot(self.ai1, calc_csl.DSC)
        self.U1 = dia_sym_mtx(np.dot(inv(DSC_ax), a20))
        self.CSL = np.dot(self.ai1, self.U1)
        red_CSL = LLL(self.CSL)
        red_CSL = get_right_hand(red_CSL)
        self.U1 = np.dot(inv(self.ai1), red_CSL)
        self.U2 = np.dot(inv(self.U), self.U1)
        self.CSL = get_right_hand(red_CSL)

    def compute_CNID(self, hkl, tol=1e-8):
        """
        compute 2D DSC

        Parameters
        ----------
        hkl :
            miller indices of the plane to compute
        tol : float
            tolerance to judge coincidence, default 1e-8
        """
        pmi_1 = hkl
        pmi_2 = get_primitive_hkl(hkl, self.ai1, self.ai2, tol)
        pb_1 = get_pri_vec_inplane(pmi_1, self.ai1)
        pb_2 = get_pri_vec_inplane(pmi_2, self.ai2)
        n = cross(pb_1[:, 0], pb_1[:, 1])
        c1 = get_right_hand(np.column_stack((pb_1, n)))
        c2 = get_right_hand(np.column_stack((pb_2, n)))
        calc_cnid = DSCcalc()
        calc_cnid.parse_int_U(c1, c2, 10000, tol)
        calc_cnid.compute_DSC(to_LLL=True, tol=tol)  # in c1 frame
        DSC = np.dot(c1, calc_cnid.DSC)
        count = 0
        CNID = np.eye(3, 2)
        for i in range(3):
            tol = 1e-10
            if norm(cross(DSC[:, i], n)) > tol:
                CNID[:, count] = DSC[:, i]
                count += 1
            if count == 2:
                break
        CNID = LLL(CNID)
        self.CNID = np.dot(inv(self.ai1), CNID)
