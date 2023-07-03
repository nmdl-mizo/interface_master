"""
Get the [001], [111], [110] symmetric tilt GBs;
Get the csl for bilayer twisted graphenes
"""

import numpy as np
from numpy.linalg import norm, det
from interfacemaster.cellcalc import MID, rot
from interfacemaster.interface_generator import core


def compute_sigma(axis, theta, filename, maxsigma=10000, verbose=True):
    """
    compute sigma values for a given disorientation

    Parameters
    __________
    axis : numpy array
        rotation axis
    theta : float
        rotation angle
    maxsigma : int
        maximum sigma value for searching
    verbose : bool, optional
        If True, the function will print additional information.
        Default is True.

    Returns
    __________
    sigma : int
    """
    if verbose:
        print(theta / np.pi * 180)
    R = rot(axis, theta)
    my_interface = core(filename,
                        filename, verbose=verbose)
    my_interface.parse_limit(
        du=1e-4, S=1e-4, sgm1=maxsigma, sgm2=maxsigma, dd=1e-4)
    my_interface.search_fixed(R, exact=True, tol=1e-3)
    return det(my_interface.U1)


def get_hkl(P, axis, tol=1e-3):
    """
    given a referring point and a rotation matrix, get the miller indices
    for this symmetric tilt GB

    Parameters
    __________
    P : numpy array
        referring point
    axis : numpy array
        rotation axis

    Returns
    __________
    hkl : numpy array
          miller indices
    """
    n = np.cross(axis, P)
    return MID(lattice=np.eye(3, 3), n=n, tol=tol)


def sample_STGB(axis, lim, maxsigma, max_index):
    """
    sampling the symmetric tilt GBs for a given rotation axis

    Parameters
    __________
    axis : numpy array
           rotation axis
    lim : int
          control the number of generated referring points
    maxsigma : int
               maximum sigma value
    max_index : int
               maximum value of miller indices
    Returns
    __________
    list1 : numpy array
           angle list
    list2 : numpy array
           sigma list
    list3 : numpy array
           hkl list
    """
    Ps, sigmas, thetas, _ = get_Ps_sigmas_thetas(lim, axis)
    sampled_indices = np.where((sigmas < maxsigma))[0]
    Ps = Ps[sampled_indices]
    sigmas = sigmas[sampled_indices]
    thetas = thetas[sampled_indices]
    hkls = []
    for i in Ps:
        hkls.append(get_hkl(i, axis, tol=1e-4))
    hkls = np.array(hkls)
    sampled_indices = np.all(np.abs(hkls) < max_index, axis=1)
    sigmas = sigmas[sampled_indices]
    thetas = thetas[sampled_indices]
    hkls = hkls[sampled_indices]
    return (
        thetas[np.argsort(thetas, kind='stable')],
        sigmas[np.argsort(thetas, kind='stable')],
        hkls[np.argsort(thetas, kind='stable')]
    )


def generate_arrays_x_y(x_min, y_min, lim):
    """
    generate x, y meshgrids

    Parameters
    __________
    x_min, y_min : int
        minimum x,y
    lim : int
        maximum x,y

    Returns
    __________
    meshgrid : numpy array
        x,y meshgrid
    """
    x = np.arange(x_min, x_min + lim, 1)
    y = np.arange(y_min, y_min + lim, 1)
    indice = (np.stack(np.meshgrid(x, y)).T).reshape(len(x) * len(y), 2)
    indice_gcd_one = []
    for i in indice:
        if np.gcd.reduce(i) == 1:
            indice_gcd_one.append(i)
    return np.array(indice_gcd_one)


def get_csl_twisted_graphenes(lim, filename, maxsigma=100, verbose=True):
    """
    get the geometric information of all the CS twisted graphene
    within a searching limitation

    Parameters
    __________
    lim : int
        control the number of generated referring points
    maxsigma : int
        maximum sigma

    Return
    __________
    list1 : numpy array
        sigma list
    list2 : numpy array
        angle list
    list3 : numpy array
        CNID areas
    list4 : numpy array
        num of atoms in supercell
    verbose : bool, optional
        If True, the function will print additional information.
        Default is True.
    """
    # mirror_plane_1
    xy_arrays = generate_arrays_x_y(1, 1, lim)
    indice = np.column_stack((xy_arrays, np.zeros(len(xy_arrays))))
    basis1 = np.column_stack(
        ([-1 / 2, 0, 1 / 2], [-1, 1 / 2, 1 / 2], [1, 1, 1]))
    P1 = np.dot(basis1, indice.T).T
    thetas1 = 2 * \
        np.arccos(np.dot(P1, [-1, 1 / 2, 1 / 2])
                  / norm(P1, axis=1) / norm([-1, 1 / 2, 1 / 2]))

    # mirror_plane_2
    xy_arrays = generate_arrays_x_y(1, 1, lim)
    indice = np.column_stack((xy_arrays, np.zeros(len(xy_arrays))))
    basis = np.column_stack(
        ([-1 / 2, 1 / 2, 0], [-1, 1 / 2, 1 / 2], [1, 1, 1]))
    P = np.dot(basis, indice.T).T
    thetas = 2 * \
        np.arccos(np.dot(P, [-1 / 2, 1 / 2, 0])
                  / norm(P, axis=1) / norm([-1 / 2, 1 / 2, 0]))
    P = np.vstack((P1, P))
    thetas = np.append(thetas1, thetas)
    sigmas = []
    for theta in thetas:
        sigmas.append(compute_sigma(
            np.array([0, 0, 1]), theta, filename, verbose=verbose))
    sigmas = np.around(sigmas)
    sigmas = np.array(sigmas, dtype=int)
    unique_sigmas = np.unique(sigmas)
    selected_thetas = []
    for i in unique_sigmas:
        selected_thetas.append(thetas[np.where(sigmas == i)[0][0]])
    selected_thetas = np.array(selected_thetas)
    sigmas = unique_sigmas
    thetas = selected_thetas[sigmas <= maxsigma]
    sigmas = sigmas[sigmas <= maxsigma]
    my_interface = core(filename,
                        filename, verbose=verbose)
    A_cnid = norm(
        np.cross(
            my_interface.lattice_1[:, 1],
            my_interface.lattice_1[:, 0]
        )) / sigmas
    anum = sigmas * 4
    return sigmas, thetas, A_cnid, anum


def get_Ps_sigmas_thetas(lim, axis, maxsigma=100000):
    """
    get the geometric information of all the symmetric tilt GBs
    for a rotation axis within a searching limitation
    arguments:
    lim --- control the number of generated referring points
    axis --- rotation axis
    maxsigma --- maximum sigma
    return:
    list of referring points, sigma list,
    angle list, original list of the 'in-plane' sigmas
    """
    if norm(np.cross(axis, [0, 0, 1])) < 1e-8:
        x = np.arange(2, 2 + lim, 1)
        indice = []
        for i in x:
            y_here = 1
            while y_here <= (i - 1):
                indice.append([i, y_here])
                y_here += 1
        indice = np.array(indice)
        indice_gcd_one = []
        for i in indice:
            if np.gcd.reduce(i) == 1:
                indice_gcd_one.append(i)
        indice_gcd_one = np.array(indice_gcd_one)
        xy_arrays = indice_gcd_one
        indice = np.column_stack((xy_arrays, np.zeros(len(xy_arrays))))
        xs = xy_arrays[:, 0]
        ys = xy_arrays[:, 1]
        basis = np.eye(3, 3)
        P = np.dot(basis, indice.T).T
        sigmas = np.array(xs**2 + ys**2)
        thetas = 2 * np.arctan(ys / xs)

    elif norm(np.cross(axis, [1, 1, 0])) < 1e-8:
        xy_arrays = generate_arrays_x_y(1, 1, lim)
        indice = np.column_stack((xy_arrays, np.zeros(len(xy_arrays))))
        xs = xy_arrays[:, 0]
        ys = xy_arrays[:, 1]
        basis = np.column_stack(([-1, 1, 0], [0, 0, 1], [1, 1, 0]))
        P = np.dot(basis, indice.T).T
        sq2 = np.sqrt(2)
        sigmas = np.sqrt((sq2 * xs)**2 + ys**2) * \
            np.sqrt((sq2 * ys / (2**(abs(ys % 2 - 1))))
                    ** 2 + (2**(ys % 2) * xs)**2) * sq2
        thetas = 2 * np.arctan(ys / sq2 / xs)

    elif norm(np.cross(axis, [1, 1, 1])) < 1e-8:
        # mirror_plane_1
        xy_arrays = generate_arrays_x_y(1, 0, lim)
        indice = np.column_stack((xy_arrays, np.zeros(len(xy_arrays))))
        xs = xy_arrays[:, 0]
        ys = xy_arrays[:, 1]
        basis1 = np.column_stack(
            ([-1 / 2, 0, 1 / 2], [-1, 1 / 2, 1 / 2], [1, 1, 1]))
        P1 = np.dot(basis1, indice.T).T
        basis2 = np.column_stack(
            (
                [-1 / 2, 0, 1 / 2] * 3 - [-1, 1 / 2, 1 / 2],
                [-1 / 2, 0, 1 / 2],
                [1, 1, 1]
            ))
        P2 = np.dot(basis2, indice.T).T
        P = np.vstack((P1, P2))
        thetas = 2 * \
            np.arccos(np.dot(P1, [-1, 1 / 2, 1 / 2])
                      / norm(P1, axis=1) / norm([-1, 1 / 2, 1 / 2]))
        sigmas = []
        for theta in thetas:
            sigmas.append(compute_sigma(
                np.array([1.0, 1.0, 1.0]), theta, maxsigma))
        sigmas = np.around(sigmas)
        sigmas = np.array(sigmas, dtype=int)
    else:
        raise RuntimeError(
            'error: only available for [001], [110] and [110] rotation axis')

    original_sigmas = 6 * norm(P, axis=1)**2
    sigmas = sigmas / (2**(abs(sigmas % 2 - 1)))
    sigmas = np.array(sigmas, dtype=int)
    sigmas = sigmas / (2**(abs(sigmas % 2 - 1)))
    sigmas = np.array(sigmas, dtype=int)
    return (
        P[np.argsort(original_sigmas, kind='stable')],
        sigmas[np.argsort(original_sigmas, kind='stable')],
        np.array(thetas[np.argsort(original_sigmas, kind='stable')]),
        original_sigmas[np.argsort(original_sigmas, kind='stable')]
    )
