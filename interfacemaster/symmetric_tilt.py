"""
Get the [001], [111], [110] symmetric tilt GBs;
Get the csl for bilayer twisted graphenes
"""

from numpy import *
from numpy.linalg import norm, det
from interfacemaster.cellcalc import MID, rot
from interfacemaster.interface_generator import core

def compute_sigma(axis, theta, filename = \
 'cif_files/Si_mp-149_conventional_standard.cif', maxsigma=10000):
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
    
    Returns
    __________
    sigma : int
    """
    print(theta/pi*180)
    R = rot(axis,theta)
    my_interface = core(filename,\
                        filename)
    my_interface.parse_limit(du = 1e-4, S  = 1e-4, sgm1=maxsigma, sgm2=maxsigma, dd = 1e-4)
    my_interface.search_fixed(R, exact=True, tol = 1e-3)
    return det(my_interface.U1)

def get_hkl(P, axis, tol = 1e-3):
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
    n = cross(axis, P)
    return MID(lattice=eye(3,3), n=n, tol=tol)

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
    Ps, sigmas, thetas, original_sigmas = get_Ps_sigmas_thetas(lim,axis)
    sampled_indices = where((sigmas < maxsigma))[0]
    Ps = Ps[sampled_indices]
    sigmas = sigmas[sampled_indices]
    thetas = thetas[sampled_indices]
    hkls = []
    for i in Ps:
        hkls.append(get_hkl(i, axis, tol =1e-4))
    hkls = array(hkls)
    sampled_indices = all(abs(hkls)<max_index, axis=1)
    sigmas = sigmas[sampled_indices]
    thetas = thetas[sampled_indices]
    hkls = hkls[sampled_indices]
    return thetas[argsort(thetas,kind='stable')], sigmas[argsort(thetas,kind='stable')], hkls[argsort(thetas,kind='stable')]
    
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
    x = arange(x_min, x_min + lim, 1)
    y = arange(y_min, y_min + lim, 1)
    indice = (stack(meshgrid(x, y)).T).reshape(len(x) *len(y), 2)
    indice_gcd_one = []
    for i in indice:
        if gcd.reduce(i) == 1:
            indice_gcd_one.append(i)
    return array(indice_gcd_one)

def get_csl_twisted_graphenes(lim, filename, maxsigma = 100):
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
    """
    #mirror_plane_1
    xy_arrays = generate_arrays_x_y(1,1,lim)
    indice = column_stack((xy_arrays,zeros(len(xy_arrays))))
    xs = xy_arrays[:,0]
    ys = xy_arrays[:,1]
    basis1 = column_stack(([-1/2, 0, 1/2], [-1, 1/2, 1/2], [1,1,1]))
    P1 = dot(basis1, indice.T).T
    thetas1 = 2*arccos(dot(P1,[-1, 1/2, 1/2])/norm(P1, axis=1)/norm([-1, 1/2, 1/2]))
    
    #mirror_plane_2
    xy_arrays = generate_arrays_x_y(1,1,lim)
    indice = column_stack((xy_arrays,zeros(len(xy_arrays))))
    xs = xy_arrays[:,0]
    ys = xy_arrays[:,1]
    basis = column_stack(([-1/2,1/2,0], [-1,1/2,1/2], [1,1,1]))
    P = dot(basis, indice.T).T
    thetas = 2*arccos(dot(P,[-1/2,1/2,0])/norm(P, axis=1)/norm([-1/2,1/2,0]))
    P = vstack((P1,P))
    thetas = append(thetas1, thetas)
    sigmas = []
    for i in range(len(thetas)):
        sigmas.append(compute_sigma(array([0,0,1]), thetas[i], filename))
    sigmas = around(sigmas)
    sigmas = array(sigmas,dtype = int)
    unique_sigmas = unique(sigmas)
    selected_thetas = []
    for i in unique_sigmas:
        selected_thetas.append(thetas[where(sigmas==i)[0][0]])
    selected_thetas = array(selected_thetas)
    sigmas = unique_sigmas
    thetas = selected_thetas[sigmas <= maxsigma]
    sigmas = sigmas[sigmas <= maxsigma]
    my_interface = core(filename,\
                        filename)
    A_cnid = norm(cross(my_interface.lattice_1[:,1], my_interface.lattice_1[:,0])) / sigmas
    anum = sigmas * 4
    return sigmas, thetas, A_cnid, anum

def get_Ps_sigmas_thetas(lim, axis, maxsigma = 100000):
    """
    for a rotation axis, get the geometric information of all the symmetric tilt GBs
    within a searching limitation
    arguments:
    lim --- control the number of generated referring points
    axis --- rotation axis
    maxsigma --- maximum sigma
    return:
    list of referring points, sigma list, angle list, original list of the 'in-plane' sigmas
    """
    if norm(cross(axis, [0,0,1])) < 1e-8:
        x = arange(2, 2 + lim, 1)
        indice = []
        for i in x:
            y_here = 1
            while y_here <= (i-1):
                indice.append([i, y_here])
                y_here+=1
        indice = array(indice)
        indice_gcd_one = []
        for i in indice:
            if gcd.reduce(i) == 1:
                indice_gcd_one.append(i)
        indice_gcd_one = array(indice_gcd_one)
        xy_arrays = indice_gcd_one
        indice = column_stack((xy_arrays,zeros(len(xy_arrays))))
        xs = xy_arrays[:,0]
        ys = xy_arrays[:,1]
        basis = eye(3,3)
        P = dot(basis, indice.T).T
        sigmas = array(xs**2 + ys**2)
        thetas = 2*arctan(ys/xs)
        
    elif norm(cross(axis, [1,1,0])) < 1e-8:
        xy_arrays = generate_arrays_x_y(1,1,lim)
        indice = column_stack((xy_arrays,zeros(len(xy_arrays))))
        xs = xy_arrays[:,0]
        ys = xy_arrays[:,1]
        basis = column_stack(([-1,1,0], [0,0,1], [1,1,0]))
        P = dot(basis, indice.T).T
        sq2 = sqrt(2)
        sigmas = sqrt( (sq2*xs)**2 + ys**2 ) * \
        sqrt( (sq2*ys / (2**(abs(ys%2-1))))**2 + (2**(ys%2)*xs)**2) * sq2
        thetas = 2*arctan(ys/sq2/xs)
        
    elif norm(cross(axis, [1,1,1])) < 1e-8:
        #mirror_plane_1
        xy_arrays = generate_arrays_x_y(1,0,lim)
        indice = column_stack((xy_arrays,zeros(len(xy_arrays))))
        xs = xy_arrays[:,0]
        ys = xy_arrays[:,1]
        basis1 = column_stack(([-1/2, 0, 1/2], [-1, 1/2, 1/2], [1,1,1]))
        P1 = dot(basis1, indice.T).T
        basis2 = column_stack(([-1/2, 0, 1/2] * 3 - [-1, 1/2, 1/2], [-1/2, 0, 1/2], [1, 1, 1]))
        P2 = dot(basis2, indice.T).T
        P = vstack((P1, P2))
        thetas = 2*arccos(dot(P1,[-1, 1/2, 1/2])/norm(P1, axis=1)/norm([-1, 1/2, 1/2]))
        sigmas = []
        for i in range(len(thetas)):
            sigmas.append(compute_sigma(array([1.0,1.0,1.0]), thetas[i], maxsigma))
        sigmas = around(sigmas)
        sigmas = array(sigmas,dtype = int)
    else:
          raise RuntimeError('error: only available for [001], [110] and [110] rotation axis')
    
    original_sigmas = 6*norm(P,axis=1)**2
    sigmas = sigmas/(2**(abs(sigmas%2-1)))
    sigmas = array(sigmas,dtype = int)
    sigmas = sigmas/(2**(abs(sigmas%2-1)))
    sigmas = array(sigmas,dtype = int)
    return P[argsort(original_sigmas, kind='stable')], sigmas[argsort(original_sigmas, kind='stable')], \
array(thetas[argsort(original_sigmas, kind='stable')]), original_sigmas[argsort(original_sigmas, kind='stable')]
