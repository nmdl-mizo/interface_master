"""
Get the [001], [111], [110] symmetric tilt GBs
"""

from numpy import *
from numpy.linalg import inv, norm, det
from cellcalc import MID, rot
from interfacemaster.interface_generator import core

def compute_sigma(axis, theta, maxsigma=100):
    print(theta/pi*180)
    R = rot(axis,theta)
    my_interface = core('cif_files/Si_mp-149_conventional_standard.cif',\
                        'cif_files/Si_mp-149_conventional_standard.cif')
    my_interface.parse_limit(du = 1e-4, S  = 1e-4, sgm1=maxsigma, sgm2=maxsigma, dd = 1e-4)
    my_interface.search_fixed(R, exact=True, tol = 1e-3)
    return det(my_interface.U1)

def get_hkl(P, axis, tol = 1e-3):
    n = cross(axis, P)
    return MID(lattice=eye(3,3), n=n, tol=tol)

def sample_STGB(axis, lim, maxsigma, max_index):
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
    x = arange(x_min, x_min + lim, 1)
    y = arange(y_min, y_min + lim, 1)
    indice = (stack(meshgrid(x, y)).T).reshape(len(x) *len(y), 2)
    indice_gcd_one = []
    for i in indice:
        if gcd.reduce(i) == 1:
            indice_gcd_one.append(i)
    return array(indice_gcd_one)

def get_Ps_sigmas_thetas(lim, axis, maxsigma = 100000):
    
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
