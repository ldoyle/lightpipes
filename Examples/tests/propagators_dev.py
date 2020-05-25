# -*- coding: utf-8 -*-
"""

Testing of Steps() reimplementation using scipy.

Created on Fri May 22 22:06:49 2020

@author: Leonard Doyle
"""


import numpy as _np

from scipy.optimize import least_squares #TODO hide from public
from scipy.sparse import coo_matrix

from LightPipes.field import Field
from LightPipes import tictoc


def _StepsScipy(z, nstep, refr, Fin):
    """Right now this is just a test and collection of code from
    https://scipy-cookbook.readthedocs.io/items/discrete_bvp.html
    which is not functional for Lightpipes!
    Also, its really really slow, so possibly not useful at all.
    """
    """
    Fout = Steps(z, nstep, refr, Fin)
                 
    :ref:`Propagates the field a distance, nstep x z, in nstep steps in a
    medium with a complex refractive index stored in the
    square array refr. <Steps>`

    Args::
    
        z: propagation distance per step
        nstep: number of steps
        refr: refractive index (N x N array of complex numbers) as n=n_r+j kappa
        Fin: input field
        
    Returns::
      
        Fout: ouput field (N x N square array of complex numbers).
        
    Example:
    
    :ref:`Propagation through a lens like medium <lenslikemedium>`
    
    """
    if Fin._curvature != 0.0:
        raise ValueError('Cannot operate on spherical coords.'
                         + 'Use Convert() first')
    Fout = Field.shallowcopy(Fin)
    N = Fout.N
    lam = Fout.lam
    size = Fout.siz
    
    Pi = _np.pi
    k0 = 2.*Pi/lam
    dz = z
    
    dx = size/(N-1.) #dx
    dy = dx
    
    
    AA= -10./dz/nstep #/* total absorption */
    band_pow=2.   #/* profile of the absorption border, 2=quadratic*/
    x = _np.arange(N)
    xx,yy = _np.meshgrid(x,x)
    # c_absorb_x[mask] = 1j* (AA*K)*_np.power(iii/i_left, band_pow)
    
    n_c = _np.asarray(refr).astype(complex, copy=True)
    n_avg = _np.mean(refr) #real and im separately, obviously, avg over X and Y
    k_avg = k0 * n_avg
    
    # n_c += 1j* _np.power(xx/(N/20), 8)
    # n_c += 1j* _np.power((N-xx)/(N/20), 8)
    # n_c += 1j* _np.power(yy/(N/20), 8)
    # n_c += 1j* _np.power((N-yy)/(N/20), 8)
    
    A = n_c**2
    A -= n_avg**2 #n^2-n_avg^2, not (n-n_avg)^2 !
    A *= k0**2
    
    a = 1/dx**2
    b = 1/dy**2
    # b = a #force square grid for now, makes functions a little easier to write
    
    V0 = Fin.field
    
    def fun(V):
        #given Vij, calculate Vij out, to find stationary solution
        #just use global params as long as possible, even though ugly
        
        assert V.size == N*N*2 #N^2 as (real,im) floats
        V = V.view(complex) #treat 2 consecutive floats each as (real, im)
        V = V.reshape((N, N))
        Vpad = _np.zeros((N+2,N+2),dtype=complex) #pad to have NxN output,
        # implicit data allocation is also good as it avoids
        # headache about accidentally changing the input data reference
        Vpad[1:-1,1:-1] = V
        V = Vpad
        vij = V[1:-1,1:-1] #makes it the original V again
        vij1 = V[2:,1:-1]   #V[j+1,i]
        vij_1 = V[:-2,1:-1] #V[j-1,i]
        vi1j = V[1:-1,2:]   #V[j,i+1]
        vi_1j = V[1:-1,:-2] #V[j,i-1]
        
        vv = a*vi1j + a*vi_1j + b*vij1 + b*vij_1
        vv += (-2*a-2*b+(2j*k_avg/dz)+A)*vij
        vv -= 2j*k_avg/dz*V0
        ret = vv.ravel().view(float)
        assert ret.size == N*N*2
        return ret
    
    #initial guess is old field
    v0 = Fin.field.ravel().view(float) #treat real and im as 2 entries to array
    v0 = v0.copy()
    # v0[:] = 0.0
    res_1 = least_squares(fun, v0, verbose=1)
    field = res_1.x.view(complex)
    field = field.reshape((N,N))
    
    Fout.field = field # * _np.exp(1j*k_avg*dz)
    return Fout, res_1


def _StepsScipyJac(z, nstep, refr, Fin):
    """Right now this is just a test and collection of code from
    https://scipy-cookbook.readthedocs.io/items/discrete_bvp.html
    which is not functional for Lightpipes!
    Also, its really really slow, so possibly not useful at all.
    """
    """
    Fout = Steps(z, nstep, refr, Fin)
                 
    :ref:`Propagates the field a distance, nstep x z, in nstep steps in a
    medium with a complex refractive index stored in the
    square array refr. <Steps>`

    Args::
    
        z: propagation distance per step
        nstep: number of steps
        refr: refractive index (N x N array of complex numbers) as n=n_r+j kappa
        Fin: input field
        
    Returns::
      
        Fout: ouput field (N x N square array of complex numbers).
        
    Example:
    
    :ref:`Propagation through a lens like medium <lenslikemedium>`
    
    """
    if Fin._curvature != 0.0:
        raise ValueError('Cannot operate on spherical coords.'
                         + 'Use Convert() first')
    Fout = Field.shallowcopy(Fin)
    N = Fout.N
    lam = Fout.lam
    size = Fout.siz
    
    Pi = _np.pi
    k0 = 2.*Pi/lam
    dz = z
    
    dx = size/(N-1.) #dx
    dy = dx
    
    c = 1
    
    n_c = _np.asarray(refr).astype(complex, copy=True)
    n_avg = _np.mean(refr) #real and im separately, obviously, avg over X and Y
    k_avg = k0 * n_avg
    
    # n_c += 1j* _np.power(xx/(N/20), 8)
    # n_c += 1j* _np.power((N-xx)/(N/20), 8)
    # n_c += 1j* _np.power(yy/(N/20), 8)
    # n_c += 1j* _np.power((N-yy)/(N/20), 8)
    
    A = n_c**2
    A -= n_avg**2 #n^2-n_avg^2, not (n-n_avg)^2 !
    A *= k0**2
    
    a = 1/dx**2
    b = 1/dy**2
    b = a #force square grid for now, makes functions a little easier to write
    
    V0 = Fin.field
    
    def f(v, ):
        ret = 1/a*(2j*k_avg/dz+A)*v-2j*k_avg/(a*dz)*V0
        return ret
    
    def f_prime(v):
        ret = 1/a*(2j*k_avg/dz+A)
        return ret
    
    def fun(V, n, f, f_prime, **kwargs):
        assert V.size == N*N*2 #N^2 as (real,im) floats
        V = V.view(complex) #treat 2 consecutive floats each as (real, im)
        V = V.reshape((N, N))
        Vpad = _np.zeros((N+2,N+2),dtype=complex) #pad to have NxN output,
        # implicit data allocation is also good as it avoids
        # headache about accidentally changing the input data reference
        Vpad[1:-1,1:-1] = V
        V = Vpad
        vij = V[1:-1,1:-1] #makes it the original V again
        vij1 = V[2:,1:-1]   #V[j+1,i]
        vij_1 = V[:-2,1:-1] #V[j-1,i]
        vi1j = V[1:-1,2:]   #V[j,i+1]
        vi_1j = V[1:-1,:-2] #V[j,i-1]
        vv = vi1j + vi_1j + vij1 + vij_1
        vv -= 4*vij
        vv += f(V)
        ret = vv.ravel().view(float)
        
        # v = _np.zeros((n + 2, n + 2))
        # u = u.reshape((n, n))
        # v[1:-1, 1:-1] = u
        # y = v[:-2, 1:-1] + v[2:, 1:-1] + v[1:-1, :-2] + v[1:-1, 2:] - 4 * u + c * f(u)
        return ret

    def compute_jac_indices(n):
        i = _np.arange(n)
        jj, ii = _np.meshgrid(i, i)
    
        ii = ii.ravel()
        jj = jj.ravel()
    
        ij = _np.arange(n**2) 
    
        jac_rows = [ij] #diagonal, i.e. self contrib
        jac_cols = [ij]
    
        mask = ii > 0
        ij_mask = ij[mask]
        jac_rows.append(ij_mask) #rows go along j
        jac_cols.append(ij_mask - n)
    
        mask = ii < n - 1
        ij_mask = ij[mask]
        jac_rows.append(ij_mask)
        jac_cols.append(ij_mask + n)
    
        mask = jj > 0
        ij_mask = ij[mask]
        jac_rows.append(ij_mask)
        jac_cols.append(ij_mask - 1)
    
        mask = jj < n - 1
        ij_mask = ij[mask]
        jac_rows.append(ij_mask)
        jac_cols.append(ij_mask + 1)
    
        return _np.hstack(jac_rows), _np.hstack(jac_cols)
    jac_rows, jac_cols = compute_jac_indices(N)
    u0 = _np.ones(N**2) * 0.5
    # u0 = Fin.field.ravel() #initial guess is old field
    
    def jac(u, n, f, f_prime, jac_rows=None, jac_cols=None):
        jac_values = _np.ones_like(jac_cols, dtype=float)
        jac_values[:n**2] = -4 + f_prime(u)
        return coo_matrix((jac_values, (jac_rows, jac_cols)),
                          shape=(n**2, n**2))
    
    res_1 = least_squares(fun, u0.real, jac=jac, gtol=1e-3,
                          args=(N, f, f_prime),
                          kwargs={'jac_rows': jac_rows,
                                  'jac_cols': jac_cols},
                          verbose=0)
    # print(res_1)
    Fout = res_1.x.reshape((N, N))
    return Fout






