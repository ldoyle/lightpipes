# -*- coding: utf-8 -*-


import numpy as _np
from scipy.special import fresnel as _fresnel

from .field import Field


def Fresnel(z, Fin):
    """
    Fout = Fresnel(z, Fin)

    :ref:`Propagates the field using a convolution method. <Fresnel>`

    Args::
    
        z: propagation distance
        Fin: input field
        
    Returns::
     
        Fout: output field (N x N square array of complex numbers).
            
    Example:
    
    :ref:`Two holes interferometer <Young>`

    """
    Fout = Field.copy(Fin)
    Fout.field = _field_Fresnel(z, Fout.field, Fout.dx, Fout.lam)
    return Fout



def _field_Fresnel(z, field, dx, lam):
    """
    Separated the "math" logic out so that only standard and numpy types
    are used.
    
    Parameters
    ----------
    z : float
        Propagation distance.
    field : ndarray
        2d complex numpy array (NxN) of the field.
    dx : float
        In units of sim (usually [m]), spacing of grid points in field.
    lam : float
        Wavelength lambda in sim units (usually [m]).

    Returns
    -------
    ndarray (2d, NxN, complex)
        The propagated field.

    """
    
    """Major differences to Cpp based LP version:
        - dx =siz/N instead of dx=siz/(N-1), more consistent with physics 
            and rest of LP package
        - fftw uses no normalization, numpy uses 1/N on ifft -> omitted
            factor of 1/(2*N)**2 in final calc before return
        - bug in Cpp version: did not touch top row/col, now we extract one
            more row/col to fill entire field. No errors noticed with the new
            method so far
    """
    
    N = field.shape[0] #assert square
    
    kz = 2*_np.pi/lam*z
    cokz = _np.cos(kz)
    sikz = _np.sin(kz)
    
    legacy = True #switch on to numerically compare oldLP/new results
    if legacy:
        siz = N*dx
        dx = siz/(N-1) #like old Cpp code, even though unlogical

    in_outF = _np.zeros((2*N, 2*N),dtype=complex)
    in_outK = _np.zeros((2*N, 2*N),dtype=complex)
    
    No2 = int(N/2) #"N over 2"
    """The following section contains a lot of uses which boil down to
    2*No2. For even N, this is N. For odd N, this is NOT redundant:
        2*No2 is N-1 for odd N, therefore sampling an even subset of the
        field instead of the whole field. Necessary for symmetry of first
        step involving Fresnel integral calc.
    """
    
    RR = _np.sqrt(1/(2*lam*z))*dx*2
    io = _np.arange(0, (2*No2)+1) #add one extra to stride fresnel integrals
    R1 = RR*(io - No2)
    fs, fc = _fresnel(R1)
    fss = _np.outer(fs, fs) #    out[i, j] = a[i] * b[j]
    fsc = _np.outer(fs, fc)
    fcs = _np.outer(fc, fs)
    fcc = _np.outer(fc, fc)

    temp_re = (fsc[1:, 1:] #s[i+1]c[j+1]
               + fcs[1:, 1:] #c[+1]s[+1]
               - fsc[:-1, 1:] #-scp [p=+1, without letter =+0]
               - fcs[:-1, 1:] #-csp
               - fsc[1:, :-1] #-spc
               - fcs[1:, :-1] #-cps
               + fsc[:-1, :-1] #sc
               + fcs[:-1, :-1]) #cs
    temp_im = (-fcc[1:, 1:] #-cpcp
               + fss[1:, 1:] # +spsp
               + fcc[:-1, 1:] # +ccp
               - fss[:-1, 1:] # -ssp
               + fcc[1:, :-1] # +cpc
               - fss[1:, :-1] # -sps
               - fcc[:-1, :-1] # -cc
               + fss[:-1, :-1])# +ss
    
    
    ii = _np.ones((2*No2),dtype=float)
    ii[1::2] = -1
    iiij = _np.outer(ii, ii)
    
    temp_K = 0.5*iiij*(temp_re + 1j* temp_im)
    
    in_outK[(N-No2):(N+No2), (N-No2):(N+No2)] = temp_K
    in_outF[(N-No2):(N+No2), (N-No2):(N+No2)] = (iiij
                        * field[(N-2*No2):N,(N-2*No2):N]) #cutting off field!
    
    in_outK = _np.fft.fft2(in_outK)
    in_outF = _np.fft.fft2(in_outF)
    
    
    ii = _np.ones((2*N),dtype=float)
    ii[1::2] = -1
    sign_pattern = _np.outer(ii, ii)
        
    cc = (in_outK.real * in_outF.real
          - in_outK.imag * in_outF.imag)
    cci = (in_outK.real * in_outF.imag
            + in_outF.real * in_outK.imag)
    in_outF[:,:] = sign_pattern * (cc + 1j * cci)
    
    in_outF = _np.fft.ifft2(in_outF)
    
    ii = _np.ones(N,dtype=float)
    ii[1::2] = -1
    iiij = _np.outer(ii, ii)
    
    Ftemp = (in_outF[No2:N+No2, No2:N+No2]
        - in_outF[No2-1:N+No2-1, No2:N+No2]
        + in_outF[No2-1:N+No2-1, No2-1:N+No2-1]
        - in_outF[No2:N+No2, No2-1:N+No2-1])
    FR = Ftemp.real
    FI = Ftemp.imag
    
    temp_re = 0.25*iiij*(FR * cokz - FI * sikz)
    temp_im = 0.25*iiij*(FI * cokz + FR * sikz)
    
    field[:,:] = (temp_re + 1j * temp_im)

    return field



