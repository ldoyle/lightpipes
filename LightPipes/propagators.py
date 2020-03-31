# -*- coding: utf-8 -*-

"""User can decide to disable dependency here. This will slow down the FFT,
but otherwise numpy.fft is a drop-in replacement so far."""
USE_PYFFTW = True
if USE_PYFFTW:
    try:
        from pyfftw.interfaces.numpy_fft import fft2 as _fft2
        from pyfftw.interfaces.numpy_fft import ifft2 as _ifft2
    except ImportError:
        import warnings
        warnings.warn('LightPipes: Cannot import pyfftw,'
                      + ' falling back to numpy.fft')
        from numpy.fft import fft2 as _fft2
        from numpy.fft import ifft2 as _ifft2
else:
    from numpy.fft import fft2 as _fft2
    from numpy.fft import ifft2 as _ifft2

import numpy as _np
from scipy.special import fresnel as _fresnel

from .field import Field
from . import tictoc

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
    tictoc.tic()
    Fout = Field.copy(Fin)
    tictoc.printtoc(prefix='copy')
    with tictoc.printtimer('field_Fresnel'):
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
    
    """ *************************************************************
    Major differences to Cpp based LP version:
        - dx =siz/N instead of dx=siz/(N-1), more consistent with physics 
            and rest of LP package
        - fftw DLL uses no normalization, numpy uses 1/N on ifft -> omitted
            factor of 1/(2*N)**2 in final calc before return
        - bug in Cpp version: did not touch top row/col, now we extract one
            more row/col to fill entire field. No errors noticed with the new
            method so far
    ************************************************************* """
    
    # Timing values recorded as min/max over 5 runs on i7-7500U laptop
    # with Young.py and N=2000
    # generally, one has to wait a few seconds between each run or else
    # they get longer and longer, probably due to GC or CPU getting hot
    # method1 = True #5.49-6.71s -> 1.4x slower
    method1 = False #3.97-4.82s
    
    N = field.shape[0] #assert square
    
    kz = 2*_np.pi/lam*z
    cokz = _np.cos(kz)
    sikz = _np.sin(kz)
    
    legacy = True #switch on to numerically compare oldLP/new results
    if legacy:
        siz = N*dx
        dx = siz/(N-1) #like old Cpp code, even though unlogical

    No2 = int(N/2) #"N over 2"
    """The following section contains a lot of uses which boil down to
    2*No2. For even N, this is N. For odd N, this is NOT redundant:
        2*No2 is N-1 for odd N, therefore sampling an even subset of the
        field instead of the whole field. Necessary for symmetry of first
        step involving Fresnel integral calc.
    """

    with tictoc.printtimer('array_2N create'):
        in_outF = _np.zeros((2*N, 2*N),dtype=complex)
        in_outK = _np.zeros((2*N, 2*N),dtype=complex)
    
    with tictoc.printtimer('sign_pattern'):
        """Create the sign-flip pattern for largest use case and 
        reference smaller grids with a view to the same data for
        memory saving."""
        ii2N = _np.ones((2*N),dtype=float)
        ii2N[1::2] = -1
        iiij2N = _np.outer(ii2N, ii2N)
        iiij2No2 = iiij2N[:2*No2,:2*No2] #slice to size used below
        iiijN = iiij2N[:N, :N]
    
    with tictoc.printtimer('frensel int'): #0.05-0.06s
        RR = _np.sqrt(1/(2*lam*z))*dx*2
        io = _np.arange(0, (2*No2)+1) #add one extra to stride fresnel integrals
        R1 = RR*(io - No2)
        fs, fc = _fresnel(R1)
        fss = _np.outer(fs, fs) #    out[i, j] = a[i] * b[j]
        fsc = _np.outer(fs, fc)
        fcs = _np.outer(fc, fs)
        fcc = _np.outer(fc, fc)
    
    with tictoc.printtimer('frensel apply'):
        # method1 = False
        if method1: #~0.26-0.33s
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
        else: #0.14-0.16s
            """Old notation (0.26-0.33s):
                temp_re = (a + b + c - d + ...)
                # numpy func add takes 2 operands A, B only
                # -> each operation needs to create a new temporary array, i.e.
                # ((((a+b)+c)+d)+...)
                # since python does not optimize to += here (at least is seems)
            New notation (0.14-0.16s):
                temp_re = (a + b) #operation with 2 operands
                temp_re += c
                temp_re -= d
                ...
            Wrong notation:
                temp_re = a #copy reference to array a
                temp_re += b
                ...
                # changing `a` in-place, re-using `a` will give corrupted
                # result
            """
            temp_re = (fsc[1:, 1:] #s[i+1]c[j+1]
                       + fcs[1:, 1:]) #c[+1]s[+1]
            temp_re -= fsc[:-1, 1:] #-scp [p=+1, without letter =+0]
            temp_re -= fcs[:-1, 1:] #-csp
            temp_re -= fsc[1:, :-1] #-spc
            temp_re -= fcs[1:, :-1] #-cps
            temp_re += fsc[:-1, :-1] #sc
            temp_re += fcs[:-1, :-1] #cs
            
            temp_im = (-fcc[1:, 1:] #-cpcp
                       + fss[1:, 1:]) # +spsp
            temp_im += fcc[:-1, 1:] # +ccp
            temp_im -= fss[:-1, 1:] # -ssp
            temp_im += fcc[1:, :-1] # +cpc
            temp_im -= fss[1:, :-1] # -sps
            temp_im -= fcc[:-1, :-1] # -cc
            temp_im += fss[:-1, :-1]# +ss
    
    with tictoc.printtimer('iiij1'): 
        # method1 = False
        if method1:#0.23-0.32s
            ii = _np.ones((2*No2),dtype=float)
            ii[1::2] = -1
            iiij = _np.outer(ii, ii)
            
            temp_K = 0.5*iiij*(temp_re + 1j* temp_im)
            
            in_outK[(N-No2):(N+No2), (N-No2):(N+No2)] = temp_K
            in_outF[(N-No2):(N+No2), (N-No2):(N+No2)] = (iiij
                                * field[(N-2*No2):N,(N-2*No2):N]) #cutting off field!
        else:#0.14-0.16s
            temp_K = 1j * temp_im # a * b creates copy
            temp_K += temp_re
            temp_K *= iiij2No2
            temp_K *= 0.5
            in_outK[(N-No2):(N+No2), (N-No2):(N+No2)] = temp_K
            
            in_outF[(N-No2):(N+No2), (N-No2):(N+No2)] \
                = field[(N-2*No2):N,(N-2*No2):N] #cutting off field!
            in_outF[(N-No2):(N+No2), (N-No2):(N+No2)] *= iiij2No2

    tictoc.tic()
    in_outK = _fft2(in_outK)
    in_outF = _fft2(in_outF)
    tictoc.printtoc(prefix='fft 1: ')
    
    tictoc.tic()
    ii = _np.ones((2*N),dtype=float)
    ii[1::2] = -1
    sign_pattern = _np.outer(ii, ii)
    # in_outF *= sign_pattern #TODO debug only!!
    tictoc.printtoc(prefix='sign_pattern')
    
    # tictoc.tic()
    # in_outF = _np.fft.fftshift(_np.fft.ifftshift(in_outF)) #TODO debug only
    # tictoc.printtoc(prefix='fftshift')
    
    with tictoc.printtimer('sign pattern apply'):
        # method1 = False
        if method1: #0.93-1.11s
            #just equals a complex multiplication:
            # K = kr + j ki, F = fr + j fi
            # K*F = kr*fr + j^2*ki*fi +j (kr*fi + ki*fr)
            #     = kr*fr - ki*fi +j (kr*fi + ki*fr)
            cc = (in_outK.real * in_outF.real
                  - in_outK.imag * in_outF.imag)
            cci = (in_outK.real * in_outF.imag
                    + in_outF.real * in_outK.imag)
            in_outF[:,:] = sign_pattern * (cc + 1j * cci)
        else: #0.06-0.10s
            in_outF *= in_outK
            in_outF *= iiij2N
            # cc = (in_outK.real * in_outF.real
            #       - in_outK.imag * in_outF.imag)
            # cci = (in_outK.real * in_outF.imag
            #         + in_outF.real * in_outK.imag)
            # in_outF[:,:] = sign_pattern * (cc + 1j * cci)
            
    with tictoc.printtimer('ifft'):
        in_outF = _ifft2(in_outF)
    
    with tictoc.printtimer('field slice'):
        # method1 = False
        if method1: #0.38-0.48s
            ii = _np.ones(N,dtype=float)
            ii[1::2] = -1
            iiij = _np.outer(ii, ii)
            
            Ftemp = (in_outF[No2:N+No2, No2:N+No2]
                - in_outF[No2-1:N+No2-1, No2:N+No2]
                + in_outF[No2-1:N+No2-1, No2-1:N+No2-1]
                - in_outF[No2:N+No2, No2-1:N+No2-1])
            FR = Ftemp.real
            FI = Ftemp.imag
            
            #TODO check normalization if USE_PYFFTW
            temp_re = 0.25*iiij*(FR * cokz - FI * sikz)
            temp_im = 0.25*iiij*(FI * cokz + FR * sikz)
            
            field[:,:] = (temp_re + 1j * temp_im)
        else: #0.08-0.11s
            Ftemp = (in_outF[No2:N+No2, No2:N+No2]
                     - in_outF[No2-1:N+No2-1, No2:N+No2])
            Ftemp += in_outF[No2-1:N+No2-1, No2-1:N+No2-1]
            Ftemp -= in_outF[No2:N+No2, No2-1:N+No2-1]
            comp = complex(cokz, sikz)
            Ftemp *= 0.25 * comp
            Ftemp *= iiijN
            # field[:,:] = Ftemp
            field = Ftemp #reassign without data copy
            #TODO check normalization if USE_PYFFTW
            
    return field



