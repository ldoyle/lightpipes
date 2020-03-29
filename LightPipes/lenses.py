# -*- coding: utf-8 -*-

import numpy as _np

from .field import Field


def Axicon(phi, n1, x_shift, y_shift, Fin):
    """
    Fout = Axicon(phi, n1, x_shift, y_shift, Fin)
   
    :ref:`Propagates the field through an axicon. <Axicon>`

    Args::
    
        phi: top angle of the axicon in radians
        n1: refractive index of the axicon material
        x_shift, y_shift: shift from the center
        Fin: input field
        
    Returns::
      
        Fout: output field (N x N square array of complex numbers).
            
    Example:
    
    :ref:`Bessel beam with axicon <BesselBeam>`

    """
    Fout = Field.copy(Fin)
    k = 2*_np.pi/Fout.lam
    theta = _np.arcsin(n1*_np.cos(phi/2)+phi/2-_np.pi/2)
    Ktheta = k * theta
    yy, xx = Fout.mgrid_cartesian
    xx -= x_shift
    yy -= y_shift
    fi = -Ktheta*_np.sqrt(xx**2+yy**2)
    Fout.field *= _np.exp(1j*fi)
    return Fout


def Lens(f, x_shift, y_shift, Fin):
    """
    Fout = Lens(f, x_shift, y_shift, Fin)

    :ref:`Propagates the field through an ideal, thin lens. <Lens>`

    It adds a phase given by:
    :math:`F_{out}(x,y)=e^{-j\\frac{2\\pi}{\\lambda}\\left(\\frac{(x-x_{shift})^2+(y-y_{shift})^2}{2f}\\right)}F_{in}(x,y)`
        
    Args::
    
        f: focal length
        x_shift, y_shift: shift from center
        Fin: input field
        
    Returns::
    
        Fout: output field (N x N square array of complex numbers).

    """
    Fout = Field.copy(Fin)
    k = 2*_np.pi/Fout.lam
    yy, xx = Fout.mgrid_cartesian
    xx -= x_shift
    yy -= y_shift
    fi = -k*(xx**2+yy**2)/(2*f)
    Fout.field *= _np.exp(1j * fi)
    return Fout


def LensFarfield(f, Fin):
    """
    Use a direct FFT approach to calculate the far field of the input field.
    Given the focal length f, the correct scaling is applied and the
    output field will have it's values for size and dx correctly set.

    Parameters
    ----------
    f : double
        Focal length in meters/ global units
    Fin : lp.Field
        The input field.

    Returns
    -------
    The output field.

    """
    """
        The focus(="far field") is related to the nearfield phase and intensity
        via the Fourier transform. Applying the correct scalings we can immediately
        calculate the focus of a measured wavefront.
        Maths relations: [e.g. Tyson Fourier Optics]
        
        lam     [m] = wavelength lambda
        f_L     [m] = focal length of lens/focusing optic
        N       [1] = grid size NxN, assume square for now
        L       [m] = size of FOV in the near field
        dx      [m] = L/N grid spacing in near field
        L'      [m] = size of FOV in focal plane
        dx'     [m] = grid spacing in focal plane
        
        lam * f_L = dx' * L
                  = dx * L'
                  = dx * dx' * N
        
        given: N, dx', lam, f_L
        lemma: L' = N * dx'
        required: L, dx
        --> L = lam * f_L / dx'
        --> dx = L / N = lam * f_L / (N * dx') = lam * f_L / L' 

    """
    Fout = Field.copy(Fin)
    dx = Fout.dx
    lam = Fout.lam
    L_prime = lam * f / dx
    focusfield = _np.fft.fftshift(_np.fft.fft2(Fout.field))
    Fout.field = focusfield
    Fout.siz = L_prime
    return Fout