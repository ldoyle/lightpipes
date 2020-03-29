# -*- coding: utf-8 -*-

import numpy as _np

from .field import Field

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