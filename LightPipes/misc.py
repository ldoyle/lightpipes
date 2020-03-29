# -*- coding: utf-8 -*-

import numpy as _np

from .field import Field


def Gain(Isat, alpha0, Lgain, Fin):
    """
    Fout = Gain(Isat, alpha0, Lgain, Fin)

    :ref:`Propagates the field through a thin saturable gain sheet. <Gain>`
        
        :math:`F_{out}(x,y) = F_{in}(x,y) e^{\\alpha L_{gain}}`, with
        :math:`\\alpha = \\dfrac{\\alpha_0}{1 + {2 I(x,y)}/{I_{sat}}}`.
         
        :math:`2\\alpha L_{gain}` is the net round-trip intensity gain. 
        :math:`\\alpha_0` is the small-signal intensity gain and 
        :math:`I_{sat}` is the saturation intensity of the gain medium 
        with a length :math:`L_{gain}`.
        
        The intensity must be doubled because of the left- and right 
        propagating fields in a normal resonator. (If only one field is propagating in one direction (ring 
        laser) you should double :math:`I_{sat}` as well to remove the factor 2 in the denominator).

        The gain sheet should be at one of the mirrors of a (un)stable laser resonator.
        
        See: Rensch and Chester (1973).
        
    Args::
    
        Isat: saturation intensity
        alpha0: small signal gain
        Lgain: length of the gain sheet
        Fin: input field
        
    Returns::
     
        Fout: output field (N x N square array of complex numbers).

    Example:
    
    :ref:`Unstable resonator <Unstab>`

    """
    Fout = Field.copy(Fin)
    Ii = _np.abs(Fout.field)**2
    """direct port from Cpp:
    if Isat == 0.0:
        Io = Ii
    else:
        Io = Ii * _np.exp(alpha0*Lgain/(1+2*Ii/Isat))
    ampl = _np.sqrt(Io/Ii)
    ampl[Ii==0.0] = 0.0 #replace nan from 0-division
    Fout.field *= ampl
    
    However this can be simplified since multiplying and dividing by Ii
    is redundant and once removed the div by zero is gone, too.
    And if Isat==0.0, Io/Ii=1 everywhere -> gain=1, just skip.
    Finally, sqrt can be put in exp() as 1/2
    """
    if Isat == 0.0:
        ampl = 1
    else:
        ampl = _np.exp(1/2*alpha0*Lgain/(1+2*Ii/Isat))
    Fout.field *= ampl
    return Fout


def Tilt(tx, ty, Fin):
    """
    Fout = Tilt(tx, ty, Fin)

    :ref:`Tilts the field. <Tilt>`

    Args::
    
        tx, ty: tilt in radians
        Fin: input field
        
    Returns::
    
        Fout: output field (N x N square array of complex numbers).

    """
    Fout = Field.copy(Fin)
    yy, xx = Fout.mgrid_cartesian
    k = 2*_np.pi/Fout.lam
    fi = -k*(tx*xx + ty*yy)
    Fout.field *= _np.exp(1j * fi)
    return Fout


