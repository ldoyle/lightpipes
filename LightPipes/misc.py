# -*- coding: utf-8 -*-

import numpy as _np

from .field import Field

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


