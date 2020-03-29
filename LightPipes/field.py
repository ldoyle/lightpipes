# -*- coding: utf-8 -*-
"""

Created on Sun Feb 23 17:53:56 2020

@author: Leonard.Doyle@physik.uni-muenchen.de
"""

import numpy as _np
import copy as _copy

class Field:
    """
    Lightpipes Field object, containing the field data and meta
    parameters as well as helper functions to change data formats etc.
    """
    
    
    @classmethod
    def begin(cls, grid_size, wavelength, N):
        """
        Initialize a new field object with the given parameters.
        This method is preferred over direct calling of constructor.
        

        Parameters
        ----------
        grid_size : float
            [m] physical size of the square grid
        wavelength : float
            [m] physical wavelength
        N : int
            number of grid points in each dimension (square)

        Returns
        -------
        The initialized Field object.

        """
        inst = cls(None, grid_size, wavelength, N)
        return inst
        
    @classmethod
    def copy(cls, Fin):
        """
        Create a copy of the input field with identical values but
        no common references to numpy fields etc.

        Parameters
        ----------
        Fin : Field
            Input field to copy/clone

        Returns
        -------
        A new Field object with identical values as Fin.

        """
        return _copy.deepcopy(Fin)
    
    def __init__(self, Fin=None, grid_size=1.0, wavelength=1.0, N=0):
        """Private, use class method factories instead."""
        if Fin is None:
            if not N:
                raise ValueError('Cannot create zero size field (N=0)')
            Fin = _np.ones((N,N),dtype=complex)
        else:
            Fin = _np.asarray(Fin, dtype=complex)
        self._field = Fin
        self._lam = wavelength
        self._siz = grid_size
        self._int1 = 0 #remembers PipFFT direction
        self._curvature = 0.0 #remembers field curvature or 0.0 for normal
    
    
    def _get_grid_size(self):
        """Get or set the grid size in [m]."""
        return self._siz
    
    def _set_grid_size(self, gridsize):
        self._siz = gridsize
    
    grid_size = property(_get_grid_size, _set_grid_size)
    siz = grid_size
    
    def _get_wavelength(self):
        """Get or set the wavelength of the field. All units in [m]."""
        return self._lam
    
    def _set_wavelength(self, wavelength):
        self._lam = wavelength
    
    wavelength = property(_get_wavelength, _set_wavelength)
    lam = wavelength
    
    @property
    def grid_dimension(self):
        return self._field.shape[0] #assert square
    
    N = grid_dimension
    
    
    @property
    def grid_step(self):
        """Distance in [m] between 2 grid points"""
        return self.siz/self.N
    
    dx = grid_step
    
    @property
    def field(self):
        """Get the complex E-field."""
        return self._field
    
    @field.setter
    def field(self, field):
        """The field must be a complex 2d square numpy array.
        """
        field = _np.asarray(field, dtype=complex)
        #will not create a new instance if already good
        self._field = field
    
    @property
    def mgrid_cartesian(self):
        """Return a meshgrid tuple (Y, X) of cartesian coordinates for each 
        pixel of the field."""
        
        """LightPipes manual/ examples Matlab and Python version:
            plotting the Intensity with imshow() yields coord sys:
                positive shift in x is right
                positive shift in y is down!!
            -> stick to this convention where possible
        
        Adapted from matplotlib.imshow convention: coords define pixel center,
        so extent will be xmin-1/2dx, xmax+1/2dx
        For an odd number of pixels this puts a pixel in the center as expected
        for an even number, the "mid" pixel shifts right and down by 1
        """
        h, w = self.N, self.N
        cy, cx = int(h/2), int(w/2)
        Y, X = _np.mgrid[:h, :w]
        Y = (Y-cy)*self.dx
        X = (X-cx)*self.dx
        return (Y, X)
    
    @property
    def mgrid_polar(self):
        """Return a meshgrid tuple (R, Phi) of polar coordinates for each
        pixel in the field (matching legacy LP convention)."""
        Y, X = self.mgrid_cartesian
        r = _np.sqrt(X**2+Y**2)
        # phi = _np.arctan2(Y, X) + _np.pi #TODO this is ported from Cpp
        phi = _np.arctan2(Y, X) + _np.pi/2 #however this matches results
        # -> something is slightly going wrong somewhere in the definition
        # of e.g. the azimuthal angle or coord axes
        return (r, phi)


