# fix: https://github.com/matthew-brett/delocate/issues/15

__all__ = [
    'Axicon',
    'BeamMix',
    'Begin',
    'CircAperture',
    'CircScreen',
    'Convert',
    'Forward',
    'Forvard',
    'Fresnel',
    'Gain',
    'GaussAperture',
    'GaussScreen',
    'GaussHermite',
    'GaussLaguerre',
    'IntAttenuator',
    'Intensity',
    'Interpol',
    'Lens',
    'LensForvard',
    'LensFresnel',
    'MultIntensity',
    'MultPhase',
    'Normal',
    'Phase',
    'PhaseUnwrap',
    'PipFFT',
    'Power',
    'RandomIntensity',
    'RandomPhase',
    'RectAperture',
    'RectScreen',
    'Steps',
    'Strehl',
    'SubIntensity',
    'SubPhase',
    'Tilt',
    'Zernike',
    'noll_to_zern',
    'ZernikeName',
    'ZernikeFit',
    'ZernikeFilter',
    'getGridSize',
    'setGridSize',
    'getWavelength',
    'setWavelength',
    'getGridDimension',
    'LPtest',
    'LPhelp',
    'GaussBeam',
    'PointSource',
    'LPdemo',
]

#physical units like m, mm, rad, deg, ...
from .units import * # noqa

from ._version import __version__
LPversion=__version__

__all__.extend([
    'm', 'cm', 'mm', 'um', 'nm',
    'rad', 'mrad', 'urad', 'deg', 'W', 'mW', 'LPversion',
])

# avoid modified
__all__ = tuple(__all__)

import functools
import numpy as np
import webbrowser

from ._LightPipes import Init
_LP = Init() # noqa

from .field import Field
from .propagators import Fresnel
from .lenses import Axicon, Lens, LensFarfield
from .zernike import ZernikeName, ZernikeNolltoMN, noll_to_zern, \
    ZernikeFilter, ZernikeFit, Zernike
from .core import CircAperture, CircScreen, RectAperture, RectScreen
from .core import Intensity, Phase, PhaseUnwrap
from .core import RandomIntensity, RandomPhase
from .core import Strehl
from .core import SubIntensity, SubPhase
from .core import BeamMix
from .core import MultIntensity, MultPhase
from .core import Normal, Power
from .core import IntAttenuator
from .misc import Tilt, Gain, PipFFT

def _apply_vals_to_LP(Fin):
    """Apply the values stored in Field to LP instance.
    Use this before calling any LP function."""
    _LP.internal_setN(Fin.N)
    _LP.setGridSize(Fin.siz)
    _LP.setWavelength(Fin.lam)
    _LP.internal_setInt1(Fin._int1)
    _LP.internal_setDoub1(Fin._curvature)

def _field_vals_from_LP(Fout):
    """Apply (in-place!) the global values stored in
    LP to the field Fout."""
    if Fout.N != _LP.getGridDimension():
        raise ValueError('Field size does not match LP global params')
    Fout.siz = _LP.getGridSize()
    Fout.lam = _LP.getWavelength()
    Fout._int1 = _LP.internal_getInt1()
    Fout._curvature = _LP.internal_getDoub1()
    

def accept_new_field(fn):
    """Decorator to wrap existint LP functions to accept new style
    Field object with numpy field.
    """
    @functools.wraps(fn)
    def fn_wrapper(*args, **kwargs):
        if 'Fin' in kwargs:
            raise NotImplementedError(
                'accept_new_field decorator: Fin must not be keyword arg')
        Fin = args[-1] #all LP functions have Fin as last arg
        args = args[:-1] #strip last arg
        Fout = Field.copy(Fin)
        ll_in = Fout.field.T.tolist() #transpose since numpy convention
        # is [y,x] and LP functions mostly assume [x,y] with some exceptions
        args = list(args) #make mutable
        args.append(ll_in)
        # args = tuple(args) #back to standard
        
        _apply_vals_to_LP(Fout)
        
        ll_out = fn(*args, **kwargs)
        
        Fout.field = np.asarray(ll_out).T #undo transpose, see above
        _field_vals_from_LP(Fout)
        
        return Fout
    
    return fn_wrapper
        

def Begin(size,labda,N):
    """
    F = Begin(GridSize, Wavelength, N)
    
    :ref:`Creates a plane wave (phase = 0.0, amplitude = 1.0). <Begin>`

    Args::
    
        GridSize: size of the grid
        Wavelength: wavelength of the field
        N: N x N grid points (N must be even)
        
    Returns::
     
        F: N x N square array of complex numbers (1+0j).
            
    Example:
    
    :ref:`Diffraction from a circular aperture <Diffraction>`
    
    """
    # return _LP.Begin(size, labda, N) #returns list of list
    Fout = Field.begin(size, labda, N) #returns Field class with all params
    _apply_vals_to_LP(Fout) #apply global params to keep consistency
    return Fout


@accept_new_field
def Convert(Fin):
    """
    Fout = Convert(Fin)

    :ref:`Converts the field from a spherical variable coordinate to a normal coordinate system. <Convert>`

    Args::
    
        Fin: input field
        
    Returns::
     
        Fout: output field (N x N square array of complex numbers).
            
    Example:
    
    :ref:`Unstable resonator <Unstab>`
    
    """
    return _LP.Convert( Fin)

@accept_new_field
def Forward(z, sizenew, Nnew, Fin):
    """
    Fout = Forward(z, sizenew, Nnew, Fin)

    :ref:`Propagates the field using direct integration. <Forward>`

    Args::
    
        z: propagation distance
        Fin: input field
        
    Returns::
     
        Fout: output field (N x N square array of complex numbers).
            
    Example:
    
    :ref:`Diffraction from a circular aperture <Diffraction>`
    
    """
    return _LP.Forward(z, sizenew, Nnew, Fin)

@accept_new_field
def Forvard(z, Fin):
    """
    Fout = Forvard(z, Fin)

    :ref:`Propagates the field using a FFT algorithm. <Forvard>`

    Args::
    
        z: propagation distance
        Fin: input field
        
    Returns::
     
        Fout: output field (N x N square array of complex numbers).
            
    Example:
    
    :ref:`Diffraction from a circular aperture <Diffraction>`
    
    """
    return _LP.Forvard(z, Fin)


@accept_new_field
def GaussAperture(w, x_shift, y_shift, T, Fin):
    """
    Fout = GaussAperture(w, x_shift, y_shift, T, Fin)
    
    :ref:`Inserts an aperture with a Gaussian shape in the field. <GaussAperture>`
    
        :math:`F_{out}(x,y)= \\sqrt{T}e^{ -\\frac{ x^{2}+y^{2} }{2w^{2}} } F_{in}(x,y)`

    Args::
    
        w: 1/e intensity width
        x_shift, y_shift: shift from center
        T: center intensity transmission
        Fin: input field
    
    Returns::
    
        Fout: output field (N x N square array of complex numbers).

    """   
    return _LP.GaussAperture( w, x_shift, y_shift, T, Fin)

@accept_new_field
def GaussScreen(w, x_shift, y_shift, T, Fin):
    """
    Fout = GaussScreen(w, x_shift, y_shift, T, Fin)
    
    :ref:`Inserts a screen with a Gaussian shape in the field. <GaussScreen>`

        :math:`F_{out}(x,y)= \\sqrt{1-(1-T)e^{ -\\frac{ x^{2}+y^{2} }{w^{2}} }} F_{in}(x,y)`

   Args::
    
        w: 1/e intensity width
        x_shift, y_shift: shift from center
        T: center intensity transmission
        Fin: input field
    
    Returns::
    
        Fout: output field (N x N square array of complex numbers).

    """   
    return _LP.GaussScreen( w, x_shift, y_shift, T, Fin)

@accept_new_field
def GaussHermite(m, n, A, w0, Fin):
    """
    Fout = GaussHermite(m, n, A, w0, Fin)
    
    :ref:`Substitutes a Gauss-Hermite mode (beam waist) in the field. <GaussHermite>`

        :math:`F_{m,n}(x,y,z=0) = A H_m\\left(\\dfrac{\\sqrt{2}x}{w_0}\\right)H_n\\left(\\dfrac{\\sqrt{2}y}{w_0}\\right)e^{-\\frac{x^2+y^2}{w_0^2}}`

    Args::
        
        m, n: mode indices
        A: Amplitude
        w0: Guaussian spot size parameter in the beam waist (1/e amplitude point)
        Fin: input field
        
    Returns::
    
        Fout: output field (N x N square array of complex numbers).            
        
    Reference::
    
        A. Siegman, "Lasers", p. 642

    """
    return _LP.GaussHermite( m, n, A, w0, Fin)

@accept_new_field
def GaussLaguerre(p, m, A, w0, Fin):
    """
    Fout = GaussLaguerre(p, m, A, w0, Fin)

    :ref:`Substitutes a Gauss-Laguerre mode (beam waist) in the field. <GaussLaguerre>`

        :math:`F_{p,m}(x,y,z=0) = A \\left(\\frac{\\rho}{2}\\right)^{\\frac{|m|}{2} }L^p_m\\left(\\rho\\right)e^{-\\frac{\\rho}{2}}\\cos(m\\theta)`,
        
        with :math:`\\rho=\\frac{2(x^2+y^2)}{w_0^2}`

    Args::
        
        p, m: mode indices
        A: Amplitude
        w0: Guaussian spot size parameter in the beam waist (1/e amplitude point)
        Fin: input field
        
    Returns::
    
        Fout: output field (N x N square array of complex numbers).            
        
    Reference::
    
        A. Siegman, "Lasers", p. 642

    """
        
    return _LP.GaussLaguerre( p, m, A, w0, Fin)


@accept_new_field            
def Interpol(new_size, new_number, x_shift, y_shift, angle, magnif, Fin):
    """
    Fout = Interpol(NewSize, NewN, x_shift, y_shift, angle, magnif, Fin)
    
    :ref:`Interpolates the field to a new grid size, grid dimension. <Interpol>`
    
    Args::
    
        NewSize: the new grid size
        NewN: the new grid dimension
        x_shift, y_shift: shift of the field
        angle: rotation of the field in degrees
        magnif: magnification of the field amplitude
        Fin: input field
        
    Returns::
        
        Fout: output field (Nnew x Nnew square array of complex numbers).
  
    """
    return _LP.Interpol(new_size, new_number, x_shift, y_shift, angle, magnif, Fin)


@accept_new_field
def LensForvard(f, z, Fin):
    """
    Fout = LensForvard(f, z, Fin)

    :ref:`Propagates the field in a variable spherical coordinate system. <LensForvard>`
        
    Args::
        
        f: focal length
        z: propagation distance
        Fin: input field
        
    Returns::
        
        Fout: output field (N x N square array of complex numbers).
        
    Example:
        
    :ref:`Spherical coordinates <SphericalCoordinates>`
        
    """
    return _LP.LensForvard(f, z, Fin)

@accept_new_field
def LensFresnel(f, z, Fin):
    """
    Fout = LensFresnel(f, z, Fin)

    :ref:`Propagates the field in a variable spherical coordinate system. <LensFresnel>`
        
    Args::
        
        f: focal length
        z: propagation distance
        Fin: input field
        
    Returns::
        
        Fout: output field (N x N square array of complex numbers).
        
    Example:
        
    :ref:`Spherical coordinates <SphericalCoordinates>`
        
    """
    return _LP.LensFresnel(f, z, Fin)


@accept_new_field
def Steps(z, nstep, refr, Fin):
    """
    Fout = Steps(z, nstep, refr, Fin)
                 
    :ref:`Propagates the field a distance, nstep x z, in nstep steps in a
    medium with a complex refractive index stored in the
    square array refr. <Steps>`

    Args::
    
        z: propagation distance per step
        nstep: number of steps
        refr: refractive index (N x N array of complex numbers)
        Fin: input field
        
    Returns::
      
        Fout: ouput field (N x N square array of complex numbers).
        
    Example:
    
    :ref:`Propagation through a lens like medium <lenslikemedium>`
    
    """
    return _LP.Steps(z, nstep, refr, Fin)


def LPtest():
    """
    Performs a test to check if the installation of the LightPipes package was successful.
    
    Args::
    
        -
        
    Returns::
    
        "LightPipes for Python: test passed." if successful,
        "Test failed" if not.

    """
    Fa=[]
    F=Begin(1,2,4)
    F=Fresnel(10,F)
    f_ll = F.field.T.tolist() #transpose to make f_ll[x,y] not [y,x]
    for i in range(0, 4):
        for j in range(0, 4):
            Fa.append('({0.real:2.7f} + {0.imag:2.7f}i)'.format(f_ll[i][j]))
    Faa=[
    '(0.0004345 + -0.0195239i)',
    '(0.0006237 + -0.0273331i)',
    '(0.0006237 + -0.0273331i)',
    '(0.0004345 + -0.0195239i)',
    '(0.0006237 + -0.0273331i)',
    '(0.0008946 + -0.0382658i)',
    '(0.0008946 + -0.0382658i)',
    '(0.0006237 + -0.0273331i)',
    '(0.0006237 + -0.0273331i)',
    '(0.0008946 + -0.0382658i)',
    '(0.0008946 + -0.0382658i)',
    '(0.0006237 + -0.0273331i)',
    '(0.0004345 + -0.0195239i)',
    '(0.0006237 + -0.0273331i)',
    '(0.0006237 + -0.0273331i)',
    '(0.0004345 + -0.0195239i)'
    ]
    if Fa==Faa:
        ret = _LP.test()
        if ret==1:
            print('Test OK')
    else:
        print('Test failed')

def LPhelp():
    """
    Go to the LightPipes documentation website on:
    
    https://opticspy.github.io/lightpipes/

    """
    webbrowser.open_new("https://opticspy.github.io/lightpipes/")

def getGridSize():
    """
    size_grid = getGridSize()
    
    Returns the value of the size of the grid in meters.
    
    Args::
        
        -
        
    Returns::
    
        size_grid: Size of the grid (real number).

    """
    return _LP.getGridSize()

def setGridSize(newSize):
    """
    setGridSize(newGridSize)
    
    Changes the value of the grid size.
    
    Args::
    
        newGridSize: New size of the grid.
        
    Returns::
    
        -

    """
    # _LP.setGridSize(newSize)
    raise NotImplementedError('Deprecated! use Field.size on object, not lib.')

def getWavelength():
    """
    wavelength = getWavelength()
    
    Returns the value of the wavelength in meters.
    
    Args::
    
        -
        
    Returns::
    
        wavelength: Value of the wavelength (real number).

    """
    return _LP.getWavelength()

def setWavelength(newWavelength):
    """
    setWavelength(newWavelength)
    
    Changes the value of the wavelength.
    
    Args::
    
        newWavelength: New value of the wavelength.
        
    Returns::
    
        -

    """ 
    # _LP.setWavelength(newWavelength)
    raise NotImplementedError('Deprecated! use Field.lam on object, not lib.')

def getGridDimension():
    """
    grid-dimension = getGridDimension()
    
    Returns the value of the grid dimension.
    The grid dimension cannot be set. Use: :ref:`Interpol. <Interpol>`
    
    Args::
    
        -
        
    Returns::
    
        grid-dimension: Value of the dimension of the grid (integer).

    """
    return _LP.getGridDimension()

def GaussBeam(size,labda,N,w,tx,ty):
    """
    F=GaussBeam(GridSize, Wavelength, N, w, tx,ty)
    :ref:`Creates a Gaussian beam in its waist (phase = 0.0, amplitude = 1.0). <Begin>`

    Args::
    
        GridSize: size of the grid
        Wavelength: wavelength of the field
        N: N x N grid points (N must be even)
        w: size of the waist
        tx, ty: tilt of the beam
        
        
    Returns::
     
        F: N x N square array of complex numbers (1+0j).
            
    Example:
    
    :ref:`Diffraction from a circular aperture <Diffraction>`
    
    """
    F=Begin(size,labda,N)
    F=GaussHermite(0,0,1,w,F)
    F=Tilt(tx,ty,F)
    return F

def PointSource(size,labda,N,x,y):
    """
    F=PointSource(GridSize, Wavelength, N, x, y)
    :ref:`Creates a point source. <Begin>`

    Args::
    
        GridSize: size of the grid
        Wavelength: wavelength of the field
        N: N x N grid points (N must be even)
        x, y: position of the point source.
        
        
    Returns::
     
        F: N x N square array of complex numbers (0+0j, or 1+0j where the pointsorce is ).
            
    Example:
    
    :ref:`Diffraction from a circular aperture <Diffraction>`
    
    """
    F = Begin(size,labda,N)
    if abs(x) >size/2 or abs(y) > size/2:
        print('error in PointSource: x and y must be inside grid!')
        return F
    F=IntAttenuator(0,F)
    nx=int(N/2*(1+2*x/size))
    ny=int(N/2*(1+2*y/size))
    F.field[ny, nx] = 1.0
    return F

def LPdemo():
    """
    LPdemo()
    Demonstrates the simulation of a two-holes interferometer.
    
    Args::
    
         -
    
    Returns::
    
        A plot of the interference pattern and a listing of the Python script.
    
    """
    import matplotlib.pyplot as plt
    import sys
    import platform
    m=1
    mm=1e-3*m
    cm=1e-2*m
    um=1e-6*m
    wavelength=20*um
    size=30.0*mm
    N=500
    F=Begin(size,wavelength,N)
    F1=CircAperture(0.15*mm, -0.6*mm,0, F)
    F2=CircAperture(0.15*mm, 0.6*mm,0, F)    
    F=BeamMix(F1,F2)
    F=Fresnel(10*cm,F)
    I=Intensity(0,F)
    #plt.contourf(I,50); plt.axis('equal')
    fig=plt.figure()
    fig.canvas.set_window_title('Interference pattern of a two holes interferometer') 
    plt.imshow(I,cmap='rainbow');plt.axis('off')
    print(
        '\n\nLightPipes for Python demo\n\n'
        'Python script of a two-holes interferometer:\n\n'
        '   import matplotlib.pyplot as plt\n'
        '   from LightPipes import *\n'
        '   wavelength=20*um\n'
        '   size=30.0*mm\n'
        '   N=500\n'
        '   F=Begin(size,wavelength,N)\n'
        '   F1=CircAperture(0.15*mm, -0.6*mm,0, F)\n'
        '   F2=CircAperture(0.15*mm, 0.6*mm,0, F)\n'
        '   F=BeamMix(F1,F2)\n'
        '   F=Fresnel(10*cm,F)\n'
        '   I=Intensity(0,F)\n'
        '   fig=plt.figure()\n'
        '   fig.canvas.set_window_title(\'Interference pattern of a two holes interferometer\')\n'
        '   plt.imshow(I,cmap=\'rainbow\');plt.axis(\'off\')\n'
        '   plt.show()\n\n'
    )
    print('Executed with python version: ' + sys.version)
    print('on a ' + platform.system() + ' ' + platform.release() + ' ' + platform.machine() +' machine')
    plt.show()


