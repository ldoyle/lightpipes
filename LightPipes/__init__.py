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
    'GaussBeam',
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
    'PhaseSpiral',
    'PhaseUnwrap',
    'PipFFT',
    'PlaneWave',
    'PointSource',
    'Power',
    'RandomIntensity',
    'RandomPhase',
    'RectAperture',
    'RectScreen',
    'Steps',
    'Strehl',
    'SubIntensity',
    'SubPhase',
    'SuperGaussAperture',
    'Tilt',
    'Zernike',
    'noll_to_zern',
    'ZernikeName',
    'ZernikeFit',
    'ZernikeFilter',
    'LPtest',
    'LPhelp',
    'LPdemo',
]

#physical units like m, mm, rad, deg, ...
from .units import * 

from ._version import __version__
LPversion=__version__

__all__.extend([
    'm', 'cm', 'mm', 'um', 'nm',
    'rad', 'mrad', 'urad', 'deg', 'W', 'mW', 'LPversion',
    'PI'
])

# avoid modified
__all__ = tuple(__all__)

import functools
import numpy as np
import webbrowser

from .field import Field
from .propagators import Fresnel, Forward, Forvard, Steps
from .lenses import Axicon, Lens, LensFarfield, LensForvard, LensFresnel, \
    Convert
from .zernike import ZernikeName, ZernikeNolltoMN, noll_to_zern, \
    ZernikeFilter, ZernikeFit, Zernike
from .core import CircAperture, CircScreen, RectAperture, RectScreen
from .core import GaussAperture, GaussScreen, GaussHermite, GaussLaguerre, SuperGaussAperture
from .core import Intensity, Phase, PhaseUnwrap, PhaseSpiral
from .core import RandomIntensity, RandomPhase
from .core import Strehl
from .core import SubIntensity, SubPhase
from .core import BeamMix
from .core import MultIntensity, MultPhase
from .core import Normal, Power
from .core import IntAttenuator
from .misc import Tilt, Gain, PipFFT
from .core import Interpol
from .sources import PointSource, GaussBeam, PlaneWave  

def Begin(size,labda,N):
    """
    F = Begin(GridSize, Wavelength, N)
    
    :ref:`Creates a plane wave (phase = 0.0, amplitude = 1.0). <Begin>`

    Args::
    
        GridSize: size of the grid
        Wavelength: wavelength of the field
        N: N x N grid points
        
    Returns::
     
        F: N x N square array of complex numbers (1+0j).
            
    Example:
    
    :ref:`Diffraction from a circular aperture <Diffraction>`
    
    """
    Fout = Field.begin(size, labda, N) #returns Field class with all params
    return Fout

def LPtest():
    """
    Performs a test to check if the installation of the LightPipes package was successful.
    
    Args::
    
        -
        
    Returns::
    
        "LightPipes for Python: test passed." if successful,
        "Test failed" if not.

    """
    F=Begin(1.8,2.5,55)
    F=Fresnel(10,F)
    I=Intensity(0,F)
    S=np.sum(I)
    Sa=16.893173606654138
    if S==Sa:
        print('Test OK')
    else:
        print('Test failed')

def LPhelp():
    """
    Go to the LightPipes documentation website on:
    
    https://opticspy.github.io/lightpipes/

    """
    webbrowser.open_new("https://opticspy.github.io/lightpipes/")

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

