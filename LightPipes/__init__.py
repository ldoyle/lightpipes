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

import functools
import numpy as np
import webbrowser

from .field import Field

from ._LightPipes import Init # noqa
from ._version import __version__

LPversion=__version__

_LP = Init() # noqa
# for name in __all__:
#     locals()[name] = getattr(LP, name)

from .units import * # noqa

__all__.extend([
    'm', 'cm', 'mm', 'um', 'nm',
    'rad', 'mrad', 'W', 'mW', 'LPversion',
])

# avoid modified
__all__ = tuple(__all__)


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
        ll_in = Fout.field.tolist()
        args = list(args) #make mutable
        args.append(ll_in)
        # args = tuple(args) #back to standard
        
        _apply_vals_to_LP(Fout)
        
        ll_out = fn(*args, **kwargs)
        
        Fout.field = np.asarray(ll_out)
        _field_vals_from_LP(Fout)
        
        return Fout
    
    return fn_wrapper
        

@accept_new_field
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
    return _LP.Axicon(phi, n1, x_shift, y_shift, Fin)


def BeamMix(Fin1, Fin2):
    """
    Fout = BeamMix(F1, F2)

    :ref:`Addition of the fields F1 and F2. <BeamMix>`

    Args::
    
        F1, F2: input fields
        
    Returns::
      
        Fout: output field (N x N square array of complex numbers).
        
    Example:
    
    :ref:`Two holes interferometer <Young>`
    
    """
    if Fin1.field.shape != Fin2.field.shape:
        raise ValueError('Field sizes do not match')
    Fout = Field.copy(Fin1)
    ll_in1 = Fout.field.tolist()
    ll_in2 = Fin2.field.tolist()
    _apply_vals_to_LP(Fout)
    ll_out = _LP.BeamMix(ll_in1, ll_in2)
    Fout.field = np.asarray(ll_out)
    _field_vals_from_LP(Fout)
    
    return Fout

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
def CircAperture(R, x_shift, y_shift, Fin):
    """
    Fout = CircAperture(R, x_shift, y_shift, Fin)
    
    :ref:`Propagates the field through a circular aperture. <CircAperture>`

    Args::
    
        R: radius of the aperture
        x_shift, y_shift: shift from the center
        Fin: input field
        
    Returns::
     
        Fout: output field (N x N square array of complex numbers).
            
    Example:
    
    :ref:`Diffraction from a circular aperture <circ_aperture>`
    
    """
    return _LP.CircAperture(R, x_shift, y_shift, Fin)

@accept_new_field
def CircScreen(R, x_shift, y_shift, Fin):
    """
    Fout = CircScreen(R, x_shift, y_shift, Fin)
                
    :ref:`Diffracts the field by a circular screen. <CircScreen>`

    Args::
    
        R: radius of the screen
        x_shift, y_shift: shift from the center
        Fin: input field
        
    Returns::
     
        Fout: output field (N x N square array of complex numbers).
            
    Example:
    
    :ref:`Spot of Poisson <Poisson>`
    
    """
    return _LP.CircScreen(R, x_shift, y_shift, Fin)

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
    return _LP.Fresnel(z, Fin) 

@accept_new_field 
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
    return _LP.Gain(Isat, alpha0, Lgain, Fin)

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
def IntAttenuator(att, Fin):
    """
    Fout = IntAttenuator(att, Fin)
    
    :ref:`Attenuates the intensity of the field. <IntAttenuator>`
        
        :math:`F_{out}(x,y)=\\sqrt{att}F_{in}(x,y)`
        
    Args::
    
        att: intensity attenuation factor
        Fin: input field
        
    Returns::
    
        Fout: output field (N x N square array of complex numbers).
   
    """    
    return _LP.IntAttenuator( att, Fin)

def Intensity(flag,Fin):
    """
    I=Intensity(flag,Fin)
    
    :ref:`Calculates the intensity of the field. <Intensity>`
    
    :math:`I(x,y)=F_{in}(x,y).F_{in}(x,y)^*`
    
    Args::
    
        flag: 0= no normalization, 1=normalized to 1, 2=normalized to 255 (for bitmaps)
        Fin: input field
        
    Returns::
    
        I: intensity distribution (N x N square array of doubles)

    """
    ll_in = Fin.field.tolist()
    _apply_vals_to_LP(Fin) #important to set N in LP
    return np.asarray(_LP.Intensity(flag, ll_in))

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
    return _LP.Lens(f, x_shift, y_shift, Fin)

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
def MultIntensity(Intens, Fin):
    """
    Fout = MultIntensity(Intens, Fin)

    :ref:`Multiplies the field with a given intensity distribution. <MultIntensity>`
        
    Args::
        
        Intens: N x N square array of real numbers
        Fin: input field
        
    Returns::
        
        Fout: output field (N x N square array of complex numbers).
  
    """
    return _LP.MultIntensity( Intens, Fin)

@accept_new_field
def MultPhase(Phase, Fin):
    """
    Fout = MultPhase(Phase, Fin)

    :ref:`Multiplies the field with a given phase distribution. <MultPhase>`
        
    Args::
        
        Phase: N x N square array of real numbers
        Fin: input field
        
    Returns::
        
        Fout: output field (N x N square array of complex numbers).
  
    """
    return _LP.MultPhase( Phase, Fin)

@accept_new_field
def Normal(Fin):
    """
    Fout = Normal(Fin)

    :ref:`Normalizes the field. <Normal>`
        
    Args::
        
        Fin: input field
        
    Returns::
        
        Fout: output field (N x N square array of complex numbers).
  
    """
    return _LP.Normal(Fin)

def Phase(Fin):
    """
    Phi=Phase(Fin)
    
    :ref:`Calculates the phase of the field. <Phase>`
    
    
    Args::
    
        Fin: input field
        
    Returns::
    
        Phi: phase distribution (N x N square array of doubles)

    """
    ll_in = Fin.field.tolist()
    _apply_vals_to_LP(Fin) #important to set N in LP
    return np.asarray(_LP.Phase(ll_in))

def PhaseUnwrap(Phi):
    """
    PhiOut=PhaseUnwrap(PhiIn)
    
    :ref:`Unwraps (removes jumps of pi radians) the phase. <PhaseUnwrap>`
    
    
    Args::
    
        PhiIn: input phase distribution
        
    Returns::
    
        PhiOut: unwrapped phase distribution (N x N square array of doubles)

    """
    ll_in = Phi.tolist()
    N = Phi.shape[0]
    _LP.internal_setN(N) #important, needs to be set correctly in LP
    return np.asarray(_LP.PhaseUnwrap(ll_in))

@accept_new_field
def PipFFT(index, Fin):
    """
    Fout = PipFFT(index, Fin)

    :ref:`Performs a 2D Fourier transform of the field. <PipFFT>`
        
    Args::
        
        index: +1 = forward transform, -1 = back transform
        Fin: input field
        
    Returns::
        
        Fout: output field (N x N square array of complex numbers).
  
    """
    return _LP.PipFFT(index, Fin)

def Power(Fin):
    """
    P = Power(Fin)

    :ref:`Calculates the total power. <Power>`
        
    Args::
        
        Fin: input field
        
    Returns::
        
        P: output power (real number).
  
    """
    ll_in = Fin.field.tolist()
    _apply_vals_to_LP(Fin) #important to set N in LP
    return _LP.Power(ll_in)

@accept_new_field
def RandomIntensity(seed, noise, Fin):
    """
    Fout = RandomIntensity(seed, noise, Fin)

    :ref:`Adds random intensity to the field <RandomIntensity>`
        
    Args::
        
        seed: seed number for the random noise generator
        noise: level of the noise
        Fin: input field
        
    Returns::
        
        Fout: output field (N x N square array of complex numbers).
  
    """
    return _LP.RandomIntensity(seed, noise, Fin)

@accept_new_field
def RandomPhase(seed, maxPhase, Fin):
    """
    Fout = RandomPhase(seed, maxPhase, Fin)

    :ref:`Adds random phase to the field <RandomPhase>`
        
    Args::
        
        seed: seed number for the random noise generator
        maxPhase: maximum phase in radians
        Fin: input field
        
    Returns::
        
        Fout: output field (N x N square array of complex numbers).
  
    """
    return _LP.RandomPhase(seed, maxPhase, Fin)      

@accept_new_field          
def RectAperture(sx, sy, x_shift, y_shift, angle, Fin):
    """
    Fout = RectAperture(w, h, x_shift, y_shift, angle, Fin)
    
    :ref:`Propagates the field through a rectangular aperture. <RectAperture>`

    Args::
    
        w: width of the aperture
        h: height of the aperture
        x_shift, y_shift: shift from the center
        angle: rotation angle in degrees 
        Fin: input field
        
    Returns::
     
        Fout: output field (N x N square array of complex numbers).

    """
    return _LP.RectAperture(sx, sy, x_shift, y_shift, angle, Fin)

@accept_new_field
def RectScreen(sx, sy, x_shift, y_shift, angle, Fin):
    """
    Fout = RectScreen(w, h, x_shift, y_shift, angle, Fin)
    
    :ref:`Diffracts the field by a rectangular screen. <RectScreen>`

    Args::
    
        w: width of the screen
        h: height of the screen
        x_shift, y_shift: shift from the center
        angle: rotation angle in degrees 
        Fin: input field
        
    Returns::
     
        Fout: output field (N x N square array of complex numbers).

    """    
    return _LP.RectScreen(sx, sy, x_shift, y_shift, angle, Fin)

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

def Strehl(Fin):
    """
    S = Strehl( Fin)

    :ref:`Calculates the Strehl value of the field <Strehl>`
        
    Args::
        
        Fin: input field
        
    Returns::
        
        S: Strehl value (real number).
  
    """
    _apply_vals_to_LP(Fin) #important to set N in LP
    ll_in = Fin.field.tolist()
    return _LP.Strehl(ll_in)

@accept_new_field
def SubIntensity(Intens, Fin):
    """
    Fout = SubIntensity(Intens, Fin)

    :ref:`Substitutes  a given intensity distribution in the field with. <SubIntensity>`
        
    Args::
        
        Intens: N x N square array of real numbers
        Fin: input field
        
    Returns::
        
        Fout: output field (N x N square array of complex numbers).
  
    """
    return _LP.SubIntensity( Intens.tolist(), Fin)

@accept_new_field
def SubPhase(Phase, Fin):
    """
    Fout = SubPhase(Phase, Fin)

    :ref:`Substitutes  a given phase distribution in the field with. <SubPhase>`
        
    Args::
        
        Phase: N x N square array of real numbers
        Fin: input field
        
    Returns::
        
        Fout: output field (N x N square array of complex numbers).
  
    """
    return _LP.SubPhase( Phase.tolist(), Fin)

@accept_new_field
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
    return _LP.Tilt( tx, ty, Fin)

@accept_new_field
def Zernike(n, m, R, A, Fin):
    """
    Fout = Zernike(n, m, R, A, Fin)

    :ref:`Substitutes a Zernike aberration phase distribution in the field. <Zernike>`
        
        :math:`F_{out}(x,y)=e^{\\phi^m_n (x,y)}F_{in}(x,y)`
        
        with:
        
        :math:`\\phi^m_n(x,y)=-j \\frac{2 \\pi }{ \\lambda } Z^m_n {(\\rho (x,y) ,\\theta (x,y)) }`
        
        :math:`\\rho(x,y)=  \\sqrt{ \\frac{x^2+y^2}{R^2} }`
        
        :math:`\\theta (x,y)=atan \\big( \\frac{y}{x} \\big)`
        
        :math:`Z^m_n(\\rho , \\theta)=A \\sqrt{ \\frac{2(n+1)}{1+\\delta_{m0}} } V^m_n(\\rho)cos(m\\theta)`
        
        :math:`Z^{-m}_n(\\rho , \\theta)=A \\sqrt{ \\frac{2(n+1)}{1+\\delta_{m0}} }V^m_n(\\rho)sin(m\\theta)`
        
        :math:`V^m_n(\\rho)= \\sum_{s=0}^{ \\frac{n-m}{2} }  \\frac{(-1)^s(n-s)!}{s!( \\frac{n+m}{2}-s)!( \\frac{n-m}{2}-s )! } \\rho^{n-2s}`
        
        :math:`\\delta_{m0}= \\begin{cases}1 & m = 0\\\\0 & m  \\neq  0\\end{cases}`
        
       
    Args::
    
        n: radial order
        m: azimuthal order, n-|m| must be even, |m|<=n
        R: radius of the aberrated aperture
        A: size of the aberration
        Fin: input field
        
    Returns::
      
        Fout: ouput field (N x N square array of complex numbers).
            
    Example:
    
    :ref:`Zernike aberration <Zernike>`
    
    Reference: https://en.wikipedia.org/wiki/Zernike_polynomials
 

    """
    return _LP.Zernike(n, m, R, A, Fin)

def noll_to_zern(j):
    """
    Convert linear Noll index to tuple of Zernike indices.
    j is the linear Noll coordinate, n is the radial Zernike index and m is the azimuthal Zernike index.
    @param [in] j Zernike mode Noll index
    @return (n, m) tuple of Zernike indices
    @see <https://oeis.org/A176988>.
    Thanks to: Tim van Werkhoven, https://github.com/tvwerkhoven
    """

    if (j == 0):
        print("Noll indices start at 1, 0 is invalid.")
        return (0,0)

    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n

    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))
    return (n, m)

def ZernikeNolltoMN(Noll):
    MN = []
    n = 0
    while (Noll > n):
        n += 1
        Noll -= n
    m = -n+2*Noll
    MN=[m,n]
    return MN

def ZernikeName(Noll):
    if (Noll >= 1 and Noll <= 21):
        name = [
            "piston",
            "horizontal tilt",
            "vertical tilt",
            "defocus",
            "oblique primary astigmatism",
            "vertical primary astigmatism",
            "vertical coma",
            "horizontal coma",
            "vertical trefoil",
            "oblique trefoil",
            "primary spherical",
            "vertical secondary astigmatism",
            "oblique secondary astigmatism",
            "vertical quadrafoil",
            "oblique quadrafoil",
            "horizontal secondary coma",
            "vertical secondary coma",
            "oblique secondary trefoil",
            "vertical secondary trefoil",
            "oblique pentafoil",
            "vertical pentafoil",
        ]
        return name[Noll-1]
    elif Noll < 1:
        print( "Error in ZernikeName(Noll): argument must be larger than 1")
        return ""
    else:
        return ""

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
    f_ll = F.field.tolist()
    for i in range(0, 4):
        for j in range(0, 4):
            Fa.append('({0.real:2.7f} + {0.imag:2.7f}i)'.format(F[i][j]))
    Faa=[
    '(0.0013726 + -0.0346812i)',
    '(0.0019701 + -0.0485514i)',
    '(0.0019701 + -0.0485514i)',
    '(0.0013726 + -0.0346812i)',
    '(0.0019701 + -0.0485514i)',
    '(0.0028259 + -0.0679688i)',
    '(0.0028259 + -0.0679688i)',
    '(0.0019701 + -0.0485514i)',
    '(0.0019701 + -0.0485514i)',
    '(0.0028259 + -0.0679688i)',
    '(0.0028259 + -0.0679688i)',
    '(0.0019701 + -0.0485514i)',
    '(0.0013726 + -0.0346812i)',
    '(0.0019701 + -0.0485514i)',
    '(0.0019701 + -0.0485514i)',
    '(0.0013726 + -0.0346812i)'
    ]
    if Fa==Faa:
        _LP.test()
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
    _LP.setGridSize(newSize)

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
    _LP.setWavelength(newWavelength)

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
    F.field[nx, ny] = 1.0
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


