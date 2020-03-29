# -*- coding: utf-8 -*-

import numpy as _np

USE_SCIPY=False
if USE_SCIPY:
    from skimage.restoration import unwrap_phase as _unwrap_phase
    #used in PhaseUnwrap, or using own implementation in .unwrap

from .units import deg
from .field import Field
from .unwrap import phaseunwrap


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
    Fout.field += Fin2.field
    return Fout

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
    
    #from
    #https://stackoverflow.com/questions/44865023/
    # circular-masking-an-image-in-python-using-numpy-arrays
    Fout = Field.copy(Fin)
    
    Y, X = Fout.mgrid_cartesian
    Y = Y - y_shift
    X = X - x_shift
    
    dist_sq = X**2 + Y**2 #squared, no need for sqrt
    
    Fout.field[dist_sq > R**2] = 0.0
    return Fout

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
    
    #from
    #https://stackoverflow.com/questions/44865023/
    # circular-masking-an-image-in-python-using-numpy-arrays
    Fout = Field.copy(Fin)
    
    Y, X = Fout.mgrid_cartesian
    Y = Y - y_shift
    X = X - x_shift
    dist_sq = X**2 + Y**2 #squared, no need for sqrt
    
    Fout.field[dist_sq <= R**2] = 0.0
    return Fout


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
    Efactor = _np.sqrt(att) #att. given as intensity
    Fout = Field.copy(Fin)
    Fout.field *= Efactor
    return Fout

def Intensity(flag, Fin):
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
    I = _np.abs(Fin.field)**2
    if flag > 0:
        Imax = I.max()
        if Imax == 0.0:
            raise ValueError('Cannot normalize because of 0 beam power.')
        I = I/Imax
        if flag == 2:
            I = I*255
    return I


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
    if Intens.shape != Fin.field.shape:
        raise ValueError('Intensity pattern shape does not match field size')
    Fout = Field.copy(Fin)
    Efield = _np.sqrt(Intens)
    Fout.field *= Efield
    return Fout


def MultPhase(Phi, Fin):
    """
    Fout = MultPhase(Phase, Fin)

    :ref:`Multiplies the field with a given phase distribution. <MultPhase>`
        
    Args::
        
        Phase: N x N square array of real numbers
        Fin: input field
        
    Returns::
        
        Fout: output field (N x N square array of complex numbers).
  
    """
    if Phi.shape != Fin.field.shape:
        raise ValueError('Phase pattern shape does not match field size')
    Fout = Field.copy(Fin)
    Fout.field *= _np.exp(1j*Phi)
    return Fout


def Normal(Fin):
    """
    Fout = Normal(Fin)

    :ref:`Normalizes the field using beam power. <Normal>`
    
        :math:`F_{out}(x,y)= \\frac{F_{in}(x,y)}{\\sqrt{P}}`
        
        with
        
        :math:`P=\\int\\int\\abs{F_{in}(x,y)\\right}^2 dx dy`
    
    Args::
        
        Fin: input field
        
    Returns::
        
        Fout: output field (N x N square array of complex numbers).
  
    """
    Fabs = _np.abs(Fin.field)**2
    Fabs *= Fin.dx**2
    Ptot = Fabs.sum()
    if Ptot == 0.0:
        raise ValueError('Error in Normal(Fin): Zero beam power!')
    Fout = Field.copy(Fin)
    Fout.field *= _np.sqrt(1/Ptot)
    return Fout


def Phase(Fin, unwrap = False, units='rad', blank_eps=0):
    """
    Phi=Phase(Fin)
    
    :ref:`Calculates the phase of the field. <Phase>`
    
    
    Args::
    
        Fin: input field
        unwrap: Call PhaseUnwarp on the extracted Phase, Default is False
        units: 'opd': returned in [meters] of optical path length
                'lam': returned in multiples of lambda
                'rad': returned in multiples of 2pi phase jumps (default)
        blank_eps: [fraction] of max. Intensity at which to blank the phase
            and replace the value with numpy.nan (e.g. 1e-3==0.1%)
            Set to 0 or None to disable
        
    Returns::
    
        Phi: phase distribution (N x N square array of doubles)

    """
    _2pi = 2*_np.pi
    Phi = _np.angle(Fin.field)
    if unwrap:
        Phi = PhaseUnwrap(Phi)
    
    if units=='opd':
        Phi = Phi/_2pi*Fin.lam #a PtV of 2pi will yield e.g. 1*lam=1e-6=1um
    elif units=='lam':
        Phi = Phi/_2pi #a PtV of 2pi=6.28 will yield 1 (as in 1 lambda)
    elif units=='rad':
        pass #a PtV of 2pi will yield 6.28 as requested
    else:
        raise ValueError('Unknown value for option units={}'.format(units))
        
    if blank_eps:
        I = Intensity(0,Fin)
        Phi[I<blank_eps*I.max()] = _np.nan
        
    return Phi



def PhaseUnwrap(Phi):
    """
    PhiOut=PhaseUnwrap(PhiIn)
    
    :ref:`Unwraps (removes jumps of pi radians) the phase. <PhaseUnwrap>`
    
    
    Args::
    
        PhiIn: input phase distribution
        
    Returns::
    
        PhiOut: unwrapped phase distribution (N x N square array of doubles)

    """
    if USE_SCIPY:
        PhiU = _unwrap_phase(Phi)
    else:
        PhiU = phaseunwrap(Phi)
        
    return PhiU


def Power(Fin):
    """
    P = Power(Fin)

    :ref:`Calculates the total power. <Power>`
        
    Args::
        
        Fin: input field
        
    Returns::
        
        P: output power (real number).
  
    """
    #TODO why does Normal() also sum dx**2 (==integral) while this does not??
    I = _np.abs(Fin.field)**2
    return I.sum()


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
    #TODO implementation error in original LP: field error, not I error!
    # need to sqrt for that
    Fout = Field.copy(Fin)
    _np.random.seed(int(seed))
    N = Fout.N
    ranint = _np.random.rand(N, N)*noise
    Fout.field += ranint
    return Fout

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
    #2020023 - ldo - tested similar result as Cpp version, although not 
    # 1:1 since seed is different in numpy
    Fout = Field.copy(Fin)
    _np.random.seed(int(seed))
    N = Fout.N
    ranphase = (_np.random.rand(N, N)-0.5)*maxPhase
    Fout.field *= _np.exp(1j * ranphase)
    return Fout


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
    Fout = Field.copy(Fin)
    yy, xx = Fout.mgrid_cartesian
    yy = yy - y_shift
    xx = xx - x_shift
    if angle!=0.0:
        ang_rad = -1*angle*deg #-1 copied from Cpp convention
        cc = _np.cos(ang_rad)
        ss = _np.sin(ang_rad)
        xxr = cc * xx + ss * yy
        yyr = -ss * xx + cc * yy
        yy, xx = yyr, xxr
    matchx = _np.abs(xx) > sx/2
    matchy = _np.abs(yy) > sy/2
    Fout.field[matchx | matchy] = 0.0
    return Fout


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
    Fout = Field.copy(Fin)
    yy, xx = Fout.mgrid_cartesian
    yy = yy - y_shift
    xx = xx - x_shift
    if angle!=0.0:
        ang_rad = -1*angle*deg #-1 copied from Cpp convention
        cc = _np.cos(ang_rad)
        ss = _np.sin(ang_rad)
        xxr = cc * xx + ss * yy
        yyr = -ss * xx + cc * yy
        yy, xx = yyr, xxr
    matchx = _np.abs(xx) <= sx/2
    matchy = _np.abs(yy) <= sy/2
    Fout.field[matchx & matchy] = 0.0
    return Fout


def Strehl(Fin):
    """
    S = Strehl( Fin)

    :ref:`Calculates the Strehl value of the field <Strehl>`
        
    Args::
        
        Fin: input field
        
    Returns::
        
        S: Strehl value (real number).
  
    """
    normsq = _np.abs(Fin.field).sum()**2
    if normsq == 0.0:
        raise ValueError('Error in Strehl: Zero beam power')
    strehl = _np.real(Fin.field).sum()**2 + _np.imag(Fin.field).sum()**2
    strehl = strehl/normsq
    return strehl

def SubIntensity(Intens, Fin):
    """
    Fout = SubIntensity(Intens, Fin)

    :ref:`Substitutes  a given intensity distribution in the field with. <SubIntensity>`
        
    Args::
        
        Intens: N x N square array of real numbers >= 0
        Fin: input field
        
    Returns::
        
        Fout: output field (N x N square array of complex numbers).
  
    """
    Fout = Field.copy(Fin)
    Intens = _np.asarray(Intens)
    if Intens.shape != Fout.field.shape:
        raise ValueError('Intensity map has wrong shape')
    phi = _np.angle(Fout.field)
    Efield = _np.sqrt(Intens)
    Fout.field = Efield * _np.exp(1j * phi)
    return Fout

def SubPhase(Phi, Fin):
    """
    Fout = SubPhase(Phi, Fin)

    :ref:`Substitutes  a given phase distribution in the field with. <SubPhase>`
        
    Args::
        
        Phase: N x N square array of real numbers
        Fin: input field
        
    Returns::
        
        Fout: output field (N x N square array of complex numbers).
  
    """
    Fout = Field.copy(Fin)
    Phi = _np.asarray(Phi)
    if Phi.shape != Fin.field.shape:
        raise ValueError('Phase map has wrong shape')
    oldabs = _np.abs(Fout.field)
    Fout.field = oldabs * _np.exp(1j * Phi)
    return Fout


