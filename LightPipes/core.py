# -*- coding: utf-8 -*-

import numpy as _np
from skimage.restoration import unwrap_phase as _unwrap_phase

from .field import Field
from .unwrap import phaseunwrap

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
    use_scipy=False
    if use_scipy:
        PhiU = _unwrap_phase(Phi)
    else:
        PhiU = phaseunwrap(Phi)
        
    return PhiU



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
    _np.random.seed(seed)
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
    _np.random.seed(seed)
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
    # return self.thisptr.RectAperture(sx, sy, x_shift, y_shift, angle, Fin)
    """CPP
        double dx,x,y,x0,y0,cc,ss;
        int i2;
        dx =size/N;
        i2=N/2+1;
        angle *= -Pi/180.;
        cc=cos(angle);
        ss=sin(angle);
    """
    if angle==0.0:
        yy, xx = Fout.mgrid_cartesian
        matchx = _np.abs(xx-x_shift) > sx/2
        matchy = _np.abs(yy-y_shift) > sy/2
        Fout.field[matchx | matchy] = 0.0
    else:
        """CPP
            for (int i=0;i<N ;i++){
                for (int j=0;j<N ;j++){
                    x0=(i-i2+1)*dx-x_shift;
                    y0=(j-i2+1)*dx-y_shift;
                    x=x0*cc+y0*ss;
                    y=-x0*ss+y0*cc; 
                    if(fabs(x) > sx/2. || fabs(y) > sy/2. ){
                        Field.at(i).at(j) = 0.0;
                    }
                }
            }
        """
        raise NotImplementedError('Currently only angle=0.0 allowed')
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
    # return self.thisptr.RectScreen(sx, sy, x_shift, y_shift, angle, Fin)
    """CPP
        double dx,x,y,x0,y0,cc,ss;
        int i2;
        dx =size/N;
        i2=N/2+1;
        angle *= -Pi/180.;
        cc=cos(angle);
        ss=sin(angle);
    """
    if angle==0.0:
        yy, xx = Fout.mgrid_cartesian
        matchx = _np.abs(xx-x_shift) <= sx/2
        matchy = _np.abs(yy-y_shift) <= sy/2
        Fout.field[matchx & matchy] = 0.0
    else:
        """
            for (int i=0;i<N ;i++){
                for (int j=0;j<N ;j++){
                    x0=(i-i2+1)*dx-x_shift;
                    y0=(j-i2+1)*dx-y_shift;
                    x=x0*cc+y0*ss;
                    y=-x0*ss+y0*cc; 
                    if(fabs(x) <= sx/2. && fabs(y) <= sy/2. ) {
                        Field.at(i).at(j) = 0.0;
                    }
                }
            }
        """
        raise NotImplementedError('Currently only angle=0.0 allowed')
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
    # return self.thisptr.Strehl(Fin)
    """CPP
        double sum,sum1r,sum1i,sum1;
    
        sum=sum1r=sum1i=0.0;
        for (int i=0; i< N ;i++){
            for (int j=0;j < N ;j++){
                sum += abs(Field.at(i).at(j));
                sum1r += Field.at(i).at(j).real();
                sum1i += Field.at(i).at(j).imag();
            }
        }
        sum1=(sum1r*sum1r+sum1i*sum1i);
        if (sum == 0){
            cout<<"error in Strehl: Zero beam power"<<endl;
            return sum;
        }
        return sum1/sum/sum;
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
    # return self.thisptr.SubPhase( Phase, Fin)
    """CPP
        double Intens2, phi;
        if ((int)Phase.at(0).size() != N || (int)Phase.size() != N){
            printf( "Error in SubPhase(Phase, Fin): array 'Phase' must be square and must have %d x %d elements\n",N,N);
            exit(1);
        }
        for (int i=0;i< N; i++){
            for (int j=0;j< N; j++){
                phi=Phase.at(j).at(i);
                Intens2=abs(Field.at(i).at(j));
                Field.at(i).at(j) = Intens2 * exp(_j * phi);
            }
        }
        return Field;
    """
    Fout = Field.copy(Fin)
    Phase = _np.asarray(Phase)
    if Phase.shape != Fin.field.shape:
        raise ValueError('Phase map has wrong shape')
    oldabs = _np.abs(Fout.field)
    Fout.field = oldabs * _np.exp(1j * Phase)
    return Fout


