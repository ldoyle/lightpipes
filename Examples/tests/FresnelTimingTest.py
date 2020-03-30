"""
Timing test based on Young.py

Run the example setting for different grid sizes and record times.
Finally, create a plot of average time over N.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from LightPipes import tictoc
from LightPipes.units import *

import LightPipes as lp
"""reference LightPipes (Cpp) renamed and installed with "setup.py develop" as
oldLightPipes"""
import oldLightPipes as olp

print(lp.LPversion)
print(olp.LPversion)


#******** Simulation parameters *************
wavelength=5*um
size=20.0*mm
z=50*cm
R=0.3*mm
d=1.2*mm

N_list=np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
        125, 126, 127, 128, 129, 130, 131, 132, 133,
        150, 200, 250,
        253, 254, 255, 256, 257, 258, 259, 260,
        300, 350, 400, 450, 500,
        501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514,
        515, 516, 517, 518, 519, 520,
        550, 600, 650, 700, 800, 900, 1000,
        1010, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029,
        1030,
        1100, 1200, 1300, 1400,
        1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500,
        1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510,
        1600, 1700, 1800, 1900,
        1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004,
        2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050,
        2200, 2400, 2600, 2800, 3000,
        ])

runs_per_N = 3


#********* Run for new python Fresnel *******

results_py = np.zeros((N_list.size, runs_per_N))

for iN, N in enumerate(N_list):
    print(N) #progress update
    for jj in range(runs_per_N):
        F=lp.Begin(size,wavelength,N)
        F1=lp.CircAperture(R/2.0,-d/2.0, 0, F)
        F2=lp.CircAperture(R/2.0, d/2.0, 0, F)    
        F=lp.BeamMix(F1,F2)
        
        tictoc.tic()
        F=lp.Fresnel(z,F)
        ttot = tictoc.toc()
        results_py[iN, jj] = ttot

tab = np.column_stack((N_list, results_py))
np.savetxt('times_py2.txt', tab)

#****** Run for reference cpp Fresnel *******

results_cpp = np.zeros((N_list.size, runs_per_N))

for iN, N in enumerate(N_list):
    print(N)
    for jj in range(runs_per_N):
        F=olp.Begin(size,wavelength,N)
        F1=olp.CircAperture(R/2.0,-d/2.0, 0, F)
        F2=olp.CircAperture(R/2.0, d/2.0, 0, F)    
        F=olp.BeamMix(F1,F2)
        
        tictoc.tic()
        F=olp.Fresnel(z,F)
        ttot = tictoc.toc()
        results_cpp[iN, jj] = ttot

tab = np.column_stack((N_list, results_cpp))
np.savetxt('times_cpp2.txt', tab)

#*********** Plot results *******************
"""use np.loadtxt() to plot existing results."""
# results_py = np.loadtxt('times_py.txt')
# results_cpp = np.loadtxt('times_cpp.txt')
# N_list = results_py[:,0]
# results_py = results_py[:,1:] #strip N_list
# results_cpp = results_cpp[:,1:]

plt.scatter(N_list, np.average(results_py, axis=1))
plt.scatter(N_list, np.average(results_cpp, axis=1))
plt.title('Comparison of reference LightPipes Cpp vs. new Python implementation')
plt.xlabel('N')
plt.ylabel('Time for 1 call to Fresnel [s]')
plt.legend(['pure Python / numpy', 'Cpp'])
