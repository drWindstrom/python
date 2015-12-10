#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""This script is used to check the curvature distribution of an airfoil and
to find the location of maximum curvature along the surface of the airfoil.

Example:
    $ python convcheck.py conv.tec steady

For support please contact Jan Winstroth
Email (winstroth@tfd.uni-hannover.de)

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Load airfoil coordinates
# samples = 1e+06
samples = 162
airf_coord = np.loadtxt('../airfoils/du_91-w2-250.dat',
                        usecols=(0, 1), skiprows=1)
x = airf_coord[:, 0]
y = airf_coord[:, 1]

# Bspline interpolation
tcka, u = interpolate.splprep([x, y], s=0.0000000, k=5)
unew = np.linspace(0.0, 1.0, samples)
b_airf_coord = interpolate.splev(unew, tcka, der=0)

# Get gradients for curvature calculation
grad1 = interpolate.splev(unew, tcka, der=1)
grad2 = interpolate.splev(unew, tcka, der=2)
dx = grad1[0]
dy = grad1[1]
ddx = grad2[0]
ddy = grad2[1]

# Get absolute airfoil surface curvature
curvature = np.abs((dx*ddy - ddx*dy)/((dx**2 + dy**2)**(3.0/2.0)))

# Output location of maximum curvature
max_curvature_pos = np.argmax(curvature)
print('Max curvature [%]: {}'.format(float(max_curvature_pos)/len(curvature)))

# Plots
plt.figure('Airfoil')
plt.plot(airf_coord[:, 0], airf_coord[:, 1], label='Input airfoil')
plt.plot(b_airf_coord[0][:], b_airf_coord[1][:], label='Bspline airfoil')
plt.plot(b_airf_coord[0][max_curvature_pos],
         b_airf_coord[1][max_curvature_pos], 'ro', label='max curvature')
plt.axis('equal')
plt.grid()
plt.legend()

plt.figure('Airfoil curvature')
plt.plot(curvature)
plt.grid()
plt.show()
