# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:56:39 2015

@author: winstroth
"""

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


num_vec = 150

# Load airfoil coordinates
airf_coord = np.loadtxt('airfoils/du_91-w2-250.dat',
                        usecols=(0, 1), skiprows=1)
x = airf_coord[:, 0]
y = airf_coord[:, 1]

# Bspline interpolation
tcka, u = interpolate.splprep([x, y], s=0.0000000, k=3)
uvec = np.linspace(0.0, 1.0, num_vec)



# Get gradients for curvature calculation
vec_loc = interpolate.splev(uvec, tcka, der=0)
grad1 = interpolate.splev(uvec, tcka, der=1)



# Plots
plt.figure('Airfoil')
plt.plot(airf_coord[:, 0], airf_coord[:, 1], label='Input airfoil')

vec_scal = -0.01

for i in range(len(vec_loc[0])):
    x = vec_loc[0][i]
    y = vec_loc[1][i]
    dx = grad1[0][i]
    dy = grad1[1][i]
    temp = dx
    dx = -dy
    dy = temp
    lx = [x, (x+dx*vec_scal)]
    ly = [y, (y+dy*vec_scal)]

    plt.plot(lx, ly, '-ro', label='tan vectors')

plt.axis('equal')
plt.grid()
plt.legend()