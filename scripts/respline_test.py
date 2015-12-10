# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:09:58 2015

@author: winstroth
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:30:38 2015

@author: winstroth
"""

import airfoiltools as aft
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

# Load airfoil from iges file
iges_file = '../airfoils/NACA643618.igs'
tck_org = aft.load_airfoil_iges(iges_file)

# Point coordinates from org spline
u = np.linspace(0.0, 1.0, 10000)

airfoil_org = interpolate.splev(u, tck_org, der=0)
curvature_org = aft.curvature_iges(tck_org)

# New spline
tck_new, u_new = interpolate.splprep(airfoil_org, k=5, s=0.000000000001)
airfoil_new = interpolate.splev(u, tck_new, der=0)
curvature_new = aft.curvature_iges(tck_new)

plt.figure('Curvature')
plt.plot(curvature_new, label='curvature new')
plt.plot(curvature_org, '-r', label='curvature old')
plt.grid()
plt.legend()

plt.figure('airfoils')
plt.plot(airfoil_new[0], airfoil_new[1], label='airfoil new')
plt.plot(airfoil_org[0], airfoil_org[1], '-r', label='airfoil old')
plt.legend()
plt.axis('equal')

plt.figure('Bspline coeffs')
plt.plot(tck_new[1][0], tck_new[1][1], 'o', label='new')
plt.plot(tck_org[1][0], tck_org[1][1], 'ro', label='old')
plt.legend()
plt.axis('equal')
