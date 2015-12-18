# -*- coding: utf-8 -*-

import airfoiltools as aft
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


# Load airfoil from iges file
iges_file = '../airfoils/airfoil_22000_rot.igs'
tck = aft.load_airfoil_iges(iges_file)

tck_norm = aft.norm_bspline_airfoil(tck)
tck_norm_mod = aft.correct_te(tck_norm, s=0.0, k=3)


# Plot airfoil
plt.figure('Airfoil')
u = np.linspace(0.0, 1.0, 1000)
airfoil_norm = interpolate.splev(u, tck_norm, der=0)
airfoil_norm_mod = interpolate.splev(u, tck_norm_mod, der=0)
plt.plot(airfoil_norm[0], airfoil_norm[1], label='norm. airfoil')
plt.plot(airfoil_norm_mod[0], airfoil_norm_mod[1], '-r',
         label='norm. mod. airfoil')
plt.axis('equal')
plt.grid()
plt.legend()

# Plot curvature
curvature_norm = aft.curvature_bspline(tck_norm, u)
curvature_norm_mod = aft.curvature_bspline(tck_norm_mod, u)
plt.figure('Curvature')
plt.plot(curvature_norm, '-bx', label='norm. airfoil')
plt.plot(curvature_norm_mod, '-rx', label='norm. mod. airfoil')
plt.grid()
plt.legend()
plt.show()
