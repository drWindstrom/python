# -*- coding: utf-8 -*-

import airfoiltools as aft
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


# Load airfoil from iges file
iges_file = '../airfoils/airfoil_7000_rot.igs'
tck = aft.load_airfoil_iges(iges_file)

tck_norm = aft.norm_bspline_airfoil(tck)
tck_norm_mod = aft.correct_te(tck_norm, s=0.0, k=3)
num_points, points = aft.bspline_to_points(tck_norm_mod, min_step=1e-4,
                                           max_step=0.01)
print('num of points: {}'.format(num_points))

fname = '/home/fred/cfd/grids/3d_reference_rotor/3d_ref_rot_r7000.dat'
aft.write_pointwise_seg(points, fname)

te_point = aft.find_te_point(tck_norm_mod)
u_le, le_point = aft.find_le_point(tck_norm_mod, te_point)

print('u location: {}'.format(u_le))

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

# Plot exported points
plt.figure('exp. points')
plt.plot(points[:, 0], points[:, 1], 'o', label='norm. airfoil')
plt.axis('equal')
plt.grid()
plt.legend()

plt.show()
