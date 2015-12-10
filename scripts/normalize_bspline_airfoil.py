# -*- coding: utf-8 -*-

import airfoiltools as aft
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

# Load airfoil from iges file
iges_file = '../airfoils/airfoil_5500_rot.igs'
tck = aft.load_airfoil_iges(iges_file)

# Find leading and trailing edge points
te_point = aft.find_te_point(tck)
u_le, le_point = aft.find_le_point(tck, te_point)

# Translate le_point to origin
tck = aft.translate_to_origin(tck, le_point)

# Scale airfoil
tck = aft.scale_airfoil(tck, le_point, te_point)

# Rotate airfoil
tck = aft.rotate_airfoil(tck, le_point, te_point)

# Plot airfoil
plt.figure('Airfoil')
u = np.linspace(0.0, 1.0, 1000)
airfoil_norm = interpolate.splev(u, tck, der=0)
plt.plot(airfoil_norm[0], airfoil_norm[1], label='norm. airfoil')
plt.axis('equal')
plt.grid()
plt.legend()

# Plot curvature
curvature_norm = aft.curvature_bspline(tck, u)
plt.figure('Curvature')
plt.plot(curvature_norm, label='norm. airfoil')
plt.grid()
plt.legend()

# Get discrete points
num_points, airfoil_points = aft.bspline_to_points(tck, min_step=2e-4,
                                                   max_step=0.01)
print('Number of points: {}'.format(num_points))

# Output airfoil as point coordinates
fname = 'test.dat'
aft.write_pointwise_seg(airfoil_points, fname)

# Plot points on airfoil
plt.figure('Airfoil')
plt.plot(airfoil_points[:, 0], airfoil_points[:, 1], 'ro',
         label='airfoil points')
plt.legend()
plt.show()
