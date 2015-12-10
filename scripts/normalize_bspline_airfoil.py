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
iges_file = '../airfoils/airfoil_20500_rot.igs'
tck = aft.load_airfoil_iges(iges_file)

# plt.figure('Airfoil')
# u = np.linspace(0.0, 1.0, 1000)
# airfoil = interpolate.splev(u, tck, der=0)
# plt.plot(airfoil[0], airfoil[1], label='org. airfoil')

# Get original curvature
org_curvature = aft.curvature_iges(tck)
# plt.figure('Curvature')
# plt.plot(org_curvature)

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
airfoil = interpolate.splev(u, tck, der=0)
plt.plot(airfoil[0], airfoil[1], '-r', label='norm. airfoil')
plt.axis('equal')
plt.grid()
plt.legend()

# Plot curvature
new_curvature = aft.curvature_iges(tck, res=1000)
#plt.figure('new curvature')
#plt.plot(new_curvature)

u = 0.0
points = []
max_curv = max(new_curvature)
scale = 1e-4 * max_curv

while u <= 1.0:
    points.append(interpolate.splev(u, tck, der=0))
    step = 1.0 / abs(aft.get_curvature(tck, u)) * scale
    if step > 0.01:
        step = 0.01
    u += step

points = np.array(points)
plt.figure('exported points')
plt.plot(points[:, 0], points[:, 1], '-r')
plt.grid()
plt.axis('equal')

curvature_exp = aft.curvature_points(points[:, 0], points[:, 1])
plt.figure('exp curvature')
plt.plot(curvature_exp)

# New spline
tck_new, u_new = interpolate.splprep(points.transpose(), k=5, s=0.000000000001)
u = np.linspace(0.0, 1.0, 1000)
airfoil_exp = interpolate.splev(u, tck_new, der=0)
curvature_exp = aft.curvature_iges(tck_new)

plt.figure('Curvature')
plt.plot(new_curvature, label='curvature new')
plt.plot(curvature_exp, '-r', label='curvature exp')
plt.grid()
plt.legend()
plt.show()
