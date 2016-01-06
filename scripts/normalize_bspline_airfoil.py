# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import os
import airfoiltools as aft
import pickle


inp_fname_igs = 'r22000.igs'

inp_dir = '/mnt/hgfs/GAeroFerRo/Referenzblatt_3d/3D-Referenzblatt/E-44_V2/2D_Profile'
out_dir_norm_bspline = '/mnt/hgfs/GAeroFerRo/Referenzblatt_3d/3D-Referenzblatt/E-44_V2/2D_Profile/bspline_norm'
out_dir_norm_pointwise = '/mnt/hgfs/GAeroFerRo/Referenzblatt_3d/3D-Referenzblatt/E-44_V2/2D_Profile/pointwise_norm'

# Load airfoil from iges file
iges_file = os.path.join(inp_dir, inp_fname_igs)
tck = aft.load_airfoil_iges(iges_file)

tck_norm, dist_le_te, rot_deg = aft.norm_bspline_airfoil(tck)
print('chord length: {}'.format(dist_le_te))
print('twist: {}'.format(-rot_deg))

tck_norm_mod = aft.correct_te(tck_norm, s=0.0000005, k=3)
num_points, points = aft.bspline_to_points(tck_norm_mod, min_step=5e-4,
                                           max_step=0.01)
print('num of points: {}'.format(num_points))

# Get name of input file
fname, _ = os.path.splitext(inp_fname_igs)

# Save airfoil in pointwise format (segment points)
pointwise_out = os.path.join(out_dir_norm_pointwise, fname + '_norm.dat')
aft.write_pointwise_seg(points, pointwise_out)

# Save bspline definition of airfoil
bspline_out = os.path.join(out_dir_norm_bspline, fname + '_bspline_norm.p')
with open(bspline_out, 'wb') as f:
    pickle.dump(tck_norm_mod, f)

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
curvature_norm = abs(aft.curvature_bspline(tck_norm, u))
curvature_norm_mod = abs(aft.curvature_bspline(tck_norm_mod, u))
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
