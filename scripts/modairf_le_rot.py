import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import modairfoil
# reload(modairfoil)


# User inputs
write_files = True
# for sin wave modification
alpha = 2.0
le_smooth = 1.0
s = 0.0
k = 5
# for input and output files
nairfoil = None
fname_airfoil = 'e44r14100.dat'
inp_dir = ('/mnt/hgfs/GAeroFerRo/Referenzblatt_3d/E-44_V2/2D_Profile/'
           'pointwise_norm')
out_dir = ('/mnt/hgfs/GAeroFerRo/Referenzblatt_3d/E-44_V2/'
           'profil_modifikationen/leading_edge_rotation')
# for point distribution of pointwise segment file
min_step = 1e-4
max_step = 0.01


# Create modification name
if nairfoil is None:
    nairfoil, _ = os.path.splitext(fname_airfoil)
mod_name = '{}_alpha_{}_le_rot'.format(nairfoil, alpha)
# Load airfoil
fname = os.path.join(inp_dir, fname_airfoil)
myfoil_org = modairfoil.ModAirfoil(airfcoords=fname,
                                   airf_name=nairfoil)

# Uncomment if we need to reorient the airfoil
myfoil_org.reorient()

# Normalize airfoil
myfoil_org.normalize()

# Correct trailing edge
myfoil_org.correct_te(k=3)

# Create copy of airfoil for modification
myfoil_mod = copy.deepcopy(myfoil_org)

# Apply leading edge rotation
myfoil_mod.rotate_le(alpha=alpha, nsamples=1000, le_smooth=le_smooth,
                     smoothing=s, degree=k)

# Normalize airfoil after modification
u_le = myfoil_mod.find_y(y_loc=0.0, u0=0.4, u1=0.6)
le_point = myfoil_mod.get_point(u=u_le)
myfoil_mod.normalize_chord(le_point=le_point)
myfoil_mod.le_to_origin(le_point=le_point)

# Correct leading edge
myfoil_mod.correct_te(k=k)

# Plots to control results
# Curvature
org_curv = myfoil_org.get_curvature(nsamples=10000)
mod_curv = myfoil_mod.get_curvature(nsamples=10000)
surf_len = myfoil_org.get_surface_len(nsamples=10000)
x = np.linspace(start=0.0, stop=surf_len, num=len(org_curv))
fig_curv = plt.figure('Curvature of {}'.format(nairfoil))
ax_curv = fig_curv.add_subplot(111)
ax_curv.set_title('Curvature of {}'.format(nairfoil))
# remove all lines
del ax_curv.lines[:]
ax_curv.plot(x, org_curv, '-b', label='org. airfoil')
ax_curv.plot(x, mod_curv, '-r', label='mod. airfoil')
ax_curv.set_xlabel('Surface location')
ax_curv.set_ylabel('Curvature')
ax_curv.legend()
ax_curv.grid(True)
fig_curv.canvas.draw()
# Airfoils
org_pts = myfoil_org.get_epoints(nsamples=10000)
mod_pts = myfoil_mod.get_epoints(nsamples=10000)
fig_airf = plt.figure('{} org. and mod.'.format(nairfoil))
ax_airf = fig_airf.add_subplot(111)
ax_airf.set_title('{} org. and mod.'.format(nairfoil))
# remove all lines
del ax_airf.lines[:]
ax_airf.plot(org_pts[:, 0], org_pts[:, 1], '-b', label='org. airfoil')
ax_airf.plot(mod_pts[:, 0], mod_pts[:, 1], '-r', label='mod. airfoil')
ax_airf.axis('equal')
ax_airf.set_xlim([-0.05, 1.05])
ax_airf.set_xlabel('x-coordinate')
ax_airf.set_ylabel('y-coordinate')
ax_airf.legend()
ax_airf.grid(True)
fig_curv.canvas.draw()

# Output pointwise segment file
if write_files:
    out_fname = os.path.join(out_dir, mod_name + '.dat')
    myfoil_mod.write_pointwise_seg(out_fname=out_fname, min_step=min_step,
                                   max_step=max_step)
    fname_curv = os.path.join(out_dir, '{}_curvature'.format(mod_name))
    fig_curv.savefig(fname_curv + '.svg')
    fig_curv.savefig(fname_curv + '.png')
    fig_curv.savefig(fname_curv + '.pdf')
    fname_airf = os.path.join(out_dir, '{}_profile'.format(mod_name))
    fig_airf.savefig(fname_airf + '.svg')
    fig_airf.savefig(fname_airf + '.png')
    fig_airf.savefig(fname_airf + '.pdf')

plt.show()
