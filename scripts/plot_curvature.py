# -*- coding: utf-8 -*-

import airfoiltools as aft
import flower.tecplot as tpl

tecdat = tpl.load('../airfoils/3d_ref_rot_r11300_curvature.dat')
x = tecdat[0]['x'][:, 0, 0]
y = tecdat[0]['y'][:, 0, 0]
curvi_dlr = tecdat[0]['curvi'][:, 0, 0]
curvi_tfd =  aft.curvature_points(x, y)
