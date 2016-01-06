# -*- coding: utf-8 -*-

import airfoiltools as aft
import flower.tecplot as tpl
import matplotlib.pyplot as plt

fname = 'grid_test.tec'
tecdat = tpl.load(fname)
x = tecdat[0]['CoordinateX'][:, 0, 0]
y = tecdat[0]['CoordinateY'][:, 0, 0]

curvi_tfd = aft.curvature_points(x, y)

plt.figure('Airfoil')
plt.plot(x, y)
plt.axis('equal')
plt.grid()

plt.figure('Curvature')
plt.plot(abs(curvi_tfd), '-1')

plt.show()
