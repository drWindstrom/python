# -*- coding: utf-8 -*-
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Path to iges file
#fname = 'test_profil.iges'
#fname = 'NACA643618.igs'
#fname= 'R17.igs'
fname = 'profil_15000mm.igs'
#fname = 'Windstrom2.igs'

# Define number of sampled points from bspline
bspline_res = 5.0e2

# Open iges file and extract txt
with open(fname, 'r') as f:
    file_txt = f.read()
# Extract the parameter data of the b-spline from the iges file.
# The text block starts with '126,' and ends with ';'. The text block
# consist of 'P', 'E', 'comma', 'dot', '0-9', '-' and whitespace.
pattern = re.compile(r'126,[PE,.0-9\-\s]*')
match = pattern.search(file_txt)
matched_txt = match.group()

# Split the string at newline characters and return a list of string
matched_list = matched_txt.splitlines()

# Reduce each string to the first 65 characters and remove trailing
# whitespace.
value_length = 65
for i in range(len(matched_list)):
    matched_list[i] = matched_list[i][0:value_length].rstrip()

# Join the list of strings and convert to float
bspline_txt = ''.join(matched_list)
entity_pars = [float(k) for k in bspline_txt.split(',')]
# Remove entity identifier
entity_pars.pop(0)
# First 6 numbers should be integers
num_of_int = 6
for i in range(num_of_int):
    entity_pars[i] = int(entity_pars[i])

# See Initial Graphics Exchange Specification 5.3 (IGES)
# for Rational B-Spline Curve Entity (Type 126)
K = entity_pars[0]
M = entity_pars[1]
N = 1 + K - M
A = N + 2*M
# Extract b-splines knots and control points
knots = np.array(entity_pars[6:7+A])
bspline_coeffs = np.array(entity_pars[8+A+K:11+A+4*K])

# Normalize knots to 1.0
max_knot = max(knots)
knots_norm = knots / max_knot
# Reshape control points
faxis = len(bspline_coeffs) / 3
bspline_coeffs = np.reshape(bspline_coeffs, (faxis, 3))
# Construct b-spline
tck = [knots, [bspline_coeffs[:, 0], bspline_coeffs[:, 1]], M]
# Define at what points the airfoil coordinates are interpolated
u = np.linspace(0.0, max(knots), bspline_res)
airfoil_points = interpolate.splev(u, tck, der=0)

tck_new, u_bak = interpolate.splprep(airfoil_points, s=0.0001)
u_new = np.linspace(0.0, 1.0, bspline_res)
airfoil_points_new = interpolate.splev(u_new, tck_new, der=0)


plt.figure('Bspline coefficients from iges')
plt.title('Bspline coefficients from iges')
plt.plot(bspline_coeffs[:, 0], bspline_coeffs[:, 1], 'ro')
plt.axis('equal')

plt.figure('New Bspline coefficients')
plt.title('New Bspline coefficients')
plt.plot(tck_new[1][0], tck_new[1][1], 'ro')
plt.axis('equal')

plt.figure('Points from original bspline')
plt.title('Points from original bspline')
plt.plot(airfoil_points[0], airfoil_points[1], 'bo')
plt.axis('equal')

plt.figure('Points from new bspline')
plt.title('Points from new bspline')
plt.plot(airfoil_points_new[0], airfoil_points_new[1], 'bo')
plt.axis('equal')

plt.figure('Original knots from iges')
plt.title('Original knots from iges')
plt.plot(tck[0], 'ro')

plt.figure('New knots')
plt.title('New knots')
plt.plot(tck_new[0], 'ro')

