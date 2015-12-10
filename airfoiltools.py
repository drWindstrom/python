# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:25:35 2015

@author: winstroth
"""

import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import interpolate


def load_airfoil_iges(iges_file):
    """Loads the bspline of a 2D-Airfoil from an iges file.

    This function can handle airfoils define inside an iges file defined as a
    Rational B-Spline Curve Entity (Type 126). The airfoil must be defined
    with a single b-spline and there can only be one b-spline inside the iges
    file. It is assumed that the airfoil is defined in the x,y-plane.
    Therefore, z = 0.0 for all airfoil coordinates.

    Args:
        iges_file (str): path to the iges file containing the airfoil

    Returns:
        tuple: A tuple (t,c,k) containing the vector of knots, the B-spline
            coefficients, and the degree of the spline. The tuple can be used
            with scipy.interpolate.splev.

    """
    with open(iges_file, 'r') as f:
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
    matched_txt = ''.join(matched_list)
    entity_pars = [float(k) for k in matched_txt.split(',')]
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
    # Extract b-splines knots and bspline coefficients
    knots = np.array(entity_pars[6:7+A])
    bcoeffs = np.array(entity_pars[8+A+K:11+A+4*K])
    # Reshape control points
    faxis = len(bcoeffs) / 3
    bcoeffs = np.reshape(bcoeffs, (faxis, 3))
    # Construct b-spline
    tck = [knots, [bcoeffs[:, 0], bcoeffs[:, 1]], M]
    return tck


def curvature(dx, ddx, dy, ddy):
    """Returns the curvature of a curve.

    Args:
        dx (array): first derivative of curve with repect to x
        ddx (array): second derivative of curve with respect to x
        dy (array): first derivative of curve with repect to y
        ddy (array): second derivative of curve with respect to y

    Returns:
        array: curvature of the curve

    """
    curvature = (dx*ddy - ddx*dy)/((dx**2 + dy**2)**(3.0/2.0))
    return curvature


def curvature_points(x, y):
    """Calculate curvature of airfoil defined by point coordinates.

    Args:
        x (array): x-coordinates of the airfoil
        y (array): y-coordinates of the airfoil

    Returns:
        array: curvature of the airfoil

    """
    # Calculate first derivative
    dx = np.gradient(x)
    dy = np.gradient(y)
    # Calculate sec derivative
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    return curvature(dx, ddx, dy, ddy)


def curvature_iges(tck, res=1000):
    """Calculate curvature of airfoil defined by a bspline.

    Args:
        tck (tuple): A tuple (t,c,k) containing the vector of knots, the
            B-spline coefficients, and the degree of the spline.
        res (int): How many discreet data point to use along the airfoil's
            surface.

    Returns:
        array: curvature of the airfoil

    """
    u = np.linspace(0.0, 1.0, res)
    grad1 = interpolate.splev(u, tck, der=1)
    grad2 = interpolate.splev(u, tck, der=2)
    dx = grad1[0]
    dy = grad1[1]
    ddx = grad2[0]
    ddy = grad2[1]
    return curvature(dx, ddx, dy, ddy)


def find_te_point(tck):
    """Returns the trailing edge point of an airfoil defined by a bspline.

    The trailing edge point is defined as the point half way between the
    beginning and the end point of the bspline defining the surface of the
    airfoil.

    Returns:
        array: [x-coordinate, y-coorindate] of te_point
    """
    u0 = np.array(interpolate.splev(0.0, tck, der=0))
    u1 = np.array(interpolate.splev(1.0, tck, der=0))
    te_point = (u1 - u0)/2.0 + u0
    return te_point


def find_le_point(tck, te_point, tol=1.0e-8):
    """Finds the le_point along the airfoil curve.

    The le_point is defined as the point along the curve of the airfoil
    which has the greatest distance from the trailing edge point te_point.

    Args:
        tck (tuple): A tuple (t,c,k) containing the vector of knots, the
            B-spline coefficients, and the degree of the spline.
        te_point (array): The x- and y-coordinate of the trailing edge point
        tol (float): Tolerance level when to stop the iteration process. The
            iteration stops one the change of u_le between iterations falls
            below this tolerance level.

    Returns:
        tuple: A tuple (u_le, le_point). u_le is the bspline coordinate that
        corresponds to the leading edge point le_point.

        """
    u0 = 0.0
    u1 = 1.0
    u_le_stor = 0.0
    res = 1000
    u = np.linspace(u0, u1, res)
    airfoil_points = np.array(interpolate.splev(u, tck, der=0))
    # find greated distance between te_point and point on airfoil surface
    dist_vec = airfoil_points.transpose() - te_point
    dist = np.linalg.norm(dist_vec, axis=1)
    max_pos = dist.argmax()
    u_le = u[max_pos]
    while abs(u_le - u_le_stor) > tol:
        # store last u_le
        u_le_stor = u_le
        u = np.linspace(u[max_pos - 2], u[max_pos + 2], res)
        airfoil_points = np.array(interpolate.splev(u, tck, der=0))
        # find greated distance between te_point and point on airfoil surface
        dist_vec = airfoil_points.transpose() - te_point
        dist = np.linalg.norm(dist_vec, axis=1)
        max_pos = dist.argmax()
        u_le = u[max_pos]

    le_point = np.array(interpolate.splev(u_le, tck, der=0))
    return u_le, le_point


def translate_to_origin(tck, le_point):
    """Translates the bspline of the airfoil so that le_point will be at the
    origin of the coordinate system.

    Args:
        tck (tuple): A tuple (t,c,k) containing the vector of knots, the
            B-spline coefficients, and the degree of the spline.
        le_point (array): This point will be at (0, 0) after translation

    Returns:
        tuple: A tuple (t,c,k) containing the vector of knots, the B-spline
            coefficients, and the degree of the spline.

    """
    vec_zero = np.array([0, 0])
    vec_le_zero = vec_zero - le_point
    bcoeffs = np.array([tck[1][0], tck[1][1]])
    # Update bspline coefficients
    bcoeffs = bcoeffs.transpose() + vec_le_zero
    tck = [tck[0], [bcoeffs[:, 0], bcoeffs[:, 1]], tck[2]]
    return tck


def scale_airfoil(tck, le_point, te_point):
    """Scales the airfoil so that the distance from le to te is 1.0."""

    vec_le_te = te_point - le_point
    dist_le_te = np.linalg.norm(vec_le_te)
    scale = 1.0/dist_le_te
    bcoeffs = np.array([tck[1][0], tck[1][1]])
    # Update bspline coefficients
    bcoeffs = bcoeffs.transpose() * scale
    tck = [tck[0], [bcoeffs[:, 0], bcoeffs[:, 1]], tck[2]]
    return tck


def rotate_airfoil(tck, le_point, te_point):
    """Rotates the airfoil so that the choord of the airfoil will be on or
    parallel to the x-axis."""

    vec_x0 = [1.0, 0.0]
    vec_le_te = te_point - le_point
    # angle of v2 relative to v1 = atan2(v2.y,v2.x) - atan2(v1.y,v1.x)
    alpha = np.arctan2(vec_x0[1], vec_x0[0]) - np.arctan2(vec_le_te[1],
                                                          vec_le_te[0])
    # Get 2D rotation matrix
    rot_mat = np.array([[np.cos(alpha), -np.sin(alpha)],
                       [np.sin(alpha),  np.cos(alpha)]])
    bcoeffs = np.array([tck[1][0], tck[1][1]])
    # Update bspline coefficients
    bcoeffs = rot_mat.dot(bcoeffs)
    bcoeffs = bcoeffs.transpose()
    tck = [tck[0], [bcoeffs[:, 0], bcoeffs[:, 1]], tck[2]]
    return tck