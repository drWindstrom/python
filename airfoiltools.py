# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:25:35 2015

@author: winstroth
"""

import numpy as np
import re
from scipy import interpolate, optimize


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
        dx (array): first derivative of curve with respect to x
        ddx (array): second derivative of curve with respect to x
        dy (array): first derivative of curve with respect to y
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


def curvature_bspline(tck, u):
    """Calculate curvature of airfoil defined by a Bspline.

    Args:
        tck (tuple): A tuple (t,c,k) containing the vector of knots, the
            B-spline coefficients, and the degree of the spline.
        res (array): u coordinate of Bspline for which to return curvature

    Returns:
        array: curvature of the airfoil

    """
    grad1 = interpolate.splev(u, tck, der=1)
    grad2 = interpolate.splev(u, tck, der=2)
    dx = grad1[0]
    dy = grad1[1]
    ddx = grad2[0]
    ddy = grad2[1]
    return curvature(dx, ddx, dy, ddy)


def find_te_point(tck):
    """Returns the trailing edge point of an airfoil defined by a Bspline.

    The trailing edge point is defined as the point half way between the
    beginning and the end point of the Bspline defining the surface of the
    airfoil.

    Returns:
        array: [x-coordinate, y-coordinate] of te_point
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
        tuple: A tuple (u_le, le_point). u_le is the Bspline coordinate that
        corresponds to the leading edge point le_point.

        """
    u0 = 0.0
    u1 = 1.0
    u_le_stor = 0.0
    res = 1000
    u = np.linspace(u0, u1, res)
    airfoil_points = np.array(interpolate.splev(u, tck, der=0))
    # find greatest distance between te_point and point on airfoil surface
    dist_vec = airfoil_points.transpose() - te_point
    dist = np.linalg.norm(dist_vec, axis=1)
    max_pos = dist.argmax()
    u_le = u[max_pos]
    while abs(u_le - u_le_stor) > tol:
        # store last u_le
        u_le_stor = u_le
        u = np.linspace(u[max_pos - 2], u[max_pos + 2], res)
        airfoil_points = np.array(interpolate.splev(u, tck, der=0))
        # find greatest distance between te_point and point on airfoil surface
        dist_vec = airfoil_points.transpose() - te_point
        dist = np.linalg.norm(dist_vec, axis=1)
        max_pos = dist.argmax()
        u_le = u[max_pos]

    le_point = np.array(interpolate.splev(u_le, tck, der=0))
    return u_le, le_point


def translate_to_origin(tck, le_point):
    """Translates the Bspline of the airfoil so that le_point will be at the
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
    # Update Bspline coefficients
    bcoeffs = bcoeffs.transpose() + vec_le_zero
    tck = [tck[0], [bcoeffs[:, 0], bcoeffs[:, 1]], tck[2]]
    return tck


def scale_airfoil(tck, le_point, te_point):
    """Scales the airfoil so that the distance from le to te is 1.0."""

    vec_le_te = te_point - le_point
    dist_le_te = np.linalg.norm(vec_le_te)
    scale = 1.0/dist_le_te
    bcoeffs = np.array([tck[1][0], tck[1][1]])
    # Update Bspline coefficients
    bcoeffs = bcoeffs.transpose() * scale
    tck = [tck[0], [bcoeffs[:, 0], bcoeffs[:, 1]], tck[2]]
    return tck, dist_le_te


def rotate_airfoil(tck, le_point, te_point):
    """Rotates the airfoil so that the chord of the airfoil will be on or
    parallel to the x-axis."""

    vec_x0 = [1.0, 0.0]
    vec_le_te = te_point - le_point
    # angle of v2 relative to v1 = atan2(v2.y,v2.x) - atan2(v1.y,v1.x)
    alpha = np.arctan2(vec_x0[1], vec_x0[0]) - np.arctan2(vec_le_te[1],
                                                          vec_le_te[0])
    rot_deg = alpha * 180.0 / np.pi

    # Get 2D rotation matrix
    rot_mat = np.array([[np.cos(alpha), -np.sin(alpha)],
                       [np.sin(alpha),  np.cos(alpha)]])
    bcoeffs = np.array([tck[1][0], tck[1][1]])
    # Update Bspline coefficients
    bcoeffs = rot_mat.dot(bcoeffs)
    bcoeffs = bcoeffs.transpose()
    tck = [tck[0], [bcoeffs[:, 0], bcoeffs[:, 1]], tck[2]]
    return tck, rot_deg


def bspline_to_points(tck, min_step=1e-4, max_step=0.01):
    """Discretizes the Bspline and returns the discrete points.

    The step width is based on the curvature of the Bspline and on min_step
    and max_step. The step width will be min_step at the point of maximum
    curvature and will never be farther than max_step.

    Args:
        tck (tuple): A tuple (t,c,k) containing the vector of knots, the
            B-spline coefficients, and the degree of the spline.
        min_step (float): Minimum step width at point of maximum curvature
        max_step (float): Maximum step width

    Returns:
        tuple: (num_points, points) The number of points num_points and an
            array points with containing the discretized points.

    """
    u = 0.0
    points = []
    # Find maximum curvature
    u_lin = np.linspace(0.0, 1.0, 10000)
    max_curv = max(curvature_bspline(tck, u_lin))
    # Get scale factor
    scale = min_step * max_curv
    # Step along Bspline
    while u < 1.0:
        points.append(interpolate.splev(u, tck, der=0))
        step = 1.0 / abs(curvature_bspline(tck, u)) * scale
        if step > max_step:
            step = max_step
        u += step
    points.append(interpolate.splev(1.0, tck, der=0))
    points = np.array(points)
    num_points, _ = points.shape
    return num_points, points


def write_pointwise_seg(points, fname):
    """Writes coordinates in points to fname in pointwise segment format."""

    num_points, num_coordinates = points.shape
    # If we only have x,y-coordinates append zeros for z
    if num_coordinates == 2:
        zeros = np.zeros((num_points, 1))
        points = np.hstack((points, zeros))

    np.savetxt(fname, points, header='{}'.format(num_points), comments='')


def norm_bspline_airfoil(tck):
    """Returns the normalized airfoil defined by tck.

    Args:
        tck (tuple): A tuple (t,c,k) containing the vector of knots, the
            B-spline coefficients, and the degree of the spline.

    Returns:
        tuple: A tuple (t,c,k) containing the vector of knots, the B-spline
            coefficients, and the degree of the spline.

    """
    # Find leading and trailing edge points
    te_point = find_te_point(tck)
    u_le, le_point = find_le_point(tck, te_point)
    # Translate le_point to origin
    tck = translate_to_origin(tck, le_point)
    # Scale airfoil
    tck, dist_le_te = scale_airfoil(tck, le_point, te_point)
    # Rotate airfoil
    tck, rot_deg = rotate_airfoil(tck, le_point, te_point)
    return tck, dist_le_te, rot_deg


def bspl_find_x(x_loc, start, end, tck):
    """Returns the u coordinate of tck that corresponds to x.

    Args:
        x_loc (float): The x-location we want to know the corresponding
            u-coordinate of the spline to
        start (float): start of the interval we want to look in
        end (float): end of the interval we want to look in
        tck (tuple): A tuple (t,c,k) containing the vector of knots, the
            B-spline coefficients, and the degree of the spline.

    Returns:
        float: The u coordinate that corresponds to x

    Raises:
        ValueError: If f(start) and f(end) do not have opposite signs or in
            other words: If the x-location is not found in the given interval.

    """
    def f(x, tck):
        points = interpolate.splev(x, tck, der=0)
        return x_loc - points[0]
    u = optimize.brentq(f=f, a=start, b=end, args=(tck,))
    return u


def correct_te(tck, k):
    """Corrects the trailing edge of a flatback airfoil.

    This corrections will make the trailing edge of the normalized flatback
    airfoil align with the y-axis.

    Args:
        tck (tuple): A tuple (t,c,k) containing the vector of knots, the
            B-spline coefficients, and the degree of the spline.
        k (int): The degree of the returned bspline

    Return:
        tuple: A tuple (t,c,k) containing the vector of knots, the
            B-spline coefficients, and the degree of the spline.

    """
    try:
        u0_x = bspl_find_x(x_loc=1.0, start=0.0, end=0.1, tck=tck)
    except ValueError:
        u0_x = None
    try:
        u1_x = bspl_find_x(x_loc=1.0, start=0.9, end=1.0, tck=tck)
    except ValueError:
        u1_x = None

    if u0_x is not None and u1_x is not None:
        u = np.linspace(u0_x, u1_x, 1000)
        points = interpolate.splev(u, tck, der=0)
        tck_norm_mod = interpolate.splprep(points, s=0.0, k=k)
    elif u0_x is None and u1_x is not None:
        u = np.linspace(0.0, u1_x, 1000)
        points = interpolate.splev(u, tck, der=0)
        p_u0 = [points[0][0], points[1][0]]
        u0_grad = interpolate.splev(0.0, tck, der=1)
        dx = 1.0 - p_u0[0]
        dy = dx * u0_grad[1] / u0_grad[0]
        p_new = [1.0, p_u0[1] + dy]
        x_pts = np.insert(points[0], 0, p_new[0])
        y_pts = np.insert(points[1], 0, p_new[1])
        tck_norm_mod, _ = interpolate.splprep([x_pts, y_pts], s=0.0, k=k)
    elif u0_x is not None and u1_x is None:
        u = np.linspace(u0_x, 1.0, 1000)
        points = interpolate.splev(u, tck, der=0)
        p_u1 = [points[0][-1], points[1][-1]]
        u1_grad = interpolate.splev(1.0, tck, der=1)
        dx = 1.0 - p_u1[0]
        dy = dx * u1_grad[1] / u1_grad[0]
        p_new = [1.0, p_u1[1] + dy]
        x_pts = np.append(points[0], p_new[0])
        y_pts = np.append(points[1], p_new[1])
        tck_norm_mod, _ = interpolate.splprep([x_pts, y_pts], s=0.0, k=k)
    else:
        raise ValueError('Something is wrong with the bspline!')
    return tck_norm_mod


def smooth_bspline(tck, num_points, s, k):
    """Corrects the trailing edge of a flatback airfoil.

    This corrections will make the trailing edge of the normalized flatback
    airfoil align with the y-axis.

    Args:
        tck (tuple): A tuple (t,c,k) containing the vector of knots, the
            B-spline coefficients, and the degree of the spline.
        num_points (int): Number of points along the bspline curve used for
            reconstruction
        s (float): Smoothing of bspline
        k (int): The degree of the returned bspline

    Return:
        tuple: A tuple (t,c,k) containing the vector of knots, the
            B-spline coefficients, and the degree of the spline.

    """
    u = np.linspace(0.0, 1.0, num_points)
    points = interpolate.splev(u, tck, der=0)
    tck_smooth, _ = interpolate.splprep(points, s=s, k=k)
    return tck_smooth

