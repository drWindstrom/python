# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:25:35 2015

@author: winstroth
"""
import re
from scipy import interpolate, optimize
import matplotlib.pyplot as plt
import numpy as np
import os


def curvature(dx, ddx, dy, ddy):
    """Returns the curvature of a curve.

    Args:
        dx (array): first derivative of curve with respect to x
        ddx (array): second derivative of curve with respect to x
        dy (array): first derivative of curve with respect to y
        ddy (array): second derivative of curve with respect to

    Returns:
        array: curvature of the curve

    """
    curvature = (dx*ddy - ddx*dy)/((dx**2 + dy**2)**(3.0/2.0))
    return curvature


class AirfGeom(object):
    """Class holds coordinates of the airfoil and modification functions."""

    def __init__(self, airfcoords, airf_name='airfoil', nsamples=10000,
                 comments='#', delimiter=None, skiprows=1, usecols=(0, 1),
                 smoothing=0.0, degree=3):
        """Loads the airfoil."""
        self.airf_name = airf_name
        self.nsamples = nsamples
        self.lpoints = None
        self.lbspline = None
        # Handle airfcoords by type
        if type(airfcoords) is tuple:
            self.tck = airfcoords
            self.lbspline = airfcoords
        elif type(airfcoords) is str:
            _, fext = os.path.splitext(airfcoords)
            if fext.lower() == '.txt' or fext.lower() == '.dat':
                self.load_point_data(points_file=airfcoords, comments=comments,
                                     delimiter=delimiter, skiprows=skiprows,
                                     usecols=usecols, smoothing=smoothing,
                                     degree=degree)
            elif fext.lower() == '.iges' or fext.lower() == '.igs':
                self.load_iges(iges_file=airfcoords)
            else:
                raise IOError('Wrong file type. Only the following extensions '
                              'are support: *.txt, *.dat, *.igs or *.iges.')
        else:
            raise TypeError('The type of airfcoorfs must either be str or '
                            'tuple but the supplied type is {}.'.format(
                                type(airfcoords)))

    def load_iges(self, iges_file):
        """Loads the bspline of a 2D-Airfoil from an iges file.

        This function can handle airfoils define inside an iges file defined
        as a Rational B-Spline Curve Entity (Type 126). The airfoil must be
        defined with a single b-spline and there can only be one b-spline
        inside the iges file. It is assumed that the airfoil is defined in the
        x,y-plane. Therefore, z = 0.0 for all airfoil coordinates.

        Args:
            iges_file (str): path to the iges file containing the airfoil

        Returns:
            tuple: A tuple (t,c,k) containing the vector of knots, the B-spline
                coefficients, and the degree of the spline. The tuple can be
                used with scipy.interpolate.splev.

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
        self.tck = [knots, [bcoeffs[:, 0], bcoeffs[:, 1]], M]
        self.lbspline = [knots, [bcoeffs[:, 0], bcoeffs[:, 1]], M]

    def load_point_data(self, points_file, comments='#', delimiter=None,
                        skiprows=1, usecols=(0, 1), smoothing=0.0, degree=3):
        """Loads the point data of a 2D-Airfoil from a text file.

        The function is more or less a wrapper for numpy.loadtxt. Please look
        there for more info.

        Args:
            points_file (str): path to the text file containing the point
            coordinates for the airfoil

        Returns:
            np.array: nx2 Array with x- and y-coordinates

        """
        self.lpoints = np.loadtxt(points_file, comments=comments,
                                  delimiter=delimiter, skiprows=skiprows,
                                  usecols=usecols)
        x = [self.lpoints[:, 0], self.lpoints[:, 1]]
        self.tck, _ = interpolate.splprep(x, s=smoothing, k=degree)

    def reorient(self):
        """Reorients the spline of the airfoil."""
        points = self.get_epoints()
        points = np.flipud(points)
        self.tck, _ = interpolate.splprep([points[:, 0], points[:, 1]],
                                          s=0.0, k=self.tck[2])

    def get_curvature(self, u=None, nsamples=None):
        """Calculate curvature of the airfoil."""
        if nsamples is None:
            nsamples = self.nsamples
        if u is None:
            u = np.linspace(0.0, 1.0, nsamples)
        grad1 = interpolate.splev(u, self.tck, der=1)
        grad2 = interpolate.splev(u, self.tck, der=2)
        dx = grad1[0]
        dy = grad1[1]
        ddx = grad2[0]
        ddy = grad2[1]
        return curvature(dx, ddx, dy, ddy)

    def get_tan_vecs(self, nsamples):
        """Returns nsamples equidistant tangent vectors along blade surface."""
        u = np.linspace(0.0, 1.0, nsamples)
        pvecs = interpolate.splev(u, self.tck, der=0)
        pvecs = np.array([pvecs[0], pvecs[1]])
        grad = interpolate.splev(u, self.tck, der=1)
        tvecs = np.array([grad[0], grad[1]])
        tnorms = np.linalg.norm(tvecs, axis=0)
        tvecs = tvecs/tnorms
        return tvecs, pvecs, u

    def get_normal_vecs(self, nsamples):
        """Returns nsamples equidistant normal vectors along blade surface."""
        tvecs, pvecs, u = self.get_tan_vecs(nsamples=nsamples)
        nvecs = np.array([tvecs[1, :], -tvecs[0, :]])
        return nvecs, pvecs, u

    def get_point(self, u):
        """Return the point on the airfoil the corresponds to coordinate u.

        Returns:
            (np.array): [x-coordinate, y-coordinate]
        """
        return np.array(interpolate.splev(u, self.tck, der=0))

    def get_epoints(self, u0=0.0, u1=1.0, nsamples=None):
        """Returns nsamples equidistant points of the airfoil.

        We can specify the interval where we want our points with u0 und u1.

        Returns:
            (np.array): nx2 where n = number of points. First column are x-
                coordinates and second column are y-coordinates.
        """
        if nsamples is None:
            nsamples = self.nsamples
        u = np.linspace(u0, u1, nsamples)
        return np.array(interpolate.splev(u, self.tck, der=0)).transpose()

    def get_dpoints(self, min_step=1e-4, max_step=0.01):
        """Discretizes the Bspline and returns the discrete points.

        The step width is based on the curvature of the Bspline and on min_step
        and max_step. The step width will be min_step at the point of maximum
        curvature and will never be farther than max_step.

        Args:
            min_step (float): Minimum step width at point of maximum curvature
            max_step (float): Maximum step width

        Returns:
            tuple: (num_points, points) The number of points num_points and an
                np.array (nx2 where n = number of points) containing the
                discretized points.

        """
        u = 0.0
        points = []
        # Find maximum curvature
        max_curv = max(abs(self.get_curvature()))
        # Get scale factor
        scale = min_step * max_curv
        # Step along Bspline
        while u < 1.0:
            points.append(interpolate.splev(u, self.tck, der=0))
            step = 1.0 / abs(self.get_curvature(u=u)) * scale
            if step > max_step:
                step = max_step
            u += step
        points.append(interpolate.splev(1.0, self.tck, der=0))
        points = np.array(points)
        num_points, _ = points.shape
        return num_points, points

    def get_te_point(self):
        """Returns the trailing edge point of an airfoil defined by a Bspline.

        The trailing edge point is defined as the point half way between the
        beginning and the end point of the Bspline defining the surface of the
        airfoil.

        Returns:
            array: [x-coordinate, y-coordinate] of te_point
        """
        u0 = np.array(interpolate.splev(0.0, self.tck, der=0))
        u1 = np.array(interpolate.splev(1.0, self.tck, der=0))
        te_point = (u1 - u0)/2.0 + u0
        return te_point

    def get_le_point(self, te_point=None, nsamples=None, tol=1.0e-8):
        """Finds the le_point along the airfoil curve.

        The le_point is defined as the point along the curve of the airfoil
        which has the greatest distance from the trailing edge point te_point.
        If te_point is not given, we use self.get_te_point() to find it.

        Args:
            te_point (array): The x- and y-coordinate of the trailing edge
                point. If not given, we use self.get_te_point()
            tol (float): Tolerance level when to stop the iteration process.
                The iteration stops one the change of u_le between iterations
                falls below this tolerance level.

        Returns:
            tuple: A tuple (u_le, le_point). u_le is the Bspline coordinate
            that corresponds to the leading edge point le_point and le_point
            is an np.array [x-coordinate, y-coordinate].

        """
        if nsamples is None:
            nsamples = self.nsamples
        if te_point is None:
            te_point = self.get_te_point()
        u0 = 0.0
        u1 = 1.0
        u_le_stor = 0.0
        u = np.linspace(u0, u1, nsamples)
        airfoil_points = np.array(interpolate.splev(u, self.tck, der=0))
        # find greatest distance between te_point and point on airfoil surface
        dist_vec = airfoil_points.transpose() - te_point
        dist = np.linalg.norm(dist_vec, axis=1)
        max_pos = dist.argmax()
        u_le = u[max_pos]
        while abs(u_le - u_le_stor) > tol:
            # store last u_le
            u_le_stor = u_le
            u = np.linspace(u[max_pos - 2], u[max_pos + 2], nsamples)
            airfoil_points = np.array(interpolate.splev(u, self.tck, der=0))
            # find greatest distance between te_point and point on airfoil
            # surface
            dist_vec = airfoil_points.transpose() - te_point
            dist = np.linalg.norm(dist_vec, axis=1)
            max_pos = dist.argmax()
            u_le = u[max_pos]

        # le_point = np.array(interpolate.splev(u_le, self.tck, der=0))
        le_point = self.get_point(u=u_le)
        return u_le, le_point

    def le_to_origin(self, le_point=None, output=False):
        """Translates the Bspline of the airfoil so that le_point will be at
        the origin of the coordinate system.

        Args:
            le_point (array): This point will be at (0, 0) after translation.
            If not given, we use self.get_le_point()

        Returns:
            tuple: A tuple (t,c,k) containing the vector of knots, the B-spline
                coefficients, and the degree of the spline.

        """
        if le_point is None:
            _, le_point = self.get_le_point()
        vec_zero = np.array([0, 0])
        vec_le_zero = vec_zero - le_point
        bcoeffs = np.array([self.tck[1][0], self.tck[1][1]])
        # Update Bspline coefficients
        bcoeffs = bcoeffs.transpose() + vec_le_zero
        self.tck = [self.tck[0], [bcoeffs[:, 0], bcoeffs[:, 1]], self.tck[2]]
        if output:
            return self.tck

    def normalize_chord(self, le_point=None, te_point=None, output=None):
        """Scales the airfoil so that the distance from le to te is 1.0."""

        if le_point is None:
            _, le_point = self.get_le_point()
        if te_point is None:
            te_point = self.get_te_point()
        vec_le_te = te_point - le_point
        dist_le_te_old = np.linalg.norm(vec_le_te)
        scale = 1.0/dist_le_te_old
        bcoeffs = np.array([self.tck[1][0], self.tck[1][1]])
        # Update Bspline coefficients
        bcoeffs = bcoeffs.transpose() * scale
        self.tck = [self.tck[0], [bcoeffs[:, 0], bcoeffs[:, 1]], self.tck[2]]
        if output:
            return self.tck, dist_le_te_old

    def derotate(self, le_point=None, te_point=None, output=None):
        """Rotates the airfoil so that the chord of the airfoil will be on or
        parallel to the x-axis."""

        if le_point is None:
            _, le_point = self.get_le_point()
        if te_point is None:
            te_point = self.get_te_point()
        vec_x0 = [1.0, 0.0]
        vec_le_te = te_point - le_point
        # angle of v2 relative to v1 = atan2(v2.y,v2.x) - atan2(v1.y,v1.x)
        alpha = np.arctan2(vec_x0[1], vec_x0[0]) - np.arctan2(vec_le_te[1],
                                                              vec_le_te[0])
        rot_deg = alpha * 180.0 / np.pi

        # Get 2D rotation matrix
        rot_mat = np.array([[np.cos(alpha), -np.sin(alpha)],
                           [np.sin(alpha),  np.cos(alpha)]])
        bcoeffs = np.array([self.tck[1][0], self.tck[1][1]])
        # Update Bspline coefficients
        bcoeffs = rot_mat.dot(bcoeffs)
        bcoeffs = bcoeffs.transpose()
        self.tck = [self.tck[0], [bcoeffs[:, 0], bcoeffs[:, 1]], self.tck[2]]
        if output:
            return self.tck, rot_deg

    def normalize(self, output=None):
        """Returns the normalized airfoil defined by tck.

        Returns:
            tuple: A tuple (t,c,k) containing the vector of knots, the B-spline
                coefficients, and the degree of the spline.

        """
        # Translate le_point to origin
        self.le_to_origin()
        # Scale chord to 1
        _, dist_le_te_old = self.normalize_chord(output=True)
        # tck, dist_le_te = scale_airfoil(tck, le_point, te_point)
        # Rotate
        _, rot_deg = self.derotate(output=True)
        # tck, rot_deg = rotate_airfoil(tck, le_point, te_point)
        if output:
            return self.tck, dist_le_te_old, rot_deg

    def find_x(self, x_loc, u0, u1):
        """Returns the u coordinate of tck that corresponds to x.

        Args:
            x_loc (float): The x-location we want to know the corresponding
                u-coordinate of the spline to
            start (float): start of the interval we want to look in
            end (float): end of the interval we want to look in

        Returns:
            float: The u coordinate that corresponds to x

        Raises:
            ValueError: If f(start) and f(end) do not have opposite signs or in
                other words: If the x-location is not found in the given
                interval.

        """
        def f(x, tck):
            points = interpolate.splev(x, tck, der=0)
            return x_loc - points[0]
        u = optimize.brentq(f=f, a=u0, b=u1, args=(self.tck,))
        return u

    def find_y(self, y_loc, u0, u1):
        """Returns the u coordinate of tck that corresponds to y.

        Args:
            y_loc (float): The y-location we want to know the corresponding
                u-coordinate of the spline to
            start (float): start of the interval we want to look in
            end (float): end of the interval we want to look in

        Returns:
            float: The u coordinate that corresponds to y

        Raises:
            ValueError: If f(start) and f(end) do not have opposite signs or in
                other words: If the y-location is not found in the given
                interval.

        """
        def f(x, tck):
            points = interpolate.splev(x, tck, der=0)
            return y_loc - points[1]
        u = optimize.brentq(f=f, a=u0, b=u1, args=(self.tck,))
        return u

    def correct_te(self, k):
        """Corrects the trailing edge of a flatback airfoil.

        This corrections will make the trailing edge of the normalized flatback
        airfoil align with the y-axis.

        Args:
            k (int): The degree of the returned bspline

        Return:
            tuple: A tuple (t,c,k) containing the vector of knots, the
                B-spline coefficients, and the degree of the spline.

        """
        try:
            u0_x = self.find_x(x_loc=1.0, u0=0.0, u1=0.1)
        except ValueError:
            u0_x = None
        try:
            u1_x = self.find_x(x_loc=1.0, u0=0.9, u1=1.0)
        except ValueError:
            u1_x = None

        if u0_x is not None and u1_x is not None:
            u = np.linspace(u0_x, u1_x, 1000)
            points = interpolate.splev(u, self.tck, der=0)
            self.tck = interpolate.splprep(points, s=0.0, k=self.tck[2])
        elif u0_x is None and u1_x is not None:
            u = np.linspace(0.0, u1_x, 1000)
            points = interpolate.splev(u, self.tck, der=0)
            p_u0 = [points[0][0], points[1][0]]
            u0_grad = interpolate.splev(0.0, self.tck, der=1)
            dx = 1.0 - p_u0[0]
            dy = dx * u0_grad[1] / u0_grad[0]
            p_new = [1.0, p_u0[1] + dy]
            x_pts = np.insert(points[0], 0, p_new[0])
            y_pts = np.insert(points[1], 0, p_new[1])
            self.tck, _ = interpolate.splprep([x_pts, y_pts], s=0.0,
                                              k=self.tck[2])
        elif u0_x is not None and u1_x is None:
            u = np.linspace(u0_x, 1.0, 1000)
            points = interpolate.splev(u, self.tck, der=0)
            p_u1 = [points[0][-1], points[1][-1]]
            u1_grad = interpolate.splev(1.0, self.tck, der=1)
            dx = 1.0 - p_u1[0]
            dy = dx * u1_grad[1] / u1_grad[0]
            p_new = [1.0, p_u1[1] + dy]
            x_pts = np.append(points[0], p_new[0])
            y_pts = np.append(points[1], p_new[1])
            self.tck, _ = interpolate.splprep([x_pts, y_pts], s=0.0,
                                              k=self.tck[2])
        else:
            raise ValueError('Something is wrong with the bspline!')

    def write_pointwise_seg(self, out_fname, min_step=1e-4, max_step=0.01,
                            verbose=True):
        """Writes a pointwise segment file cotaining the airfoil shape."""

        num_points, points = self.get_dpoints(min_step=min_step,
                                              max_step=max_step)
        if verbose:
            print('Number of points to write: {}'.format(num_points))
        # Append zeros for z
        zeros = np.zeros((num_points, 1))
        points = np.hstack((points, zeros))
        np.savetxt(out_fname, points, header='{}'.format(num_points),
                   comments='')

    def smooth(self, smoothing, degree=None):
        """Smooth the airfoil curve."""
        points = self.get_epoints()
        if degree is None:
            degree = self.tck[2]
        self.tck, _ = interpolate.splprep([points[:, 0], points[:, 1]],
                                          s=smoothing, k=degree)

    def get_surface_len(self, nsamples=10000):
        surf_pts = self.get_epoints(nsamples=nsamples).transpose()
        pt_to_pt_vecs = surf_pts[:, 0:-1] - surf_pts[:, 1:]
        pt_dist_vec = np.linalg.norm(pt_to_pt_vecs, axis=0)
        return np.sum(pt_dist_vec)

    def plot(self, lformat='-r'):
        """bla."""
        # Plot airfoil
        plt.figure(self.airf_name)
        u = np.linspace(0.0, 1.0, self.nsamples)
        points = interpolate.splev(u, self.tck, der=0)
        plt.plot(points[0], points[1], lformat, label=self.airf_name)
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_curvature(self, lformat='-r'):
        """bla."""
        plt.figure('Curvature of {}'.format(self.airf_name))
        plt.plot(abs(self.get_curvature()), lformat)
        plt.grid(True)
        plt.show()

    def plot_dpoints(self, min_step=1e-4, max_step=0.01, lformat='or'):
        """bla."""
        num_points, points = self.get_dpoints(min_step=min_step,
                                              max_step=max_step)
        # Plot new point distribution
        plt.figure('New point distribution for {}'.format(self.airf_name))
        plt.plot(points[:, 0], points[:, 1], lformat, label=self.airf_name)
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_le_te_points(self):
        """bla."""
        te_point = self.get_te_point()
        u_le, le_point = self.get_le_point()
        # Plot leading and trailing edge points
        plt.figure('Leading and trailing edge of {}'.format(self.airf_name))
        points = self.get_epoints()
        plt.plot(points[:, 0], points[:, 1], label=self.airf_name)
        plt.plot(te_point[0], te_point[1], 'or', label='te_point')
        plt.plot(le_point[0], le_point[1], 'og', label='le_point')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.show()

    def plot_ss_ps(self):
        """bla."""
        u_le, le_point = self.get_le_point()
        pts_ss = self.get_epoints(u0=0.0, u1=u_le, nsamples=1000)
        pts_ps = self.get_epoints(u0=u_le, u1=1.0, nsamples=1000)
        # Plot pressure and suction side
        plt.figure('Pressure and suction side for {}'.format(self.airf_name))
        plt.plot(pts_ss[:, 0], pts_ss[:, 1], '-r', label='suction side')
        plt.plot(pts_ps[:, 0], pts_ps[:, 1], '-b', label='pressure side')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.show()

    def plot_tvec(self, nsamples, vec_scal=0.01):
        # Plot airfoil
        plt.figure('Tangent vectors for {}.'.format(self.airf_name))
        u = np.linspace(0.0, 1.0, 10000)
        points = interpolate.splev(u, self.tck, der=0)
        plt.plot(points[0], points[1], label=self.airf_name)
        plt.axis('equal')
        plt.grid(True)
        # Get tangent vectors and point vectors
        tvecs, pvecs = self.get_tan_vecs(nsamples=nsamples)
        _, vec_len = tvecs.shape
        for i in range(vec_len):
            x = pvecs[0, i]
            y = pvecs[1, i]
            dirx = tvecs[0, i]
            diry = tvecs[1, i]
            lx = [x, x+dirx*vec_scal]
            ly = [y, y+diry*vec_scal]
            if i == 0:
                plt.plot(lx, ly, '-ro', label='tangent vectors')
            else:
                plt.plot(lx, ly, '-ro')

        plt.legend()
        plt.show()

    def plot_nvec(self, nsamples, vec_scal=0.01):
        # Plot airfoil
        plt.figure('Normal vectors for {}.'.format(self.airf_name))
        u = np.linspace(0.0, 1.0, 10000)
        points = interpolate.splev(u, self.tck, der=0)
        plt.plot(points[0], points[1], label=self.airf_name)
        plt.axis('equal')
        plt.grid(True)
        # Get normal vectors and point vectors
        nvecs, pvecs = self.get_normal_vecs(nsamples=nsamples)
        _, vec_len = nvecs.shape
        for i in range(vec_len):
            x = pvecs[0, i]
            y = pvecs[1, i]
            dirx = nvecs[0, i]
            diry = nvecs[1, i]
            lx = [x, x+dirx*vec_scal]
            ly = [y, y+diry*vec_scal]
            if i == 0:
                plt.plot(lx, ly, '-ro', label='normal vectors')
            else:
                plt.plot(lx, ly, '-ro')

        plt.legend()
        plt.show()
