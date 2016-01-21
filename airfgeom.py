# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:25:35 2015

@author: winstroth
"""
import re
import os
from scipy import interpolate, optimize
import matplotlib.pyplot as plt
import numpy as np


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


class AirfGeom(object):
    """Class holds coordinates of the airfoil and modification functions."""

    def __init__(self, tck=None, airf_name='airfoil'):
        """Loads the airfoil."""
        self.tck = tck
        self.airf_name = airf_name
        self.lpoints = None
        self.lbspline = None

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
        self.tck, _ = interpolate.splprep(x, s=smoothing,
                                          k=degree)

    def get_curvature(self, u=None, nsamples=1000):
        """Calculate curvature of the airfoil."""
        if u is None:
            u = np.linspace(0.0, 1.0, nsamples)
        grad1 = interpolate.splev(u, self.tck, der=1)
        grad2 = interpolate.splev(u, self.tck, der=2)
        dx = grad1[0]
        dy = grad1[1]
        ddx = grad2[0]
        ddy = grad2[1]
        return curvature(dx, ddx, dy, ddy)

    def get_dpoints(self, min_step=1e-4, max_step=0.01, nsamples=1000):
        """Discretizes the Bspline and returns the discrete points.

        The step width is based on the curvature of the Bspline and on min_step
        and max_step. The step width will be min_step at the point of maximum
        curvature and will never be farther than max_step.

        Args:
            min_step (float): Minimum step width at point of maximum curvature
            max_step (float): Maximum step width

        Returns:
            tuple: (num_points, points) The number of points num_points and an
                array points with containing the discretized points.

        """
        u = 0.0
        points = []
        # Find maximum curvature
        max_curv = np.max(self.get_curvature(nsamples))
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

    def plot(self, nsamples=1000, lformat='-r'):
        """bla."""
        # Plot airfoil
        plt.figure(self.airf_name)
        u = np.linspace(0.0, 1.0, nsamples)
        points = interpolate.splev(u, self.tck, der=0)
        plt.plot(points[0], points[1], lformat, label=self.airf_name)
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.show()

    def plot_curvature(self, nsamples=1000, lformat='-r'):
        """bla."""
        curvature = self.get_curvature(nsamples)
        plt.figure('Curvature of {}'.format(self.airf_name))
        plt.plot(abs(curvature), lformat, )
        plt.grid()
        plt.show()

    def plot_dpoints(self, min_step=1e-4, max_step=0.01, nsamples=1000,
                     lformat='or'):
        """bla."""
        num_points, points = self.get_dpoints(min_step=min_step,
                                              max_step=max_step,
                                              nsamples=nsamples)

        plt.figure('New point distribution for {}'.format(self.airf_name))
        plt.plot(points[0], points[1], lformat, label=self.airf_name)
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.show()
