# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:25:35 2015

@author: winstroth
"""
import numpy as np
import airfgeom as afg
from scipy import interpolate
import matplotlib.pyplot as plt


class ModAirfoil(afg.AirfGeom):
    """Class for adding modifications to airfoils."""

    def __init__(self, airfcoords, airf_name='airfoil', nsamples=10000,
                 comments='#', delimiter=None, skiprows=1, usecols=(0, 1),
                 smoothing=0.0, degree=3):
        """Init ModAirfoil."""
        super(ModAirfoil, self).__init__(airfcoords=airfcoords,
                                         airf_name=airf_name,
                                         nsamples=nsamples,
                                         comments=comments,
                                         delimiter=delimiter,
                                         skiprows=skiprows,
                                         usecols=usecols,
                                         smoothing=smoothing,
                                         degree=degree)

    def _rotate_around_point(self, alpha, u0, u1, rot_pt, nsamples):
        """Rotates the airfoil part from u0 to u1 around point by angle degs.

        """
        points = self.get_epoints(u0=u0, u1=u1, nsamples=nsamples)
        # Translate points
        points = points - rot_pt
        alpha = alpha * np.pi / 180.0
        # Get 2D rotation matrix
        rot_mat = np.array([[np.cos(alpha), np.sin(alpha)],
                           [-np.sin(alpha),  np.cos(alpha)]])
        # Rotate points
        points = np.dot(points, rot_mat)
        # Translate back
        points = points + rot_pt
        return points

    def rotate_le(self, alpha, nsamples, te_smooth=1, smoothing=0.0, degree=5,
                  ins_pt=None):
        """Splits the airfoil in two parts at the leading edge, rotates both
        parts by alpha/2.0 around the leading edge and than reconnects both
        parts again."""
        u_le, le_point = self.get_le_point()
        te_smoothing = 1.0*te_smooth/100.0
        u0 = 0.0
        u1 = u_le - te_smoothing
        u2 = u_le + te_smoothing
        u3 = 1.0
        ss_pts = self._rotate_around_point(alpha=alpha/2.0, u0=u0, u1=u1,
                                           rot_pt=le_point, nsamples=nsamples)
        ps_pts = self._rotate_around_point(alpha=-alpha/2.0, u0=u2, u1=u3,
                                           rot_pt=le_point, nsamples=nsamples)
        if ins_pt is None:
            new_pts = np.vstack((ss_pts, le_point, ps_pts))
        else:
            new_pts = np.vstack((ss_pts, ins_pt, ps_pts))
        x = [new_pts[:, 0], new_pts[:, 1]]
        self.tck, _ = interpolate.splprep(x, s=smoothing, k=degree)
        return ss_pts, ps_pts

    def rotate_te(self, alpha, nsamples, te_smooth=1, smoothing=0.0, degree=5,
                  ins_pt=None):
        """Splits the airfoil in two parts at the leading edge, rotates both
        parts by alpha/2.0 around the trailing edge and than reconnects both
        parts again."""
        te_point = self.get_te_point()
        u_le, le_point = self.get_le_point()
        te_smoothing = 1.0*te_smooth/100.0
        u0 = 0.0
        u1 = u_le - te_smoothing
        u2 = u_le + te_smoothing
        u3 = 1.0
        ss_pts = self._rotate_around_point(alpha=-alpha/2.0, u0=u0, u1=u1,
                                           rot_pt=te_point, nsamples=nsamples)
        ps_pts = self._rotate_around_point(alpha=alpha/2.0, u0=u2, u1=u3,
                                           rot_pt=te_point, nsamples=nsamples)
        if ins_pt is None:
            new_pts = np.vstack((ss_pts, ps_pts))
        else:
            new_pts = np.vstack((ss_pts, ins_pt, ps_pts))
        x = [new_pts[:, 0], new_pts[:, 1]]
        self.tck, _ = interpolate.splprep(x, s=smoothing, k=degree)
        return ss_pts, ps_pts

    def normal_transform(self, tck_transf, nsamples, s=0.0, k=3, plot=False):
        """bla."""
        nvecs, pvecs, u = self.get_normal_vecs(nsamples=nsamples)
        amps = interpolate.splev(u, tck_transf)
        pts = []
        _, vec_len = nvecs.shape
        for i in range(vec_len):
            x = pvecs[0, i]
            y = pvecs[1, i]
            dirx = nvecs[0, i]
            diry = nvecs[1, i]
            amp = amps[i]
            lx = x+dirx*amp
            ly = y+diry*amp
            pts.append([lx, ly])
        pts = np.array(pts)
        if plot:
            plt.figure('Comparison org. airfoil vs mod. airfoil')
            plt.plot(pvecs[0, :], pvecs[1, :], label='org. airfoil')
            plt.plot(pts[:, 0], pts[:, 1], '-r', label='mod. airfoil')
            plt.axis('equal')
            plt.grid(True)
            plt.legend()
            plt.show()
        # Create bspline representation
        x = [pts[:, 0], pts[:, 1]]
        self.tck, _ = interpolate.splprep(x, s=s, k=k)

    def sin_normal_transform(self, u0, u1, ampl, num_periods,
                             trans_before=1.0, trans_mod_start=0.4,
                             trans_mod_end=0.4, trans_after=1.0,
                             samples_p_period=40, s=0.0, k=3, plot=False):
        """bla."""
        # Get period ans frequenzy of the oscillation
        mod_len = u1 - u0
        period = mod_len/num_periods
        freq = 1.0/period
        # Adjust the modification interval to get a smoother transition
        qperiod = period/4.0
        u_mod_start = qperiod*trans_mod_start
        u_mod_end = mod_len - qperiod*trans_mod_end
        # Get modification part
        u_mod = np.linspace(start=u_mod_start, stop=u_mod_end,
                            num=samples_p_period*num_periods)
        y = ampl*np.sin(2*np.pi*freq*u_mod)
        u_mod = u_mod + u0
        # Get number of zero points before and after the mod part
        pts_u_mod = len(u_mod)
        pts_p_len = float(pts_u_mod)/(u_mod_end - u_mod_start)
        u_len_before = u0 - qperiod*trans_before
        pts_before = int(pts_p_len*u_len_before)
        u_len_after = 1.0 - u1 - qperiod*trans_after
        pts_after = int(pts_p_len*u_len_after)
        # Create zero vectors before and after the mod part
        u_before = np.linspace(start=0.0,
                               stop=(u0 - qperiod*trans_before),
                               num=pts_before)
        u_before_zeros = np.zeros(len(u_before))
        u_after = np.linspace(start=(u1 + qperiod*trans_after),
                              stop=1.0,
                              num=pts_after)
        u_after_zeros = np.zeros(len(u_after))
        u = np.hstack((u_before, u_mod, u_after))
        y = np.hstack((u_before_zeros, y, u_after_zeros))
        tck = interpolate.splrep(x=u, y=y, k=k, s=s)
        if plot:
            # Plot complete modification with bspline fit
            u_bspl = np.linspace(start=0.0, stop=1.0, num=10000)
            y_bspl = interpolate.splev(x=u_bspl, tck=tck)
            plt.figure('Sine wave modification')
            plt.plot(u, y, 'or', label='data points')
            plt.plot(u_bspl, y_bspl, '-b', label='bspline fit')
            plt.axis('equal')
            plt.grid(True)
            plt.legend()
            plt.show()
        return tck




















