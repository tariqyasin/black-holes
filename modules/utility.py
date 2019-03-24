"""A file containing auxiliary functions."""

import numpy as np
import scipy as sp

from imbh_config import imbh_data


def dist_arr(size_y, size_x, spax):
    '''Function that generates array with distances from centre

    Inputs:

        - size: length of array in pixels
        - spax: spaxel size [mas/spaxel]

    Outputs:

        - distance_array: shape (size_y,size_x)

    '''

    arra = np.zeros((size_y, size_x), dtype=float)
    y, x = arra.shape
    x_inds, y_inds = np.ogrid[:y, :x]
    mid_x, mid_y = (sp.array(arra.shape[:2]) - 1) / float(2)
    n_arra = ((y_inds - mid_y) ** 2 + (x_inds - mid_x) ** 2) ** 0.5
    n_arra *= spax

    # Ensure central value is tiny for Airy function
    r = np.where(n_arra < 1.E-15, 0.001, n_arra)
    return r


def x_arr(size, spax):
    '''Generates an array with distance from x-axis.'''

    arra = np.zeros((size, size), dtype=float)
    y, x = arra.shape
    x_inds, y_inds = np.ogrid[:y, :x]
    mid_x, mid_y = (sp.array(arra.shape[:2]) - 1) / float(2)
    n_arra = ((y_inds - mid_y)*0 + (x_inds - mid_x) ** 2) ** 0.5
    n_arra *= spax

    # Ensure central value is tiny for Airy function.
    r = np.where(n_arra < 1.E-15, 0.001, n_arra)
    return r


def y_arr(size, spax):
    '''Generates an array with distance from y-axis.'''

    arra = np.zeros((size, size), dtype=float)
    y, x = arra.shape
    x_inds, y_inds = np.ogrid[:y, :x]
    mid_x, mid_y = (sp.array(arra.shape[:2]) - 1) / float(2)
    n_arra = ((y_inds - mid_y) ** 2 + (x_inds - mid_x) * 0) ** 0.5
    n_arra *= spax
    # Quick and dirty fix to ensure central value is tiny for Airy function to handle.
    r = np.where(n_arra < 1.E-15, 0.001, n_arra)
    return r


def parsec(mas, d):
    """Returns the length (in parsecs) subtended by the angle mas (in milliarseconds) at
    a distance d (in parsecs)."""
    return (mas * imbh_data['rad_msec']) * d


def trap(f, a, b, n, *args):
    """ Returns the integral of the function f(*args) between the
    limits a and b by using the trapezium rule with n segments. """

    h = (b - a) / n
    s = 0.0
    s += f(a, *args)/2.0
    for i in xrange(1, n):
        s += f(a + i*h, *args)
    s += f(b, *args)/2.0
    return s * h


def trap2d(f, xa, xb, ya, yb, n, *args):
    """2-Dimensional integration using the trapezium rule."""

    z = np.zeros((n+1, n+1))
    dx = (xb - xa) / n
    dy = (yb - ya) / n
    for j in xrange(0, n+1):
        for i in xrange(0, n+1):
            z[i, j] = f(xa + (i*dx), ya + (j*dy), *args)

    sum = np.sum

    s1 = z[0, 0] + z[-1, 0] + z[0, -1] + z[-1, -1]
    s2 = sum(z[1:-1, 0]) + sum(z[1:-1, -1]) + sum(z[0, 1:-1]) + sum(z[-1, 1:-1])
    s3 = sum(z[1:-1, 1:-1])

    return 0.25*dx*dy*(s1 + 2*s2 + 4*s3)
