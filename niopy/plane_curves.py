#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy

from numpy import dot, cos, sin, array, arcsin, sqrt, pi, zeros, arctan
from scipy.interpolate import UnivariateSpline

from niopy.geometric_transforms import dst, ang_h, ang_p, V, nrm, R

from pathlib import Path


class PlaneCurve:

    def __init__(self, curve_pts: array, curve_center: array = None):

        pts = deepcopy(curve_pts)
        # Ensure that points of the plane curve are in ascending order at the x coordinate.
        # This is needed to generate splines interpolations for the curve.
        if pts[0][0] > pts[-1][0]:
            pts = pts[::-1]

        self.x = pts[:, 0]
        self.y = pts[:, -1]

        self.curve_pts = zeros(shape=(self.x.shape[0], 2))
        self.curve_pts.T[0][:], self.curve_pts.T[1][:] = self.x, self.y

        if curve_center is None:
            n_m = int((len(pts) + 1) / 2)
            hc = pts[n_m]
        else:
            hc = curve_center

        # Translated [x, y] points (relative position from the curve center -- a translated reference frame)
        self.x_t = self.x - hc[0]
        self.y_t = self.y - hc[-1]

    def spline(self, centered=False):
        spl = UnivariateSpline(x=self.x_t, y=self.y_t, s=0) if centered else UnivariateSpline(x=self.x, y=self.y, s=0)
        return spl

    def normals2surface(self):
        normals = zeros(shape=(self.x.shape[0], 2))
        spl = self.spline()
        d1 = spl.derivative()

        normals[:] = [nrm(V(arctan(d1(p)))) for p in self.x]
        normals = R(pi / 2).dot(normals.T).T

        return normals

    def curvature(self):
        x = self.x_t
        f = self.spline(centered=True)
        df = f.derivative()
        d2f = f.derivative(n=2)

        kappa = zeros(x.shape[0])
        kappa[:] = [abs(d2f(v)) / (1 + df(v) ** 2) ** (3 / 2) for v in x]

        return kappa.round(10)

    def spline2soltrace(self, file_path: Path, file_name: str):
        # It gets the translated (to curve_center) points of the plane curve, and convert to meters.
        x_t = self.x_t / 1000
        y_t = self.y_t / 1000

        # constructs the spline from the translated points and calculate the first derivative values at both edges knots
        df = self.spline(centered=True).derivative()
        df_1 = df(x_t[0])
        df_n = df(x_t[-1])

        # creates the surface cubic spline file (a 'csi' extension file for SolTrace to correctly read it)
        full_file_path = Path(file_path, f"{file_name}.csi")

        file = open(full_file_path, 'w')
        file.write(f"{len(x_t)}\n")  # the first line must contain the number of points which defines the surface
        for i in range(len(x_t)):
            # write in the file the point coordinates values in meters
            file.write(f"{x_t[i]} {y_t[i]}\n")

        # the last line should contain the first derivatives at both edge knots.
        file.write(f"{df_1} {df_n}")  # writes the first derivatives at both edges knots
        file.close()  # closes the file

        return full_file_path


########################################################################################################################
########################################################################################################################


def par(alpha: float, f: array, p: array):
    """
    This function returns a parametric function of a parabola tilted by an angle 'alpha' to the horizontal,
    with focus 'f' and that goes through a point 'p'.

    :param alpha: Parabola's tilt angle from the horizontal, in radians.
    :param f: Parabola's focal point, an array.
    :param p: Point which the parabola goes through, an array.

    :return: A parametric function.
    """

    def fc(x):
        num = dst(p, f) - dot(p - f, V(alpha))
        den = 1 - cos(x)
        return (num / den) * V(x + alpha) + f

    return fc


def eli(f: array, g: array, p: array):

    """
    :param f: Ellipse focus
    :param g: Ellipse focus
    :param p: Point which the ellipse goes through
    :return: This function returns a parametric function of an ellipse with focus at 'f' and 'g'
    and that goes through a point 'p'.
    """
    # ToDo: check if it works
    alpha = ang_h(g - f)
    k = dst(f, p) + dst(p, g)
    d_fg = dst(f, g)
    num = k ** 2 - d_fg ** 2

    def fc(x):
        den = 2 * (k - d_fg * cos(x))

        return f + (num / den) * V(x + alpha.rad)

    return fc


def hyp(f: array, g: array, p: array):
    """
    This function returns a parametric function of a hyperbola with foci 'f' and 'g'
    and passing through point 'p'

    :param f: Hyperbola focus, an array point
    :param g: Hyperbola focus, an array point
    :param p: A point the that the hyperbola passes through

    :return: A parametric function
    """

    alpha = ang_h(g - f)
    k = abs(dst(f, p) - dst(p, g))
    d_fg = dst(f, g)
    num = k ** 2 - d_fg ** 2

    def fc(y):
        den = 2 * (k - d_fg * cos(y))

        return f + (num / den) * V(y + alpha.rad)

    return fc


def winv(p: array, f: array, r: float):
    """
    This function returns a parametric function of a winding involute to a circle centered at point 'f' with
    radius 'r', and that goes through point 'p'.

    :param p: A point that the involute passes through.
    :param f: Center point of the circle.
    :param r: Radius of the circle

    :return: A parametric function
    """

    d_pf = dst(p, f)
    phi_p = ang_h(p - f).rad + arcsin(r / d_pf)
    k = sqrt(d_pf ** 2 - r ** 2) + r * phi_p

    def fc(x):
        return r * array([sin(x), -cos(x)]) + (k - r * x) * V(x) + f

    return fc


def uinv(p: array, f: array, r: float):
    """
    This function returns a parametric function of an unwinding involute to a circle with center at 'f' with
    radius 'r', and that goes through point 'p'.

    :param p: A point that the involute passes through.
    :param f: Center point of the circle.
    :param r: Radius of the circle.

    :return: A parametric function.
    """

    d_pf = dst(p, f)

    phi_p = ang_h(p - f).rad - arcsin(r / d_pf)
    phi_p = 2 * pi + phi_p if phi_p < 0 else phi_p
    k = sqrt(d_pf ** 2 - r ** 2) - r * phi_p

    def fc(x):
        return r * array([-sin(x), cos(x)]) + (k + r * x) * V(x) + f

    return fc


def wmp(alpha: float, f: array, r: float, p: array):
    """
    This function returns a parametric function of a winding macrofocal parabola tilted by an angle 'alpha'
    to the horizontal, with macrofocus centered at 'f' and radius 'r', and that goes through point 'p'.

    :param alpha: Macrofocal parabola tilt angle from the horizontal, in radians.
    :param f: Macrofocus center point.
    :param r: Macrofocus radius.
    :param p: A point the macrofocal parabola passes through.

    :return: A parametric function.
    """

    phi_p = ang_p(p - f, V(alpha)).rad + arcsin(r / dst(p, f))

    k = sqrt(dst(p, f) ** 2 - r ** 2) * (1 - cos(phi_p)) + r * (1 + phi_p - sin(phi_p))

    def fc(x):
        num = k + r * (sin(x) - 1 - x)
        den = 1 - cos(x)

        return r * array([sin(x + alpha), -cos(x + alpha)]) + (num / den) * V(x + alpha) + f

    return fc


def ump(alpha: float, f: array, r: float, p: array):
    """
    This function returns a parametric function of an unwinding macrofocal parabola tilted by an angle 'alpha'
    to the horizontal, with macrofocus centered at 'f' and radius 'r', and that goes through point 'p'.

    :param alpha: Macrofocal parabola tilt angle from the horizontal, in radians.
    :param f: Macrofocus center point.
    :param r: Macrofocus radius.
    :param p: A point the macrofocal parabola passes through.

    :return: A parametric function.
    """

    phi_p = ang_p(p - f, V(alpha)).rad - arcsin(r / dst(p, f))
    phi_p = 2 * pi + phi_p if phi_p < 0 else phi_p

    k = sqrt(dst(p, f) ** 2 - r ** 2) * (1 - cos(phi_p)) + r * (1 - phi_p + sin(phi_p))

    def fc(x):
        num = k + r * (x - 1 - sin(x))
        den = 1 - cos(x)

        return r * array([-sin(x + alpha), cos(x + alpha)]) + (num / den) * V(x + alpha) + f

    return fc


def wme(f: array, r: float, g: array, p: array):
    """
    This function returns a parametric function of a winding macrofocal ellipse with macrofocus centered at 'f'
    with radius 'r', point focus at 'g' and that goes through point 'p'.

    :param f: Ellipse macrofocus center point.
    :param r: Macrofocus radius.
    :param g: Ellipse point focus.
    :param p: A point that the macrofocal ellipse passes through.

    :return: A parametric function.
    """

    alpha = ang_h(g - f).rad
    ff = dst(g, f)

    phi_p = ang_p(p - f, V(alpha)).rad + arcsin(r / dst(p, f))
    tp = sqrt(dst(p, f) ** 2 - r ** 2)
    k = tp + r * phi_p + sqrt(ff ** 2 + r ** 2 + tp ** 2 - 2 * ff * (tp * cos(phi_p) + r * sin(phi_p)))

    def fc(x):
        num = (k - r * x) ** 2 + 2 * ff * r * sin(x) - ff ** 2 - r ** 2
        den = 2 * (k - r * x - ff * cos(x))

        return r * array([sin(x + alpha), -cos(x + alpha)]) + (num / den) * V(x + alpha) + f

    return fc


def ume(f: array, r: float, g: array, p: array):
    """
    This function returns a parametric function of an unwinding macrofocal ellipse with macrofocus centered
    at 'f' with radius 'r', point focus at 'g' and that goes through point 'p'.

    :param f: Ellipse macrofocus center point.
    :param r: Macrofocus radius.
    :param g: Ellipse point focus.
    :param p: A point that the macrofocal ellipse passes through.

    :return: A parametric function.
    """

    alpha = ang_h(g - f).rad
    ff = dst(g, f)
    tp = sqrt(dst(p, f) ** 2 - r ** 2)

    phi = ang_p(p - f, V(alpha)).rad - arcsin(r / dst(p, f))
    phi_p = 2 * pi + phi if phi < 0 else phi

    k = tp - r * phi_p + sqrt(ff ** 2 + r ** 2 + tp ** 2 - 2 * ff * (tp * cos(phi_p) - r * sin(phi_p)))

    def fc(x):
        num = (k + r * x) ** 2 - 2 * ff * r * sin(x) - ff ** 2 - r ** 2
        den = 2 * (k + r * x - ff * cos(x))

        return r * array([-sin(x + alpha), cos(x + alpha)]) + (num / den) * V(x + alpha) + f

    return fc
