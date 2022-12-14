#####
#####
# Set of functions implemented by André Santos (andrevitoras@gmail.com / avas@uevora.pt)
# For more details see J. Chaves (2016), Introduction to nonimaging optics, 2nd Ed., CRC Press, Chapter 21.

from typing import Any

from numpy.linalg import norm
from numpy import arccos, array, dot, pi, sign, cos, sin, tan, cross


class Angle:

    def __init__(self, rad=None, deg=None):

        if deg is None and rad is not None:
            self.rad = rad
            self.deg = rad * 180. / pi
        elif rad is None and deg is not None:
            self.deg = deg
            self.rad = deg * pi / 180.
        else:
            print('No value was inputted for the angle. Please add in radians (rad=) or degrees(deg=)')

    def __str__(self):
        return "An Angle: in radians = " + str(self.rad) + " | In degrees = " + str(self.deg)

    def sin(self):
        return sin(self.rad)

    def cos(self):
        return cos(self.rad)

    def tan(self):
        return tan(self.rad)


def V(x: float):
    """
    :param x: an angle in radians
    :return: This function returns a 2D unit array vector for angle x
    """
    return array([cos(x), sin(x)])


def mag(v: array) -> float:
    """
    :param v: a array vector
    :return: the magnitude (length) of a vector v.
    """
    return norm(v)


def nrm(v: array) -> array:
    """
    :param v:
    :return: a unit vector in the same direction as v.
    """
    return v / mag(v)


def dst(p: array, q: array) -> array:
    """
    :param p: a point array
    :param q: a point array
    :return: the Euclidian distance between two points 'p' and 'q'.
    """

    return mag(p - q)


def ang(v: array, u: array) -> Angle:
    """
    :param v:
    :param u:
    :return: This function returns the angle between two vectors v and u in the range from 0 to pi.";
    """

    beta = arccos(nrm(v).dot(nrm(u)))

    return Angle(rad=beta)


def ang_p(v: array, u: array, ) -> Angle:

    """
    :param v: an array vector in the xy plane
    :param u: an array vector in the xy plane

    :return: This function returns the angle that vector 'v' makes relative to vector 'u' in the positive direction
    and in the range from 0 to 2*pi.
    """

    alpha = ang(v=v, u=u).rad

    p = alpha if (u[0] * v[1] - u[1] * v[0] >= 0) else 2. * pi - alpha

    return Angle(rad=p)


def ang_h(v: array) -> Angle:
    # TODO: check if it works
    """
    :param v: a array vector

    :return: the angle vector v makes to the horizontal {1,0} in the positive direction and in the range from 0 to 2pi.
    """

    return ang_p(v=v, u=array([1, 0]))


def ang_pn(v: array, u: Any = None) -> Angle:
    """
    :param v:
    :param u:
    :return: the angle to the horizontal axis: positive if the vector points up and negative if the vector points down.
             the angle of vector v relative to vector u: positive if u is clockwise from v and negative otherwise.
    """
    if u is None:
        alpha = sign(v[1]) * ang_h(v=v).rad
    else:
        alpha = ang(v=v, u=u).rad if (u[0] * v[1] - u[1] * v[0] >= 0) else -ang(v=v, u=u).rad

    return Angle(rad=alpha)


def ang_pnd(u: array, v: array, n: array):
    return sign(dot(n, cross(u, v))) * ang(v=v, u=u).rad


def R(alpha: float, v: array = None) -> array:
    """
    R(alpha) is a rotation matrix of an angle alpha. R(alpha,v)
    is a rotation matrix of an angle alpha around axis v.
    Such rotations are pivoted from origin [0,0] or [0,0,0].
    """
    if v is None:
        rm = array(
            [
                [cos(alpha), -sin(alpha)],
                [sin(alpha), cos(alpha)],
            ]
        )
    else:
        if v.shape[0] != 3:
            raise Exception(f'Wrong dimension of v. Found dimension {v.shape[0]} where should be 3.')
        vn = nrm(v)
        rm = array(
            [
                [
                    cos(alpha) + vn[0] ** 2 * (1 - cos(alpha)),
                    vn[0] * vn[1] * (1 - cos(alpha)) - vn[2] * sin(alpha),
                    vn[0] * vn[2] * (1 - cos(alpha)) + vn[1] * sin(alpha),
                ],
                [
                    vn[1] * vn[0] * (1 - cos(alpha)) + vn[2] * sin(alpha),
                    cos(alpha) + vn[1] ** 2 * (1 - cos(alpha)),
                    vn[1] * vn[2] * (1 - cos(alpha)) - vn[0] * sin(alpha),
                ],
                [
                    vn[2] * vn[0] * (1 - cos(alpha)) - vn[1] * sin(alpha),
                    vn[2] * vn[1] * (1 - cos(alpha)) + vn[0] * sin(alpha),
                    cos(alpha) + vn[2] ** 2 * (1 - cos(alpha)),
                ],
            ]
        )
    return rm


def Rot(v: array, alpha: float):
    """
    rotates a vector v by an angle alpha.
    :param v:
    :param alpha:
    :return: This function returns the rotated vector v by the angle alpha
    """
    return dot(R(alpha=alpha), v)


def islp(p, v, q, n):
    """
    If the geometry is three-dimensional, returns the intersection point between a straight line defined by point P
        and vector v and a plane defined by point Q and normal vector n.
    If the geometry is two-dimensional, the function returns the intersection of a straight line defined by point P
        and vector v and another straight line trough point Q with normal n.
    """
    vn = nrm(v)
    nn = nrm(n)

    return p + vn * dot((q - p), nn) / dot(vn, nn)


def isl(p, v, q, u):
    return islp(p, v, q, Rot(v=u, alpha=pi / 2))


def mid_point(p: array, q: array):
    """
    :param p: a point in space
    :param q: a point in space
    :return: the mid point between p and q
    """
    return (p + q) * 0.5


def Sy(v: array):
    """
    :param v: a 2D (x-y) point, or vector.
    :return: returns the symmetrical of v relative to the y-axis
    """
    m = array([[-1, 0], [0, 1]])

    return dot(m, v)


def Sx(v: array):
    """
    :param v: a 2D (x-y) point, or vector.
    :return: returns the symmetrical of v relative to the x-axis
    """
    m = array([[1, 0], [0, - 1]])

    return dot(m, v)


def proj_vector_plane(v: array, n: array) -> array:
    """
    :param v: a 3D vector
    :param n: a 3D vector that represents a normal to a plane
    :return: This function returns the projection of v onto a plane defined by its normal vector n
    """
    u = n * dot(v, n) / dot(n, n)

    return v - u


def tg_pts(f: array, r: float, p: array) -> list:
    """
    :param f: tube center point
    :param r: tube radius
    :param p: point out the tube
    :return: This function returns the two circle's tangent points that goes through point p
    """
    beta = Angle(rad=arccos(r / dst(p, f)))

    return [f + r * dot(R(beta.rad), nrm(p - f)), f + r * dot(R(-beta.rad), nrm(p - f))]
