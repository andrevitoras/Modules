from datetime import datetime

from numpy import array, cos, sin, deg2rad, cross, sign, arccos, arctan, pi
from numpy.linalg import norm
from pvlib.location import Location
from pytz import timezone
from tzwhere import tzwhere

# PECS geographical data, both in degree. From: maps.google.com
pecs_latitude, pecs_longitude = 38.53044837789431, -8.010462146558064

# Gets PECS location timezone
tz_where = tzwhere.tzwhere()
timezone_str = tz_where.tzNameAt(latitude=pecs_latitude, longitude=pecs_longitude)
tz = timezone(timezone_str)

# Creates the PECS Location object from the 'pvlib' library to calculate the sun position
pecs_site = Location(latitude=pecs_latitude,
                     longitude=pecs_longitude,
                     tz=tz)


def nrm(v: array) -> array:
    """
    :param v:
    :return: a unit vector in the same direction as v.
    """
    return v / norm(v)


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


def platform_to_inertial(pecs_azimuth: float, pecs_tilt: float):

    """
    This function considers two coordinate systems.
    One is the inertial frame, which has directions as follows: North points to positive Y; East points to positive X;
    and the Zenith points to positive Z

    The other coordinate system is attached to the PECS platform movement -- the two axis tracking system.
    Considering the platform with both azimuth and tilt angle as zero, the identity matrix is the rotation matrix
    that represents the relation between these two reference frames.

    Therefore, this function returns the rotation matrix that defines the axis of the PECS frame regarding the
    inertial coordinate system.

    :param pecs_azimuth: PECS azimuth angle regarding the North, in degrees
    :param pecs_tilt: PECS tilt angle regarding the horizontal plane, in degrees

    :return: It returns the rotation matrix that defines the axis of the PECS frame regarding the inertial
    coordinate system.
    """

    pecs_azimuth_rad = deg2rad(pecs_azimuth)
    pecs_tilt_rad = deg2rad(pecs_tilt)
    R1 = R(pecs_azimuth_rad, array([0, 0, 1]))
    R2 = R(pecs_tilt_rad, array([1, 0, 0]))

    return R1.dot(R2).round(10)


def inertial_to_platform(pecs_azimuth: float, pecs_tilt: float):

    """
    This is the inverse of the 'platform_to_inertial' function. In the case of rotation matrix, the inverse is equal to
    its transpose.

    Please, See documentation of the function 'platform_to_inertial'.

    :param pecs_azimuth: PECS azimuth angle regarding the North, in degrees
    :param pecs_tilt: PECS tilt angle regarding the horizontal plane, in degrees

    :return: It returns the rotation matrix represents the inertial reference frame regarding the PECS frame
    """

    return platform_to_inertial(pecs_azimuth=pecs_azimuth, pecs_tilt=pecs_tilt).T


def get_local_time():

    """
    This function returns the local time as a datetime object
    """

    return datetime.now()


def sun_angles(local_time):

    """
    :param local_time: A datetime object with the local time

    :return: It returns the sun angle positions: zenith and azimuth, both in degrees.
    """

    sol_pos = pecs_site.get_solarposition(local_time)
    sun_zenith = sol_pos.iloc[0]['zenith']
    sun_azimuth = sol_pos.iloc[0]['azimuth']

    return sun_zenith, sun_azimuth


def inertial_frame_sun_vector(sun_zenith, sun_azimuth):

    zen = deg2rad(sun_zenith)
    azi = deg2rad(sun_azimuth)
    vi = array([sin(zen) * sin(azi), sin(zen) * cos(azi), cos(zen)]).round(10)

    return vi


def ang(v: array, u: array):
    """
    :param v:
    :param u:
    :return: This function returns the angle between two vectors v and u in the range from 0 to pi.";
    """

    beta = arccos(nrm(v).dot(nrm(u)))

    return beta


def angular_position(center: array, aim: array):
    Iz = array([0, 0, 1])

    sm = array([aim[0], 0, aim[-1]])
    hc = array([center[0], 0, center[-1]])

    aim_vector = sm - hc
    lamb = sign(cross(Iz, aim_vector)[1]) * ang(aim_vector, Iz)

    return lamb


receiver_aperture_s1 = array([195, 0, 4000])
receiver_aperture_s2 = array([444.9723708, 0, 4387.556976])
tracking_aim_point = 0.5 * (receiver_aperture_s1 + receiver_aperture_s2)

farthest_mirror_center_point = array([7875, 0, 0])
farthest_mirror_angular_position = angular_position(center=farthest_mirror_center_point, aim=tracking_aim_point)


def tracking_angle_farthest_mirror(local_time, pecs_azimuth: float, pecs_tilt: float):

    # Calculates the sun zenith and azimuth regarding the inertial coordinate system
    sun_zenith, sun_azimuth = sun_angles(local_time=local_time)
    # Calculates the sun vector in the inertial coordinate system
    sun_vector = inertial_frame_sun_vector(sun_zenith=sun_zenith, sun_azimuth=sun_azimuth)

    # Calculates the rotation matrix that maps a vector in the inertial frame into the platform frame
    rm = inertial_to_platform(pecs_azimuth=pecs_azimuth, pecs_tilt=pecs_tilt)
    # Transform the sun vector into the platform frame
    sun_vector_pecs_frame = rm.dot(sun_vector)

    # Calculates the transversal incidence angle, in radians.
    theta_t = arctan(sun_vector_pecs_frame[0] / sun_vector_pecs_frame[-1])
    # Then calculate the correspondent tracking angle and returns it in degrees.
    tau = 0.5 * (farthest_mirror_angular_position + theta_t)

    return tau * 180.0 / pi
