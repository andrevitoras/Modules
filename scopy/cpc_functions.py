"""
Created by André Santos (andrevitoras@gmail.com / avas@uevora.pt)
"""

from numpy import arccos, sin, pi, tan
from niopy.geometric_transforms import Angle


def real_cpc_tube_data(theta_a: Angle, tube_radius: float, outer_glass_radius: float):
    beta = Angle(rad=arccos(tube_radius / outer_glass_radius))
    s = outer_glass_radius * sin(beta.rad)

    a = 2 * (pi * tube_radius - beta.rad * tube_radius + s) / sin(theta_a.rad)
    h = a / (2 * tan(theta_a.rad)) + tube_radius / sin(theta_a.rad)

    return a, h


def ideal_cpc_tube_data(theta_a: Angle, tube_radius: float):

    a = 2 * pi * tube_radius / sin(theta_a.rad)
    h = a / (2 * tan(theta_a.rad) + tube_radius / sin(theta_a.rad))

    return a, h
