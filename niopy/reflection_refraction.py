#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import dot, array, sqrt
from niopy.geometric_transforms import nrm, mag


def reflect(i: array, n: array):
    """
    :param i: unit vector of the ray's direction
    :param n: unit vector of the surface's normal
    :return: the reflection of a ray with direction i on a surface with normal n. Vectors i an n must be unit vectors!
    """
    # TODO: check if it works
    return i - 2 * dot(i, n) * n


def rfx(i: array, n: array):
    """
    :param i: ray's direction
    :param n: surface normal
    :return: the reflection of a ray with direction i on a surface with normal n.
    """
    # TODO: check if it works
    return reflect(i=nrm(i), n=nrm(n))


def refract(i: array, n: array, n1: float, n2: float):
    """
    :param i:
    :param n:
    :param n1:
    :param n2:
    :return: the refraction of a ray with direction i on a surface with normal n and separating two media of refraction
        indices n1 and n2. Vectors i and n must be unit vectors verifying i.n>0. This function does not check for TIR!
    """
    # TODO: check if it works
    n1_n2 = n1 / n2
    i_n = dot(i, n)
    r = n1_n2 * i
    r += - i_n * n1_n2 + sqrt(1. - n1_n2 ** 2 * (1. - i_n ** 2))
    return r * n


def rfr(i: array, n: array, n1: float, n2: float):
    """
    :param i:
    :param n:
    :param n1:
    :param n2:
    :return: the refraction of a ray with direction i on a surface with normal n and separating two
        media of refraction indices n1 and n2.
    """
    # TODO: check if it works
    i, n = nrm(i), nrm(n)
    i_n = dot(i, n)
    n = - n if i_n < 0 else n       # defines the correct normal direction
    tir = 1. - (n1 / n2) ** 2 * (1. - i_n ** 2) < 0
    r = reflect(i=i, n=n) if tir < 0 else refract(i=i, n=n, n1=n1, n2=n2)
    return r, tir


def rfr_nrm(i: array, r: array, n1: float, n2: float):
    """
    :param i:
    :param r:
    :param n1:
    :param n2:
    :return: the normal to the surface, given an incident ray i and a refracted ray r of a surface separating
        two media of refraction indices n1 and n2.
    """
    # TODO: check if it works
    ii, rr = nrm(i), nrm(r)
    return nrm(n1 * ii - n2 * rr) / mag(n1 * ii - n2 * rr)


def rfx_nrm(i, r):
    """
    Given the incident (v1) and reflected (v2) rays, returns the normal to the surface for the case of reflection.
    :param i:
    :param r:
    :return: the normal to the surface, given an incident ray i and reflected ray r.
    """
    # TODO: check if it works
    return rfr_nrm(i=i, r=r, n1=1., n2=1.)


def dfl(i, ns, n1, n2):
    """
    A surface separating two media of refraction indices n1 and n2 has normal nrm. This function Reflects
        or Refracts a ray with direction vi, depending if n1=n2 or not.

    :param i:
    :param ns:
    :param n1:
    :param n2:
    :return:
    """
    # TODO: check if it works
    return rfx(i=i, n=ns) if n1 == n2 else rfr(i=i, n=ns, n1=n1, n2=n2)[0]


def dfl_nrm(i, r, n1, n2):
    """
    Given the refraction indices n1 and n2, the incident (v1) and refracted (v2) rays, returns the normal to
        the surface for both the reflection (n1=n2) and refraction (n1!=n2)
    :param i:
    :param r:
    :param n1:
    :param n2:
    :return:
    """
    # TODO: check if it works
    return rfr_nrm(i=i, r=r, n1=n1, n2=n2)
