# -*- coding: utf-8 -*-
"""
Created on Thu, Dec 3,2020 15:47:43
New version: Feb 8, 2022 09:00:55
@author: André Santos (andrevitoras@gmail.com / avas@uevora.pt)

"""

import json
from multiprocessing import cpu_count, Pool
from pathlib import Path

from numpy import array, linspace, pi, zeros, cos, sin, sign, cross, power, identity, dot, sqrt, tan, arctan, \
    absolute, arccos, arcsin, deg2rad, ones
from pandas import read_csv
from portion import closed, empty
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.optimize import fsolve

from niopy.geometric_transforms import nrm, ang, Angle, R, dst, islp, ang_pnd, V, ang_pn, mid_point, ang_h, ang_p, isl
from niopy.plane_curves import PlaneCurve, par, winv, uinv, wme, ume, hyp
from niopy.reflection_refraction import rfx_nrm
from scopy.sunlight import sun_direction, sun2lin, RadialSunshape
from soltracepy import OpticalSurface, Element, Trace, Stage, Geometry, Optics, \
    soltrace_script, ElementStats, run_soltrace, read_element_stats, reflective_surface, absorber_surface, \
    glass_cover_surfaces

from utils import read_trnsys_tmy2


class OpticalProperty:

    """
    This class aims to represent the common optical properties in the context of the design and analysis of the
    linear Fresnel collector. It contains reflective, absorptive and transmissive properties.

    """

    class reflector:

        def __init__(self, name: str, rho=1.0, slope_error=0.0, spec_error=0.0):
            """
            :param name: The name of the property
            :param rho: The reflector hemispherical reflectance. It should be a value between 0 and 1.
            :param slope_error: The slope error of the reflector surface, in mrad.
            :param spec_error: THe specular error of the reflector surface, in mrad.
            """

            assert 0 <= abs(rho) <= 1, ValueError('Reflectance value must be between 0 and 1.')

            self.rho = abs(rho)
            self.type = 'reflect'
            self.slope_error = abs(slope_error)
            self.spec_error = abs(spec_error)
            self.name = name

        def to_soltrace(self):

            """
            :return: This method returns an equivalent SolTrace Optic.
            """

            return reflective_surface(name=self.name, rho=self.rho,
                                      slope_error=self.slope_error*1000, spec_error=self.spec_error*1000)

    class flat_absorber:

        def __init__(self, name: str, alpha=1.):
            """
            :param alpha: The absorbance of the absorber surface. It must be a value between 0 and 1.
            """

            assert 0 <= abs(alpha) <= 1, ValueError('Absorbance value must be between 0 and 1.')

            self.type = 'reflect'
            self.alpha = abs(alpha)
            self.name = name

        def to_soltrace(self):
            """

            :return: This method returns an equivalent SolTrace Optic.
            """

            return absorber_surface(name=self.name, alpha=self.alpha)

    class absorber_tube:

        def __init__(self, name: str, alpha=1.):
            """
            :param alpha: The absorbance of the absorber surface. It must be a value between 0 and 1.
            """

            assert 0 <= abs(alpha) <= 1, ValueError('Absorbance value must be between 0 and 1.')

            self.type = 'reflect'
            self.alpha = abs(alpha)
            self.name = name

        def to_soltrace(self):
            """

            :return: This method returns an equivalent SolTrace Optic.
            """

            return absorber_surface(name=self.name, alpha=self.alpha)

    class transmitter:

        def __init__(self, name: str, tau=1., refract_index=1.52):
            """

            :param tau: The transmittance of the transmitter element. It should be a value between 0 and 1.
            :param refract_index: The refractive index of the transmitter element. It should be greater than 1.
            """

            assert 0 <= abs(tau) <= 1, ValueError('Transmittance value must be between 0 and 1.')
            assert abs(refract_index) >= 1, ValueError('Refractive index must be greater than 1.')

            self.name = name
            self.type = 'refract'
            self.tau = abs(tau)
            self.refract_index = abs(refract_index)

        def to_soltrace(self):

            return glass_cover_surfaces(tau=self.tau, name=self.name, refractive_index=self.refract_index)

    class evacuated_tube:

        def __init__(self):
            pass

        def to_soltrace(self):
            pass


class Secondary:

    class trapezoidal:

        def __init__(self, aperture_center: array, aperture_width: float, tilt: float, height: float,
                     name='trapezoidal_secondary'):

            self.name = name

            Ix = array([1, 0])
            self.aperture_center = array([aperture_center[0], aperture_center[-1]])
            self.aperture_width = abs(aperture_width)
            self.tilt = abs(tilt)
            self.height = abs(height)

            self.ap_left = - 0.5 * self.aperture_width * Ix + self.aperture_center
            self.ap_right = + 0.5 * self.aperture_width * Ix + self.aperture_center

            self.back_left = self.ap_left + V(tilt) * self.height
            self.back_right = self.ap_right + V(pi - tilt) * self.height

            self.left_edge = self.ap_left
            self.right_edge = self.ap_right

            self.segments = array([self.ap_left, self.back_left, self.back_right, self.ap_right])

        def as_soltrace_element(self, length: float, optic: OpticalSurface):

            return trapezoidal2soltrace(geometry=self, name=self.name, length=length, optic=optic)


class Absorber:

    class flat:

        def __init__(self, width: float, center: array, axis=array([1, 0]), name='flat_absorber'):

            self.name = name

            self.width = abs(width)
            self.center = array([center[0], center[-1]])

            self.axis = array([axis[0], axis[-1]])

            self.s1 = - 0.5 * self.width * nrm(self.axis) + self.center
            self.s2 = + 0.5 * self.width * nrm(self.axis) + self.center

            self.left_edge = self.s1
            self.right_edge = self.s2

        def as_soltrace_element(self, length: float, optic: OpticalSurface, name=None):

            if name is None:
                elem = flat_absorber2soltrace(geometry=self, name=self.name, optic=optic, length=length)
            else:
                elem = flat_absorber2soltrace(geometry=self, name=name, optic=optic, length=length)

            return elem

    class tube:

        def __init__(self, radius: float, center: array, name='absorber_tube', nbr_pts=121):

            self.name = name

            self.radius = abs(radius)
            self.center = array([center[0], center[-1]])

            self.n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts

            unit_circle = array([[cos(x), sin(x)] for x in linspace(start=0, stop=2 * pi, num=self.n_pts)])
            self.tube = unit_circle * self.radius + self.center

            self.left_edge = self.center - array([self.radius, 0])
            self.right_edge = self.center + array([self.radius, 0])

        def as_soltrace_element(self, length: float, optic: OpticalSurface):

            aim_vec = array([0, 0, 0])
            hc = transform_vector(self.center)
            elem = tube2soltrace(name=self.name, radius=self.radius, center=hc, length=length, aim=aim_vec, optic=optic)

            return elem

    class evacuated_tube:

        def __init__(self, radius: float, inner_cover_radius: float, outer_cover_radius: float, center: array,
                     nbr_pts=121):

            self.radius = abs(radius)
            self.center = array([center[0], center[-1]])

            self.radius = abs(radius)
            self.inner_radius = min(abs(outer_cover_radius), abs(inner_cover_radius))
            self.outer_radius = max(abs(outer_cover_radius), abs(inner_cover_radius))

            self.n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts

            unit_circle = array([[cos(x), sin(x)] for x in linspace(start=0, stop=2 * pi, num=self.n_pts)])
            self.tube = self.radius * unit_circle + self.center
            self.inner_cover = self.inner_radius * unit_circle + self.center
            self.outer_cover = self.outer_radius * unit_circle + self.center

            self.left_edge = self.center - array([self.radius, 0])
            self.right_edge = self.center + array([self.radius, 0])

        def as_soltrace_element(self, name: str, length: float,
                                cover_optic: OpticalSurface, absorber_optic: OpticalSurface) -> list:
            pass


class Receiver:

    def __init__(self, absorber, secondary=None):
        self.secondary = secondary
        self.absorber = absorber

    def as_soltrace_element(self, length: float,
                            absorber_property: OpticalSurface, secondary_property: OpticalSurface):

        if self.secondary is None:
            elements = self.absorber.to_soltrace(name=self.absorber.name, length=length, optic=absorber_property)
        else:
            elements = self.secondary.to_soltrace(name=self.secondary.name, length=length, optic=secondary_property)
            elements += self.absorber.to_soltrace(name=self.absorber.name, length=length, optic=absorber_property)

        return elements


class Heliostat:

    """
    The class Heliostat aims to represent a linear Fresnel primary mirror. It can be flat or cylindrical.
    Parabolic primaries were not included since they are equivalent to the cylindrical ones and are simpler to design.
    For a further understanding, one must read Refs. [1-5].

    The Heliostat object has the following attributes: shape, width, center point (2D), and radius.

    Furthermore, due to the analytical calculations of intercept factor (optical efficiency), other attributes are
    defined: hel_pts are XY (2D), ZX_center (3D), ZX_pts (3D), and normals (3D).

    The curvature() method was also implemented, and also other methods related to SolTrace implementations, and others.

    [1] Abbas R., Montes MJ., Piera M., Martínez-Val JM. 2012. https://doi.org/10.1016/j.enconman.2011.10.010.
    [2] Qiu Y, He YL, Cheng ZD, Wang K. 2015. https://doi.org/10.1016/j.apenergy.2015.01.135.
    [3] Abbas R, Martínez-Val JM. 2017. https://doi.org/10.1016/j.apenergy.2016.01.065.
    [4] Boito, P., Grena, R. 2017. https://doi.org/10.1016/j.solener.2017.07.079.
    [5] Cheng ZD, Zhao XR, He YL, Qiu Y. 2018. https://doi.org/10.1016/j.renene.2018.06.019.
    """

    def __init__(self, width: float, center: array, radius: float, nbr_pts=121):

        self.width = abs(width)
        self.radius = abs(radius)

        self.center = array([center[0], center[-1]])

        if self.radius == 0:
            self.shape = 'flat'
        else:
            self.shape = 'cylindrical'

        # Ensure an odd number of points to discretize the heliostat surface. In this way, the center point is part of
        # the set of points.
        self.n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts

        # The design of the set of points which represents the heliostat surface at a horizontal position (i.e, normal
        # vector at the center should point only to the vertical direction)
        # It calculates the points within the heliostat surface. The edges and the center are in this set.
        if self.shape == 'cylindrical':
            self.hel_pts = design_cylindrical_heliostat(hc=self.center, w=self.width, rc=self.radius,
                                                        nbr_pts=self.n_pts)  # XY points
        else:
            self.hel_pts = design_flat_heliostat(hc=self.center, w=self.width, nbr_pts=self.n_pts)  # XY points

        # transforms the previous XY center in a ZX point.
        self.ZX_center = transform_vector(self.center)

        # Attribute that holds the Heliostat object as a PlaneCurve object.
        # It also returns the XY normals at the heliostat surface points.
        self.curve = self.as_plane_curve()
        self.normals = self.curve.normals2surface()

        ###############################################################
        # 3D surface points: as [x, 0, z] arrays.
        # They define the surface. The edges and the center are in this set.
        self.ZX_pts = transform_heliostat(self.hel_pts)  # transforms the previous XY pts in ZX points.
        ################################################################

        # Points which define segments of the heliostat surface. Their projected width in the aperture are equal.
        seg_x_range = array([0.5 * (self.curve.x[i] + self.curve.x[i + 1]) for i in range(len(self.curve.x) - 1)])
        hel_as_spline = self.curve.spline()
        self.seg_pts = zeros(shape=(seg_x_range.shape[0], 2))

        # Data (points and normal vectors) of the segments in which the heliostat surface is divided for computations.
        # They are [x, y] arrays.
        self.seg_pts.T[0][:] = seg_x_range
        self.seg_pts.T[1][:] = hel_as_spline(seg_x_range)
        self.seg_normals = PlaneCurve(curve_pts=self.seg_pts, curve_center=self.center).normals2surface()
        ################################################################

    def as_plane_curve(self):
        """
        A method to return the Heliostat points as a PlaneCurve object.

        :return: It returns the xy points of the heliostat as a PlaneCurve object.
        """
        return PlaneCurve(curve_pts=self.hel_pts, curve_center=self.center)

    def angular_position(self, aim: array):

        """
        This function calculates the heliostat angular position regarding a tracking aim-point.
        It assumes the ZX plane as the transversal plane, i.e., the one which defines the LFC geometry.

        :param aim: The aim-point at the receiver, an array-point.

        :return: The heliostat angular position, in radians.
        """

        # Old version #################################################
        # This version considers the transversal plane as the XY plane.

        # sm = array([aim[0], aim[-1]])
        # vf = sm - self.center
        # lamb = sign(self.center[0]) * ang(vf, array([0, 1])).rad
        ################################################################

        # New version in 2022-11-08 ####################################
        # This version considers the transversal plane as the ZX plane. Therefore, a different equation is used.

        sm = array([aim[0], 0, aim[-1]])
        lamb = angular_position(center=self.ZX_center, aim=sm)
        ################################################################

        return lamb

    def tracking_angle(self, aim: array, theta_t: float):
        """
        This function calculates the heliostat tracking angle regarding an aim-point at the receiver for a particular
        value of transversal incidence angle.

        It assumes the ZX plane as the transversal plane, i.e., the one which defines the LFC geometry.

        :param aim: The aim-point at the receiver, an array-point.
        :param theta_t: The transversal incidence angle, in degrees.

        :return: The heliostat tracking angle, in radians.
        """

        theta_t_rad = theta_t * pi / 180.
        lamb = self.angular_position(aim=aim)

        tau = (lamb + theta_t_rad) / 2

        return tau

    def rotated_points(self, aim: array, theta_t='horizontal', return_xy=True, to_calculations=True):

        """
        This method returns the surface points of the rotated heliostat for a given transversal incidence angle
        ('theta_t'), in degrees, considering a particular tracking aim-point at the receiver ('aim')

        :param aim: The tracking aim-point at the receiver, an array.
        :param theta_t: The transversal incidence angle, in degrees.
        :param return_xy: A boolean sign to whether return [x, y] or [x, 0, z] points.

        :param to_calculations: A boolean sign to identify whether points in the surface will be used to efficiency
        calculations or not. If 'True', points represent segments of the surface that has the same projected width on
        the aperture plane of the heliostat.

        When the param 'theta_t' = 'horizontal', the heliostat is returned in its horizontal position.

        :return: It returns an array of array-points.

        """

        # Check whether horizontal or a particular incidence was inputted. ###################################
        if isinstance(theta_t, (float, int)):
            # A particular transversal incidence was selected.
            # All operations are done considering a ZX plane. Thus, mirrors rotated around the y-axis.
            # and the heliostats points are the kind [x, 0, z]
            Iy = array([0, 1, 0])

            # The correspondent tracking angle for the transversal incidence given by 'theta_t'.
            tau = self.tracking_angle(aim=aim, theta_t=theta_t)

            # Calculating the rotated points of the heliostat ################################################
            # Verify if the points will be used in efficiency calculations or not ############################
            if to_calculations:
                points = transform_heliostat(self.seg_pts)
            else:
                points = self.ZX_pts
            rotated_points = rotate_points(points=points, center=self.ZX_center, tau=tau, axis=Iy)
        #######################################################################################################

        # Return heliostat at the horizontal position #########################################################
        else:
            rotated_points = transform_heliostat(self.hel_pts)

        # Checking to return [x, y] or [x, 0, z] array-points #################################################
        if return_xy:
            rotated_points = transform_heliostat(rotated_points)
        #######################################################################################################

        return rotated_points

    def rotated_normals(self, aim: array, theta_t: float, return_xy=False, to_calculations=True):

        """
        This method returns the normal vectors to the points of a rotated heliostat for a given
        transversal incidence angle ('theta_t'), in degrees.
        It considers a particular tracking aim-point at the receiver ('aim') -- an array-point.

        :param aim: The tracking aim-point at the receiver, an array.
        :param theta_t: The transversal incidence angle, in degrees.
        :param return_xy: A boolean sign to whether return [x, y] or [x, 0, z] array-vectors.

        :param to_calculations: A boolean sign to identify whether points in the surface will be used to efficiency
        calculations or not. If 'True', points represent segments of the surface that has the same projected width on
        the aperture plane of the heliostat.

        :return: It returns an array of array-vectors.
        """

        Iy = array([0, 1, 0])
        # The correspondent tracking angle for the transversal incidence given by theta_t
        tau = self.tracking_angle(aim=aim, theta_t=theta_t)

        # Calculating the rotated normals vectors of the heliostat.
        if to_calculations:
            normals = transform_heliostat(self.seg_normals)
        else:
            normals = transform_heliostat(self.normals)
        rotated_normals = rotate_vectors(vectors=normals, tau=tau, axis=Iy)

        if return_xy:
            rotated_normals = transform_heliostat(rotated_normals)

        return rotated_normals

    def local_slope(self, weighted=True):

        slope_f = self.curve.spline().derivative()
        l_slope = arctan(slope_f(self.seg_pts.T[0])) if weighted else arctan(slope_f(self.hel_pts.T[0]))

        return l_slope

    def aim_vector(self, aim_point: array, SunDir: array):
        """
        This method computes the direction of the normal vector at mirror center point for a given Sun direction.

        :param aim_point: The aim point at the receiver used in the tracking procedure
        :param SunDir: A 3D vector which represents the Sun vector
        :return: This function returns a [x, 0, z] vector array.

        This is a method used to calculate the aim vector needed for a SolTrace script. Since it is only needed in this
        context, the sun vector is represented by a [x, y, z] array vector, as usual in SolTrace.

        The aim vector represents a direction of the normal at mirror center point, and with a module equals
        to the focusing distance. This computation follows the definition presented by Said et al. [1].

        [1] Said Z, Ghodbane M, Hachicha AA, Boumeddane B.
        Optical performance assessment of a small experimental prototype of linear Fresnel reflector.
        Case Studies in Thermal Engineering 2019;16:100541. https://doi.org/10.1016/j.csite.2019.100541.

        """

        # Computing the aim point at the receiver as a [x, 0, z] vector array.
        sm = array([aim_point[0], 0, aim_point[-1]])
        # Calculating the focusing distance of the heliostat, i.e., the Euclidian distance between the center point and
        # the aim at the receiver.
        f = dst(p=self.ZX_center, q=sm)

        # Computing the projection of the sun vector in the transversal plane, the ZX plane. Only this projection is
        # used in the tracking calculations.
        st = nrm(array([SunDir[0], 0, SunDir[-1]]))
        n = nrm(rfx_nrm(i=st, r=self.ZX_center - aim_point))

        aim_vec = self.ZX_center + f * array([n[0], 0, n[-1]])

        return aim_vec

    def as_soltrace_element(self, name: str, length: float, aim_pt: array, sun_dir: array,
                            optic: OpticalSurface, par_approx=True, file_path=None):

        aim = array([aim_pt[0], 0, aim_pt[-1]])

        elem = heliostat2soltrace(hel=self, name=name, length=length, aim_pt=aim, sun_dir=sun_dir, optic=optic,
                                  par_approx=par_approx, file_path=file_path)
        return elem


class ParabolicHeliostat:

    """
    The class ParabolicHeliostat aim to represent primary mirrors with a parabolic shape. There are two design
    possibilities:
    (1) A design was proposed by Häberle [1], which considers a design position and an aim point;
    (2) the vertical design, which considers just the focal length of the parabola, here define by the boolean argument
    'forced_design'.


    [1] Häberle A. Linear Fresnel Collectors. Solar Energy, New York, NY: Springer New York; 2013, p. 72–8.
    https://doi.org/10.1007/978-1-4614-5806-7_679.

    """

    def __init__(self, center: array, width: float, theta_d: Angle = None, aim: array = None,
                 forced_design=False, focal_length=None,
                 nbr_pts=121):

        self.shape = 'parabolic'
        self.width = abs(width)

        self.center = array([center[0], center[-1]])

        # Ensure an odd number of points
        self.n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts

        if not forced_design and theta_d is not None and aim is not None:

            aim_pt = array([aim[0], aim[-1]])
            self.hel_pts = design_parabolic_heliostat(center=self.center, width=self.width, nbr_pts=self.n_pts,
                                                      aim=aim_pt, theta_d=theta_d)

        elif forced_design and focal_length is not None:
            self.hel_pts = design_nested_parabolic_heliostat(center=self.center, focal_length=focal_length,
                                                             width=self.width, n_pts=self.n_pts)
        else:
            raise ValueError('Invalid arguments. Please, see ParabolicHeliostat class documentation.')

        self.ZX_center = transform_vector(self.center)  # transforms the previous XY center in a ZX point
        # Attribute that holds the Heliostat object as a PlaneCurve object.
        self.curve = self.as_plane_curve()

        # 3D surface points: as [x, 0, z] arrays.
        # They define the surface. The edges and the center are in this set.
        self.ZX_pts = transform_heliostat(self.hel_pts)  # transforms the previous XY pts in ZX points.
        self.normals = transform_heliostat(self.curve.normals2surface())
        ################################################################

        # Points which define segments of the heliostat surface. Their projected width in the aperture are equal.
        seg_x_range = array([0.5 * (self.curve.x[i] + self.curve.x[i + 1]) for i in range(len(self.curve.x) - 1)])
        hel_as_spline = self.curve.spline()
        self.seg_pts = zeros(shape=(seg_x_range.shape[0], 2))

        # Data (points and normal vectors) of the segments in which the heliostat surface is divided for computations.
        # They are [x, y] arrays.
        self.seg_pts.T[0][:] = seg_x_range
        self.seg_pts.T[1][:] = hel_as_spline(seg_x_range)
        self.seg_normals = PlaneCurve(curve_pts=self.seg_pts, curve_center=self.center).normals2surface()
        ################################################################

    def angular_position(self, aim: array):

        """
        This function calculates the heliostat angular position regarding a tracking aim-point.
        It assumes the ZX plane as the transversal plane, i.e., the one which defines the LFC geometry.

        :param aim: The aim-point at the receiver, an array-point.

        :return: The heliostat angular position, in radians.
        """

        # Old version #################################################
        # This version considers the transversal plane as the XY plane.

        # sm = array([aim[0], aim[-1]])
        # vf = sm - self.center
        # lamb = sign(self.center[0]) * ang(vf, array([0, 1])).rad
        ################################################################

        # New version in 2022-11-08 ####################################
        # This version considers the transversal plane as the ZX plane. Therefore, a different equation is used.

        sm = array([aim[0], 0, aim[-1]])
        lamb = angular_position(center=self.ZX_center, aim=sm)
        ################################################################

        return lamb

    def tracking_angle(self, aim: array, theta_t: float):
        """
        This function calculates the heliostat tracking angle regarding an aim-point at the receiver for a particular
        value of transversal incidence angle.

        It assumes the ZX plane as the transversal plane, i.e., the one which defines the LFC geometry.

        :param aim: The aim-point at the receiver, an array-point.
        :param theta_t: The transversal incidence angle, in degrees.

        :return: The heliostat tracking angle, in radians.
        """

        theta_t_rad = theta_t * pi / 180.
        lamb = self.angular_position(aim=aim)

        tau = (lamb + theta_t_rad) / 2

        return tau

    def as_plane_curve(self):
        return PlaneCurve(curve_pts=self.hel_pts, curve_center=self.center)

    def curvature(self):
        curve = self.as_plane_curve()

        x = curve.x_t
        f = curve.spline(centered=True)
        df = f.derivative()
        d2f = f.derivative(n=2)

        kappa = zeros(x.shape[0])
        kappa[:] = [abs(d2f(v)) / (1 + df(v) ** 2) ** (3 / 2) for v in x]

        return kappa.round(10)

    def local_slope(self, weighted=True):

        slope_f = self.curve.spline().derivative()
        l_slope = arctan(slope_f(self.seg_pts.T[0])) if weighted else arctan(slope_f(self.hel_pts.T[0]))

        return l_slope


class PrimaryField:

    def __init__(self, heliostats: list):

        self.primaries = []
        for i, hel in enumerate(heliostats):
            if isinstance(hel, (Heliostat, ParabolicHeliostat)):
                self.primaries.append(hel)
            else:
                raise f'A non Heliostat instance was inputted. Please, check the {i + 1}-element of the inputted list'

        self.nbr_mirrors = len(self.primaries)
        self.radius = array([hel.radius for hel in self.primaries])

        self.widths = zeros(self.nbr_mirrors)
        self.widths[:] = [hel.width for hel in self.primaries]

        # XY attributes #################################################################
        # These attributes are point and vector arrays with the format [x, y]
        self.centers = zeros(shape=(self.nbr_mirrors, 2))
        self.centers[:] = [hel.center for hel in self.primaries]

        self.heliostats = array([hel.hel_pts for hel in self.primaries])
        #################################################################################

        # ZX attributes #################################################################
        # These attributes are point and vector arrays with the format [x, 0, z]

        self.ZX_centers = zeros(shape=(self.nbr_mirrors, 3))
        self.ZX_centers[:] = [hel.ZX_center for hel in self.primaries]

        self.ZX_primaries = array([hel.ZX_pts for hel in self.primaries])
        self.normals = transform_field(array([hel.normals for hel in self.primaries]))

        self.seg_primaries = transform_field([hel.seg_pts for hel in self.primaries])
        self.seg_normals = transform_field([hel.seg_normals for hel in self.primaries])
        #################################################################################

    def rotated_mirrors(self, theta_t, aim: array, return_xy=True, to_calculations=True):

        mirrors = array([hc.rotated_points(theta_t=theta_t, aim=aim,
                                           return_xy=return_xy, to_calculations=to_calculations)
                         for hc in self.primaries])

        return mirrors

    def rotated_normals(self, theta_t: float, aim: array, return_xy=True, to_calculations=True):

        normals = array([hc.rotated_normals(theta_t=theta_t, aim=aim,
                                            return_xy=return_xy, to_calculations=to_calculations)
                         for hc in self.primaries])

        return normals

    def intercept_factor(self, flat_absorber: Absorber.flat, theta_t: float, theta_l: float, length: float,
                         aim: array = None, cum_eff='collimated', end_losses='no', full_out=False):

        # Check if an aim point was inputted as an argument of the method. If not, it will use the mid-point between
        # the edges of the receiver.

        s1 = flat_absorber.s1
        s2 = flat_absorber.s2

        aim_pt = mid_point(p=s1, q=s2) if aim is None else aim

        r = intercept_factor(field=self.seg_primaries, normals=self.seg_normals,
                             centers=self.ZX_centers, widths=self.widths,
                             s1=s1, s2=s2, aim=aim_pt,
                             theta_t=theta_t, theta_l=abs(theta_l),
                             length=length, cum_eff=cum_eff, end_losses=end_losses)

        out = r if full_out else r[0]

        return out

    def optical_analysis(self, flat_absorber: Absorber.flat, length: float, aim: array = None,
                         cum_eff='collimated', end_losses='no', symmetric=False):

        s1 = flat_absorber.s1
        s2 = flat_absorber.s2

        aim_pt = mid_point(p=s1, q=s2) if aim is None else aim

        angles, t_values, l_values = fac_optical_analysis(field=self, s1=s1, s2=s2,
                                                          aim=aim_pt, length=length,
                                                          cum_eff=cum_eff, end_losses=end_losses)

        transversal_values, longitudinal_values = zeros(shape=(19, 2)), zeros(shape=(19, 2))
        transversal_values[-1, 0], longitudinal_values[-1, 0] = 90, 90

        transversal_values.T[0][:-1][:], longitudinal_values.T[0][:-1][:] = angles, angles
        transversal_values.T[1][:-1][:], longitudinal_values.T[1][:-1][:] = t_values, l_values

        if not symmetric:
            n_theta_t = -angles[::-1][:-1]
            inputs = [(self, s1, s2, aim_pt, theta, length,
                       cum_eff, end_losses) for theta in n_theta_t]

            n_cores = cpu_count()
            with Pool(n_cores - 2) as p:
                n_trans_values = p.starmap(transversal_if, inputs)

            t_angles = (n_theta_t.tolist() + angles.tolist())
            t_values = n_trans_values + transversal_values.T[1][:-1].tolist()

            transversal_values = zeros(shape=(len(t_angles) + 2, 2))
            transversal_values[0, 0], transversal_values[-1, 0] = -90, 90

            transversal_values.T[0][1:-1][:] = array(t_angles)
            transversal_values.T[1][1:-1][:] = array(t_values)
        else:
            pass

        return transversal_values, longitudinal_values

    def acceptance_function(self, theta_t: float, flat_absorber: Absorber.flat, aim: array = None, cum_eff='collimated',
                            lvalue=0.60, dt=0.1):

        s1 = flat_absorber.s1
        s2 = flat_absorber.s2

        aim_pt = mid_point(s1, s2) if aim is None else aim

        off_axis_angles, norm_if = acceptance_function(field=self.seg_primaries, normals=self.seg_normals,
                                                       centers=self.ZX_centers, widths=self.widths,
                                                       theta_t=theta_t, s1=s1, s2=s2, aim=aim_pt,
                                                       cum_eff=cum_eff, lvalue=lvalue, dt=dt)

        return off_axis_angles, norm_if

    def acceptance_angle(self, theta_t: float, flat_absorber: Absorber.flat, aim: array, cum_eff='collimated',
                         lvalue=0.60, dt=0.1):

        off_axis_angles, norm_if = self.acceptance_function(theta_t=theta_t, flat_absorber=flat_absorber, aim=aim,
                                                            cum_eff=cum_eff, lvalue=lvalue, dt=dt)

        acc_angle = acceptance_angle(off_axis_angles, norm_if)

        return acc_angle

    def annual_eta(self, length: float, location: str, flat_absorber: Absorber.flat, aim: array,
                   cum_eff='collimated', end_losses='no', NS=True):

        transversal_data, longitudinal_data = self.optical_analysis(length=length, aim=aim, flat_absorber=flat_absorber,
                                                                    cum_eff=cum_eff, end_losses=end_losses)

        eta = annual_eta(transversal_data=transversal_data, longitudinal_data=longitudinal_data,
                         location=location, NS=NS)

        return eta


class LFR:

    def __init__(self, primary_field: PrimaryField, flat_absorber: Absorber.flat):

        # Primary field data
        self.field = primary_field  # the PrimaryField object
        self.radius = self.field.radius

        # Flat receiver data
        self.receiver = flat_absorber

    def rotated_mirrors(self, theta_t, aim: array = None, return_xy=True, to_calculations=True):

        aim_pt = mid_point(self.receiver.s1, self.receiver.s2) if aim is None else aim

        mirrors = array([hc.rotated_points(theta_t=theta_t, aim=aim_pt,
                                           return_xy=return_xy, to_calculations=to_calculations)
                         for hc in self.field.primaries])

        return mirrors

    def rotated_normals(self, theta_t: float, aim: array = None, return_xy=True, to_calculations=True):

        aim_pt = mid_point(self.receiver.s1, self.receiver.s2) if aim is None else aim

        normals = array([hc.rotated_normals(theta_t=theta_t, aim=aim_pt,
                                            return_xy=return_xy, to_calculations=to_calculations)
                         for hc in self.field.primaries])

        return normals

    def intercept_factor(self, theta_t: float, theta_l: float, length: float, aim: array = None,
                         cum_eff='collimated', end_losses='no'):

        aim_pt = mid_point(self.receiver.s1, self.receiver.s2) if aim is None else aim

        gamma = self.field.intercept_factor(flat_absorber=self.receiver, aim=aim_pt, theta_t=theta_t, theta_l=theta_l,
                                            length=length, cum_eff=cum_eff, end_losses=end_losses, full_out=False)

        return gamma

    def optical_analysis(self, length: float, aim: array = None,
                         cum_eff='collimated', end_losses='no', symmetric=False):

        if aim is None:
            aim_pt = mid_point(self.receiver.s1, self.receiver.s2)
        else:
            aim_pt = aim

        angles, t_values, l_values = fac_optical_analysis(field=self.field, s1=self.receiver.s1, s2=self.receiver.s2,
                                                          aim=aim_pt, length=length,
                                                          cum_eff=cum_eff, end_losses=end_losses)

        transversal_values, longitudinal_values = zeros(shape=(19, 2)), zeros(shape=(19, 2))
        transversal_values[-1, 0], longitudinal_values[-1, 0] = 90, 90

        transversal_values.T[0][:-1][:], longitudinal_values.T[0][:-1][:] = angles, angles
        transversal_values.T[1][:-1][:], longitudinal_values.T[1][:-1][:] = t_values, l_values

        if not symmetric:
            n_theta_t = -angles[::-1][:-1]
            inputs = [(self.field, self.receiver.s1, self.receiver.s2, aim_pt, theta, length,
                       cum_eff, end_losses) for theta in n_theta_t]

            n_cores = cpu_count()
            with Pool(n_cores - 2) as p:
                n_trans_values = p.starmap(transversal_if, inputs)

            t_angles = (n_theta_t.tolist() + angles.tolist())
            t_values = n_trans_values + transversal_values.T[1][:-1].tolist()

            transversal_values = zeros(shape=(len(t_angles) + 2, 2))
            transversal_values[0, 0], transversal_values[-1, 0] = -90, 90

            transversal_values.T[0][1:-1][:] = array(t_angles)
            transversal_values.T[1][1:-1][:] = array(t_values)
        else:
            pass

        return transversal_values, longitudinal_values

    def acceptance_function(self, theta_t: float, aim: array, cum_eff='collimated', lvalue=0.60, dt=0.1):

        off_axis_angles, norm_if = self.field.acceptance_function(theta_t=theta_t, flat_absorber=self.receiver, aim=aim,
                                                                  cum_eff=cum_eff, lvalue=lvalue, dt=dt)

        return off_axis_angles, norm_if

    def acceptance_angle(self, theta_t: float, aim, cum_eff='collimated', lvalue=0.60, dt=0.1):

        theta_a = self.field.acceptance_angle(theta_t=theta_t, flat_absorber=self.receiver, aim=aim, cum_eff=cum_eff,
                                              lvalue=lvalue, dt=dt)

        return theta_a

    def annual_eta(self, length: float, location: str,
                   cum_eff='collimated', end_losses='no', NS=True, symmetric=False):

        transversal_data, longitudinal_data = self.optical_analysis(length=length,
                                                                    cum_eff=cum_eff, end_losses=end_losses,
                                                                    symmetric=symmetric)

        eta = annual_eta(transversal_data=transversal_data,
                         longitudinal_data=longitudinal_data,
                         location=location, NS=NS)

        return eta


class LinearFresnel:

    def __init__(self, primary_field: PrimaryField, receiver: Receiver, aim: array = None):

        self.field = primary_field
        self.secondary = receiver.secondary
        self.absorber = receiver.absorber

        if aim is not None:
            self.aim_pt = aim
        else:
            if self.secondary is not None:
                self.aim_pt = mid_point(p=self.secondary.left_edge, q=self.secondary.rigth_edge)
            else:
                self.aim_pt = mid_point(p=self.absorber.left_edge, q=self.absorber.rigth_edge)

    # def optical_efficiency(self, primaries_property: OpticalSurface, secondary_property: OpticalSurface,
    #                        absorber_property: OpticalSurface, sun: SoltraceSun, trace_options: Trace,
    #                        file_path: Path, file_name: str):
    #
    #
    #
    #     pass

########################################################################################################################
# General auxiliary functions ##########################################################################################


def transform_vector(v: array):
    if v.shape[0] == 3 and v[1] == 0:
        return array([v[0], v[2]])
    elif v.shape[0] == 2:
        return array([v[0], 0, v[1]])
    else:
        raise Exception(f'The input must be an array of the kind [x, 0, z] or [x, y].')


def transform_heliostat(hel: array):
    n = len(hel)
    if hel.shape[-1] == 3:
        vectors = zeros(shape=(n, 2))
    elif hel.shape[-1] == 2:
        vectors = zeros(shape=(n, 3))
    else:
        raise Exception(f'The input must be arrays of the kind [x, 0, z] or [x, y].')

    vectors[:] = [transform_vector(v) for v in hel]
    return vectors


def transform_field(heliostats: array):
    return array([transform_heliostat(hel) for hel in heliostats])


def rotate_points(points: array, center: array, tau: float, axis: array):
    assert center.shape[0] == 2 or center.shape[0] == 3, ValueError("The center point is not a [x, y] or [x, 0, z] "
                                                                    "point array.")
    assert points.shape[1] == center.shape[0], ValueError("Dimensions of 'points' and 'center' are not equal.")

    translated_pts = points - center

    if center.shape[0] == 3 and (axis == array([0, 1, 0])).all():
        rm = R(alpha=tau, v=axis)
    elif center.shape[0] == 2 and (axis == array([0, 0, 1])).all():
        rm = R(alpha=tau)
    else:
        raise ValueError('Arrays shape and rotating axis are not properly inputted')

    rotated_pts = rm.dot(translated_pts.T).T + center
    return rotated_pts


def rotate_vectors(vectors: array, tau: float, axis: array):
    rm = R(alpha=tau, v=axis)
    rotated_vec = rm.dot(vectors.T).T

    return rotated_vec


########################################################################################################################
# LFR design functions #################################################################################################


def heliostat_angular_position(center: array, aim: array):
    sm = array([aim[0], aim[-1]])
    hc = array([center[0], center[-1]])

    vf = sm - hc

    lamb = sign(hc[0]) * ang(vf, array([0, 1])).rad

    return lamb


def heliostat_tracking_angle(center: array, aim: array, theta_t):
    theta_t_rad = theta_t * pi / 180

    lamb = heliostat_angular_position(center=center, aim=aim)
    tau = (lamb + theta_t_rad) / 2

    return tau


def design_cylindrical_heliostat(hc: array, w: float, rc: float, nbr_pts=51):
    """
    This function returns the surface points of a cylindrical shape heliostat.

    :param hc: Heliostat center point
    :param w: Heliostat width
    :param rc: Heliostat cylindrical curvature radius
    :param nbr_pts: Number of points in which the surface is discretized

    :return: An array of points that define the cylindrical surface.
    """

    ####################################################################################################################
    # Ensure an odd number of point to represent the heliostat surface #################################################
    # This is needed to ensure that the center of the mirror is also an element in the array of points which describes
    # the heliostat surface.
    n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts
    ####################################################################################################################

    ####################################################################################################################
    # Calculations #####################################################################################################

    # The array of x-values which the heliostat ranges.
    x_range = linspace(start=-0.5 * w, stop=+0.5 * w, num=n_pts)

    # Ensure that the center point is a XY array point.
    center = array([hc[0], hc[-1]])

    # the function which analytically describes the cylindrical surface which comprises the heliostat
    def y(x): return -sqrt(rc ** 2 - x ** 2) + rc

    # the computation of the points which discretize the heliostat surface
    hel_pts = array([[x, y(x)] for x in x_range]) + center
    ####################################################################################################################

    return hel_pts


def design_flat_heliostat(hc: array, w: float, nbr_pts: int):
    """
    :param hc: heliostat center point
    :param w: heliostat width
    :param nbr_pts: number of point to parametrize
    :return: This function returns a list of points from the function of the heliostat.
       """
    n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts
    center = array([hc[0], hc[-1]])

    hel_pts = zeros(shape=(n_pts, 2))
    hel_pts[:, 0] = linspace(start=-0.5 * w, stop=0.5 * w, num=n_pts) + center[0]

    return hel_pts


def design_nested_parabolic_heliostat(center: array, focal_length: float, width: float, n_pts: int):
    nbr_pts = n_pts if n_pts % 2 != 0 else n_pts + 1

    hc = array([center[0], center[-1]])
    w = abs(width)
    f = abs(focal_length)

    hel_pts = zeros(shape=(nbr_pts, 2))
    x_range = linspace(start=-0.5 * w, stop=0.5 * w, num=nbr_pts)

    def par_f(x): return power(x, 2) / (4 * f)

    hel_pts[:] = [[v, par_f(v)] for v in x_range]

    return PlaneCurve(curve_pts=hel_pts + hc, curve_center=hc).curve_pts


def design_parabolic_heliostat(center: array, aim: array, width: float, nbr_pts: int, theta_d=Angle(rad=0)):
    # The number of points to discretize the parabolic heliostat.
    # It must be an odd number for the heliostat center be a point in the array
    n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts

    # Picking values and changing vectors to a 2D dimension
    w = abs(width)
    hc = array([center[0], center[-1]])
    f = array([aim[0], aim[-1]])

    # calculates the tracking angle
    tau = heliostat_tracking_angle(center=hc, aim=f, theta_t=theta_d.deg)

    # the angle which the parabola optical axis makes with the horizontal axis
    alpha = 0.5 * pi + theta_d.rad
    optical_axis = V(alpha)

    # Check for the specific reference design for a nested parabola design or not
    if nrm(f - hc).dot(optical_axis) == 1.0:
        hel_pts = design_nested_parabolic_heliostat(center=hc, focal_length=dst(f, hc), width=w, n_pts=n_pts)
    else:
        # parabola function to compute the vertex point
        par_f = par(alpha=alpha, f=f, p=hc)
        vertex = par_f(pi)

        # nested parabola focal distance. For the calculation in the rotated reference frame.
        fm = dst(vertex, f)

        # nested parabola equation.
        def centered_par(x):
            return (x ** 2) / (4 * fm)

        # Map the heliostat center in the rotated reference frame
        # In this sense, a tilted parabola becomes a nested one
        pn = R(alpha=-theta_d.rad).dot(hc - vertex)

        # Derivative of the nested parabola for the mapped point in the rotated frame.
        d_pn = pn[0] / (2 * fm)

        # points distant half-width of the mapped point in the rotated frame.
        p1 = pn + 0.5 * w * V(arctan(d_pn))
        p2 = pn - 0.5 * w * V(arctan(d_pn))

        # The edge points which define the parabolic heliostat are calculated by the interception between
        # the nested parabola equation and straight lines which are normal to the tangent at the
        # heliostat center mapped at the rotated reference frame, i.e., point 'pn'.

        def sl1(z):
            return p1[1] + (z - p1[0]) * tan(0.5 * pi + arctan(d_pn))

        def sl2(z):
            return p2[1] + (z - p2[0]) * tan(0.5 * pi + arctan(d_pn))

        e1 = fsolve(lambda z: centered_par(z) - sl1(z), pn[0])[0]
        e2 = fsolve(lambda z: centered_par(z) - sl2(z), pn[0])[0]

        # creates the range of x values comprised between the x components of the edges, i.e., 'e1' and 'e2'
        x_values = linspace(start=min(e1, e2), stop=max(e1, e2), num=n_pts)

        # Calculate the array of points which comprises the heliostat in the rotated frame
        rot_pts = zeros(shape=(n_pts, 2))
        rot_pts[:] = [[x, centered_par(x)] for x in x_values]

        # Calculating the points in the fixed reference frame
        rm = R(alpha=theta_d.rad)
        rot_pts = rm.dot(rot_pts.T).T + vertex

        # Rotating the heliostat to the horizontal position
        rm = R(alpha=-tau)
        hel_pts = rm.dot((rot_pts - hc).T).T + hc

    return hel_pts


def uniform_centers(total_width: float, mirror_width: float, number_mirrors: int) -> array:
    """
    :param total_width: THe total width of the primary field. The distance between the outer edges of the edge mirrors.
    :param mirror_width: The width of the mirrors in the primary field
    :param number_mirrors: number of heliostats
    :return: This function returns a list with all mirrors center point in the x-y plane
    in the form of [xc, 0]. It considers a uniform shift between primaries.
    """

    centers = zeros((number_mirrors, 2))
    centers[:, 0] = linspace(start=0.5 * (total_width - mirror_width), stop=-0.5 * (total_width - mirror_width),
                             num=number_mirrors)

    return centers


def rabl_curvature(center: array, aim: array, theta_d=Angle(rad=0)) -> float:
    """
    A function to calculate the ideal curvature radius of a cylindrical heliostat as defined by Rabl [1, p.179]

    :param center: heliostat's center point.
    :param aim: aim point at the receiver.
    :param theta_d: design position, a transversal incidence angle.
    :return: This function returns the ideal cylindrical curvature.

    References:
    [1] Rabl A. Active Solar Collectors and Their Applications. New York: Oxford University Press; 1985.

    It is important to highlight that calculations are for a xy-plane, where transversal incidence angles are positive
    on the left side of the y-axis direction (a positive rotation about the z-axis). The same definition is used
    for the heliostat angular position computed by the relation between the center point and the aim-point at the
    receiver.

    """

    # Angle from the horizontal which defines the direction of the incoming sunlight at the transversal plane.
    alpha = 0.5 * pi + theta_d.rad
    vi = V(alpha)

    # forcing the center and aim as 2D array points: [x, y]
    hc = array([center[0], center[-1]])
    f = array([aim[0], aim[-1]])

    # Check if the direction of the incoming sunlight is aligned with the mirror focusing vector since
    # the function 'ang(u, v)' used here sometimes calculates a wrong value when u || v.
    # Then, calculate the curvature radius.
    if cross(f - hc, vi).round(7) == 0:
        r = 2 * dst(hc, f)
    else:
        mi = 0.5 * ang(f - hc, vi).rad
        r = 2. * dst(hc, f) / cos(mi)

    return r


def boito_curvature(center: array, aim: array, lat: Angle) -> float:
    """
    Equation proposed by Boito and Grena (2017) for the optimum curvature radius of an LFR cylindrical primary.
    For a further understanding, one must read:
    Boito, P., Grena, R., 2017. https://doi.org/10.1016/j.solener.2017.07.079.

    :param center: heliostat's center point
    :param aim: aim point at the receiver
    :param lat: local latitude
    :return: the cylindrical curvature radius of an LFR primary mirror
    """

    hc = array([center[0], center[-1]])
    sm = array([aim[0], aim[-1]])

    a = 1.0628 + 0.0467 * power(lat.rad, 2)
    b = 0.7448 + 0.1394 * power(lat.rad, 2)

    v = sm - hc
    x, h = absolute(v)

    r = 2 * h * (a + b * power(x / h, 1.6))

    return r


########################################################################################################################
# Functions for the analytical optical method ##########################################################################

def angular_position(center: array, aim: array):
    Iz = array([0, 0, 1])

    sm = array([aim[0], 0, aim[-1]])
    hc = array([center[0], 0, center[-1]])

    aim_vector = sm - hc
    lamb = sign(cross(Iz, aim_vector)[1]) * ang(aim_vector, Iz).rad

    return lamb


def reft(i: array, n: array):
    return 2 * dot(i, n) * n - i


def define_neighbors(theta_t: float, i: int, centers: array, rotated_field: list):
    n_hel = len(centers)
    # select mirror's neighbor to account for blocking
    if centers[i][0] > 0:
        neighbor_b = rotated_field[i + 1]
    else:
        neighbor_b = rotated_field[i - 1]
    # select the edge point on blocking neighbor
    if neighbor_b[0][2] > neighbor_b[-1][2]:
        edge_pt_b = neighbor_b[0]
    else:
        edge_pt_b = neighbor_b[-1]

    # select mirror's neighbor to account for shading
    # the first and last heliostat are never shaded for theta_t > 0 and theta_t < 0, respectively.
    # this is accounted in a different computation, but here it is needed to select one mirror for the algorithm does
    # not get an out of index error
    if theta_t == 0 or (i == 0 and theta_t > 0) or (i == n_hel - 1 and theta_t < 0):
        neighbor_s = neighbor_b
    else:
        neighbor_s = rotated_field[i - int(1 * sign(theta_t))]

    # select the edge point on shading neighbor
    if neighbor_s[0][2] > neighbor_s[-1][2]:
        edge_pt_s = neighbor_s[0]
    else:
        edge_pt_s = neighbor_s[-1]

    return neighbor_b, edge_pt_b, neighbor_s, edge_pt_s


def flat_receiver_limiting_vectors(p: array, sal: array, sar: array, nr: array):
    yll = - (sal - p).dot(nr) / nr[1]
    ylr = - (sar - p).dot(nr) / nr[1]

    vll = array([sal[0], yll, sal[2]]) - p
    vlr = array([sar[0], ylr, sar[2]]) - p

    return vll, vlr


def flat_receiver_shading_vectors(p: array, ni: array, sal: array, sar: array):
    vll_t, vlr_t = sal - p, sar - p

    yl = - vll_t.dot(ni) / ni[1]
    yr = - vlr_t.dot(ni) / ni[1]

    vl_rs = array([vll_t[0], yl, vll_t[2]])
    vr_rs = array([vlr_t[0], yr, vlr_t[2]])

    return vl_rs, vr_rs


def define_neighbor_vectors(p: array, edge_pt_b: array, edge_pt_s: array, ni: array, nr: array):
    # Calculating the neighbor vector for blocking analysis
    ymb = - (edge_pt_b - p).dot(nr) / nr[1]
    vmb = array([edge_pt_b[0], ymb, edge_pt_b[2]]) - p

    # Calculation of the neighbor vector for shading analysis
    yms = - (edge_pt_s - p).dot(ni) / ni[1]
    vms = array([edge_pt_s[0], yms, edge_pt_s[2]]) - p

    return vms, vmb


def receiver_shading_analysis(p: array, ni: array, vi: array, sal: array, sar: array, length: float):
    vl_rs, vr_rs = flat_receiver_shading_vectors(p=p, ni=ni, sal=sal, sar=sar)
    ym = abs(vl_rs[1] + vr_rs[1]) / 2.

    if dot(ni, cross(vl_rs, vi)) >= 0 and dot(ni, cross(vr_rs, vi)) <= 0 and ym < length:
        ns_len_rec = ym / length
    else:
        ns_len_rec = 1

    return ns_len_rec


def neighbor_shading_analysis(p: array, ni: array, vi: array, vms: array, theta_t: float, length: float):
    if theta_t != 0:
        shaded = 1 if sign(theta_t) * dot(ni, cross(vms, vi)) >= 0 else 0
    else:
        shaded = 1 if sign(p[0]) * dot(cross(vms, vi), ni) <= 0 else 0

    if shaded == 1 and abs(vms[1]) < length:
        n_sha_len = abs(vms[1]) / length  # vms points as vi, so its 'y' component is negative.
    else:
        n_sha_len = 1

    return n_sha_len


def blocking_analysis(p: array, vn: array, nr: array, vmb: array):
    if sign(p[0]) * dot(cross(vn, vmb), nr) >= 0:
        blocked = 0
    else:
        blocked = 1

    return blocked


def focusing_analysis(vn: array, nr: array, vll: array, vlr: array):
    if dot(cross(vn, vlr), nr) <= 0 and dot(cross(vn, vll), nr) >= 0:
        focused = 1
    else:
        focused = 0

    return focused


def collimated_rays_analysis(theta_t: float, i: int, n: int, p: array, vi: array, ni: array, vn: array, nr: array,
                             vms: array, vmb: array, sal: array, sar: array, length: float):
    # calculate the non shaded length due to receiver and neighbor
    ns_len_receiver = receiver_shading_analysis(p=p, ni=ni, vi=vi, sal=sal, sar=sar, length=length)
    ns_len_nei = neighbor_shading_analysis(p=p, ni=ni, vi=vi, vms=vms, theta_t=theta_t, length=length)

    # for theta_t > 0, the first heliostat (i == 0) is never shaded
    if theta_t > 0 and i == 0:
        ns_len = 1
    # for theta_t < 0, the last heliostat (i == n - 1) is never shaded
    elif theta_t < 0 and i == n - 1:
        ns_len = 1
    else:
        ns_len = min(ns_len_receiver, ns_len_nei)

    # If all heliostat is being shaded (ns_len == 0) no energy could be collected nor other losses could exist
    if ns_len == 0:
        pt_if, blo, de, acc, sha = 0, 0, 0, 0, 1
    else:
        sha = 1 - ns_len
        vll, vlr = flat_receiver_limiting_vectors(p=p, sal=sal, sar=sar, nr=nr)
        blo = ns_len * blocking_analysis(p=p, vn=vn, nr=nr, vmb=vmb)
        de = ns_len * (1 - focusing_analysis(vn=vn, nr=nr, vll=vll, vlr=vlr)) if blo == 0 else 0
        acc = 0
        pt_if = ns_len if blo == 0 and de == 0 else 0

    return pt_if, sha, blo, de, acc, ns_len


def collimated_end_losses(end_losses: str, pt_if: float, theta_l: float, length: float, ns_len: float,
                          p: array, vn: array, sm: array):
    Iz = array([0, 0, 1])

    if end_losses == 'no' or pt_if == 0 or theta_l == 0:
        elo = 0
    else:
        lost_length = islp(p, vn, sm, Iz)[1] / length
        if lost_length < ns_len:
            elo = lost_length / ns_len
        else:
            elo = pt_if
    return elo


def inc_beam_analysis(p: array, theta_t: float, vi: array, ni: array, sal: array, sar: array, vms: array):
    vl_rs, vr_rs = flat_receiver_shading_vectors(p=p, ni=ni, sal=sal, sar=sar)

    signal = 1 if p[0] < 0 else -1

    theta1 = ang_pnd(u=vi, v=vr_rs, n=ni)
    theta2 = ang_pnd(u=vi, v=vl_rs, n=ni)
    theta3 = ang_pnd(u=vi, v=vms, n=ni)

    if theta_t == 0:
        r_shading = closed(min(theta1, theta2), max(theta1, theta2))
        n_shading = closed(min(pi * signal, theta3), max(pi * signal, theta3))
    else:
        r_shading = closed(min(theta1, theta2), max(theta1, theta2))
        n_shading = closed(min(sign(theta_t) * pi, theta3), max(sign(theta_t) * pi, theta3))

    return r_shading, n_shading


def ref_beam_analysis(p: array, vn: array, nr: array, vmb: array, vll: array, vlr: array):
    signal = 1 if p[0] > 0 else -1

    theta1 = ang_pnd(u=vn, v=vlr, n=nr)
    theta2 = ang_pnd(u=vn, v=vll, n=nr)
    theta3 = ang_pnd(u=vn, v=vmb, n=nr)

    blocking = closed(min(pi * signal, theta3), max(pi * signal, theta3))
    intercepted = closed(min(theta1, theta2), max(theta1, theta2))

    return blocking, intercepted


def reduce_interval(a, b):
    """
    :param a: continuous interval [a1, a2]: a2 > a1 -- portion.closed(a1, a2)
    :param b: continuous interval [b1, b2]: b2 > b1 -- portion.closed(b1, b2)
    :return: This functions returns the difference between 'a' and its intersection with 'b',
    and checks for an empty interval
    """
    c = a - (a & b)
    if c == empty():
        c = closed(0, 0)

    return c


def Flux(interval, g):
    u_val = interval.upper
    l_val = interval.lower

    flux = abs(g(abs(u_val)) - sign(u_val) * sign(l_val) * g(abs(l_val)))

    return flux


def non_shaded_lengths(p: array, ni: array, vms: array, sal: array, sar: array, length: float):
    vl_rs, vr_rs = flat_receiver_shading_vectors(p=p, ni=ni, sal=sal, sar=sar)
    ym = abs(vl_rs[1] + vr_rs[1]) / 2

    ns_len_receiver = ym / length if ym < length else 1
    ns_len_nei = abs(vms[1]) / length if abs(vms[1]) < length else 1

    return ns_len_receiver, ns_len_nei


def one_segment_analysis(r_shading, n_shading, blocking, intercepted):
    n_sha = reduce_interval(a=n_shading, b=r_shading)
    # update neighbor blocking interval
    n_blo = reduce_interval(a=blocking, b=r_shading)
    n_blo = reduce_interval(a=n_blo, b=n_sha)
    # update intercept interval
    i1 = reduce_interval(a=intercepted, b=r_shading)
    i1 = reduce_interval(a=i1, b=n_sha)
    i1 = reduce_interval(a=i1, b=n_blo)

    return n_sha, n_blo, i1


def segments_analysis(r_shading, n_shading, blocking, intercepted):
    # First length section -- shading by the neighbor and the receiver
    n_sha1 = reduce_interval(a=n_shading, b=r_shading)
    # update neighbor blocking interval
    n_blo1 = reduce_interval(a=blocking, b=r_shading)
    n_blo1 = reduce_interval(a=n_blo1, b=n_sha1)
    # update intercept interval
    i1 = reduce_interval(a=intercepted, b=r_shading)
    i1 = reduce_interval(a=i1, b=n_sha1)
    i1 = reduce_interval(a=i1, b=n_blo1)
    # Second length section -- shaded only by the neighbor
    # update neighbor blocking interval
    n_blo2 = reduce_interval(a=blocking, b=n_shading)
    # update intercept interval
    i2 = reduce_interval(a=intercepted, b=n_shading)
    i2 = reduce_interval(a=i2, b=n_blo2)
    # calculate the intercepted flux for the second length section
    # Third section of segment -- not shaded length
    # update intercept interval
    i3 = reduce_interval(a=intercepted, b=blocking)
    # calculate the intercepted flux for the third length section

    return n_sha1, n_blo1, n_blo2, i1, i2, i3


def flux_analysis_lost_length(end_losses: str, theta_l: float, pt_if: float, vll: array, vlr: array, length: float):
    if end_losses == 'no' or theta_l == 0 or pt_if == 0:
        lost_length = 0
    else:
        y_end = max(vll[1], vlr[1])
        if y_end >= length:
            lost_length = 1
        else:
            lost_length = y_end / length
    return lost_length


def segments_end_losses(lost_length: float, l1: float, l2: float, Int: list, Blo: list):
    I1, I2, I3 = Int
    B1, B2, B3 = Blo

    if lost_length >= l2 + l1:
        elo = l2 * I2 + l1 * I1 + (lost_length - l1 - l2) * I3
        elo_b = l2 * B2 + l1 * B1 + (lost_length - l1 - l2) * B3
    elif lost_length >= l1:
        elo = l1 * I1 + (lost_length - l1) * I2
        elo_b = l1 * B1 + (lost_length - l1) * B2
    else:
        elo = lost_length * I1
        elo_b = lost_length * B1
    return elo, elo_b


def rotated_field_data(field: array, normals: array, centers: array, sm: array, theta_t: float):
    # calculates angular position (lamb)
    lamb = zeros(len(centers))
    lamb[:] = [angular_position(center=hc, aim=sm) for hc in centers]

    # mirrors tracking angle
    tau = 0.5 * (lamb + deg2rad(theta_t))

    # calculating the points of each mirror after the proper rotation due to the tracking procedure
    rotated_field = array([rotate_points(points=hel, center=hc, tau=tau[i], axis=array([0, 1, 0]))
                           for i, (hel, hc) in enumerate(zip(field, centers))
                           ])

    # calculating the direction of the normal vectors after the rotation of the tracking procedure
    rotated_normals = array([rotate_vectors(vectors=n, tau=tau[i], axis=array([0, 1, 0]))
                             for i, n in enumerate(normals)
                             ])

    return tau, rotated_field, rotated_normals


def intercept_factor(field: array, normals: array, centers: array, widths: array,
                     s1: array, s2: array, aim: array, theta_t: float, theta_l: float,
                     length: float, cum_eff='collimated', end_losses='no', longitudinal_segments=False):
    """
    This function returns the bi-axial intercept factor of a linear fresnel concentrator with a flat receiver.
    It is based on the method published by Santos et al. [1].

    :param field: A list of arrays. Each array defined a heliostat composed of a finite number of
    [x, 0, z] point arrays.
    :param normals: A list of arrays.
    Each array defined the normal vector to the heliostat in a vertical position composed
    of a number of [x, 0, z] vector arrays.
    :param centers: An array with the center points of all heliostats. Each center is defined by an
    [x, 0, z] point array
    :param widths: An array with the widths of all heliostats.

    :param s1: One edge point of the flat receiver.
    :param s2: The other edge point of the flat receiver.
    :param aim: The aim point at the receiver which the heliostats use in the tracking procedure.

    :param theta_t: Transversal incidence angle, in degrees.
    :param theta_l: Longitudinal incidence angle, in degrees.

    :param length: LFR length in the longitudinal direction, im mm.
    :param cum_eff: Linear effective source cumulative function.
    :param end_losses: A sign to calculate or not the end-losses

    :param longitudinal_segments: A boolean sign to account or not with the
    non-shaded and non-blocked longitudinal segments.

    :return: It returns the bi-axial intercept factor of a linear fresnel concentrator with a flat receiver.
    A value between 0 and 1

    [1] Santos AV, Canavarro D, Horta P, Collares-Pereira M. An analytical method for the optical analysis of
    Linear Fresnel Reflectors with a flat receiver. Solar Energy 2021;227:203–16.
    https://doi.org/10.1016/j.solener.2021.08.085.

    """

    Ix = array([1, 0, 0])

    vi = sun_direction(Angle(deg=theta_t), Angle(deg=abs(theta_l)))  # direction of incident sunlight
    ni = nrm(cross(vi, Ix))  # normal vector that defines the incidence plane

    # checks edges sides: right (x > 0) or left (x < 0)
    if s2[0] > s1[0]:
        sar, sal = s2, s1
    else:
        sar, sal = s1, s2

    sar = array([sar[0], 0, sar[-1]])
    sal = array([sal[0], 0, sal[-1]])
    sm = array([aim[0], 0, aim[-1]])

    # calculating the points of each mirror after the proper rotation due to the tracking procedure
    tau, rotated_field, rotated_normals = rotated_field_data(field=field, normals=normals, centers=centers,
                                                             theta_t=theta_t, sm=sm)

    if longitudinal_segments:
        results = segments_efficiency_computation(rotated_field=rotated_field, rotated_normals=rotated_normals,
                                                  sm=sm, tau=tau, centers=centers, widths=widths,
                                                  theta_t=theta_t, theta_l=theta_l, vi=vi, ni=ni, sar=sar, sal=sal,
                                                  length=length, cum_eff=cum_eff, end_losses=end_losses)
    else:
        results = one_segment_efficiency_computation(rotated_field=rotated_field, rotated_normals=rotated_normals,
                                                     sm=sm, tau=tau, centers=centers, widths=widths,
                                                     theta_t=theta_t, theta_l=theta_l, vi=vi, ni=ni, sar=sar, sal=sal,
                                                     length=length, cum_eff=cum_eff, end_losses=end_losses)

    lfr_if, shading_losses, blocking_losses, defocusing_losses, cosine_losses, end_loss = results

    return lfr_if, shading_losses, blocking_losses, defocusing_losses, cosine_losses, end_loss


def transversal_if(field: PrimaryField, s1: array, s2: array, aim: array, theta_t: float, length: float,
                   cum_eff='collimated', end_losses='no'):
    return intercept_factor(field=field.ZX_primaries, normals=field.normals,
                            centers=field.ZX_centers, widths=field.widths, s1=s1, s2=s2, aim=aim,
                            theta_t=theta_t, theta_l=0, length=length,
                            cum_eff=cum_eff, end_losses=end_losses)[0]


def longitudinal_if(field: PrimaryField, s1: array, s2: array, aim: array, theta_l: float, length: float,
                    cum_eff='collimated', end_losses='no'):
    return intercept_factor(field=field.ZX_primaries, normals=field.normals,
                            centers=field.ZX_centers, widths=field.widths, s1=s1, s2=s2, aim=aim,
                            theta_t=0, theta_l=theta_l, length=length,
                            cum_eff=cum_eff, end_losses=end_losses)[0]


def fac_optical_analysis(field: PrimaryField, s1: array, s2: array, aim: array, length: float,
                         cum_eff='collimated', end_losses='no'):
    angles = array([x for x in range(0, 90, 5)])
    inputs = [(field, s1, s2, aim, theta, length, cum_eff, end_losses) for theta in angles]

    n_cores = cpu_count()

    with Pool(n_cores - 2) as p:
        t_values = p.starmap(transversal_if, inputs)
        l_values = p.starmap(longitudinal_if, inputs)

    return angles, t_values, l_values


def acceptance_function(field: array, normals: array, centers: array, widths: array,
                        s1: array, s2: array, aim: array, theta_t: float, cum_eff='collimated',
                        lvalue=0.60, dt=0.1):
    """
    :param field: A list of arrays. Each array defined a heliostat composed of a finite number of
    [x, 0, z] point arrays.

    :param normals: A list of arrays.
    Each array defined the normal vector to the heliostat in a vertical position composed
    of a number of [x, 0, z] vector arrays.

    :param centers: An array with the center points of all heliostats. Each center is defined by an
    [x, 0, z] point array

    :param widths: An array with the widths of all heliostats.
    :param s1: One of the edges of the flat receiver. It is a [x, y] array or a [x, 0, z] array.
    :param s2: The other edge of the flat receiver. It is a [x, y] array or a [x, 0, z] array.
    :param aim: The aim point of the heliostats to track the sun movement. It is a [x, y] array or a [x, 0, z] array.
    :param theta_t: The transversal incidence angle, in degrees
    :param cum_eff: A linear cumulative effective source.
    :param lvalue: Normalized efficiency lower value up to the off-axis calculations will be accounted.
    :param dt: the off-axis angle step.
    :return: This function returns the arrays of angles (0 and off-axis), and the correspondent array of the
    normalized (by the on-axis value) efficiencies.
    """

    Ix, Iy, Iz = identity(3)

    # checks edges sides: right (x > 0) or left (x < 0)
    if s2[0] > s1[0]:
        sar, sal = s2, s1
    else:
        sar, sal = s1, s2

    # ensure that arrays are the kind [x, 0, z]
    sar = array([sar[0], 0, sar[-1]])
    sal = array([sal[0], 0, sal[-1]])
    sm = array([aim[0], 0, aim[-1]])

    # calculating the points of each mirror after the proper rotation due to the tracking procedure
    tau, rotated_field, rotated_normals = rotated_field_data(field=field, normals=normals, centers=centers,
                                                             theta_t=theta_t, sm=sm)

    # Calculates the on-axis power, i.e., the efficiency at an incidence [theta_t, 0].
    on_axis_flux = intercept_factor(field=field, normals=normals,
                                    centers=centers, widths=widths,
                                    s1=s1, s2=s2, aim=sm, theta_t=theta_t, theta_l=0,
                                    cum_eff=cum_eff, length=60000, end_losses='no')[0]

    # The on-axis values. Normalized power and off-axis angles are 1 and 0 for the on-axis position, logically.
    norm_if = [1.0]
    off_axis_angles = [0]

    # variable to account the number of loops in the while section.
    k = 1

    # a loop for positive off-axis incidences
    # This will run until the off-axis efficiency reaches the 'lvalue' argument of the function.
    while norm_if[-1] >= lvalue:

        # Off-axis incidence. It considers the on-axis angle ('theta_t') and a displacement to give the off-axis value.
        # See that no additional tracking was considered. Therefore, this incidence is an off-axis one.
        angle = theta_t + k * dt  # the off-axis angle to be analyzed.
        vi = sun_direction(Angle(deg=angle), Angle(deg=0))  # direction of incident sunlight.
        ni = nrm(cross(vi, Ix))  # normal vector that defines the incidence plane.

        # compute the intercept factor for the incidence direction given by 'vi' and 'ni'.
        lfr_if = acceptance_flux_computation(rotated_field=rotated_field, rotated_normals=rotated_normals,
                                             centers=centers, widths=widths, theta_t=theta_t, vi=vi, ni=ni,
                                             sar=sar, sal=sal, cum_eff=cum_eff)

        # adds to respective list the off-axis efficiency and incidence angle
        norm_if.append(lfr_if / on_axis_flux)
        off_axis_angles.append(angle - theta_t)

        # update the deviation from the on-axis angle.
        k += 1

        # check if the last two values were zero. The concentrator efficiency will never be lower than zero.
        if (array(norm_if[-2:]).round(4) == array([0., 0.])).all():
            break

    # variable to account the number of loops in the while section.
    k = 1
    # a loop for negative off-axis incidences
    # This will run until the off-axis efficiency reaches the 'lvalue' argument of the function.
    while norm_if[0] >= lvalue:

        # Off-axis incidence. It considers the on-axis angle ('theta_t') and a displacement to give the off-axis value.
        # See that no additional tracking was considered. Therefore, this incidence is an off-axis one.
        angle = theta_t - k * dt  # the off-axis angle to be analyzed.
        vi = sun_direction(Angle(deg=angle), Angle(deg=0))  # direction of incident sunlight.
        ni = nrm(cross(vi, Ix))  # normal vector that defines the incidence plane.

        # compute the intercept factor for the incidence direction given by 'vi' and 'ni'.
        lfr_if = acceptance_flux_computation(rotated_field=rotated_field, rotated_normals=rotated_normals,
                                             centers=centers, widths=widths, theta_t=theta_t, vi=vi, ni=ni,
                                             sar=sar, sal=sal, cum_eff=cum_eff)

        # adds to respective list the off-axis efficiency and incidence angle
        norm_if.insert(0, lfr_if / on_axis_flux)
        off_axis_angles.insert(0, angle - theta_t)

        # update the deviation from the on-axis angle.
        k += 1
        # check if the last two values were zero. The concentrator efficiency will never be lower than zero.
        if (array(norm_if[:2]).round(4) == array([0., 0.])).all():
            break

    return array(off_axis_angles), array(norm_if)


def acceptance_angle(angles: array, norm_if: array, ref_value=0.9):
    acc_function = InterpolatedUnivariateSpline(angles, norm_if - ref_value, k=3)
    roots = acc_function.roots()

    theta_a = 0.5 * abs(roots[0] - roots[1])

    return theta_a


########################################################################################################################

def annual_eta(transversal_data: array, longitudinal_data: array, location: str, NS=True):
    """
    A function to compute the annual optical efficiency of a linear concentrator.

    :param transversal_data: Transversal optical efficiency values, in the form of arrays of [angle, efficiency].
    :param longitudinal_data: Longitudinal optical efficiency values, in the form of arrays of [angle, efficiency].
    :param location: A file with the site data where the concentrator is installed.
    :param NS: A sign to inform whether a NS (North-South) or EW (East-West) mounting was considered.

    :return: It returns the annual optical efficiency (a value between 0 and 1)

    --------------------------------------------------------------------------------------------------------------------

    This function assumes the sun azimuth as measured regarding the South direction, where displacements East of South
    are negative and West of South are positive [3]. Moreover, the inertial XYZ coordinates systems is aligned as
    follows: X points to East, Y to North, and Z to Zenith.

    [1] IEC (International Electrotechnical Commission). Solar thermal electric plants
    - Part 5-2: Systems and components - General requirements and test methods for large-size linear Fresnel collectors.
    Solar thermal electric plants, 2021.
    [2] Hertel JD, Martinez-Moll V, Pujol-Nadal R. Estimation of the influence of different incidence angle modifier
    models on the bi-axial factorization approach. Energy Conversion and Management 2015;106:249–59.
    https://doi.org/10.1016/j.enconman.2015.08.082.
    [3] Duffie JA, Beckman WA. Solar Engineering of Thermal Processes. 4th Ed. New Jersey: John Wiley & Sons; 2013.
    """

    ####################################################################################################################
    # Creating optical efficiency (intercept factor) functions     #####################################################

    # Creating both transversal and longitudinal optical efficiencies functions for the calculations.
    # Ref. [1] suggest that a linear interpolation should be considered.
    gamma_t = interp1d(x=transversal_data.T[0], y=transversal_data.T[1], kind='linear')
    gamma_l = interp1d(x=longitudinal_data.T[0], y=longitudinal_data.T[1], kind='linear')
    # Taking the value of optical efficiency at normal incidence.
    gamma_0 = gamma_t(0)
    ####################################################################################################################

    ####################################################################################################################
    # Checking if a symmetric linear concentrator is being considered or not.
    symmetric_lfr = True if transversal_data.shape[0] == longitudinal_data.shape[0] else False
    ####################################################################################################################

    ####################################################################################################################
    # Importing sun position and irradiation data from external files ##################################################

    # It will only consider DNI values greater than zero, which imply that -90º <= zenith <= +90º.

    # from Meteonorm files which file extension is 'dat'
    # In this case, the DataFrame has the following columns: sun altitude, sun azimuth, and DNI.
    # However, the DataFrame has no headers.
    file_format = str(location)[-3:]
    if file_format == 'dat':
        site_data = read_csv(location, header=None)
        df = site_data[site_data[2] > 0]

        zenith = 90 - array(df[0])
        azimuth = array(df[1])
        dni = array(df[2])

    # from TRNSYS tm2 files that were converted to a csv data files.
    # In this case, the DataFrame has the headers shown in the below code.
    elif file_format == 'csv':
        site_data = read_trnsys_tmy2(location)
        df = site_data[site_data['DNI [W/m2]'] > 0]

        zenith = array(df['Solar Zenith [degrees]'])
        azimuth = array(df['Solar Azimuth [degrees]'])
        dni = array(df['DNI [W/m2]'])

    # Furthermore, more extensions can be included.
    # The data should be saved as arrays to use the Numpy vectorization approach and give fast results.
    # Data must be in the variables 'zenith', 'azimuth', and 'dni'.
    else:
        raise 'Input error in the location file'
    ####################################################################################################################

    ####################################################################################################################
    # Calculating the linear incidence angles ##########################################################################

    # Here, it considers transversal and solar longitudinal incidence angles, as defined in Refs. [1,2].
    theta_t, _, theta_i = sun2lin(zenith=zenith, azimuth=azimuth, degree=True, NS=NS, solar_longitudinal=True)
    ####################################################################################################################

    ####################################################################################################################
    # Energetic computations ###########################################################################################
    # It does not matter if transversal incidence angle is positive or negative if the concentrator is symmetric.
    # Nevertheless, the sign of the longitudinal angle does not matter at all.
    # Since vector operations were used, it only has few lines of code.
    if symmetric_lfr:
        energy_sum = (gamma_t(absolute(theta_t)) * gamma_l(absolute(theta_i)) / gamma_0).dot(dni)
    else:
        energy_sum = (gamma_t(theta_t) * gamma_l(absolute(theta_i)) / gamma_0).dot(dni)
    ####################################################################################################################

    # energetic sum is converted to annual optical efficiency by the division for the annual sum of DNI.
    # the annual sum of dni is the available energy to be collected.
    return energy_sum / dni.sum()


########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Analytical method core calculations ##################################################################################

def one_segment_efficiency_computation(rotated_field: array, rotated_normals: array, sm: array, tau: array,
                                       centers: array, widths: array, theta_t: float, theta_l: float,
                                       vi: array, ni: array, sar: array, sal: array,
                                       length: float, cum_eff, end_losses):
    # incidence angles in radians to be used
    theta_t_rad = theta_t * pi / 180
    theta_l_rad = abs(theta_l * pi / 180)

    n_hel = len(centers)
    # creates the arrays that will hold values of losses and intercept factor for each mirror
    gamma = zeros(n_hel)  # intercept factor
    elf_s = zeros(n_hel)  # shading losses
    elf_b = zeros(n_hel)  # blocking losses
    elf_d = zeros(n_hel)  # defocusing losses
    elf_c = zeros(n_hel)  # cosine effect losses
    elf_e = zeros(n_hel)  # end losses intercept factor

    # It runs an analysis for each mirror in heliostats field
    for i, hel in enumerate(rotated_field):

        n_pts = len(hel)
        # creates the arrays that will hold values of losses and intercept factor for each point in the mirror
        hel_if = zeros(n_pts)  # intercept factor
        hel_s = zeros(n_pts)  # shading losses
        hel_b = zeros(n_pts)  # blocking losses
        hel_d = zeros(n_pts)  # defocusing losses
        hel_end = zeros(n_pts)  # end losses intercept factor

        # Calculates the neighbors mirrors, and the edge points for losses analysis
        neighbor_b, edge_pt_b, neighbor_s, edge_pt_s = define_neighbors(theta_t=theta_t, i=i, centers=centers,
                                                                        rotated_field=rotated_field)

        # It runs an analysis for each point in the mirror being analyzed
        for j, p in enumerate(hel):

            ns = rotated_normals[i][j]  # the normal vector to mirror's surface at point 'p'
            vn = reft(vi, ns)  # direction of the reflected ray on that point
            nr = reft(ni, ns)  # normal vector that defines the reflection plane

            # calculate the neighbor vectors to compute shading and blocking losses
            vms, vmb = define_neighbor_vectors(p=p, edge_pt_b=edge_pt_b, edge_pt_s=edge_pt_s, ni=ni, nr=nr)

            if cum_eff == 'collimated':
                pt_if, sha, blo, de, acc, ns_len = collimated_rays_analysis(theta_t=theta_t, i=i, n=n_hel, p=p,
                                                                            vi=vi, ni=ni, vn=vn, nr=nr,
                                                                            vms=vms, vmb=vmb, sal=sal,
                                                                            sar=sar, length=length)
                # End-losses calculations for collimated sunlight model
                elo = collimated_end_losses(end_losses=end_losses, pt_if=pt_if, theta_l=theta_l, length=length,
                                            ns_len=ns_len, p=p, vn=vn, sm=sm)
                pt_if = pt_if - elo if pt_if > elo else 0
            else:
                # Incident and Reflected sunlight as beam fluxes
                # The analysis presented here does not consider the non-shaded and non-blocked lengths of the module due
                # to longitudinal effects of the incident and reflected beams.

                # the first mirror will never be shaded (receiver or neighbor)
                r_shading, n_shading = inc_beam_analysis(p=p, theta_t=theta_t, vi=vi, ni=ni, sal=sal, sar=sar, vms=vms)
                if (theta_t > 0 and i == 0) or (theta_t < 0 and i == n_hel - 1):
                    n_shading = closed(0, 0)

                # Force the neighbor shading interval to zero.
                # This is useful to correctly compute the central mirror (below the receiver) at normal incidence.
                if theta_t == 0 and n_hel % 2 != 0 and i == int((n_hel - 1) / 2):
                    n_shading = closed(0, 0)

                # Compute the angular intervals of the reflected beam related to blocking and intercepted.
                vll, vlr = flat_receiver_limiting_vectors(p=p, sal=sal, sar=sar, nr=nr)
                blocking, intercepted = ref_beam_analysis(p=p, vn=vn, nr=nr, vmb=vmb, vll=vll, vlr=vlr)

                # the central heliostat, right below the receiver is never blocked by its neighbors
                # on the other hand, it can be shaded by the receiver and its neighbors
                if centers[i][0].round(5) == 0:
                    blocking = closed(0, 0)

                n_sha_interval, n_blo_interval, intercept_interval = one_segment_analysis(r_shading=r_shading,
                                                                                          n_shading=n_shading,
                                                                                          blocking=blocking,
                                                                                          intercepted=intercepted)

                sha = Flux(interval=r_shading, g=cum_eff) + Flux(interval=n_sha_interval, g=cum_eff)
                blo = Flux(interval=n_blo_interval, g=cum_eff)
                pt_if = Flux(interval=intercept_interval, g=cum_eff)

                # Computing end-losses
                # Calculating the segment length that is not being used due to the finite length of the receiver.
                lost_length = flux_analysis_lost_length(end_losses=end_losses, theta_l=theta_l, pt_if=pt_if,
                                                        vll=vll, vlr=vlr, length=length)
                # Computing end-losses.
                elo = pt_if * lost_length
                # updating the intercepted fraction of the beam
                pt_if = pt_if - elo

                # Computing the defocusing losses,
                # which includes finite acceptance losses (sun shape and optical errors) [1].
                de = 1 - (pt_if + sha + blo + elo)
            ############################################################################################################
            # Updating the array with the values for each point of the heliostat
            hel_if[j] = pt_if
            hel_s[j] = sha
            hel_b[j] = blo
            hel_d[j] = de
            hel_end[j] = elo

        # Cosine losses accounting #####################################################################################
        # Analytical expression for the cosine effect, derived from a vector analysis from the definition.
        # It considers mirror's center point due to its central symmetry, instead of account for each point.

        num = cos(theta_l_rad) * cos(theta_t_rad - tau[i])
        den = cos(theta_t_rad) * sqrt(1 + cos(theta_l_rad) ** 2 * tan(theta_t_rad) ** 2)
        cosine_effect = num / den

        # Updates the values of each loss and intercept factor for the mirror, an average of all its points
        gamma[i] = hel_if.mean() * cosine_effect  # includes cosine effect in heliostat's intercept factor
        elf_s[i] = hel_s.mean()  # heliostat's energy loss factor due to shading
        elf_b[i] = hel_b.mean()  # heliostat's energy loss factor due to blocking
        elf_d[i] = hel_d.mean()  # heliostat's energy loss factor due to defocusing
        elf_c[i] = 1 - cosine_effect  # heliostat's energy loss factor due to cosine effect
        elf_e[i] = hel_end.mean()  # heliostat's intercept factor for end losses

    # calculate the sum of mirrors width -- the weights used to compute averages for the whole LFR.
    total_widths = widths.sum()

    # Parameter values for the whole concentrator are weighted averages of its mirrors
    lfr_if = widths.dot(gamma) / total_widths  # LFR intercept factor
    shading_losses = widths.dot(elf_s) / total_widths  # shading losses
    blocking_losses = widths.dot(elf_b) / total_widths  # blocking losses
    defocusing_losses = widths.dot(elf_d) / total_widths  # defocusing losses
    cosine_losses = widths.dot(elf_c) / total_widths  # cosine effect losses
    end_loss = widths.dot(elf_e) / total_widths  # LFR end losses

    return lfr_if, shading_losses, blocking_losses, defocusing_losses, cosine_losses, end_loss


def segments_efficiency_computation(rotated_field: array, rotated_normals: array,
                                    centers: array, widths: array, tau: array, sm: array,
                                    theta_t: float, theta_l: float,
                                    vi: array, ni: array, sar: array, sal: array,
                                    length: float, cum_eff, end_losses):
    # incidence angles in radians to be used
    theta_t_rad = theta_t * pi / 180
    theta_l_rad = abs(theta_l * pi / 180)

    n_hel = len(centers)
    # creates the arrays that will hold values of losses and intercept factor for each mirror
    gamma = zeros(n_hel)  # intercept factor
    elf_s = zeros(n_hel)  # shading losses
    elf_b = zeros(n_hel)  # blocking losses
    elf_d = zeros(n_hel)  # defocusing losses
    elf_c = zeros(n_hel)  # cosine effect losses
    elf_e = zeros(n_hel)  # end losses intercept factor

    # It runs an analysis for each mirror in heliostats field
    for i, hel in enumerate(rotated_field):

        n_pts = len(hel)
        # creates the arrays that will hold values of losses and intercept factor for each point in the mirror
        hel_if = zeros(n_pts)  # intercept factor
        hel_s = zeros(n_pts)  # shading losses
        hel_b = zeros(n_pts)  # blocking losses
        hel_d = zeros(n_pts)  # defocusing losses
        hel_end = zeros(n_pts)  # end losses intercept factor

        # Calculates the neighbors mirrors, and the edge points for losses analysis
        neighbor_b, edge_pt_b, neighbor_s, edge_pt_s = define_neighbors(theta_t=theta_t, i=i, centers=centers,
                                                                        rotated_field=rotated_field)

        # It runs an analysis for each point in the mirror being analyzed
        for j, p in enumerate(hel):

            ns = rotated_normals[i][j]  # the normal vector to mirror's surface at point 'p'
            vn = reft(vi, ns)  # direction of the reflected ray on that point
            nr = reft(ni, ns)  # normal vector that defines the reflection plane

            # calculate the neighbor vectors to compute shading and blocking losses
            vms, vmb = define_neighbor_vectors(p=p, edge_pt_b=edge_pt_b, edge_pt_s=edge_pt_s, ni=ni, nr=nr)

            if cum_eff == 'collimated':
                pt_if, sha, blo, de, acc, ns_len = collimated_rays_analysis(theta_t=theta_t, i=i, n=n_hel, p=p,
                                                                            vi=vi, ni=ni, vn=vn, nr=nr,
                                                                            vms=vms, vmb=vmb, sal=sal,
                                                                            sar=sar, length=length)
                # End-losses calculations for collimated sunlight model
                elo = collimated_end_losses(end_losses=end_losses, pt_if=pt_if, theta_l=theta_l, length=length,
                                            ns_len=ns_len, p=p, vn=vn, sm=sm)
                pt_if = pt_if - elo if pt_if > elo else 0
            else:
                # Incident and Reflected sunlight as beam fluxes
                # Longitudinal effects that creates non-shaded sections of the concentrator length.
                ns_len_rec, ns_len_nei = non_shaded_lengths(p=p, ni=ni, vms=vms, sal=sal, sar=sar, length=length)

                # Calculating the angular intervals of receiver and neighbor shading effects.
                r_shading, n_shading = inc_beam_analysis(p=p, theta_t=theta_t, vi=vi, ni=ni, sal=sal, sar=sar, vms=vms)

                # The first mirror will never be shaded (receiver or neighbor).
                if (theta_t > 0 and i == 0) or (theta_t < 0 and i == n_hel - 1):
                    n_shading, ns_len_nei = closed(0, 0), 1

                # Computation to force a correct account for neighbor shading
                # of the central heliostat at normal incidence
                if theta_t == 0 and n_hel % 2 != 0 and i == int((n_hel - 1) / 2):
                    n_shading, ns_len_nei = closed(0, 0), 1

                # Calculating limiting vectors for a flat receiver.
                vll, vlr = flat_receiver_limiting_vectors(p=p, sal=sal, sar=sar, nr=nr)
                # Calculating the angular intervals related with the reflected beam.
                blocking, intercepted = ref_beam_analysis(p=p, vn=vn, nr=nr, vmb=vmb, vll=vll, vlr=vlr)

                # The central heliostat, right below the receiver is never blocked by its neighbors
                # on the other hand, it can be shaded by the receiver and its neighbors
                if centers[i][0].round(5) == 0:
                    blocking = closed(0, 0)

                ########################################################################################################
                # A non-shaded and non-blocked segments analysis #######################################################
                l1 = 1 - ns_len_rec  # fraction of segment surface subjected to all losses
                l2 = ns_len_rec - ns_len_nei  # fraction of the segment not subjected to receiver shading
                l3 = 1 - l1 - l2  # fraction of the segment not subjected to shading -- receiver or neighbor

                # Calculating the angular intervals for each one of the non-shaded and non-blocked segments.
                n_sha1, n_blo1, n_blo2, i1, i2, i3 = segments_analysis(r_shading=r_shading, n_shading=n_shading,
                                                                       blocking=blocking, intercepted=intercepted)
                # Intercepted flux by the receiver
                I1, I2, I3 = Flux(interval=i1, g=cum_eff), Flux(interval=i2, g=cum_eff), Flux(interval=i3, g=cum_eff)
                pt_if = l1 * I1 + l2 * I2 + l3 * I3

                # Shading losses #######################################################################################
                #  for the first interval, where all losses occur -- sums receiver and neighbor shading.
                SH1 = Flux(interval=r_shading, g=cum_eff) + Flux(interval=n_sha1, g=cum_eff)
                # Shading losses for the second interval, where it does not have receiver shading.
                SH2 = Flux(interval=n_shading, g=cum_eff)
                # Final computation of shading losses -- the composition of angular intervals and sections lengths.
                sha = l1 * SH1 + l2 * SH2

                # Blocking losses
                B1, B2, B3 = Flux(interval=n_blo1, g=cum_eff), Flux(interval=n_blo2, g=cum_eff), Flux(interval=blocking,
                                                                                                      g=cum_eff)
                blo = l1 * B1 + l2 * B2 + l3 * B3

                # End-losses calculations.
                # Calculation of the lost length due to end-effect
                lost_length = flux_analysis_lost_length(end_losses=end_losses, theta_l=theta_l, pt_if=pt_if,
                                                        vll=vll, vlr=vlr, length=length)
                elo, elo_b = segments_end_losses(lost_length=lost_length, l1=l1, l2=l2,
                                                 Int=[I1, I2, I3], Blo=[B1, B2, B3])

                # update of the point intercept factor and blocking losses -- discount of end-losses
                if elo > 0:
                    pt_if = pt_if - elo if pt_if > elo else 0  # discount the end-losses from the intercepted flux
                    blo = blo - elo_b if blo > elo_b else 0  # discount the end-losses from blocking losses
                else:
                    pt_if = pt_if

                de = 1 - (pt_if + sha + blo + elo)
            ###########################################################################################################
            hel_if[j] = pt_if
            hel_s[j] = sha
            hel_b[j] = blo
            hel_d[j] = de
            hel_end[j] = elo

        # Analytical expression for the cosine effect --- derived from a vector analysis from the definition.
        # It considers mirror's center point due to its central symmetry, instead of account for each point.
        num = cos(theta_l_rad) * cos(theta_t_rad - tau[i])
        den = cos(theta_t_rad) * sqrt(1 + cos(theta_l_rad) ** 2 * tan(theta_t_rad) ** 2)
        cosine_effect = num / den

        # Updates the values of each loss and intercept factor for the mirror, an average of all its points
        gamma[i] = hel_if.mean() * cosine_effect  # includes cosine effect in heliostat's intercept factor
        elf_s[i] = hel_s.mean()  # heliostat's energy loss factor due to shading
        elf_b[i] = hel_b.mean()  # heliostat's energy loss factor due to blocking
        elf_d[i] = hel_d.mean()  # heliostat's energy loss factor due to defocusing
        elf_c[i] = 1 - cosine_effect  # heliostat's energy loss factor due to cosine effect
        elf_e[i] = hel_end.mean()  # heliostat's intercept factor for end losses

    # calculate the sum of mirrors width -- the weights used to compute averages for the whole LFR.
    total_widths = widths.sum()

    # Parameter values for the whole concentrator are weighted averages of its mirrors
    lfr_if = widths.dot(gamma) / total_widths  # LFR intercept factor
    shading_losses = widths.dot(elf_s) / total_widths  # shading losses
    blocking_losses = widths.dot(elf_b) / total_widths  # blocking losses
    defocusing_losses = widths.dot(elf_d) / total_widths  # defocusing losses
    cosine_losses = widths.dot(elf_c) / total_widths  # cosine effect losses
    end_loss = widths.dot(elf_e) / total_widths  # LFR end losses

    return lfr_if, shading_losses, blocking_losses, defocusing_losses, cosine_losses, end_loss


def acceptance_flux_computation(rotated_field: array, rotated_normals: array, centers: array, widths: array,
                                theta_t: float, vi: array, ni: array, sar: array, sal: array, cum_eff):
    n_hel = len(rotated_field)
    total_widths = widths.sum()
    gamma = zeros(n_hel)

    for i, hel in enumerate(rotated_field):
        # number of points in the discretized heliostat.
        n_pts = hel.shape[0]

        # creates the arrays that will hold values of losses and intercept factor for each point of the mirror
        hel_if = zeros(n_pts)  # intercept factor

        # calculate the neighbors and neighbors edge points.
        neighbor_b, edge_pt_b, neighbor_s, edge_pt_s = define_neighbors(theta_t=theta_t, i=i, centers=centers,
                                                                        rotated_field=rotated_field)

        # calculations for all points of the heliostat
        for j, p in enumerate(hel):
            # Reflection directions
            ns = rotated_normals[i][j]  # the normal vector to mirror's surface at point 'p'
            vn = reft(vi, ns)  # direction of the reflected ray on that point
            nr = reft(ni, ns)  # normal vector that defines the reflection plane

            # the neighbor vectors to compute shading and blocking
            vms, vmb = define_neighbor_vectors(p=p, edge_pt_b=edge_pt_b, edge_pt_s=edge_pt_s, ni=ni, nr=nr)

            # efficiency calculations for collimated rays or flux analyses
            if cum_eff == 'collimated':
                pt_if, _, _, _, _, _ = collimated_rays_analysis(theta_t=theta_t, i=i, n=n_hel, p=p, vi=vi, ni=ni,
                                                                vn=vn, nr=nr, vms=vms, vmb=vmb, sal=sal, sar=sar,
                                                                length=60000)
            else:
                # calculating the angular intervals for receiver and neighbor shading losses
                r_shading, n_shading = inc_beam_analysis(p=p, theta_t=theta_t, vi=vi, ni=ni,
                                                         sal=sal, sar=sar, vms=vms)

                # first and last mirrors are never shaded if 'theta_t' > 0 and 'theta_t' < 0, respectively.
                if (theta_t > 0 and i == 0) or (theta_t < 0 and i == n_hel - 1):
                    n_shading = closed(0, 0)

                # a forced code to correctly compute the central heliostat.
                if theta_t == 0 and n_hel % 2 != 0 and i == int((n_hel - 1) / 2):
                    n_shading = closed(0, 0)

                # calculating the angular intervals for blocking losses and intercepted flux
                vll, vlr = flat_receiver_limiting_vectors(p=p, sal=sal, sar=sar, nr=nr)
                blocking, intercepted = ref_beam_analysis(p=p, vn=vn, nr=nr, vmb=vmb, vll=vll, vlr=vlr)

                # the central heliostat, right below the receiver is never blocked by its neighbors
                # on the other hand, it can be shaded by the receiver and its neighbors
                if centers[i][0].round(5) == 0:
                    blocking = closed(0, 0)

                _, _, intercept_interval = one_segment_analysis(r_shading=r_shading, n_shading=n_shading,
                                                                blocking=blocking, intercepted=intercepted)
                # intercepted flux
                pt_if = Flux(interval=intercept_interval, g=cum_eff)

            hel_if[j] = pt_if * cos(ang(vi, ns))  # accounting for cosine losses

        gamma[i] = hel_if.mean()  # the efficiency of the heliostat is an average of all its points

    # the concentrator efficiency is a weighted average of all its heliostats
    lfr_if = gamma.dot(widths) / total_widths

    return lfr_if


########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# An attempt to use the numpy.vectorize function to speed up the code ##################################################
#
#
# def point_flux_calculations(j: int, i: int,
#                             rotated_field: array, rotated_normals: array, centers: array,
#                             sm: array, sar: array, sal: array,
#                             theta_t: float, theta_l: float, vi: array, ni: array,
#                             length: float, cum_eff, end_losses):
#     # The number of heliostats in the primary field
#     n_hel = centers.shape[0]
#
#     p = rotated_field[i][j]  # a point at the heliostat surface
#     ns = rotated_normals[i][j]  # the normal vector to mirror's surface at point 'p'
#
#     # Reflected ray at the heliostat surface and normal vector to the reflection plane
#     vn = reft(vi, ns)  # direction of the reflected ray on that point
#     nr = reft(ni, ns)  # normal vector that defines the reflection plane
#
#     # It determines the neighbor mirrors to compute shading and blocking losses.
#     neighbor_b, edge_pt_b, neighbor_s, edge_pt_s = define_neighbors(theta_t=theta_t, i=i, centers=centers,
#                                                                     rotated_field=rotated_field)
#
#     # It calculates the vectors to calculate the shading and blocking from neighbor
#     vms, vmb = define_neighbor_vectors(p=p, edge_pt_b=edge_pt_b, edge_pt_s=edge_pt_s, ni=ni, nr=nr)
#
#     if cum_eff == 'collimated':
#         pt_if, sha, blo, de, acc, ns_len = collimated_rays_analysis(theta_t=theta_t, i=i, n=n_hel, p=p,
#                                                                     vi=vi, ni=ni, vn=vn, nr=nr,
#                                                                     vms=vms, vmb=vmb, sal=sal,
#                                                                     sar=sar, length=length)
#         # End-losses calculations for collimated sunlight model
#         elo = collimated_end_losses(end_losses=end_losses, pt_if=pt_if, theta_l=theta_l, length=length,
#                                     ns_len=ns_len, p=p, vn=vn, sm=sm)
#
#         pt_if = pt_if - elo if pt_if > elo else 0
#         de += acc
#     else:
#         # Incident and Reflected sunlight as beam fluxes
#         # Longitudinal effects that creates non-shaded sections of the concentrator length.
#         ns_len_rec, ns_len_nei = non_shaded_lengths(p=p, ni=ni, vms=vms, sal=sal, sar=sar, length=length)
#
#         # Calculating the angular intervals of receiver and neighbor shading effects.
#         r_shading, n_shading = inc_beam_analysis(p=p, theta_t=theta_t, vi=vi, ni=ni, sal=sal, sar=sar, vms=vms)
#
#         # The first mirror will never be shaded (receiver or neighbor).
#         if (theta_t > 0 and i == 0) or (theta_t < 0 and i == n_hel - 1):
#             n_shading, ns_len_nei = closed(0, 0), 1
#
#         # Computation to force a correct account for neighbor shading
#         # of the central heliostat at normal incidence
#         if theta_t == 0 and n_hel % 2 != 0 and i == int((n_hel - 1) / 2):
#             n_shading, ns_len_nei = closed(0, 0), 1
#
#         # Calculating limiting vectors for a flat receiver.
#         vll, vlr = flat_receiver_limiting_vectors(p=p, sal=sal, sar=sar, nr=nr)
#         # Calculating the angular intervals related with the reflected beam.
#         blocking, intercepted = ref_beam_analysis(p=p, vn=vn, nr=nr, vmb=vmb, vll=vll, vlr=vlr)
#
#         # The central heliostat, right below the receiver is never blocked by its neighbors
#         # on the other hand, it can be shaded by the receiver and its neighbors
#         if centers[i][0].round(5) == 0:
#             blocking = closed(0, 0)
#
#         ########################################################################################################
#         # A non-shaded and non-blocked segments analysis #######################################################
#         l1 = 1 - ns_len_rec  # fraction of segment surface subjected to all losses
#         l2 = ns_len_rec - ns_len_nei  # fraction of the segment not subjected to receiver shading
#         l3 = 1 - l1 - l2  # fraction of the segment not subjected to shading -- receiver or neighbor
#
#         # Calculating the angular intervals for each one of the non-shaded and non-blocked segments.
#         n_sha1, n_blo1, n_blo2, i1, i2, i3 = segments_analysis(r_shading=r_shading, n_shading=n_shading,
#                                                                blocking=blocking, intercepted=intercepted)
#         # Intercepted flux by the receiver
#         I1, I2, I3 = Flux(interval=i1, g=cum_eff), Flux(interval=i2, g=cum_eff), Flux(interval=i3, g=cum_eff)
#         pt_if = l1 * I1 + l2 * I2 + l3 * I3
#
#         # Shading losses
#         #  for the first interval, where all losses occur -- sums receiver and neighbor shading.
#         SH1 = Flux(interval=r_shading, g=cum_eff) + Flux(interval=n_sha1, g=cum_eff)
#         # Shading losses for the second interval, where it does not have receiver shading.
#         SH2 = Flux(interval=n_shading, g=cum_eff)
#         # Final computation of shading losses -- the composition of angular intervals and sections lengths.
#         sha = l1 * SH1 + l2 * SH2
#
#         # Blocking losses
#         B1, B2, B3 = Flux(interval=n_blo1, g=cum_eff), Flux(interval=n_blo2, g=cum_eff), Flux(interval=blocking,
#                                                                                               g=cum_eff)
#         blo = l1 * B1 + l2 * B2 + l3 * B3
#
#         # End-losses calculations.
#         # Calculation of the lost length due to end-effect
#         lost_length = flux_analysis_lost_length(end_losses=end_losses, theta_l=theta_l, pt_if=pt_if,
#                                                 vll=vll, vlr=vlr, length=length)
#         elo, elo_b = segments_end_losses(lost_length=lost_length, l1=l1, l2=l2,
#                                          Int=[I1, I2, I3], Blo=[B1, B2, B3])
#
#         # update of the point intercept factor and blocking losses -- discount of end-losses
#         if elo > 0:
#             pt_if = pt_if - elo if pt_if > elo else 0  # discount the end-losses from the intercepted flux
#             blo = blo - elo_b if blo > elo_b else 0  # discount the end-losses from blocking losses
#         else:
#             pt_if = pt_if
#
#         de = 1 - (pt_if + sha + blo + elo)
#
#     return pt_if, sha, blo, de, elo
#
#
# def vectorized_flux_computation():
#     return vectorize(point_flux_calculations, excluded=['i', 'rotated_field', 'rotated_normals', 'centers',
#                                                         'sm', 'sar', 'sal', 'theta_t', 'theta_l', 'vi', 'ni',
#                                                         'length', 'cum_eff', 'end_losses'])
#
#
# def hel_flux_calculation(i: int, rotated_field: array, rotated_normals: array, centers: array,
#                          sm: array, sar: array, sal: array,
#                          theta_t: float, theta_l: float, vi: array, ni: array,
#                          length: float, cum_eff, end_losses):
#     hel_pts = rotated_field[i].shape[0]
#
#     points_list = [x for x in range(hel_pts)]
#
#     f = vectorized_flux_computation()
#
#     results = f(j=points_list, i=i,
#                 rotated_field=rotated_field, rotated_normals=rotated_normals, centers=centers,
#                 sm=sm, sar=sar, sal=sal,
#                 theta_t=theta_t, theta_l=theta_l,
#                 vi=vi, ni=ni,
#                 length=length, cum_eff=cum_eff, end_losses=end_losses)
#
#     return results


########################################################################################################################
########################################################################################################################

########################################################################################################################
# Secondary optics functions ###########################################################################################


def primary_field_edge_points(primaries: array):
    f1 = primaries[0][-1] if primaries[0][-1][0] > primaries[0][-1][-1] else primaries[0][0]  # right side edge
    f2 = primaries[-1][-1] if primaries[-1][-1][0] < primaries[-1][0][0] else primaries[-1][0]  # left side edge

    return f1, f2


def primary_field_edge_mirrors_centers(primaries: array):
    n_pts = len(primaries[0])
    k = int((n_pts - 1) // 2)

    f1 = primaries[0][k] if primaries[0][k][0] > primaries[-1][k][0] else primaries[-1][k]
    f2 = primaries[-1][k] if primaries[-1][k][0] < primaries[0][k][0] else primaries[0][k]

    return f1, f2


def tgs2tube(point: array, tube_center: array, tube_radius: float):
    """

    :param point: A point in the xy plane
    :param tube_center: Tube center point
    :param tube_radius: Tube radius

    :return: It returns the two points in the tube surface which tangent lines passes through the outer point.
    """
    beta = arccos(tube_radius / dst(p=point, q=tube_center))

    p1 = tube_center + R(beta).dot(nrm(point - tube_center)) * tube_radius
    p2 = tube_center + R(-beta).dot(nrm(point - tube_center)) * tube_radius

    return p1, p2


# Zhu's adaptative secondary optic design ##############################################################################

"""
The following functions comprises implementations of Zhu's adaptative method to design an LFC secondary optic.
In this sense, it contains a main function and other auxiliary functions.
"""


def zhu_adaptative_secondary(primaries: array, centers: array, widths: array,
                             tube_center: array, tube_radius: float,
                             delta_h: float, source_rms_width: float, ds=0.1,
                             flat_mirrors=False, central_incidence=False):
    # ToDo: Finish the implementation of Zhu's adaptative design of a secondary optic for an LFC with a tubular absorber
    """

    This function is based on the Zhu's [1] adaptative method to design a linear Fresnel secondary optic for a tubular
    absorber.


    [1] Zhu G. New adaptive method to optimize the secondary reflector of linear Fresnel collectors.
    Solar Energy 2017;144:117–26. https://doi.org/10.1016/j.solener.2017.01.005.
    """

    aim = array([tube_center[0], tube_center[-1]])

    ####################################################################################################################
    # The starting point calculations ##################################################################################

    # Secondary aperture ########################################
    # the angular aperture which comprises 95% of total power of the effective source beam.
    beta = 4 * source_rms_width  # the 95% criterion

    # Data of the farthest mirror
    cn = centers[0] if centers[0][0] > centers[-1][0] else centers[-1]  # center
    ln = dst(cn, aim)  # distance between the center and the aim point
    wn = widths[0] if centers[0][0] > centers[-1][0] else widths[-1]  # width
    tau_n = arctan(cn[0] / aim[-1])  # tracking angle for a normal incidence

    # Secondary optic aperture. It considers flat and bent mirrors (parabolical or cylindrical)
    a = ln * sin(beta) + wn * cos(tau_n) if flat_mirrors else ln * sin(beta)
    ###############################################################

    # The starting point ###########################################################
    # Zhu's criterion for the starting point. It is one of the edges of the aperture.
    # Here, it is considered the edged on the left side of the absorber tube
    p = aim - array([a / 2, delta_h])
    # The starting point ###########################################################

    # A list to hold the calculated points for the secondary optic surface
    secondary_pts = [p]
    ####################################################################################################################

    ####################################################################################################################
    # The adaptative calculations for the secondary optic surface points ###############################################
    while secondary_pts[-1][0] < aim[0]:
        # Principal direction calculation
        # That are two options: a simple approach with the central incidence, and a more complicated one.
        principal_direction = principal_incidence(primaries=primaries, tube_center=tube_center,
                                                  tube_radius=tube_radius, point=p, central_incidence=central_incidence)
        principal_direction = nrm(principal_direction)
        target_direction = nrm(aim - p)

        # Normal and tangent direction at the point
        # It considers the incident ray as the principal incidence, and the reflected one as the target direction
        normal_v = principal_direction + target_direction
        # In this case, both vector are from the point in the secondary's surface to the points in the primary field,
        # and aim at the absorber, respectively.
        # Therefore, by such an equation the normal vector at the point also points out of it, i.e, towards the field.

        # The tangent vector at the surface point.
        tangent_v = R(0.5 * pi).dot(normal_v)

        # Calculating the new point in the secondary surface and append it to the list of points.

        # # Zhu's proposition for a better surface approximation ############################################
        # p2 = p + ds * nrm(tangent_v)
        # pd2 = principal_incidence(primaries=primaries, tube_center=tube_center,
        #                           tube_radius=tube_radius, point=p, central_incidence=central_incidence)
        # pd2 = nrm(pd2)
        # target_direction2 = nrm(aim - p2)
        # normal2 = pd2 + target_direction2
        #
        # theta = ang(normal_v, normal2).rad
        # tangent_v = R(0.5 * pi + 0.5*theta).dot(normal_v)
        #####################################################################################################

        # The next point in the secondary's surface
        p = p + ds * nrm(tangent_v)

        secondary_pts.append(p)

    ####################################################################################################################

    return array(secondary_pts)


def principal_incidence(primaries: array, tube_center: array, tube_radius: float, point: array,
                        central_incidence=False):
    # A unit vector which defines the x-axis.
    Ix = array([1, 0])

    ####################################################################################################################
    # The angular view in which the point sees the receiver ############################################################

    # limiting vector -- from the point to the absorber surface
    tg1, tg2 = tgs2tube(point=point, tube_center=tube_center, tube_radius=tube_radius)
    vll, vlr = tg1 - point, tg2 - point

    # The angles in which the limiting vectors regarding the x-axis.
    # A counterclockwise direction means a positive angle. A clockwise one means a negative values.
    # With this definition, angles are within -pi to +pi
    theta_1, theta_2 = ang_pn(v=vll, u=Ix).rad, ang_pn(v=vlr, u=Ix).rad

    # The angular interval defined by the above two angles.
    # It is the finite interval from the range between -pi to +pi in which the point in the secondary's surface
    # sees the tubular absorber
    blockage_interval = closed(min(theta_1, theta_2), max(theta_1, theta_2))

    ####################################################################################################################

    ####################################################################################################################
    # The angular view in which the point sees the primary field #######################################################

    # edges points of the primary field
    # the first one is the right edge, the other one is the left edge.
    # f1, f2 = primary_field_edge_points(primaries)
    f1, f2 = primary_field_edge_mirrors_centers(primaries)

    # incidence vectors to the edge. They define the range of incidences in the point on the secondary surface
    v1 = nrm(f1 - point)
    v2 = nrm(f2 - point)

    # Calculating the angular view in which the point sees the primary field
    theta_3, theta_4 = ang_pn(v=v1, u=Ix).rad, ang_pn(v=v2, u=Ix).rad
    incidence_interval = closed(min(theta_3, theta_4), max(theta_3, theta_4))
    ####################################################################################################################

    ####################################################################################################################
    # Accounting for the absorber blockage #############################################################################
    net_interval = reduce_interval(a=incidence_interval, b=blockage_interval)

    # The angular interval defined by the receiver acceptance is the blockage interval.
    # If it lies within the range of the angular interval in which the point sees the primary field, the 'net_interval'
    # is divided in two intervals.

    # Calculating the principal incidence ##############################################################################
    # If the 'net_interval' has only one interval, only it must be used to compute the principal incidence direction-
    if len(net_interval) == 1:
        incidence_interval = closed(net_interval.lower, net_interval.upper)
    # If two intervals are produced by the blockage calculations, then We only consider one.
    # The problem is: which one?
    elif len(net_interval) == 2:
        # One proposition ########################################################################
        # A point in the left side of the secondary's surface will just considers the incidence range from the left.
        # This is suggested by Zhu's Fig. 2 and Fig. 3.
        if point[0] < tube_center[0]:
            incidence_interval = closed(net_interval[0].lower, net_interval[0].upper)
        else:
            incidence_interval = closed(net_interval[-1].lower, net_interval[-1].upper)
        ##########################################################################################

        # Other proposition ######################################################################
        # It does not mater which side the point is. It only accounts for the farthest interval from the point.
        # That is, the one which has the highest deviation from the position -pi/2
        # if net_interval[0].lower + pi/2 > net_interval[1].upper + pi/2:
        #     incidence_interval = closed(net_interval[0].lower, net_interval[0].upper)
        # else:
        #     incidence_interval = closed(net_interval[1].lower, net_interval[1].upper)
        # Results of this proposition does not seem right, nonetheless.
        ###########################################################################################
    # If more than two intervals are generated, it raises an error.
    else:
        print(point)
        raise ValueError('More than two intervals were produced by the receiver blockage calculations.')
    ####################################################################################################################

    ####################################################################################################################
    # Determining the principal incidence ##############################################################################

    # If the central incidence approach is considered. It is the most straightforward one since the intensity analysis
    # is neglected. Thus, the principal incidence is the bisector vector.
    if central_incidence:
        a = 0.5 * (incidence_interval.lower + incidence_interval.upper)
    # If an intensity analysis is considered. Then, the acceptance of the point is considered.
    else:
        point_acceptance = 2 * arcsin(tube_radius / dst(p=point, q=tube_center))
        # If the point is on the left side of the secondary's surface, the most distant edge is the left one.
        if point[0] < tube_center[0]:
            a = incidence_interval.upper - 0.5 * point_acceptance
        else:
            a = incidence_interval.lower + 0.5 * point_acceptance

        # if abs(incidence_interval.lower + pi/2) > abs(incidence_interval.upper + pi/2):
        #     most_distant_direction = incidence_interval.lower
        #     a = most_distant_direction + 0.5 * point_acceptance
        # else:
        #     most_distant_direction = incidence_interval.upper
        #     a = most_distant_direction - 0.5 * point_acceptance

    return V(a)


########################################################################################################################
########################################################################################################################

# CPC secondary for LFCs ###############################################################################################


def virtual_receiver_perimeter(tube_radius: float, cover_outer_radius: float):
    r = abs(tube_radius)
    rg = abs(cover_outer_radius)

    beta = arccos(r / rg)

    return 2 * r * (pi - beta + tan(beta))


def edges2tube(f1: array, f2: array, tube_center: array, tube_radius):
    tg1, tg2 = tgs2tube(point=f1, tube_center=tube_center, tube_radius=tube_radius)
    t1 = tg1 if tg1[0] < tube_center[0] else tg2

    tg3, tg4 = tgs2tube(point=f2, tube_center=tube_center, tube_radius=tube_radius)
    t2 = tg3 if tg3[0] > tube_center[0] else tg4

    return t1, t2


def hotel_strings(f1, f2, s1, s2):
    a = dst(f1, s1) - dst(f1, s2)
    b = dst(f2, s1) - dst(f2, s2)

    return abs(a) + abs(b)


def aperture_from_flow_line(f1, f2, e1, e2, flow_line, phi):
    p = flow_line(phi)
    s1 = isl(p=f1, v=p - f1, q=f2, u=e2)
    s2 = isl(p=f2, v=p - f2, q=f1, u=e1)

    return s1, s2


def cpc_aperture(f1: array, f2: array, tube_center: array, tube_radius):
    t1, t2 = edges2tube(f1=f1, f2=f2, tube_center=tube_center, tube_radius=tube_radius)
    e1, e2 = t1 - f1, t2 - f2

    A = isl(p=f1, v=e1, q=f2, u=e2)

    if dst(f1, tube_center) > dst(f2, tube_center):
        flow_line = hyp(f=f1, g=f2, p=A)
        phi_0 = ang_h(A - f1).rad - pi
    else:
        flow_line = hyp(f=f2, g=f1, p=A)
        phi_0 = ang_h(A - f2).rad

    def delta_etendue(phi):

        a1, a2 = aperture_from_flow_line(f1=f1, f2=f2, e1=e1, e2=e2, flow_line=flow_line, phi=phi)
        u = hotel_strings(f1=f1, f2=f2, s1=a1, s2=a2)

        return u - 4 * pi * tube_radius

    phi_a = fsolve(delta_etendue, x0=phi_0)[0] - pi
    s1, s2 = aperture_from_flow_line(f1=f1, f2=f2, e1=e1, e2=e2, flow_line=flow_line, phi=phi_a)

    return s1, s2


def cpc_secondary_tubular_absorber(primaries: array, tube_center: array, tube_radius: float, section_points=121):
    # edges of the primary field which will define the edge-rays of the secondary optic
    f1, f2 = primary_field_edge_points(primaries)
    # f1, f2 = primary_field_edge_mirrors_centers(primaries)

    # The tangent points in the absorber tube which define the edge rays from the primary field.
    # tg1, tg2 = tgs2tube(point=f1, tube_center=tube_center, tube_radius=tube_radius)
    # t1 = tg1 if tg1[1] < tg2[1] else tg2
    #
    # tg3, tg4 = tgs2tube(point=f2, tube_center=tube_center, tube_radius=tube_radius)
    # t2 = tg3 if tg3[1] < tg4[1] else tg4
    t1, t2 = edges2tube(f1=f1, f2=f2, tube_center=tube_center, tube_radius=tube_radius)

    # The starting point for the involute, that is, the intersection point between the optic and the absorber.
    A = tube_center + array([0, tube_radius])

    # The right-side involute
    i1 = winv(p=A, f=tube_center, r=tube_radius)
    phi_1 = ang_h(array([-1, 0])).rad
    phi_2 = ang_h(t2 - f2).rad
    r_inv = array([i1(x) for x in linspace(start=phi_1, stop=phi_2, num=section_points)])
    s3 = r_inv[-1]

    # The left-side involute
    i2 = uinv(p=A, f=tube_center, r=tube_radius)
    phi_1 = ang_h(array([1, 0])).rad
    phi_2 = ang_h(t1 - f1).rad
    l_inv = array([i2(x) for x in linspace(start=phi_1, stop=phi_2, num=section_points)])
    s4 = l_inv[-1]

    # The right-side macrofocal ellipse
    e1 = wme(f=tube_center, r=tube_radius, g=f2, p=s3)
    phi_1 = ang_p(s4 - f2, f2 - tube_center).rad
    phi_2 = ang_p(f1 - t1, f2 - tube_center).rad
    r_ell = array([e1(x) for x in linspace(start=phi_1, stop=phi_2, num=section_points)])

    # The left-side macrofocal ellipse
    e2 = ume(f=tube_center, r=tube_radius, g=f1, p=s4)
    phi_1 = ang_p(s3 - f1, f1 - tube_center).rad
    phi_2 = ang_p(f2 - t2, f1 - tube_center).rad
    l_ell = array([e2(x) for x in linspace(start=phi_1, stop=phi_2, num=section_points)])

    return l_ell, l_inv, r_inv, r_ell


def cpc_secondary_evacuated_tubular_absorber(primaries: array, tube_center: array, tube_radius: float,
                                             outer_radius: float, section_points=121):
    """
    This function considers the virtual receiver design as the solution for the gap losses [1].

    :param primaries:
    :param tube_center:
    :param tube_radius:
    :param outer_radius:
    :param section_points:

    :return:


    References:
    [1] Winston R. Ideal flux concentrators with reflector gaps. Applied Optics 1978;17:1668–9.
        https://doi.org/10.1364/AO.17.001668.

    """

    # edges of the primary field which will define the edge-rays of the secondary optic
    f1, f2 = primary_field_edge_points(primaries)
    # f1, f2 = primary_field_edge_mirrors_centers(primaries)

    # The tangent points in the absorber tube which define the edge rays from the primary field.
    tg1, tg2 = tgs2tube(point=f1, tube_center=tube_center, tube_radius=tube_radius)
    t1 = tg1 if tg1[1] < tg2[1] else tg2

    tg3, tg4 = tgs2tube(point=f2, tube_center=tube_center, tube_radius=tube_radius)
    t2 = tg3 if tg3[1] < tg4[1] else tg4

    # The cusp point of the CPC optic
    # The starting point for the involute.
    A = tube_center + array([0, outer_radius])

    p1, p2 = tgs2tube(point=A, tube_center=tube_center, tube_radius=tube_radius)

    if p1[0] < p2[0]:
        p1, p2 = p2, p1

    # The right-side involute
    i1 = winv(p=A, f=tube_center, r=tube_radius)
    phi_1 = ang_h(A - p1).rad
    phi_2 = ang_h(t2 - f2).rad
    r_inv = array([i1(x) for x in linspace(start=phi_1, stop=phi_2, num=section_points)])
    s3 = r_inv[-1]

    # The left-side involute
    i2 = uinv(p=A, f=tube_center, r=tube_radius)
    phi_1 = ang_h(A - p2).rad
    phi_2 = ang_h(t1 - f1).rad
    l_inv = array([i2(x) for x in linspace(start=phi_1, stop=phi_2, num=section_points)])
    s4 = l_inv[-1]

    # The right-side macrofocal ellipse
    e1 = wme(f=tube_center, r=tube_radius, g=f2, p=s3)
    phi_1 = ang_p(s4 - f2, f2 - tube_center).rad
    phi_2 = ang_p(f1 - t1, f2 - tube_center).rad
    r_ell = array([e1(x) for x in linspace(start=phi_1, stop=phi_2, num=section_points)])

    # The left-side macrofocal ellipse
    e2 = ume(f=tube_center, r=tube_radius, g=f1, p=s4)
    phi_1 = ang_p(s3 - f1, f1 - tube_center).rad
    phi_2 = ang_p(f2 - t2, f1 - tube_center).rad
    l_ell = array([e2(x) for x in linspace(start=phi_1, stop=phi_2, num=section_points)])

    return l_ell, l_inv, r_inv, r_ell


########################################################################################################################
########################################################################################################################

########################################################################################################################
# Soltrace functions ###################################################################################################


def heliostat2soltrace(hel: Heliostat, name: str, length: float, aim_pt: array, sun_dir: array,
                       optic: OpticalSurface, par_approx=True, file_path=None):

    center = hel.ZX_center / 1000
    aim_vec = hel.aim_vector(aim_point=aim_pt, SunDir=sun_dir) / 1000
    L = length / 1000  # convert to meters
    aperture = list([0] * 9)
    surface = list([0] * 9)

    if hel.shape == 'flat':
        aperture[0:3] = 'r', hel.width / 1000, L
        surface[0] = 'f'
    else:
        if par_approx:
            aperture[0:3] = 'r', hel.width / 1000, L
            rc = hel.radius / 1000  # convert to meters
            # 'c' is the parabola's gradient, as defined in SolTrace
            c = 1 / rc
            surface[0:2] = 'p', c
        else:
            if file_path is None:
                raise "The .csi filepath for the surface must be inputted"
            else:
                curve = hel.as_plane_curve()
                x1 = round(curve.x_t[0] / 1000, 6) + 0.0000001
                x2 = round(curve.x_t[-1] / 1000, 6) - 0.0000001
                aperture[0:4] = 'l', x1, x2, L

                full_file_path = curve.spline2soltrace(file_path=file_path, file_name=name)
                surface[0:2] = 'i', str(full_file_path)
                surface = surface[0:2]

    elem = Element(name=name, orig=center, aim_vec=aim_vec, z_rot=0,
                   aperture=aperture, surface=surface, optic=optic, reflect=True)

    return elem


def flat_absorber2soltrace(geometry: Absorber.flat, name: str, optic: OpticalSurface, length: float) -> list:

    # Converting units to meters ###
    w = geometry.width / 1000
    L = length / 1000
    hc = array([geometry.center[0], 0, geometry.center[-1]]) / 1000
    ################################

    # Setting the aperture #########
    aperture = list([0] * 9)
    aperture[0:3] = 'r', w, L
    ################################

    # Setting the surface ##########
    surface = list([0] * 9)
    surface[0] = 'f'
    ################################

    elem = Element(name=name, orig=hc, aim_vec=array([0, 0, 1]), z_rot=0,
                   aperture=aperture, surface=surface,
                   optic=optic, reflect=True)

    return [elem]


def tube2soltrace(name: str, radius: float, center: array, aim: array, optic: OpticalSurface, length: float) -> list:

    # Converting the units to meters
    r = radius / 1000
    L = length / 1000
    hc = center / 1000
    v_aim = aim / 1000
    ################################

    # Setting the aperture ###########
    aperture = list([0] * 9)
    aperture[0], aperture[3] = 'l', L
    ##################################

    # Setting the surface ############
    surface = list([0] * 9)
    surface[0], surface[1] = 't', 1/r
    ##################################

    elem = Element(name=name, orig=hc, aim_vec=v_aim, z_rot=0,
                   aperture=aperture, surface=surface,
                   optic=optic, reflect=True)

    return [elem]


def trapezoidal2soltrace(geometry: Secondary.trapezoidal, name: str, length: float, optic: OpticalSurface) -> list:

    Iy = array([0, 1, 0])

    w_r = dst(p=geometry.back_right, q=geometry.ap_right) / 1000
    w_l = dst(p=geometry.back_left, q=geometry.ap_left) / 1000
    w_b = dst(p=geometry.back_left, q=geometry.back_right) / 1000
    L = length / 1000

    # Setting the aperture #########
    aperture_r, aperture_l, aperture_b = list([0] * 9), list([0] * 9), list([0] * 9)
    aperture_r[0:3] = 'r', w_r, L
    aperture_l[0:3] = 'r', w_l, L
    aperture_b[0:3] = 'r', w_b, L
    ################################

    # Setting the surface ##########
    surface = list([0] * 9)
    surface[0] = 'f'
    ################################

    r_hc = transform_vector(mid_point(p=geometry.back_right, q=geometry.ap_right)) / 1000
    r_ab = transform_vector(geometry.ap_right - geometry.back_right) / 1000
    r_aim = R(pi/2, Iy).dot(r_ab) + r_hc

    l_hc = transform_vector(mid_point(p=geometry.back_left, q=geometry.ap_left)) / 1000
    l_ba = transform_vector(geometry.back_left - geometry.ap_left) / 1000
    l_aim = R(pi/2, Iy).dot(l_ba) + l_hc

    b_hc = transform_vector(mid_point(p=geometry.back_left, q=geometry.back_right)) / 1000
    b_bb = transform_vector(geometry.back_right - geometry.back_left) / 1000
    b_aim = R(pi/2, Iy).dot(b_bb) + b_hc

    right_side_element = Element(name=f"{name}_right", orig=r_hc, aim_vec=r_aim, z_rot=0,
                                 aperture=aperture_r, surface=surface, optic=optic, reflect=True, enable=True)

    left_side_element = Element(name=f"{name}_left", orig=l_hc, aim_vec=l_aim, z_rot=0,
                                aperture=aperture_l, surface=surface, optic=optic, reflect=True, enable=True)

    back_side_element = Element(name=f"{name}_back", orig=b_hc, aim_vec=b_aim, z_rot=0,
                                aperture=aperture_b, surface=surface, optic=optic, reflect=True, enable=True)

    return [right_side_element, back_side_element, left_side_element]

########################################################################################################################
########################################################################################################################

########################################################################################################################
# LFR construction class and functions #################################################################################


class uniform_lfr_geometry:

    def __init__(self, name: str, absorber_height: float, absorber_width: float,
                 mirror_width: float, nbr_mirrors=int, total_width: float = None, center_distance: float = None):

        self.name = name

        self.rec_height = abs(absorber_height)
        self.rec_width = abs(absorber_width)
        self.receiver = Absorber.flat(width=abs(absorber_width),
                                      center=array([0, 0, abs(absorber_height)]))

        self.mirror_width = abs(mirror_width)
        self.nbr_mirrors = abs(nbr_mirrors)
        self.widths = ones(self.nbr_mirrors) * self.mirror_width

        if total_width is not None and center_distance is None:

            self.total_width = abs(total_width)
            self.center_distance = (self.total_width - self.mirror_width) / (self.nbr_mirrors - 1)
            self.centers = uniform_centers(total_width=self.total_width,
                                           mirror_width=self.mirror_width,
                                           number_mirrors=self.nbr_mirrors)

        elif center_distance is not None and total_width is None:

            self.center_distance = abs(center_distance)
            self.total_width = self.center_distance * (self.nbr_mirrors - 1) + self.mirror_width
            self.centers = uniform_centers(total_width=self.total_width,
                                           mirror_width=self.mirror_width,
                                           number_mirrors=self.nbr_mirrors)

        elif center_distance is not None and total_width is not None:

            if center_distance == (total_width - self.mirror_width) / (self.nbr_mirrors - 1):

                self.center_distance = abs(center_distance)
                self.total_width = abs(total_width)
                self.centers = uniform_centers(total_width=self.total_width,
                                               mirror_width=self.mirror_width,
                                               number_mirrors=self.nbr_mirrors)
            else:
                raise ValueError('Class argument error: Values do not make sense')
        else:
            raise ValueError('Class argument error: Please add a total_width or center_distance')

        self.filling_factor = self.widths.sum() / self.total_width
        self.rim_angle = arctan(0.5 * self.total_width / self.rec_height) * 180 / pi

    def export_geometry(self, file_path):

        file_full_path = Path(file_path, f'{self.name}_geometry.json')

        d = {'name': self.name, 'width': self.mirror_width, 'nbr_mirrors': self.nbr_mirrors,
             'total_width': self.total_width,  'center_distance': self.center_distance,
             'receiver_height': self.rec_height, 'receiver_width': self.rec_width,
             'units': 'millimeters and degrees'}

        with open(file_full_path, 'w') as file:
            json.dump(d, file)

        return file_full_path


def rabl_design_lfr(lfr_geometry, design_position, nbr_pts=121):

    n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts

    centers = lfr_geometry.centers
    widths = lfr_geometry.widths

    sm = transform_vector(mid_point(lfr_geometry.receiver.s1, lfr_geometry.receiver.s2))

    heliostats = []

    for w, hc in zip(widths, centers):

        if isinstance(design_position, Angle):
            rr = rabl_curvature(center=hc,
                                aim=sm,
                                theta_d=design_position)
        elif design_position == 'SR':
            rr = 2 * dst(hc, sm)

        else:
            raise ValueError("Design position must be an Angle or 'SR'")

        hel = Heliostat(center=hc,
                        width=w,
                        radius=rr,
                        nbr_pts=n_pts)

        heliostats.append(hel)

    primary_field = PrimaryField(heliostats=heliostats)

    lfr_concentrator = LFR(primary_field=primary_field,
                           flat_absorber=lfr_geometry.receiver)

    return lfr_concentrator


def boito_design_lfr(lfr_geometry, latitude: Angle, nbr_pts=121):
    n_pts = nbr_pts + 1 if nbr_pts % 2 == 0 else nbr_pts

    centers = lfr_geometry.centers
    widths = lfr_geometry.widths
    sm = mid_point(lfr_geometry.receiver.s1, lfr_geometry.receiver.s2)

    heliostats = []

    for w, hc in zip(widths, centers):
        rr = boito_curvature(center=hc,
                             aim=sm,
                             lat=latitude)

        hel = Heliostat(center=hc,
                        width=w,
                        radius=rr,
                        nbr_pts=n_pts)

        heliostats.append(hel)

    primary_field = PrimaryField(heliostats=heliostats)

    lfr_concentrator = LFR(primary_field=primary_field, flat_absorber=lfr_geometry.receiver)

    return lfr_concentrator


def uniform_lfr_flat_receiver(name: str, absorber_height: float, absorber_width: float,
                              mirror_width: float, nbr_mirrors=int, total_width: float = None,
                              center_distance: float = None):

    return uniform_lfr_geometry(name=name, absorber_height=absorber_height, absorber_width=absorber_width,
                                mirror_width=mirror_width, nbr_mirrors=nbr_mirrors,
                                total_width=total_width, center_distance=center_distance)


def design_lfr(lfr_geometry: uniform_lfr_geometry, curvature_design='sun_ref', nbr_pts=121):

    if curvature_design == 'sun_ref':
        lfr = rabl_design_lfr(lfr_geometry=lfr_geometry, design_position=Angle(deg=0), nbr_pts=nbr_pts)
    else:
        lfr = rabl_design_lfr(lfr_geometry=lfr_geometry, design_position='SR', nbr_pts=nbr_pts)

    return lfr


def lfr_analytical_optical_efficiency(lfr: LFR, theta_t: float, theta_l: float, aim: array, length: float,
                                      sun_shape: RadialSunshape,
                                      primaries_property: OpticalProperty.reflector,
                                      absorber_property: OpticalProperty.flat_absorber,
                                      end_losses='no'):

    rho_p = primaries_property.rho
    alpha_a = absorber_property.alpha

    cum_eff = sun_shape.linear_effective_source(specular_error=primaries_property.spec_error,
                                                slope_error=primaries_property.slope_error)

    gamma = lfr.intercept_factor(theta_t=theta_t, theta_l=theta_l, aim=aim,
                                 length=length, cum_eff=cum_eff, end_losses=end_losses)

    optical_efficiency = rho_p * alpha_a * gamma

    return optical_efficiency


def lfr_raytracing_optical_efficiency(lfr: LFR, theta_t: float, theta_l: float, aim: array, length: float,
                                      sun_shape: RadialSunshape,
                                      primaries_property: OpticalProperty.reflector,
                                      absorber_property: OpticalProperty.flat_absorber,
                                      trace_options: Trace, file_path: Path, file_name: str):

    sun_dir = sun_direction(theta_t=Angle(deg=theta_t), theta_l=Angle(deg=theta_l))

    st_primaries_property = primaries_property.to_soltrace()
    st_absorber_property = absorber_property.to_soltrace()

    primary_mirrors = [hel.as_soltrace_element(name=f'heliostat_{i + 1}', length=length,
                                               aim_pt=aim, sun_dir=sun_dir, optic=st_primaries_property)
                       for i, hel in enumerate(lfr.field.primaries)]

    absorber = lfr.receiver.as_soltrace_element(length=length, optic=st_absorber_property)

    st_elements = primary_mirrors + absorber

    st_sun = sun_shape.to_soltrace(sun_dir=sun_dir)
    st_optics = Optics(properties=[st_primaries_property, st_absorber_property])
    st_stages = [Stage(name='linear_fresnel', elements=st_elements)]
    st_geometry = Geometry(stages=st_stages)

    stats = ElementStats(stats_name=f'absorber_{theta_t}_{theta_l}',
                         stage_index=0, element_index=len(st_elements) - 1,
                         dni=1000, x_bins=150, y_bins=150, final_rays=True)

    if sun_shape.profile == 'collimated':
        trace_options.sunshape = 'false'

    if primaries_property.spec_error == 0.0 and primaries_property.slope_error == 0.0:
        trace_options.errors = 'false'
    else:
        trace_options.errors = 'true'

    script_full_path = soltrace_script(file_path=file_path, file_name=file_name,
                                       sun=st_sun, optics=st_optics, geometry=st_geometry,
                                       trace=trace_options, stats=stats)

    run_soltrace(script_full_path)
    absorber_stats = read_element_stats(stats.file_full_path)

    absorber_flux = absorber_stats['power_per_ray'] * array(absorber_stats['flux']).flatten().sum()

    optical_efficiency = absorber_flux / (stats.dni * lfr.field.widths.sum() * length * 1e-6)

    return optical_efficiency


# def raytracing_optical_efficiency(field: PrimaryField, receiver: Receiver,
#                                   theta_t: float, theta_l: float, aim: array, length: float,
#                                   sun_shape: RadialSunshape,
#                                   primaries_property: OpticalProperty.reflector,
#                                   absorber_property: OpticalProperty.flat_absorber,
#                                   trace_options: Trace, file_path: Path, file_name: str):
#
#     sun_dir = sun_direction(theta_t=Angle(deg=theta_t), theta_l=Angle(deg=theta_l))
#
#     st_primaries_property = primaries_property.to_soltrace()
#     st_absorber_property = absorber_property.to_soltrace()
#
#     primary_mirrors = [hel.as_soltrace_element(name=f'Heliostat_{i + 1}', length=length,
#                                                aim_pt=aim, sun_dir=sun_dir, optic=st_primaries_property)
#                        for i, hel in enumerate(field.primaries)]
#
#     pass
#
#     receiver_elements = receiver.as_soltrace_element(length=length,
#                                                      absorber_property=st_absorber_property,
#                                                      secondary_property=st_secondary_property)
#
#     st_elements = primary_mirrors + receiver_elements
#
#     st_sun = sun_shape.to_soltrace(sun_dir=sun_dir)
#     st_optics = Optics(properties=[st_primaries_property, st_absorber_property])
#     st_stages = [Stage(name='linear_fresnel', elements=st_elements)]
#     st_geometry = Geometry(stages=st_stages)
#
#     stats = ElementStats(stats_name=f'Absorber_{theta_t}_{theta_l}',
#                          stage_index=0, element_index=len(st_elements) - 1,
#                          dni=1000, x_bins=150, y_bins=150, final_rays=True)
#
#     if sun_shape.profile == 'collimated':
#         trace_options.sunshape = 'false'
#
#     if primaries_property.spec_error == 0.0 and primaries_property.slope_error == 0.0:
#         trace_options.errors = 'false'
#     else:
#         trace_options.true = 'false'
#
#     script_full_path = soltrace_script(file_path=file_path, file_name=file_name,
#                                        sun=st_sun, optics=st_optics, geometry=st_geometry,
#                                        trace=trace_options, stats=stats)
#
#     run_soltrace(script_full_path)
#     absorber_stats = read_element_stats(stats.file_full_path)
#
#     absorber_flux = absorber_stats['power_per_ray'] * array(absorber_stats['flux']).flatten().sum()
#
#     optical_efficiency = absorber_flux / (stats.dni * field.widths.sum() * length * 1e-6)
#
#     return optical_efficiency
