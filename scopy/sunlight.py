# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:40:11 2020
@author: André Santos (andrevitoras@gmail.com/avas@uevora.pt)
"""

# ToDo: Implement models of sun vector and DNI as function of location, latitude, hour angle, among others.
# In the work of Abbas et al. [https://doi.org/10.1016/j.apenergy.2016.01.065.], there are a comments about the models
# available in the literature to calculate the sun position, and the DNI. See Sections 2.3.1 and 2.3.2.

from numpy import sin, cos, tan, arctan, array, dot, pi, log, exp, linspace, zeros, absolute, sqrt, identity
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import norm
from niopy.geometric_transforms import R, Angle
from soltracepy import SoltraceSun


########################################################################################################################
# Sun direction functions ##############################################################################################


def inc_direction(zen: Angle, azi: Angle):
    """
    :param zen: Sun zenith angle, as an Angle object
    :param azi: Sun azimuth angle, as an Angle object
    :return: This function returns the 3D incidence direction of sunlight
    for the pair [sun_zenith, sun_azimuth] angles.

    This function assumes the sun azimuth as measured regarding the South direction, where displacements East of
    South are negative and West of South are positive [1]. Moreover, the inertial XYZ coordinates systems is aligned as
    follows: X points to East, Y to North, and Z to Zenith.

    [1] Duffie JA, Beckman WA. Solar Engineering of Thermal Processes. 4th Ed. New Jersey: John Wiley & Sons; 2013.
    """

    vi = array(
        [
            sin(zen.rad) * sin(azi.rad),
            sin(zen.rad) * cos(azi.rad),
            cos(zen.rad)
        ]
    )
    return vi


def sun_vector(zenith: float, azimuth: float, degrees=True):
    """
    :param zenith: Sun zenith angle.
    :param azimuth: Sun azimuth angle.
    :param degrees: Boolean to define solar zenith and azimuth are in degrees or radians.

    :return: This function returns the 3D incidence direction of sunlight.
    """

    v = inc_direction(Angle(deg=zenith), Angle(deg=azimuth)) \
        if degrees else inc_direction(Angle(rad=zenith), Angle(rad=azimuth))

    return v


def sun_direction(theta_t: Angle, theta_l: Angle):
    """
    :param theta_t: Transversal incidence angle, as an Angle object
    :param theta_l: Longitudinal incidence angle, as an Angle object
    :return: This function returns a 3D vector which represents the incidence direction of sunlight -- the Sun vector --
    for the pair [theta_t, theta_l]

    This function assumes the sun azimuth as measured regarding the South direction, where displacements East of
    South are negative and West of South are positive [1]. Moreover, the inertial XYZ coordinates systems is aligned as
    follows: X points to East, Y to North, and Z to Zenith.

    [1] Duffie JA, Beckman WA. Solar Engineering of Thermal Processes. 4th Ed. New Jersey: John Wiley & Sons; 2013.
    """

    Ix, Iy, Iz = identity(3)

    r1 = R(theta_l.rad, Ix)
    projected_angle = arctan(tan(theta_t.rad) * cos(theta_l.rad))
    r2 = R(projected_angle, Iy)

    return dot(r1, dot(r2, Iz))


def sun2lin(zenith, azimuth, degree=True, NS=True, solar_longitudinal=False):
    """
    A vectorized function to convert sun positions given by solar zenith and azimuth to linear concentrators incidence
    angles: transversal and longitudinal incidence angles (and solar longitudinal [1, 2]).
    This function accepts solar angles in radians or degrees, as given by the bool argument 'degrees'. Then it returns
    both linear angles in degrees, as them are usually mentioned in degrees.

    :param zenith: Solar zenith angle.
    :param azimuth: Solar azimuth angle.
    :param degree: A boolean sign to inform whether solar angles are in degree or radians.
    :param NS: A sign to inform whether a NS (North-South) or EW (East-West) mounting for the linear concentrator.
    :param solar_longitudinal: A sing to return or not the solar longitudinal angle
    :return: Returns a tuple of linear incidence angles, in degrees.
    If an array of solar angles is given, then a tuple of arrays is returned.

    This function assumes the sun azimuth as measured regarding the South direction, where displacements East of
    South are negative and West of South are positive [3]. Moreover, the inertial XYZ coordinates systems is aligned as
    follows: X points to East, Y to North, and Z to Zenith.

    [1] IEC (International Electrotechnical Commission). Solar thermal electric plants
    - Part 5-2: Systems and components - General requirements and test methods for large-size linear Fresnel collectors.
    Solar thermal electric plants, 2021.
    [2] Hertel JD, Martinez-Moll V, Pujol-Nadal R. Estimation of the influence of different incidence angle modifier
    models on the bi-axial factorization approach. Energy Conversion and Management 2015;106:249–59.
    https://doi.org/10.1016/j.enconman.2015.08.082.
    [3] Duffie JA, Beckman WA. Solar Engineering of Thermal Processes. 4th Ed. New Jersey: John Wiley & Sons; 2013.

    """

    # Calculating the linear incidence angles based on the solar angles arguments are in degree or radians
    if degree:
        tt = arctan(tan(zenith * pi / 180) * sin(-azimuth * pi / 180))
        tl = arctan(tan(zenith * pi / 180) * cos(azimuth * pi / 180))
    else:
        tt = arctan(tan(zenith) * sin(-azimuth))
        tl = arctan(tan(zenith) * cos(azimuth))

    # Accounting for a NS or EW mounting
    tt, tl = (tt, tl) if NS else (tl, tt)

    # To return or not the solar longitudinal incidence angle.
    # It also returns everything in degrees, the more usual unit for the linear incidence angles.
    if not solar_longitudinal:
        angles = (tt * 180 / pi, tl * 180 / pi)
    else:
        ti = arctan(tan(absolute(tl)) * cos(tt))
        angles = (tt * 180 / pi, tl * 180 / pi, ti * 180 / pi)

    return angles


def ZenAzi2TranLong(zenith: Angle, azimuth: Angle, NS=True):
    """
    :param zenith: Sun zenith, as an angle object
    :param azimuth: Sun azimuth, as an angle object
    :param NS: Indicates if is a North-South (NS) or East-West (EW) mounting of a linear concentrator
    :return: Returns a pair of angle objects: transversal and longitudinal incidence angles.

    This function assumes the sun azimuth as measured regarding the South direction, where displacements East of
    South are negative and West of South are positive [1]. Moreover, the inertial XYZ coordinates systems is aligned as
    follows: X points to East, Y to North, and Z to Zenith.


    [1] Duffie JA, Beckman WA. Solar Engineering of Thermal Processes. 4th Ed. New Jersey: John Wiley & Sons; 2013.
    """

    tt = arctan(tan(zenith.rad) * sin(-azimuth.rad))
    tl = arctan(tan(zenith.rad) * cos(azimuth.rad))

    angles = [Angle(rad=tt), Angle(rad=tl)] if NS else [Angle(rad=tl), Angle(rad=tt)]

    return angles


########################################################################################################################
########################################################################################################################

########################################################################################################################
# Classes ##############################################################################################################


class RadialSunshape:

    """
    This class represent the different sun shape profiles and their properties.

    """

    def __init__(self, profile: str = None, size: float = None, user_data=None):
        """
        :param profile: Indicates the profile of the sun shape, i.e, pillbox, Gaussian or collimated (None).
        :param size: The size of the sun shape, in radians.
        e.g., the half-width for a pillbox, the standard deviation for a Gaussian, and the circumsolar ratio for a Buie;
        for a collimated model, None is the input.

        :param user_data:
        """

        if size is None and profile is None:
            self.profile = 'collimated'
            self.size = 0
            self.rms_width = 0
            self.distribution = 'collimated'
            self.radial_distribution = 'collimated'
        elif profile == 'pillbox' or profile == 'p':
            self.profile = 'pillbox'
            self.size = abs(size)
            self.rms_width = self.size / sqrt(2)
            self.distribution = cumulative_uniform_sunshape(sun_disk=self.size)
            # self.radial_distribution =
        elif profile == 'gaussian' or profile == 'g':
            self.profile = 'gaussian'
            self.size = abs(size)
            self.rms_width = self.size * sqrt(2)
            self.distribution = cumulative_gaussian_sunshape(std_source=self.size)
            # self.radial_distribution =
        elif profile == 'buie' or profile == 'b':
            self.profile = 'buie'
            self.size = abs(size)
            ss = BuieSunshape(csr=self.size, csr_calibration=None)
            self.radial_distribution = ss.radial_distribution
            self.rms_width = ss.rms_width
        elif profile == 'user' or profile == 'u':
            self.profile = "'u'"
            self.values = user_data
        else:
            raise ValueError("Please, input a valid profile: 'gaussian', 'pillbox', 'buie', "
                             "or all empty for a collimated sunlight model")

    def to_soltrace(self, sun_dir: array):
        return SoltraceSun(sun_dir=sun_dir, profile=self.profile, size=self.size*1e3)

    def linear_effective_source(self, specular_error: float, slope_error: float):

        # ToDo: Study and implement a convolution approach to this linear effective source.

        if self.profile != 'collimated':
            # It composes the r.m.s width of the sun shape and optical errors
            # For a Gaussian distribution, the r.m.s width is sqrt(2) times the standard deviation
            # And the slope error has a double effect, as well as the tracking.
            source_rms_width = self.rms_width ** 2 + 2 * (
                    4 * slope_error ** 2 + specular_error ** 2)

            source_rms_width = sqrt(source_rms_width)

            # It returns a gaussian effective source with the same r.m.s width as the combination of sun and errors.
            source_sigma = source_rms_width / sqrt(2)
            phi = cumulative_gaussian_sunshape(std_source=source_sigma)
        else:
            phi = 'collimated'

        return phi


class BuieSunshape:
    """

    [1] Buie et al. Solar Energy 2003;74:113–22. https://doi.org/10.1016/S0038-092X(03)00125-7.
    [2] Wang et al. Solar Energy 2020;195:461–74. https://doi.org/10.1016/j.solener.2019.11.035.
    [3] Rabl A. Active Solar Collectors and Their Applications. New York: Oxford University Press; 1985.

    """

    def __init__(self, csr: float, csr_calibration=None):

        # Angles defined by Buie et al. when defining the sun shape profile.
        self.theta_1 = 4.65e-3  # extension of the solar disk, in rad.
        self.theta_2 = 43.6e-3  # extension of the solar aureole, in rad.

        # inputted circumsolar ratio.
        self.csr = csr

        #########################################################################################################
        # Calibration of the circumsolar ratio ##################################################################

        # As observed by Buie et al. [1], the inputted csr is not the same value as the CSR that can be calculated from
        # the distribution function. Remember that the distribution function is a curve-fitting statistical analysis
        # of real measurements. Therefore, errors are normal.
        # However, these propositions to correct the csr were found in Wang et al. [2] study.
        # Precisely in https://github.com/anustg/Tracer/blob/master/tracer/sources.py.
        if csr_calibration == 'CA':
            self.crs_cali = self.CSR_calibration('CA')
        elif csr_calibration == 'tonatiuh':
            self.crs_cali = self.CSR_calibration('tonatiuh')

        self.radial_distribution = \
            buie_sunshape(csr=self.crs_cali) if csr_calibration is not None else buie_sunshape(csr=self.csr)

        #############################################################################################################

        #############################################################################################################
        # Calculations of the r.m.s width of the sun shape. See Rabl's formula [3, pp. 134-136] #####################
        def N(x):
            return self.radial_distribution(x) * (x ** 3)

        def D(x):
            return self.radial_distribution(x) * x

        num = quad(N, 0, 43.6e-3)[0]
        den = quad(D, 0, 43.6e-3)[0]

        self.rms_width = sqrt(num / den)
        ############################################################################################################

    def CSR_calibration(self, source):
        """
        pre proceed CSR to true value
        source - 'CA' from Charles Charles-Alexis Asselineau at ANU; or 'tonatiuh' from Manuel Blanco at CyI.

        Source code: https://github.com/anustg/Tracer/blob/master/tracer/sources.py.
        """

        csr_g = self.csr

        if source == 'CA':
            if csr_g <= 0.1:
                csr_cali = -2.245e+3 * csr_g ** 4. + 5.207e+2 * csr_g ** 3. - 3.939e+1 * csr_g ** 2. \
                           + 1.891e+0 * csr_g + 8e-3
            else:
                csr_cali = 1.973 * csr_g ** 4. - 2.481 * csr_g ** 3. + 0.607 * csr_g ** 2. + 1.151 * csr_g - 0.020
        elif source == 'tonatiuh':
            if csr_g > 0.145:
                csr_cali = -0.04419909985804843 + csr_g * (1.401323894233574 + csr_g * (
                        -0.3639746714505299 + csr_g * (-0.9579768560161194 + 1.1550475450828657 * csr_g)))
            elif 0.035 < csr_g <= 0.145:
                csr_cali = 0.022652077593662934 + csr_g * (0.5252380349996234 +
                                                           (2.5484334534423887 - 0.8763755326550412 * csr_g)) * csr_g
            else:
                csr_cali = 0.004733749294807862 + csr_g * (4.716738065192151 + csr_g * (-463.506669149804 + csr_g * (
                        24745.88727411664 + csr_g * (-606122.7511711778 + 5521693.445014727 * csr_g))))
        else:
            csr_cali = csr_g

        return csr_cali


# class IncidentSun:
#
#     def __int__(self, vector: array, shape: RadialSunshape):
#         self.sun_dir = vector
#         self.sun_shape = shape
#
#     def to_soltrace(self):
#         return SoltraceSun(sun_dir=self.sun_dir, profile=self.sun_shape.profile, size=self.sun_shape.size)


########################################################################################################################
########################################################################################################################

########################################################################################################################
# Functions ############################################################################################################


def uniform_sunshape(sun_disk=4.65 * 1.e-3):  # in mrad # sun_disk = 4.65 mrad
    def phi(theta):
        return 0 if abs(theta) > sun_disk else 1

    return phi


def buie_sunshape(csr: float):
    """
    This function implements Buie's model [1] for the radiance distribution of the sun.
    It returns a normalized intensity (or radiance) profile as function of the angular deviation from the sun's center,
    in radians [2].

    Buie's model is a radial sun shape profile, based on the definition of the circumsolar ratio -- the function's
    only argument.

    :param csr: The circumsolar ratio, a value between 0 and 1.
    :return: Returns a radial sun shape profile function (callable).

    [1] Buie et al. Solar Energy 2003;74:113–22. https://doi.org/10.1016/S0038-092X(03)00125-7.
    [2] Wang et al. Solar Energy 2020;195:461–74. https://doi.org/10.1016/j.solener.2019.11.035.

    """

    gamma = - 0.1 + 2.2 * log(0.52 * csr) * (csr ** 0.43)
    k = 0.9 * log(13.5 * csr) * (csr ** -0.3)

    def phi(x):  # x is an angle in radians
        return cos(326 * x) / cos(308 * x) if abs(x) <= 4.65 * 1.e-3 else exp(k) * (abs(x * 1.e3) ** gamma)

    return phi


def cumulative_source(phi):
    def E(y: float):  # y is an angle in radians
        return phi(y) * y

    sun_aureole = 43.6 * 1.e-3

    total_area = quad(E, 0, sun_aureole)[0]
    theta = linspace(0, pi / 2, 50000)
    cumulus_values = zeros(len(theta))

    for i in range(len(theta)):
        if theta[i] > sun_aureole:
            cumulus_values[i] = 0.5
        else:
            cumulus_values[i] = quad(E, 0, theta[i])[0] / (2 * total_area)

    cumulative_distribution = interp1d(theta, cumulus_values, kind='cubic')

    # total_area = quad(E, 0, inf)[0]
    #
    # def cumulative_distribution(theta: float):  # theta is an angle in radians
    #     return quad(E, 0., theta)[0] / (2 * total_area)

    return cumulative_distribution


def cumulative_uniform_sunshape(sun_disk=4.65 * 1.e-3):
    delta = sun_disk * (3 ** 0.5) / 2

    x = linspace(0, pi, 10000)
    y = zeros(len(x))

    for i in range(len(x)):
        y[i] = abs(x[i]) / (2 * delta) if abs(x[i]) <= delta else 0.5

    cumulative_function = interp1d(x, y, kind='cubic')

    return cumulative_function


def cumulative_gaussian_sunshape(std_source):
    std = std_source

    x = linspace(0, pi, 10000)
    cv = norm.cdf(x, scale=std) - 0.5

    cumulative_function = interp1d(x, cv, kind='cubic')

    return cumulative_function
