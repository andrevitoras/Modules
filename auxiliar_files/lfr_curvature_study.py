
import json
import os
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from pathlib import Path

import time

from geopy import Nominatim
from numpy import array, ones, arange, zeros, tan, pi, arctan, linspace
from pandas import DataFrame
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

from scipy.optimize import fsolve

from tqdm import tqdm

from niopy.geometric_transforms import dst, mid_point, Angle
from scopy.linear_fresnel import Absorber, uniform_centers, transform_vector, rabl_curvature, Heliostat, \
    PrimaryField, LFR, boito_curvature, ParabolicHeliostat, annual_eta, rabl_design_lfr, boito_design_lfr

from scopy.sunlight import RadialSunshape, sun2lin
from utils import read_trnsys_tmy2, dic2json, rmse


########################################################################################################################
########################################################################################################################

########################################################################################################################
# Equivalence between parabolic and cylindrical primaries ##############################################################


def design_mirrors(width: float, center: array, aim: array, theta_d: Angle, nbr_pts: int):

    radius = rabl_curvature(center=center, aim=aim, theta_d=theta_d)

    cyl = Heliostat(width=width,
                    center=center,
                    radius=radius,
                    nbr_pts=nbr_pts)

    par = ParabolicHeliostat(width=width,
                             center=center,
                             aim=aim,
                             theta_d=theta_d,
                             nbr_pts=nbr_pts)

    return cyl, par


def center_from_lambda(lamb: Angle, h: float):
    xc = h * tan(lamb.rad)

    return array([xc, 0, 0])


def slope_dev(angles, w_ratio, nbr_pts):

    h = 8000
    width = h * w_ratio

    aim = array([0, 0, h])

    lamb, theta_d = Angle(deg=angles[0]), Angle(deg=angles[1])
    center = center_from_lambda(lamb=lamb, h=h)

    cyl, par = design_mirrors(width=width, center=center, aim=aim, theta_d=theta_d, nbr_pts=nbr_pts)

    cyl_slope = cyl.local_slope(weighted=True)
    par_slope = par.local_slope(weighted=True)

    std_slope_dev = (cyl_slope - par_slope).std() * 1000  # in mrad
    rms_slope_dev = rmse(predictions=cyl_slope, targets=par_slope) * 1000  # in mrad

    return std_slope_dev, rms_slope_dev


def parallel_slope_dev(inputs):

    angles, w_ratio, nbr_pts = inputs

    return slope_dev(angles, w_ratio, nbr_pts)


def vertical_design_dev(wi: float, we: float, n: int):

    w_range = linspace(start=min(wi, we), stop=max(wi, we), num=n)
    rms_slope_dev = zeros(len(w_range))

    f = 1
    r = 2
    center = array([0, 0])

    for i, w in enumerate(w_range):

        cyl = Heliostat(width=w, center=center, radius=r, nbr_pts=401)
        par = ParabolicHeliostat(width=w, center=center, forced_design=True, focal_length=f, nbr_pts=401)

        cyl_slope = cyl.local_slope(weighted=True)
        par_slope = par.local_slope(weighted=True)

        rms_slope_dev[i] = rmse(predictions=cyl_slope, targets=par_slope) * 1000  # in mrad

    return w_range, rms_slope_dev


def one_ratio_slope_dev(angles, ratio, nbr_pts):

    n_cores = cpu_count()

    inputs = [[a, ratio, nbr_pts] for a in angles]

    with Pool(n_cores - 2) as p:
        results = list(tqdm(p.imap(parallel_slope_dev, inputs), total=len(inputs)))

    results = array(results)

    df = DataFrame(angles, columns=['lambda', 'theta_d'])
    df['w_ratio'] = ones(df.shape[0]) * ratio
    df['n_pts'] = ones(df.shape[0]) * nbr_pts
    df['std_slope_dev'] = results.T[0]
    df['rms_slope_dev'] = results.T[1]

    return df


def sun_ref_max_slope_dev(design_position: float, w_ratio: float, nbr_pts: int):

    lambdas = arange(-75, 77.5, 2.5)
    results = [slope_dev(angles=[lb, design_position], w_ratio=w_ratio, nbr_pts=nbr_pts)[-1] for lb in lambdas]

    return max(results)


def parallel_sun_ref_analysis(inputs):

    design_position, w_ratio, nbr_pts = inputs

    return sun_ref_max_slope_dev(design_position=design_position, w_ratio=w_ratio, nbr_pts=nbr_pts)


def sun_reference_rms_deviations(design_positions: array, w_ratios: array, nbr_pts: int, file_name, file_path):

    file_full_path = Path(file_path, f'{file_name}.csv')

    n_cores = cpu_count()

    inputs = [[dp, wr, nbr_pts] for dp in design_positions for wr in w_ratios]

    with Pool(n_cores - 2) as p:
        results = list(tqdm(p.imap(parallel_sun_ref_analysis, inputs), total=len(inputs)))

    results = array(results)

    df = DataFrame(inputs, columns=['theta_d', 'w_ratio', 'nbr_pts'])
    df['rms_slope_dev'] = results

    df.to_csv(file_full_path, index=False)

    return df


########################################################################################################################
########################################################################################################################

########################################################################################################################
# Non-Uniform and Uniform designs ######################################################################################


class nonuniform_lfr_geometry:

    def __init__(self, rec_height: float, rec_width: float, centers: array, widths: array):

        self.rec_height = abs(rec_height)
        self.rec_width = abs(rec_width)

        self.receiver = Absorber.flat(width=abs(rec_width),
                                      center=array([0, 0, abs(rec_height)]))

        if centers.shape[0] != widths.shape[0]:

            raise ValueError('The number of mirrors do not correspond for "centers" and "widths"')

        else:

            self.centers = centers
            self.widths = widths

            self.nbr_mirrors = self.centers.shape[0]
            self.center_distance = [dst(self.centers[i], self.centers[i + 1]) for i in range(centers.shape[0] - 1)]


########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################


def rabl_design_as_json(lst: list, name: str, file_path=Path.cwd()):
    to_export = deepcopy(lst)

    for i, d in enumerate(to_export):
        d['dp'] = d['dp'].deg if type(d['dp']) == Angle else d['dp']
        d['tran_eta'] = d['tran_eta'].tolist()
        d['long_eta'] = d['long_eta'].tolist()

    file_full_path = Path(file_path, f'{name}.json')

    with open(file_full_path, 'w') as jsonfile:
        json.dump(to_export, jsonfile)

    return file_full_path


def read_rabl_design(file):
    with open(file, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())

    for dic in data:
        dic['tran_eta'] = array(dic['tran_eta'])
        dic['long_eta'] = array(dic['long_eta'])

    return data


def location_data(file):

    name = str(Path(file).name)[3:-10]
    name = name[:-1] if name[-1] == '-' else name

    geo_locator = Nominatim(user_agent="MyApp")
    lat = geo_locator.geocode(name).latitude

    data = read_trnsys_tmy2(file)
    df = data[data['DNI [W/m2]'] > 0]

    zenith = array(df['Solar Zenith [degrees]'])
    azimuth = array(df['Solar Azimuth [degrees]'])
    dni = array(df['DNI [W/m2]'])

    transversal_angles, longitudinal_angles = sun2lin(zenith, azimuth, degree=True, NS=True)

    avg_tran = transversal_angles.dot(dni) / dni.sum()
    avg_long = longitudinal_angles.dot(dni) / dni.sum()

    dic = {'file': file, 'name': name,
           'latitude': lat, 'dni_sum': dni.sum() / 1000, 'avg_tran': avg_tran, 'avg_long': avg_long,
           'units': 'kWh and degrees'}

    return dic


def geometry_full_analysis(lfr_geometry, cum_eff, nbr_pts, location: str, file_name: str,
                           files_path=Path.cwd(), export=True):

    # count the number of cores of the computer. It will be used to parallel computations
    n_cores = cpu_count()

    loc = location_data(location)
    files_path = Path(files_path)

    # creates a directory to export results files
    if not files_path.is_dir():
        os.makedirs(files_path)

    etas_path = Path(files_path, f'{file_name}_design_etas.json')
    if not etas_path.is_file():

        print(f'It is now {time.strftime("%H:%M:%S", time.localtime())}')
        t0 = time.time()
        print('Starting optical analysis for the whole range of design positions!')

        design_positions = arange(0, 87.5, 2.5)
        fac_etas = [rabl_design_lfr(lfr_geometry,
                                    Angle(deg=x),
                                    nbr_pts).optical_analysis(cum_eff=cum_eff, symmetric=False)
                    for x in tqdm(design_positions)]

        etas = [{'dp': Angle(deg=x), 'tran_eta': eta[0], 'long_eta': eta[1]}
                for x, eta in zip(design_positions, fac_etas)]

        spec_ref_eta = rabl_design_lfr(lfr_geometry, 'SR', nbr_pts).optical_analysis(cum_eff=cum_eff, symmetric=False)
        etas += [{'dp': 'SR', 'tran_eta': spec_ref_eta[0], 'long_eta': spec_ref_eta[1]}]

        boito_design = boito_design_lfr(lfr_geometry=lfr_geometry,
                                        latitude=Angle(deg=loc['latitude']),
                                        nbr_pts=nbr_pts)
        boito_design_etas = boito_design.optical_analysis(cum_eff=cum_eff, symmetric=False)
        etas += [{'dp': 'BG', 'tran_eta': boito_design_etas[0], 'long_eta': boito_design_etas[1]}]

        dt = round((time.time() - t0) / 60, 2)
        print(f'Optical analyzes took {dt} minutes.')

        etas_path = rabl_design_as_json(lst=etas, name=etas_path.name[:-5], file_path=files_path)
        etas = read_rabl_design(file=etas_path)
    else:
        print(f'Reading the json file with the results of the optical analyzes!')
        etas = read_rabl_design(file=etas_path)

    etas_dic = etas[:-2]
    spec_ref_dic = etas[-2]
    boito_etas = etas[-1]

    design_positions = array([d['dp'] for d in etas_dic])

    # The 'annual_energy' functions accepts as its arguments the following:
    # transversal_data: array, longitudinal_data: array, location: str, NS=True
    ns_inputs = [(d['tran_eta'], d['long_eta'], loc['file']) for d in etas_dic]
    ew_inputs = [(d['tran_eta'], d['long_eta'], loc['file'], False) for d in etas_dic]

    print('Starting annual energetic calculations!')
    with Pool(n_cores - 2) as p:
        ns_etas = p.starmap(annual_eta, ns_inputs)
        ew_etas = p.starmap(annual_eta, ew_inputs)

    ns_etas = array(ns_etas)

    etas_function = InterpolatedUnivariateSpline(design_positions, ns_etas, k=4)
    roots = etas_function.derivative().roots()
    theta_d_max = roots if roots.size > 0 else fsolve(etas_function.derivative(), design_positions[0])
    eta_max = etas_function(theta_d_max)

    spec_ref_annual_eta = annual_eta(spec_ref_dic['tran_eta'],
                                     spec_ref_dic['long_eta'],
                                     loc['file'], NS=True)

    boito_ann_eta = annual_eta(boito_etas['tran_eta'],
                               boito_etas['long_eta'],
                               loc['file'], NS=True)

    ns_out_dic = {'etas_path': str(etas_path), 'weather_file': str(loc['file']),
                  'location': loc['name'], 'latitude': loc['latitude'], 'nbr_pts': nbr_pts,
                  'theta_d': design_positions.tolist(), 'annual_etas': ns_etas.tolist(),
                  'theta_d_max': theta_d_max.tolist(), 'eta_max': eta_max.tolist(),
                  'sun_ref_theta': 0.0, 'sun_ref_eta': etas_function(0.0).tolist(),
                  'spec_ref_eta': spec_ref_annual_eta,
                  'boito_eta': boito_ann_eta}

    ew_etas = array(ew_etas)

    etas_function = InterpolatedUnivariateSpline(design_positions, ew_etas, k=4)
    roots = etas_function.derivative().roots()
    theta_d_max = roots if roots.size > 0 else fsolve(etas_function.derivative(), design_positions[0])
    eta_max = etas_function(theta_d_max)

    spec_ref_annual_eta = annual_eta(spec_ref_dic['tran_eta'],
                                     spec_ref_dic['long_eta'],
                                     loc['file'], NS=False)

    boito_ann_eta = annual_eta(boito_etas['tran_eta'],
                               boito_etas['long_eta'],
                               loc['file'], NS=False)

    ew_out_dic = {'etas_path': str(etas_path), 'weather_file': str(loc['file']),
                  'location': loc['name'], 'latitude': loc['latitude'], 'nbr_pts': nbr_pts,
                  'theta_d': design_positions.tolist(), 'annual_etas': ew_etas.tolist(),
                  'theta_d_max': theta_d_max.tolist(), 'eta_max': eta_max.tolist(),
                  'spec_ref_eta': spec_ref_annual_eta,
                  'sun_ref_theta': loc['avg_long'], 'sun_ref_eta': etas_function(loc['avg_long']).tolist(),
                  'boito_eta': boito_ann_eta}

    if export:
        results_path = Path(files_path, f'{file_name}_design_results.json')
        to_export_dic = {'ns': ns_out_dic, 'ew': ew_out_dic}
        with open(results_path, 'w') as jsonfile:
            json.dump(dic2json(to_export_dic), jsonfile)

    return ns_out_dic, ew_out_dic


def lfr_design(lfr_geometry, radius: array, nbr_pts=121):

    if radius.shape[0] != lfr_geometry.centers.shape[0]:
        raise ValueError('Number of mirrors of the lfr geometry does not match with the number of curvature radius')

    widths = lfr_geometry.widths
    centers = lfr_geometry.centers

    heliostats = []

    for w, hc, rr in zip(widths, centers, radius):
        hel = Heliostat(width=w,
                        center=hc,
                        radius=rr,
                        nbr_pts=nbr_pts)
        heliostats.append(hel)

    primary_field = PrimaryField(heliostats)
    lfr_concentrator = LFR(primary_field=primary_field, flat_absorber=lfr_geometry.receiver)

    return lfr_concentrator


def univariate_parametric_optimization(lfr_geometry, cum_eff, location, nbr_pts: int, file_name: str,
                                       files_path=Path.cwd(), export=True):

    files_path = Path(files_path)
    # creates a directory to export results files
    if not files_path.is_dir():
        os.makedirs(files_path)

    centers = lfr_geometry.centers
    aim = mid_point(lfr_geometry.receiver.s1, lfr_geometry.receiver.s2)
    sm = array([aim[0], aim[-1]])

    radius = array([2 * dst(hc, sm) for hc in centers])
    ns_radius = zeros(radius.shape[0])
    ew_radius = zeros(radius.shape[0])

    ns_out_dic = {}
    ew_out_dic = {}

    print(f'It is now {time.strftime("%H:%M:%S", time.localtime())}.')
    print(f'Starting the univariate parametric analysis to optimize the curvature radius of each primary.')
    for i, hc in enumerate(tqdm(centers)):

        hel_radius = radius[i] * arange(0.7, 2.5075, 0.075)
        ns_etas = zeros(hel_radius.shape[0])
        ew_etas = zeros(hel_radius.shape[0])

        for k, r in enumerate(hel_radius):
            radius[i] = r
            lfr_concentrator = lfr_design(lfr_geometry, radius, nbr_pts=21)
            trans, long = lfr_concentrator.optical_analysis(cum_eff=cum_eff, symmetric=False)

            ns_etas[k] = annual_eta(transversal_data=trans, longitudinal_data=long,
                                    location=location, NS=True)
            ew_etas[k] = annual_eta(transversal_data=trans, longitudinal_data=long,
                                    location=location, NS=False)

        if export:
            ns_out_dic[f'hel_{i + 1}'] = {'radius': hel_radius.tolist(),
                                          'etas': ns_etas.tolist()}

            ew_out_dic[f'hel_{i + 1}'] = {'radius': hel_radius.tolist(),
                                          'etas': ew_etas.tolist()}

            detailed_results_path = Path(files_path, f'{file_name}_univariate_optimization_data.json')
            to_export_dic = {'ns': ns_out_dic, 'ew': ew_out_dic}
            with open(detailed_results_path, 'w') as jsonfile:
                json.dump(dic2json(to_export_dic), jsonfile)

        # NS-mounting optimum values of curvature radius and annual optical efficiency (eta)
        ns_eta_function = InterpolatedUnivariateSpline(hel_radius, ns_etas, k=4)
        ns_eta_roots = ns_eta_function.derivative().roots()
        ns_opt_rad = ns_eta_roots[0] if ns_eta_roots.size > 0 else fsolve(ns_eta_function, hel_radius[0])[0]

        # EW-mounting optimum values of curvature radius and annual optical efficiency (eta)
        ew_eta_function = InterpolatedUnivariateSpline(hel_radius, ew_etas, k=4)
        ew_eta_roots = ew_eta_function.derivative().roots()
        ew_opt_rad = ew_eta_roots[0] if ew_eta_roots.size > 0 else fsolve(ew_eta_function, hel_radius[0])[0]

        ns_radius[i], ew_radius[i] = ns_opt_rad, ew_opt_rad

    ns_opt_eta = lfr_design(lfr_geometry=lfr_geometry,
                            radius=ns_radius,
                            nbr_pts=nbr_pts).annual_eta(cum_eff=cum_eff, location=location,
                                                        NS=True, symmetric=False)

    ew_opt_eta = lfr_design(lfr_geometry=lfr_geometry,
                            radius=ew_radius,
                            nbr_pts=nbr_pts).annual_eta(cum_eff=cum_eff, location=location,
                                                        NS=False, symmetric=False)

    out = {'ns_radius': ns_radius, 'ns_eta': ns_opt_eta,
           'ew_radius': ew_radius, 'ew_eta': ew_opt_eta}

    if export:
        results_path = Path(files_path, f'{file_name}_univariate_optimization_results.json')
        with open(results_path, 'w') as jsonfile:
            json.dump(dic2json(out), jsonfile)

    return out


def nonuniform_analysis(lfr_geometry, nbr_pts: int, source: RadialSunshape, location, file_name: str,
                        files_path=Path.cwd(), export=True):

    cum_eff = source.distribution

    designs_file_path = Path(files_path, f'{file_name}_design_results.json')
    if not designs_file_path.is_file():
        ns_out, ew_out = geometry_full_analysis(lfr_geometry=lfr_geometry, cum_eff=cum_eff, nbr_pts=nbr_pts,
                                                location=location, file_name=file_name, files_path=files_path,
                                                export=export)
    else:
        with open(designs_file_path, encoding='utf-8') as data_file:
            designs_data = json.loads(data_file.read())
        ns_out, ew_out = designs_data['ns'], designs_data['ew']

    ns_out['sun shape'], ew_out['sun shape'] = source.profile, source.profile
    ns_out['sun width'], ew_out['sun width'] = source.size, source.size

    univariate_results_path = Path(files_path, f'{file_name}_univariate_optimization_results.json')
    if not univariate_results_path.is_file():
        or_results = univariate_parametric_optimization(lfr_geometry=lfr_geometry, nbr_pts=nbr_pts, cum_eff=cum_eff,
                                                        location=location, file_name=file_name, files_path=files_path,
                                                        export=export)
        ns_out['or_radius'], ns_out['or_eta'] = or_results['ns_radius'].tolist(), or_results['ns_eta']
        ew_out['or_radius'], ew_out['or_eta'] = or_results['ew_radius'].tolist(), or_results['ew_eta']

    else:
        with open(univariate_results_path, encoding='utf-8') as data_file:
            or_results = json.loads(data_file.read())

        ns_out['or_radius'], ns_out['or_eta'] = or_results['ns_radius'], or_results['ns_eta']
        ew_out['or_radius'], ew_out['or_eta'] = or_results['ew_radius'], or_results['ew_eta']

    out_dic = dict({'ns': ns_out, 'ew': ew_out})

    file_full_path = Path(files_path, f'{file_name}_results.json')
    with open(file_full_path, 'w') as jsonfile:
        json.dump(dic2json(out_dic), jsonfile)

    return file_full_path


def uniform_analysis(lfr_geometry, nbr_pts: int, source: RadialSunshape, location, file_name: str,
                     file_path=Path.cwd(), export=True):

    cum_eff = source.distribution

    or_file_name = f'novatec_{source.profile}_univariate_optimization_results.json'
    or_path = Path(file_path, or_file_name)

    with open(or_path, encoding='utf-8') as data_file:
        or_design = json.loads(data_file.read())

    or_ns_radius = or_design['ns_radius']
    or_ew_radius = or_design['ew_radius']
    un_ns_radius = min(or_ns_radius) * linspace(start=1, stop=5, num=45)
    un_ew_radius = min(or_ew_radius) * linspace(start=1, stop=5, num=45)

    n_hel = lfr_geometry.centers.shape[0]

    file_full_path = Path(file_path, f'{file_name}_uniform_results.json')
    if not file_full_path.is_file():

        print('Starting Uniform optimization for a NS mounting.')
        ns_etas = [
            lfr_design(lfr_geometry=lfr_geometry,
                       radius=r * ones(n_hel), nbr_pts=19).annual_eta(location=location,
                                                                      cum_eff=cum_eff,
                                                                      NS=True)
            for r in tqdm(un_ns_radius)]

        print('Starting Uniform optimization for a EW mounting.')
        ew_etas = [
            lfr_design(lfr_geometry=lfr_geometry,
                       radius=r * ones(n_hel), nbr_pts=19).annual_eta(location=location,
                                                                      cum_eff=cum_eff,
                                                                      NS=False)
            for r in tqdm(un_ew_radius)]

        ns_eta_function = InterpolatedUnivariateSpline(un_ns_radius, ns_etas, k=4)
        ns_eta_roots = ns_eta_function.derivative().roots()
        ns_opt_rad = ns_eta_roots[0] if ns_eta_roots.size > 0 else fsolve(ns_eta_function, un_ns_radius[0])[0]
        ns_opt_eta = lfr_design(lfr_geometry=lfr_geometry,
                                radius=ns_opt_rad * ones(n_hel),
                                nbr_pts=nbr_pts).annual_eta(location=location,
                                                            cum_eff=cum_eff,
                                                            NS=True)

        ew_eta_function = InterpolatedUnivariateSpline(un_ew_radius, ew_etas, k=4)
        ew_eta_roots = ew_eta_function.derivative().roots()
        ew_opt_rad = ew_eta_roots[0] if ew_eta_roots.size > 0 else fsolve(ns_eta_function, un_ew_radius[0])[0]
        ew_opt_eta = lfr_design(lfr_geometry=lfr_geometry,
                                radius=ew_opt_rad * ones(n_hel),
                                nbr_pts=nbr_pts).annual_eta(location=location,
                                                            cum_eff=cum_eff,
                                                            NS=False)

        out_dic = {'ns_radius': un_ns_radius, 'ns_etas': ns_etas, 'ns_opt_radius': ns_opt_rad, 'ns_opt_eta': ns_opt_eta,
                   'ew_radius': un_ew_radius, 'ew_etas': ew_etas, 'ew_opt_radius': ew_opt_rad, 'ew_opt_eta': ew_opt_eta}
        if export:
            with open(file_full_path, 'w') as jsonfile:
                json.dump(dic2json(out_dic), jsonfile)

    return file_full_path


def optimization_analysis(lfr_geometry, nbr_pts: int, source: RadialSunshape, location, file_name: str,
                          file_path=Path.cwd(), export=True):

    non_uniform_results_path = Path(file_path, Path(file_path, f'{file_name}_results.json'))
    if not non_uniform_results_path.is_file():
        non_uniform_results_path = nonuniform_analysis(lfr_geometry=lfr_geometry, nbr_pts=nbr_pts, source=source,
                                                       location=location, export=export,
                                                       file_name=file_name, files_path=file_path)
    with open(non_uniform_results_path, encoding='utf-8') as data_file:
        non_uniform_results = json.loads(data_file.read())

    uniform_results_path = Path(file_path, f'{file_name}_uniform_results.json')
    if not uniform_results_path.is_file():
        uniform_results_path = uniform_analysis(lfr_geometry=lfr_geometry, nbr_pts=nbr_pts, source=source,
                                                location=location, export=export,
                                                file_name=file_name, file_path=file_path)

    with open(uniform_results_path, encoding='utf-8') as data_file:
        uniform_results = json.loads(data_file.read())

    out_dic = {'non_un': non_uniform_results, 'un': uniform_results}

    return out_dic


def lfr_acceptance(inputs):
    lfr_geometry, theta, aim, cum_eff, lvalue, dt = inputs
    return lfr_geometry.acceptance_angle(theta_t=theta, aim=aim, cum_eff=cum_eff, lvalue=lvalue, dt=dt)


def acceptance_full_analysis(lfr_geometry, location, source: RadialSunshape, nbr_pts,
                             lvalue: float, dt: float, file_name, file_path):

    n_hel = lfr_geometry.centers.shape[0]

    cum_eff = source.distribution
    aim = mid_point(lfr_geometry.receiver.s1, lfr_geometry.receiver.s2)
    loc = location_data(file=location)

    files_path = Path(file_path)
    # creates a directory to export results files
    if not files_path.is_dir():
        os.makedirs(files_path)

    # Loading the optimum non-uniform curvature radii.
    or_file_name = f'novatec_{source.profile}_univariate_optimization_results.json'
    or_path = Path(file_path, or_file_name)
    with open(or_path, encoding='utf-8') as data_file:
        or_design = json.loads(data_file.read())
    or_ns_radius = array(or_design['ns_radius'])
    or_ew_radius = array(or_design['ew_radius'])

    # Loading optimum uniform design files to get the optimum curvature radius for NS and EW mountings.
    un_file_path = Path(files_path, f'novatec_{source.profile}_uniform_results.json')
    with open(un_file_path) as data_file:
        un_dic = json.loads(data_file.read())
    un_ns_opt_radius, un_ew_opt_radius = un_dic['ns_opt_radius'], un_dic['ew_opt_radius']

    # Non-uniform sun reference designs (NS and EW)
    ns_sun_ref = rabl_design_lfr(lfr_geometry=lfr_geometry,
                                 design_position=Angle(deg=0),
                                 nbr_pts=nbr_pts)

    ew_sun_ref = rabl_design_lfr(lfr_geometry=lfr_geometry,
                                 design_position=Angle(deg=loc['avg_long']),
                                 nbr_pts=nbr_pts)

    # Non-uniform specific reference design
    spec_ref = rabl_design_lfr(lfr_geometry=lfr_geometry,
                               design_position='SR',
                               nbr_pts=nbr_pts)

    # Boito design for non-uniform configuration
    boito_design = boito_design_lfr(lfr_geometry=lfr_geometry,
                                    latitude=Angle(deg=loc['latitude']),
                                    nbr_pts=nbr_pts)

    # NS optimum non-uniform configuration through a univariate parametric analysis
    ns_or_lfr = lfr_design(lfr_geometry=lfr_geometry,
                           radius=array(or_ns_radius),
                           nbr_pts=nbr_pts)

    # EW optimum non-uniform configuration through a univariate parametric analysis
    ew_or_lfr = lfr_design(lfr_geometry=lfr_geometry,
                           radius=array(or_ew_radius),
                           nbr_pts=nbr_pts)

    # NS optimum uniform configuration through a univariate parametric analysis
    ns_un_lfr = lfr_design(lfr_geometry=lfr_geometry,
                           radius=un_ns_opt_radius * ones(n_hel),
                           nbr_pts=nbr_pts)

    # EW optimum uniform configuration through a univariate parametric analysis
    ew_un_lfr = lfr_design(lfr_geometry=lfr_geometry,
                           radius=un_ew_opt_radius * ones(n_hel),
                           nbr_pts=nbr_pts)

    transversal_angles = arange(-85, 85.5, 5)
    designs = [ns_sun_ref, ew_sun_ref, spec_ref, boito_design, ns_or_lfr, ew_or_lfr, ns_un_lfr, ew_un_lfr]
    labels = ['NS sun reference', 'EW sun reference', 'Specific reference',
              'Boito design', 'NS OR design', 'EW OR design', 'NS UN design', 'EW UN design']
    keys = ['ns_sun_ref', 'ew_sun_ref', 'spec_ref', 'boito', 'ns_or', 'ew_or', 'ns_un', 'ew_un']

    assert len(designs) == len(labels) == len(keys), 'Number of designs differ of the labels and/or keys'

    file_full_path = Path(files_path, f'{file_name}_designs_acceptance.json')

    if not file_full_path.is_file():
        dic = {'lfr': lfr_geometry.name, 'nbr_pts': nbr_pts, 'location': loc['name'], 'latitude': loc['latitude'],
               'sun shape': source.profile, 'sun width': source.size,
               'angles': transversal_angles}
    else:
        with open(file_full_path) as data_file:
            dic = json.loads(data_file.read())

    n_cores = cpu_count()
    for lb, k, des in zip(labels, keys, designs):

        if k not in dic.keys():

            print(f'Running acceptance angle calculations for the {lb} design.')
            inputs = [[des, theta, aim, cum_eff, lvalue, dt] for theta in transversal_angles]
            with Pool(n_cores - 2) as p:
                acc_results = list(tqdm(p.imap(lfr_acceptance, inputs), total=len(inputs)))

            dic[k] = acc_results

            with open(file_full_path, 'w') as jsonfile:
                json.dump(dic2json(dic), jsonfile)

        else:
            print(f'{lb} simulations were already done!')

    return file_full_path


def avg_acceptance_analysis(lfr_geometry, location, source: RadialSunshape, nbr_pts: int,
                            file_name, file_path, acceptance_file: Path):

    n_hel = lfr_geometry.centers.shape[0]
    cum_eff = source.distribution

    loc = location_data(file=location)
    site_data = read_trnsys_tmy2(location)
    df = site_data[site_data['DNI [W/m2]'] > 0]

    zenith = array(df['Solar Zenith [degrees]'])
    azimuth = array(df['Solar Azimuth [degrees]'])
    dni = array(df['DNI [W/m2]'])

    ns_theta_t, _ = sun2lin(zenith=zenith, azimuth=azimuth, degree=True, NS=True, solar_longitudinal=False)
    ew_theta_t, _ = sun2lin(zenith=zenith, azimuth=azimuth, degree=True, NS=False, solar_longitudinal=False)

    files_path = Path(file_path)
    # creates a directory to export results files
    if not files_path.is_dir():
        os.makedirs(files_path)

    or_file_name = f'novatec_{source.profile}_univariate_optimization_results.json'
    or_path = Path(file_path, or_file_name)

    with open(or_path, encoding='utf-8') as data_file:
        or_design = json.loads(data_file.read())

    or_ns_radius = array(or_design['ns_radius'])
    or_ew_radius = array(or_design['ew_radius'])

    # Loading optimum uniform design files to get the optimum curvature radius for NS and EW mountings.
    un_file_path = Path(files_path, f'novatec_{source.profile}_uniform_results.json')
    with open(un_file_path) as data_file:
        un_dic = json.loads(data_file.read())
    un_ns_opt_radius, un_ew_opt_radius = un_dic['ns_opt_radius'], un_dic['ew_opt_radius']

    # creating the LFR concentrators...
    ns_sun_ref = rabl_design_lfr(lfr_geometry=lfr_geometry,
                                 design_position=Angle(deg=0),
                                 nbr_pts=nbr_pts)

    ew_sun_ref = rabl_design_lfr(lfr_geometry=lfr_geometry,
                                 design_position=Angle(deg=loc['avg_long']),
                                 nbr_pts=nbr_pts)

    spec_ref = rabl_design_lfr(lfr_geometry=lfr_geometry,
                               design_position='SR',
                               nbr_pts=nbr_pts)
    boito_design = boito_design_lfr(lfr_geometry=lfr_geometry,
                                    latitude=Angle(deg=loc['latitude']),
                                    nbr_pts=nbr_pts)

    ns_or_lfr = lfr_design(lfr_geometry=lfr_geometry,
                           radius=array(or_ns_radius),
                           nbr_pts=nbr_pts)

    ew_or_lfr = lfr_design(lfr_geometry=lfr_geometry,
                           radius=array(or_ew_radius),
                           nbr_pts=nbr_pts)

    # NS optimum uniform configuration through a univariate parametric analysis
    ns_un_lfr = lfr_design(lfr_geometry=lfr_geometry,
                           radius=un_ns_opt_radius * ones(n_hel),
                           nbr_pts=nbr_pts)

    # EW optimum uniform configuration through a univariate parametric analysis
    ew_un_lfr = lfr_design(lfr_geometry=lfr_geometry,
                           radius=un_ew_opt_radius * ones(n_hel),
                           nbr_pts=nbr_pts)

    designs = [ns_sun_ref, ew_sun_ref, spec_ref, boito_design, ns_or_lfr, ew_or_lfr, ns_un_lfr, ew_un_lfr]
    keys = ['ns_sun_ref', 'ew_sun_ref', 'spec_ref', 'boito', 'ns_or', 'ew_or', 'ns_un', 'ew_un']
    labels = ['NS sun reference', 'EW sun reference', 'Specific reference',
              'Boito design', 'NS OR design', 'EW OR design', 'NS UN design', 'EW UN design']

    assert len(designs) == len(keys) == len(labels), 'Number of designs differ of the labels and/or keys'

    file_full_path = Path(file_path, f'{file_name}.json')
    if not file_full_path.is_file():

        out_dic = {}
        with open(acceptance_file, encoding='utf-8') as data_file:
            acc_results = json.loads(data_file.read())

        for k, d, lb in tqdm(zip(keys, designs, labels), total=len(keys)):

            print(f'Starting calculations for {lb}!')
            transversal_data = d.optical_analysis2(cum_eff=cum_eff)[0]
            eta_trn_function = interp1d(x=transversal_data.T[0], y=transversal_data.T[1], kind='linear')

            angles = deepcopy(acc_results['angles'])
            angles.insert(0, -90)
            angles.insert(len(angles), 90)

            acceptance_data = deepcopy(acc_results[k])
            acceptance_data.insert(0, 0)
            acceptance_data.insert(len(acceptance_data), 0)
            acc_function = interp1d(x=angles, y=acceptance_data, kind='linear')

            if k == 'boito':
                ns_acc = (acc_function(ns_theta_t) * eta_trn_function(ns_theta_t)).dot(dni) / eta_trn_function(
                    ns_theta_t).dot(dni)

                ew_acc = (acc_function(ew_theta_t) * eta_trn_function(ew_theta_t)).dot(dni) / eta_trn_function(
                    ew_theta_t).dot(dni)

                out_dic['ns_boito'] = ns_acc
                out_dic['ew_boito'] = ew_acc
            else:
                if 'ew' in k:
                    theta_t = ew_theta_t
                else:
                    theta_t = ns_theta_t

                avg_acc = \
                    (acc_function(theta_t) * eta_trn_function(theta_t)).dot(dni) / eta_trn_function(theta_t).dot(dni)

                out_dic[k] = avg_acc

        with open(file_full_path, 'w') as jsonfile:
            json.dump(dic2json(out_dic), jsonfile)
            
    else:
        with open(file_full_path, encoding='utf-8') as data_file:
            out_dic = json.loads(data_file.read())

    return out_dic

########################################################################################################################
########################################################################################################################
