"""
Set of functions created by André Santos (andrevitoras@gmail.com / avas@uevora.pt)
The codes are related to SolTrace commands and scripts to run ray tracing simulations.

They are related to writing a script file (.lk) which configures, trace rays and export the simulation results.
They include all steps needed to perform a ray trace simulation: (1) Sun configuration (rays source), (2) surfaces
optical properties, (3) geometrical elements definitions,(4) rays configurations, (5) export results.
"""

import json
import os
import subprocess
from pathlib import Path

from numpy import array, round, cross
from pandas import DataFrame

from datatable import fread

# from niopy.plane_curves import PlaneCurve
# from scopy.linear_fresnel import Heliostat, OpticalProperty, Absorber
# from scopy.sunlight import Sun

soltrace_paths = {'strace': Path('C:\\SolTrace\\3.1.0\\x64\\strace.exe'),
                  'soltrace2': Path('C:\\SolTrace\\2012.7.9\\SolTrace.exe'),
                  'soltrace3': Path('C:\\SolTrace\\3.1.0\\x64\\soltrace.exe')}

# The 'soltrace' key refers to the path of the version 3.1.0, available in the NREL website. This version is a GUI which
# cannot run an LK script from the prompt. The 'soltrace2' key refers to an GUI old version (2012.7.9) of Soltrace
# which can run an LK script from the prompt -- this version was sent by Thomas Fasquelle. Finally, the 'soltrace3' is a
# CLI version that can only run stinput files from the prompt following a precise command structure.
# For more details, please see the Notion documentation.

# ToDo: Check the variables in the implementation of glass covers and tubes

#######################################################################################################################
# Classes, Objects, and Methods #######################################################################################


class SoltraceSun:

    def __init__(self, sun_dir: array, profile: str, size: float, user_data=None):

        self.vector = sun_dir

        if profile == 'gaussian' or profile == 'g':
            self.shape = "'g'"
            self.sigma = size
        elif profile == 'pillbox' or profile == 'p':
            self.shape = "'p'"
            self.sigma = size
        elif profile == 'collimated':
            self.shape = "'p'"
            self.sigma = 0.5
        elif profile == 'user' or profile == 'u':
            self.shape = "'u'"
            self.values = user_data
        else:
            raise f"Please, input a valid profile: 'gaussian', 'pillbox', or 'user' with its list of values"

    def as_script(self, script):
        """
        :param script: A file object to append the LK lines of code.
        :return: This method writes the lines of code in an LK script to define a SolTrace Sun
        """
        # By definition, this function does not consider a point source at a finite distance, neither uses Latitude,
        # Day, and Hour option -- this is why 'ptsrc'=false, and 'useldh' = false, respectively.

        script.write("// ---- Setting the Sun (source of rays) options  -----------------------\n\n")

        script.write('sunopt({\n')
        script.write(f"'useldh'=false, 'ptsrc'=false,\n")
        script.write(f"'x'={self.vector[0]}, 'y'={self.vector[1]}, 'z'={self.vector[2]},\n")
        # Set the sun radiance profile and its size (standard deviation or half-width)
        if self.shape != "'u'":
            script.write(f"'shape'={self.shape}, 'sigma'={self.sigma}" + "});\n")  # not a used defined sun shape
        else:
            # A user defined sun profile (e.g., Buie's sun shape)
            script.write(f"'shape'={self.shape},\n")
            # The value used for this sun shape. It should be as a Python list to print it correctly in the script file.
            script.write(f"'userdata'={list(self.values)}" + "});\n")

        script.write("// -----------------------------------------------------------------------\n\n")

    def as_stinput(self, file):
        """
        :param file: The stinput file in which the code will be written
        :return: This function writes the lines of code related to a stinput file which configures a SolTrace Sun
        """

        if self.shape == "'g'":
            file.write(f"SUN \t PTSRC  0 \t SHAPE g \t SIGMA {self.sigma} \t HALFWIDTH 0\n")
            file.write(f"XYZ {self.vector[0]} {self.vector[1]} {self.vector[2]} \t USELDH 0 \t LDH 0 0 0\n")
            file.write(f"USER SHAPE DATA 0\n")
        elif self.shape == "'p'":
            file.write(f"SUN \t PTSRC  0 \t SHAPE p \t SIGMA 0 \t HALFWIDTH  {self.sigma}\n")
            file.write(f"XYZ {self.vector[0]} {self.vector[1]} {self.vector[2]} \t USELDH 0 \t LDH 0 0 0\n")
            file.write(f"USER SHAPE DATA 0\n")
        elif self.shape == "'u'":
            file.write(f"SUN \t PTSRC  0 \t SHAPE d \t SIGMA 0 \t HALFWIDTH  0\n")
            file.write(f"XYZ {self.vector[0]} {self.vector[1]} {self.vector[2]} \t USELDH 0 \t LDH 0 0 0\n")
            file.write(f"USER SHAPE DATA {len(self.values)}\n")
            for v in self.values:
                file.write(f"{v[0]}\t{v[1]}\n")
        else:
            raise 'Error in the inputted SolTrace Sun (Source of rays) data'


class OpticInterface:
    """
    An OpticInterface object is the main element of an Optical Surface, and is the core element of the Optics box.
    It defines the front or back side of an OpticalSurface object. Its instances are related to the input data needed.

    Its methods implement the lines of code for both a LK script or STINPUT file
    """

    # ToDo: Check how to implement an angular variable reflectivity

    def __init__(self, name: str, reflectivity: float, transmissivity: float, slp_error: float, spec_error: float,
                 front=True, real_refractive_index=1.0, img_refractive_index=1.2):
        self.name = name
        self.ref = round(reflectivity, 6)
        self.tran = round(transmissivity, 6)
        self.err_slop = slp_error
        self.err_spec = spec_error
        self.n_real = real_refractive_index
        self.n_img = img_refractive_index
        self.side = 1 if front else 2
        self.ap_stop = 3  # static value from SolTrace GUI. I do not know what this parameter means.
        self.surf_num = 1  # static value from SolTrace GUI. I do not know what this parameter means.
        self.diff_ord = 4  # static value from SolTrace GUI. I do not know what this parameter means.
        self.grt_cff = [1.1, 1.2, 1.3, 1.4]  # static value from SolTrace GUI. I do not know what this parameter means.

        self.st_parameters = [0] * 15  # useful to construct the stinput code lines

    def as_script(self, script):
        script.write("// ---- Set Surface Property  -------------------\n")
        # Adds a surface optical property with a given name
        script.write(f"opticopt('{self.name}', {self.side}, " + "{'dist'='g',\n")
        script.write(f"'refl'={self.ref}, 'trans'={self.tran}, 'errslope'={self.err_slop},'errspec'={self.err_spec},\n")
        script.write(f"'refractr'={self.n_real}, 'refracti'={self.n_img}, 'apstop'={self.ap_stop},\n")
        script.write(f"'difford'={self.diff_ord}, 'grating'={list(self.grt_cff)}" + "});\n")
        script.write("// --------------------------------------------\n\n")

    def as_stinput(self, file):
        self.st_parameters[0:3] = self.ap_stop, self.surf_num, self.diff_ord
        self.st_parameters[3:7] = self.ref, self.tran, self.err_slop, self.err_spec
        self.st_parameters[7:9] = self.n_real, self.n_img
        self.st_parameters[9:13] = self.grt_cff

        file.write(f"OPTICAL.v2\tg")
        for p in self.st_parameters:
            file.write(f"\t{p}")
        file.write(f"\n")


class OpticalSurface:
    """
    An OpticalSurface object is the main element of the Optics box of SolTrace. Basically, it must have a name, and the
    properties of its front and back sides.

    The methods implement the correspondent lines of code for both a LK script or STINPUT file
    """

    def __init__(self, name: str, front_side: OpticInterface, back_side: OpticInterface):
        self.name = name
        self.front = front_side
        self.back = back_side

    def as_script(self, script):
        script.write("// ---- Add Surface Property -----------------------------------------------\n\n")
        script.write(f"addoptic('{self.name}');\n")
        self.front.as_script(script=script)
        self.back.as_script(script=script)

        script.write("// -------------------------------------------------------------------------\n\n")

    def as_stinput(self, file):
        file.write(f"OPTICAL PAIR\t{self.name}\n")
        self.front.as_stinput(file=file)
        self.back.as_stinput(file=file)


class Optics:
    """
    The class Optics represents the "Optics" box of SolTrace. It should contain a list of OpticalProperties objects to
    be included in a LK script or in a STINPUT file.
    """

    def __init__(self, properties: list):
        self.properties = properties

    def as_script(self, script):

        script.write(f"// ---- Setting the Optics box ----------------------------------------------------------- \n\n")
        script.write(f"clearoptics();\n")

        for prop in self.properties:
            prop.as_script(script=script)

        script.write(f"// --------------------------------------------------------------------------------------- \n\n")

    def as_stinput(self, file):

        file.write(f"OPTICS LIST COUNT    {len(self.properties)}\n")
        for prop in self.properties:
            prop.as_stinput(file=file)


class Element:

    def __init__(self, name: str, orig: array, aim_vec: array, z_rot: float, aperture: list, surface: list,
                 optic: OpticalSurface, reflect=True, enable=True):

        self.name = name
        self.x, self.y, self.z = orig
        self.ax, self.ay, self.az = aim_vec
        self.z_rot = z_rot
        self.aperture = aperture
        self.surface = surface
        self.optic_name = optic.name
        self.interaction = "Reflection" if reflect else "Refraction"
        self.en = 'true' if enable else 'false'

        self.EN = 1 if enable else 0
        self.INT = 2 if reflect else 1

        # self.st_parameters = list(round(orig, 6)) + list(round(aim_vec, 6))
        self.st_parameters = list(orig) + list(aim_vec) + list([z_rot]) + self.aperture

    def as_script(self, script, el_index: int):
        script.write(f"// -- Add an element to the current stage -----------------\n\n")
        script.write(f"addelement();\n")  # It appends a new element in the current stage
        script.write(
            f"elementopt({el_index}, " + "{" + f"'en'={self.en}, " + f"'x'={self.x}, 'y'={self.y}, 'z'={self.z},\n")
        script.write(f"'ax'={self.ax}, 'ay'={self.ay}, 'az'={self.az}, 'zrot'={self.z_rot},\n")
        script.write(f"'aper'={self.aperture},\n")
        script.write(f"'surf'={self.surface}, 'interact'='{self.interaction}',\n")
        script.write(f"'optic'='{self.optic_name}', 'comment'='{self.name}'" + "});\n")
        script.write(f"// --------------------------------------------------------\n\n")

    def as_stinput(self, file):

        file.write(f"{self.EN}")
        for p in self.st_parameters:
            file.write(f"\t{p}")

        if len(self.surface) != 2:
            for p in self.surface:
                file.write(f"\t{p}")
            file.write(f"\t\t{self.optic_name}")
        else:
            s, path = self.surface
            surface_parameters = list([s]) + list([0] * 8)

            for p in surface_parameters:
                file.write(f"\t{p}")

            file.write(f"\t{path}")
            file.write(f"\t{self.optic_name}")

        file.write(f"\t{self.INT}")
        file.write(f"\t{self.name}\n")


class Stage:

    def __init__(self, name: str, elements: list, orig=array([0, 0, 0]), aim_vec=array([0, 0, 1]), z_rot=0, active=True,
                 virtual=False, rays_multi_hit=True, trace_through=False):

        for el in elements:
            if type(el) != Element:
                raise "A non Element object was added in the elements argument of the Stage instance"

        self.name = name
        self.x, self.y, self.z = orig
        self.x_aim, self.y_aim, self.z_aim = aim_vec
        self.z_rot = z_rot
        self.virtual = 'true' if virtual else 'false'
        self.multi_hit = 'true' if rays_multi_hit else 'false'
        self.trace_through = 'true' if trace_through else 'false'
        self.active = active
        self.elements = elements

        self.st_parameters = list(orig) + ['AIM'] + list(aim_vec) + ['ZROT', z_rot]
        VT = 1 if virtual else 0
        MH = 1 if rays_multi_hit else 0
        TT = 1 if trace_through else 0

        self.st_parameters += ['VIRTUAL', VT] + ['MULTIHIT', MH] + ['ELEMENTS', len(elements)] + ['TRACETHROUGH', TT]

    def as_script(self, script):

        if len(self.elements) == 0:
            raise "The Stage has no elements to interact with the rays. Please, add Elements to the elements argument"

        script.write("// ---- Setting a Stage for the Geometries to be added --------------\n\n")
        script.write(f"addstage('{self.name}');\n")
        script.write(f"stageopt('{self.name}', " + "{" + f"'x'={self.x}, 'y'={self.y}, 'z'={self.z},\n")
        script.write(f"'ax'={self.x_aim}, 'ay'={self.y_aim}, 'az'={self.z_aim}, 'zrot'={self.z_rot},\n")
        script.write(f"'virtual'={self.virtual}, 'multihit'={self.multi_hit}, 'tracethrough'={self.trace_through}")
        script.write("});\n")

        if self.active:
            script.write(f"activestage('{self.name}');\n")

        for i, el in enumerate(self.elements):
            el.as_script(script=script, el_index=i)

        script.write("// -----------------------------------------------------------------\n\n")

    def as_stinput(self, file):
        file.write(f"STAGE\tXYZ")
        for p in self.st_parameters:
            file.write(f"\t{p}")
        file.write("\n")
        file.write(f"{self.name}\n")

        for el in self.elements:
            el.as_stinput(file=file)


class Geometry:

    def __init__(self, stages: list):
        self.stages = stages

    def as_script(self, script):
        script.write(f"// ---- Setting the Geometry box --------------------------------------------------------- \n\n")
        script.write(f"clearstages();\n")

        for stg in self.stages:
            stg.as_script(script=script)

        script.write(f"// --------------------------------------------------------------------------------------- \n\n")

    def as_stinput(self, file):
        file.write(f"STAGE LIST COUNT \t{len(self.stages)}\n")

        for stg in self.stages:
            stg.as_stinput(file=file)


class Trace:

    """
    This class attributes holds all data needed to configurate and trace the SolTrace simulation.
    """

    def __init__(self, rays: int, cpus: int, seed=-1, sunshape=True, optical_errors=True, point_focus=False):

        self.rays = int(rays)
        self.max_rays = 100 * self.rays
        self.cpus = int(cpus)
        self.sunshape = 'true' if sunshape else 'false'
        self.errors = 'true' if optical_errors else 'false'
        self.seed = seed
        self.point_focus = 'true' if point_focus else 'false'

    def as_script(self, script, simulate=True):
        script.write('// ---- Setting the Ray Tracing simulation -------------------------------------------------\n\n')

        script.write('traceopt({ \n')
        script.write(f"'rays'={self.rays}, 'maxrays'={self.max_rays},\n")
        script.write(f"'include_sunshape'={self.sunshape}, 'optical_errors'={self.errors},\n")
        script.write(f"'cpus'={self.cpus}, 'seed'={self.seed}, 'point_focus'={self.point_focus}" + "});\n")

        if simulate:
            script.write('trace();\n')

        script.write(f"//-----------------------------------------------------------------------------------------\n\n")


class ElementStats:

    def __init__(self, stats_name: str, stage_index: int, element_index: int,
                 dni=1000, x_bins=15, y_bins=15, final_rays=True):

        """
        This class holds the data to select an element from a particular stage, and collect its stats (flux) information
        and then export it to an external json file.

        This json file is easily read as a Python dictionary.

        :param stage_index: The index number of the Stage
        :param element_index: The index number of the element within the Stage
        :param stats_name: The name of the json file to be exported with the element stats.

        :param dni: The solar direct normal irradiance, in W/m2
        :param x_bins: The number of grid elements in the x-axis to split the element surface.
        :param y_bins: The number of grid elements in the y-axis to split the element surface.

        """

        self.name = stats_name
        self.stg_index = abs(stage_index)
        self.ele_index = abs(element_index)

        self.dni = abs(dni)
        self.x_bins = abs(x_bins)
        self.y_bins = abs(y_bins)
        self.final_rays = 'true' if final_rays else 'false'

        self.file_full_path = None

    def as_script(self, script, file_path: Path, soltrace_version=2012):

        """
        This method writes the script lines to collect the information of an Element in a particular Stage and then
        export this to a json file.

        At the end, it also returns the full path of the exported json file.

        Furthermore, it cannot be written as script commands for a '.stinput' file but just as a '.lk' file.

        :param script: A script file to append the lines of code.
        :param file_path: The path of the file to be exported with the flux results.
        :param soltrace_version:

        :return It returns the full path of the exported json file with the element stats.

        """

        script.write('//---- Setting the results outputs ----------------------------------------\n\n')
        # Set the source irradiance to compute flux calculations. A standard value of 1000 W/m2 was chosen.
        script.write(f"absorber_data = elementstats({self.stg_index}, {self.ele_index}, ")
        script.write(f"{self.x_bins}, {self.y_bins}, {self.dni}, {self.final_rays});\n\n")

        # Setting the SolTrace cwd as the file_path.
        script.write("cwd('" + str([str(file_path)])[2:-2] + "');\n")

        # For the SolTrace 3.1.0 version #############################################################
        # SolTrace 2012 version does not have the 'json_file()' function #############################
        if soltrace_version != 2012:
            script.write(f"json_file('{self.name}_stats.json', absorber_data);\n")
            script.write(f"//--------------------------------------------------------------------\n\n")

        # For the 2012 SolTrace version ##############################################################
        else:
            # Writing a json file to export the element stats
            script.write(f"filename = '{self.name}_stats.json';\n")
            script.write("stats_file = open(filename, 'w');\n")

            # The starting-braces to define a Python dictionary in the json file
            script.write("write_line(stats_file, '{');\n\n")

            # Selecting some keys from the LK tabel 'absorber_data' previously defined. These are float data keys.
            # These keys will be exported to the json file.
            script.write('float_keys = ["min_flux", "bin_size_x", "power_per_ray", "bin_size_y", "peak_flux",\n')
            script.write('"sigma_flux", "peak_flux_uncertainty", "uniformity", "ave_flux",\n')
            script.write('"ave_flux_uncertainty", "radius", "num_rays"];\n')

            # Writing the LK code to write in the json file the keys and values of the LK-Table ###############
            # Each key is writen in a line.
            script.write("for (i=0; i<#float_keys; i++)\n")
            script.write("{\n")
            script.write("write_line(stats_file, " + "'" + '"' + "'" + ' + float_keys[i] + ' + "'" + '": ' + "'")
            script.write(" + absorber_data{float_keys[i]} + ',');\n")
            script.write("}\n\n")

            # Writing the LK code to write in the json file the vector keys and values of the LK-Table ###############
            # Each vector-key is writen in a line.
            script.write("vector_keys = ['centroid', 'xvalues', 'yvalues'];\n")
            script.write("for (i=0; i<#vector_keys; i++)\n")
            script.write("{\n")
            script.write("write_line(stats_file, " + "'" + '"' + "'" + ' + vector_keys[i] + ' + "'" + '": [' + "'")
            script.write(" + absorber_data{vector_keys[i]} + '],');\n")
            script.write("}\n\n")

            # Writing the LK code to write the 'flux' key of the elementstats LK-Table.
            script.write("rays_bins = absorber_data{'flux'};\n")
            script.write("for (i = 0; i < #rays_bins; i++)")
            script.write("{\n")

            script.write("\tif (i==0)\n")
            script.write("\t{\n")
            script.write('\tstring_to_write = ' + "'" + '"flux": [[' + "' + rays_bins[i] + '],';\n")
            script.write("\twrite(stats_file, string_to_write, #string_to_write);\n")
            script.write("\t}\n")

            script.write("\telseif (i > 0 && i < #rays_bins - 1)\n")
            script.write("\t{\n")
            script.write("\tstring_to_write = '[' + rays_bins[i] + '],';\n")
            script.write("\twrite(stats_file, string_to_write, #string_to_write);\n")
            script.write("\t}\n")

            script.write("\telse\n")
            script.write("\t{\n")
            script.write("\tstring_to_write = '[' + rays_bins[i] + ']]\\n';\n")
            script.write("\twrite(stats_file, string_to_write, #string_to_write);\n")
            script.write("\t}\n")
            script.write("}\n")

            # The end-braces to define the Python dictionary and closing the exported file from the Soltrace.
            script.write("write_line(stats_file, '}');\n")
            script.write("close(stats_file);\n")
            script.write(f"//---------------------------------------------------------------------\n\n")

        element_stats_full_path = Path(file_path, f"{self.name}_stats.json")

        self.file_full_path = element_stats_full_path

        return element_stats_full_path
        ####################################################################################################

########################################################################################################################
########################################################################################################################

########################################################################################################################
# Sun functions ########################################################################################################

# def sun2soltrace(sun: Sun):
#
#     return SoltraceSun(sun_dir=sun.sun_vector, profile=sun.sun_shape.profile, size=sun.sun_shape.size)


########################################################################################################################
########################################################################################################################

########################################################################################################################
# Optics functions #####################################################################################################


def reflective_surface(name: str, rho: float, slope_error: float, spec_error: float):

    front = OpticInterface(name=name, reflectivity=rho, transmissivity=0, slp_error=slope_error, front=True,
                           spec_error=spec_error)
    back = OpticInterface(name=name, reflectivity=0, transmissivity=0, slp_error=0, spec_error=0, front=False)

    return OpticalSurface(name=name, front_side=front, back_side=back)


def absorber_surface(name: str, alpha: float):

    front = OpticInterface(name=name, reflectivity=1 - alpha, transmissivity=0, slp_error=0, spec_error=0)
    back = OpticInterface(name=name, reflectivity=1, transmissivity=0, slp_error=0, spec_error=0, front=False)

    return OpticalSurface(name=name, front_side=front, back_side=back)


def transmissive_surface(name: str, tau: float, nf: float, nb: float):

    front = OpticInterface(name=name, reflectivity=1 - tau, transmissivity=tau, slp_error=0, spec_error=0, front=True,
                           real_refractive_index=nf)
    back = OpticInterface(name=name, reflectivity=1 - tau, transmissivity=tau, slp_error=0, spec_error=0, front=False,
                          real_refractive_index=nb)

    return OpticalSurface(name=name, front_side=front, back_side=back)


def glass_cover_surfaces(tau: float, name='glass cover', refractive_index=1.52):

    if tau > 1:
        raise "An invalid value (> 1) for the transmissivity was inputted"
    else:
        tau_s = tau ** 0.5

    if refractive_index < 1:
        raise "An invalid value (< 1) for the refractive index was inputted"
    else:
        n_index = refractive_index

    out_surf = transmissive_surface(name=f'outer_{name}', tau=tau_s, nf=1, nb=n_index)
    inn_surf = transmissive_surface(name=f'inner_{name}', tau=tau_s, nf=n_index, nb=1)

    return out_surf, inn_surf


perfect_mirror = reflective_surface(name='perfect_mirror', rho=1, slope_error=0, spec_error=0)
perfect_absorber = absorber_surface(name='perfect_absorber', alpha=1)


# def reflector2Surface(prop: OpticalProperty.reflector):
#
#     front_interface = OpticInterface(name=f'{prop.name}_front', reflectivity=prop.rho, transmissivity=0,
#                                      slp_error=prop.slope_error, spec_error=prop.spec_error, front=True)
#     back_interface = OpticInterface(name=f'{prop.name}_back', reflectivity=0, transmissivity=0,
#                                     slp_error=0, spec_error=0, front=False)
#
#     return OpticalSurface(name=prop.name, front_side=front_interface, back_side=back_interface)
#
#
# def absorber2Surface(prop: OpticalProperty.absorber):
#
#     front_interface = OpticInterface(name=f'{prop.name}_front', reflectivity=1 - prop.alpha,
#                                      transmissivity=0, slp_error=0, spec_error=0)
#     back_interface = OpticInterface(name=f'{prop.name}_back', reflectivity=1,
#                                     transmissivity=0, slp_error=0, spec_error=0, front=False)
#
#     return OpticalSurface(name=prop.name, front_side=front_interface, back_side=back_interface)

#######################################################################################################################
#######################################################################################################################

########################################################################################################################
# Geometry functions ###################################################################################################


# def create_absorber_tube(name: str, radius: float, center: array, aim: array, optic: OpticalSurface, length=60000):
#
#     # Converting the units to meters
#     R = radius / 1000
#     L = length / 1000
#     hc = center / 1000
#     v_aim = aim / 1000
#     ################################
#
#     # Setting the aperture ###########
#     aperture = list([0] * 9)
#     aperture[0], aperture[3] = 'l', L
#     ##################################
#
#     # Setting the surface ############
#     surface = list([0] * 9)
#     surface[0], surface[1] = 't', 1/R
#     ##################################
#
#     elem = Element(name=name, orig=hc, aim_vec=v_aim, z_rot=0,
#                    aperture=aperture, surface=surface,
#                    optic=optic, reflect=True)
#
#     return elem


# def create_flat_absorber(name: str, geometry: Absorber.flat, optic: OpticalSurface, length=60000):
#
#     # Converting units to meters ###
#     w = geometry.width / 1000
#     L = length / 1000
#     hc = array([geometry.center[0], 0, geometry.center[-1]]) / 1000
#     ################################
#
#     # Setting the aperture #########
#     aperture = list([0] * 9)
#     aperture[0:3] = 'r', w, L
#     ################################
#
#     # Setting the surface ##########
#     surface = list([0] * 9)
#     surface[0] = 'f'
#     ################################
#
#     elem = Element(name=name, orig=hc, aim_vec=array([0, 0, 1]), z_rot=0,
#                    aperture=aperture, surface=surface,
#                    optic=optic, reflect=True)
#
#     return elem


# def create_primary_mirror(hel: Heliostat, name: str, aim_pt: array, length: float,
#                           sun: SoltraceSun, optic: OpticalSurface,
#                           par_approx=True, file_path=None):
#
#     center = hel.ZX_center / 1000
#     aim_vec = hel.aim_vector(aim_point=aim_pt, SunDir=sun.vector) / 1000
#     L = length / 1000  # convert to meters
#     aperture = list([0] * 9)
#     surface = list([0] * 9)
#
#     if hel.shape == 'flat':
#         aperture[0:3] = 'r', hel.width / 1000, L
#         surface[0] = 'f'
#     else:
#         if par_approx:
#             aperture[0:3] = 'r', hel.width / 1000, L
#             rc = hel.radius / 1000  # convert to meters
#             # parabola gradient -- as defined in SolTrace
#             c = 1 / rc
#             surface[0:2] = 'p', c
#         else:
#             if file_path is None:
#                 raise "The .csi filepath for the surface must be inputted"
#             else:
#                 curve = hel.as_plane_curve()
#                 x1 = round(curve.x_t[0] / 1000, 6) + 0.0000001
#                 x2 = round(curve.x_t[-1] / 1000, 6) - 0.0000001
#                 aperture[0:4] = 'l', x1, x2, L
#
#                 spline_surface_file(file_path=file_path, file_name=name, curve=curve)
#                 full_file_path = Path(file_path, f"{name}.csi")
#                 surface[0:2] = 'i', str(full_file_path)
#                 surface = surface[0:2]
#
#     el = Element(name=name, orig=center, aim_vec=aim_vec, z_rot=0,
#                  aperture=aperture, surface=surface, optic=optic, reflect=True)
#     return el

########################################################################################################################
########################################################################################################################

#######################################################################################################################
# Surface types functions #############################################################################################

# def spline_surface_file(file_path, file_name: str, curve: PlaneCurve):
#     # It gets the translated (to curve_center) points of the plane curve, and convert to meters.
#     x_t = curve.x_t / 1000
#     y_t = curve.y_t / 1000
#
#     # constructs the spline from the translated points and calculate the first derivative values at both edges knots
#     df = curve.spline(centered=True).derivative()
#     df_1 = df(x_t[0])
#     df_n = df(x_t[-1])
#
#     # creates the surface cubic spline file (a 'csi' extension file for SolTrace to correctly read it)
#     full_file_path = Path(file_path, f"{file_name}.csi")
#
#     file = open(full_file_path, 'w')
#     file.write(f"{len(x_t)}\n")  # the first line must contain the number of points which defines the surface
#     for i in range(len(x_t)):
#         # write in the file the point coordinates values in meters
#         file.write(f"{x_t[i]} {y_t[i]}\n")
#
#     # the last line should contain the first derivatives at both edge knots.
#     file.write(f"{df_1} {df_n}")  # writes the first derivatives at both edges knots
#     file.close()  # closes the file
#
#     return full_file_path

########################################################################################################################
# Script functions #####################################################################################################


def script_change_element_aim(script, element_index: int, aim_vector: array):

    ax, ay, az = aim_vector
    script.write(f"elementopt({element_index}, " + "{" + f"'ax'={ax}, 'ay'={ay}, 'az'={az}" + "});\n")

    pass

########################################################################################################################
########################################################################################################################

########################################################################################################################
# Files functions ######################################################################################################


def create_script(file_path, file_name='optic'):
    """
    This function creates a lk script file with the inputted name. This file will be created at the current work
    directory if a full path is not inputted at the file_name parameter -- the standard directory in Python.

    :param file_path: THe Path where the script should be created
    :param file_name: The name of the lk script file to be created
    :return: It returns an opened LK file where the lines of code will be writen.
    """

    full_file_path = Path(file_path, f"{file_name}.lk")
    script = open(full_file_path, 'w')
    script.write('// ----------------- This set of commands will configure a SolTrace LK script ----------------\n\n\n')

    return script


def soltrace_script(file_path: Path, file_name: str, sun: SoltraceSun, optics: Optics, geometry: Geometry,
                    trace: Trace, stats: ElementStats):

    if not file_path.is_dir():
        os.makedirs(file_path)

    script = create_script(file_path=file_path, file_name=file_name)

    sun.as_script(script=script)
    optics.as_script(script=script)
    geometry.as_script(script=script)
    trace.as_script(script=script, simulate=True)
    stats.as_script(script=script, file_path=file_path)
    script.close()

    script_full_path = Path(file_path, f"{file_name}.lk")

    return script_full_path


def run_soltrace(lk_file_full_path: Path):
    """
    :param lk_file_full_path: The full path of the LK file.
    :return: This function opens the SolTrace version from 2012 and runs the LK script.
    """
    soltrace = soltrace_paths['soltrace2']
    cmd = f"{soltrace}" + ' -s ' + f"{lk_file_full_path}"
    subprocess.run(cmd)


def read_element_stats(full_file_path: Path):
    """
    :param full_file_path: The full path of the json element stats file.
    :return: A Python dictionary with the element stats.
    """

    with open(full_file_path, encoding='utf-8') as data_file:
        stats = json.loads(data_file.read())

    return stats


def create_stinput(file_path, file_name='optic'):
    full_file_path = Path(file_path, f"{file_name}.stinput")
    file = open(full_file_path, 'w')
    file.write("# SOLTRACE VERSION 3.1.0 INPUT FILE\n")

    return file


def soltrace_file(file_path: Path, file_name: str, sun: SoltraceSun, optics: Optics, geometry: Geometry):

    if not file_path.is_dir():
        os.makedirs(file_path)

    st_file = create_stinput(file_path=file_path, file_name=file_name)

    sun.as_stinput(file=st_file)
    optics.as_stinput(file=st_file)
    geometry.as_stinput(file=st_file)

    st_file.close()

    full_file_path = Path(file_path, f'{file_name}.stinput')

    return full_file_path


def run_strace(file_path, trace: Trace, strace_path='C:\\SolTrace\\3.1.0\\x64\\strace.exe'):

    """
    :param file_path: Full path of a .stinput file to be run.
    :param trace: An object Trace. It contains the data to run the simulations.
    :param strace_path: Full path of the SolTrace CLI version, i.e., the strace.exe file.

    :return: This function runs the CLI version of the SolTrace and also runs the .stinput file. At the end, it creates
    a ray data file -- as the CSV file that can be saved from the SolTrace GUI version.
    """

    ss = 1 if trace.sunshape == 'true' else 0
    er = 1 if trace.errors == 'true' else 0
    pf = 1 if trace.point_focus == 'true' else 0

    if file_path.is_file():
        cmd = f'{strace_path} {str(file_path)} {trace.rays} {trace.max_rays} {trace.seed} {ss} {er} {pf}'
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
    else:
        print("The path passed as argument is not of a file")


#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Rays File ###########################################################################################################


class RaysFile:

    def __init__(self, source_data: dict, rays_data: DataFrame):
        self.source_area = source_data['area']
        self.number_of_rays = source_data['number_of_rays']
        self.rays_data = rays_data

    def flux(self, stage: int, element: int, DNI=1000):
        power_per_ray = self.source_area * DNI / self.number_of_rays
        hit_rays = \
            self.rays_data[(self.rays_data.stage == str(stage)) & (self.rays_data.element == str(-element))].count()[0]
        flux = hit_rays * power_per_ray

        return flux


def read_RaysFile(file_path):
    """
    :param file_path: A .rays file to be interpreted. It is the output of a ray tracing simulation in strace
    (SolTrace CLI version)
    :return:
    """

    if str(file_path)[-5:] == '.rays':
        file = file_path
    else:
        file = Path(f'{str(file_path)}.rays')

    data = fread(file).to_pandas()
    source_data = list(map(float, data.iloc[0]))

    x_min, x_max, y_min, y_max, number_of_rays = source_data[0:5]
    s1, s2, s3, s4 = array([x_max, y_max]), array([x_min, y_max]), array([x_min, y_min]), array([x_max, y_min])
    source_area = abs(cross(s4 - s1, s2 - s1))

    source = {'area': source_area, 'number_of_rays': number_of_rays}

    headers = ['stage', 'element', 'ray']
    rays = DataFrame()
    for i, k in enumerate(data.keys()[6:-1]):
        rays.insert(loc=i, column=headers[i], value=data[k][2:])

    return RaysFile(source_data=source, rays_data=rays)


########################################################################################################################
########################################################################################################################
