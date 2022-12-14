"""
Author: André Santos (andrevitoas@gmail.com / avas@uevora.pt)

Functions related to the analysis of the data from the Reduced Database of sun shape measurements
from the Lawrence Berkeley Laboratory (RDB-LBL).
The Reader must see the bellow reference for a further explanation:

Noring, J.E., Grether, D.F., Hunt, A.J., 1991.
Circumsolar Radiation Data: The Lawrence Berkeley Laboratory Reduced Data Base,
Final Subcontract Report. https://doi.org/10.2172/6125786.

The 11 ASCII files with the recorded data can be downloaded from:
https://www.nrel.gov/grid/solar-resource/circumsolar.html.
"""

import time
from tqdm import tqdm
from numpy import array, zeros, pi
from scipy.integrate import quad
from scipy.interpolate import interp1d


class Sunshape:
    # This class creates an object Sunshape which has the main information about the sun shape profiles
    # recorded within the context of the reduced database of the Lawrence Berkeley Laboratory (RDB-LBL) measurements.

    def __init__(self, SunS: dict):
        # save the raw dictionary data as extracted and read from the ASCII files
        self.raw_data = SunS

        # instances for the city database, the profile record number, and the error flags.
        self.city = SunS['City']
        self.index = SunS['ProfileNumber']
        self.flags = DetectFlags(SunS)

        # an instance for the acr reading recorded at the sun shape profile measurement
        self.acr = SunS['ACR']

        # The angular steps of the disk intensity is 1.5' (arc minutes), and a total of 20 points.
        # The angular steps of circumsolar intensity is 4.5' (arc minutes), and a total of 36 points.
        # However, Intensity values are related to interval mid point.
        # For instance, the first intensity value is for the interval between 0 and 1.5', i.e., for radius 0.75'.
        # An example of that can be found in Rabl (1985, Appendix E, Table E2, pp. 485-486)
        # instances for the disk and circumsolar angular radius from the sun center

        self.disk_radius = [0.75 + i * 1.5 for i in range(20)]  # in arc minutes
        self.circumsolar_radius = [0.5 * (30 + 34.5) + i * 4.5 for i in range(36)]  # in arc minutes
        # creates the whole array of sun radius (disk + circumsolar) and converts it to radians.
        self.radius = array(self.disk_radius + self.circumsolar_radius) * 0.00029088820866572

        # instances for the RDB calculated values for disk and circumsolar irradiances.
        self.rdb_disk_irradiance = SunS['DiskRad']
        self.rdb_circumsolar_irradiance = SunS['CircumsolarRad']
        # the total irradiance from the RDB data of disk and circumsolar irradiances
        self.rdb_irradiance = self.rdb_disk_irradiance + self.rdb_circumsolar_irradiance
        # an instance for the RDB calculated circumsolar ratio: CSR = C / (C + S)
        self.rdb_csr = SunS['CSR']

        # instances for the radiance (the so-called "intensity") values recorded in the sun shape measurements
        self.disk_intensity = [float(val) for val in SunS['DiskIntensity']]
        self.circumsolar_intensity = [float(val) for val in SunS['CircumsolarIntensity']]
        # an instance for the whole array of radiance values
        self.intensity = array(self.disk_intensity + self.circumsolar_intensity)

    def __str__(self):
        return f'This is sun shape number {self.index} from {self.city} database'


def Get_LBLSunShapes(file: str):
    """
    :param file: The ASCII file for each site where the measurements were taken.
    :return: This function returns a list with all the sun shape measurement data that can be
    taken from the file. Each element of the list is a dictionary with all major measurement data.
    """
    ###########
    # As explained in Noring (1991), each sun shape measurement takes a total of 20 lines in the ASCII file.
    # The total number of sun shape is an integer value.
    # Thus, the total number of lines in the ASCII file must be a multiple of 20.

    start_time = time.time()
    Lines = open(file, mode='r').readlines()

    if len(Lines) % 20 != 0:
        print(f'The total number of lines is not a multiple of 20. The read data might not be correct. '
              f'Please, check it')

    NumSunShapes = int(len(Lines) / 20)
    SunShapes = [0] * NumSunShapes

    for i in range(NumSunShapes):

        DiskIntensity = []
        CircumsolarIntensity = []

        ACR = float(Lines[3 + i * 20][44:49])
        DiskRad = float(Lines[5 + i * 20][35:41])
        CircumsolarRad = float(Lines[5 + i * 20][52:56])
        CSR = float(Lines[5 + i * 20][68:77])
        Flags = Lines[1 + i * 20][43:77].replace(" ", "")

        for j in range(7, 11):
            DiskIntensity += Lines[j + i * 20][28:77].split()

        for j in range(11, 18):
            CircumsolarIntensity += Lines[j + i * 20][28:77].split()
        CircumsolarIntensity += Lines[18 + i * 20][28:37].split()

        SunShapeData = \
            {
                'City': file[:-4],
                'ProfileNumber': i + 1,
                'Flags': Flags,
                'ACR': ACR, 'DiskRad': DiskRad, 'CircumsolarRad': CircumsolarRad, 'CSR': CSR,
                'DiskIntensity': DiskIntensity,
                'CircumsolarIntensity': CircumsolarIntensity
            }

        SunShapes[i] = Sunshape(SunShapeData)

    print(f"Collecting the data took {round(time.time() - start_time, 1)} seconds")

    return SunShapes


def DetectFlags(SunShape: dict):
    Flags = SunShape['Flags']
    indexes = []
    for i in range(len(Flags)):
        if Flags[i] == '1':
            indexes.append(i + 1)

    return indexes


def IntegrateProfile(sun: Sunshape):

    # Creates an interpolated function from the whole data of sun radius and measured radiance values.
    # Radius range does not starts at 0, but at 0.00021825 rad (0.21825 mrad);
    # and does not ends at 3.2º, but at 0.05521725 rad (55.2 mrad).
    Int_Function = interp1d(x=sun.radius, y=sun.intensity, kind='cubic')
    # create an instance with the interpolated function for further calculations
    sun.intensity_function = Int_Function

    # integrate the intensity for the whole domain.
    sun.calc_irradiance = 2 * pi * quad(lambda x: Int_Function(x) * x, sun.radius[0], sun.radius[-1],
                                        full_output=1)[0]

    # integrate the intensity until the acceptance of the ACR pyrheliometer (2.5º ~ 43.6 mrad)
    sun.calc_irradiance_acr = 2 * pi * quad(lambda x: Int_Function(x) * x, sun.radius[0], 0.0436,
                                            full_output=1)[0]


def IntegrateDatabase(SunShapes: list):
    start_time = time.time()

    for i, ss in enumerate(tqdm(SunShapes)):
        IntegrateProfile(sun=ss)

    print(f'It is now {time.strftime("%H:%M:%S")}')
    print(f'Calculations took {round((time.time() - start_time) / 60, 1)} minutes')


def FilterDueFlag(SunShapes: list, Flag=int):
    FilteredSunShapes = []
    for i, SunS in enumerate(SunShapes):
        if Flag not in SunS.flags:
            FilteredSunShapes += [SunShapes[i]].copy()

    return FilteredSunShapes


def FilterDueClearSky(SunShapes: list):
    FilteredSunShapes = []
    for i, SunS in enumerate(SunShapes):
        signal = True
        for j in range(len(SunS.intensity) - 1):
            if SunS.intensity[j] <= SunS.intensity[j + 1]:
                signal = False
                break

        if signal:
            FilteredSunShapes += [SunShapes[i]].copy()

    return FilteredSunShapes


def FilterDueACR(SunShapes: list, acr_min=0.0):
    FilteredSunShapes = []
    for i, SunS in enumerate(SunShapes):
        if SunS.acr > acr_min:
            FilteredSunShapes += [SunShapes[i]].copy()

    return FilteredSunShapes


def ACR_Correlation(SunShapes: list):

    NbrSunShapes = len(SunShapes)
    ACR_Reading = zeros(NbrSunShapes)
    Integrated_Profile = zeros(NbrSunShapes)

    for i, SS in enumerate(SunShapes):
        Integrated_Profile[i] = SS.calc_irradiance_acr
        ACR_Reading[i] = SS.acr

    return Integrated_Profile, ACR_Reading


def Integrated_Correlation(SunShapes: list):

    NbrSunShapes = len(SunShapes)
    Integrated_RDB = zeros(NbrSunShapes)
    Integrated_Profile = zeros(NbrSunShapes)

    for i, ss in enumerate(SunShapes):
        Integrated_Profile[i] = ss.calc_irradiance
        Integrated_RDB[i] = ss.rdb_irradiance

    return Integrated_Profile, Integrated_RDB


def RadAcr_Correlation(SunShapes: list):
    NbrSunShapes = len(SunShapes)
    ACR_Reading = zeros(NbrSunShapes)
    RDB_Irradiance = zeros(NbrSunShapes)

    for i, ss in enumerate(SunShapes):
        ACR_Reading[i] = ss.acr
        RDB_Irradiance[i] = ss.rdb_irradiance

    return RDB_Irradiance, ACR_Reading
