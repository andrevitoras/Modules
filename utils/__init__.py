"""
Created by André Santos: andrevitoras@gmail.com / avas@uevora.pt
"""
from copy import deepcopy

from numpy import zeros, sqrt, array, ndarray
from pandas import DataFrame

from niopy.geometric_transforms import Angle


# class AngularInterval:
#
#     def __init__(self, lower: float, upper: float):
#         if lower != upper:
#             self.lower = min(lower, upper)
#             self.upper = max(lower, upper)
#         else:
#             self.lower = 0.0
#             self.upper = 0.0
#
#     def __repr__(self):
#         return f'({self.lower}, {self.upper})'
#
#     def is_empty(self):
#         return True if self.lower == self.upper else False
#
#     def contains(self, other):
#         return True if self.lower <= other.lower <= self.upper and self.lower <= other.upper <= self.upper else False
#
#     def __and__(self, other):
#
#         if self.upper < other.lower or self.lower > other.upper:
#             # Early out for non-overlapping intervals
#             return AngularInterval(0, 0)
#         else:
#             lower = max(self.lower, other.lower)
#             upper = min(self.upper, other.upper)
#
#         return AngularInterval(lower, upper)
#
#     def __sub__(self, other):
#         if not self.contains(other) or other.is_empty():
#             return self
#         elif other.contains(self):
#             return AngularInterval(0, 0)
#         else:
#             if self.lower == other.lower:
#                 lower = other.upper
#                 upper = self.upper
#             elif self.upper == other.upper:
#                 lower = self.lower
#                 upper = other.lower
#             else:
#                 raise ValueError('Something is wrong!!')
#
#             return AngularInterval(lower, upper)

########################################################################################################################
########################################################################################################################


def closest(my_list, my_number):
    return min(my_list, key=lambda x: abs(x - my_number))


def chop(expr, *, maxi=1.e-6):
    return [i if abs(i) > maxi else 0 for i in expr]


def arrays_to_contour(data: DataFrame, x_col: str, y_col: str, z_col: str):

    df = data.sort_values(by=[x_col, y_col])
    x = df[x_col].unique()
    y = df[y_col].unique()
    z = df[z_col].values.reshape(len(x), len(y)).T

    return x, y, z


def read_trnsys_tmy2(file):

    lines = open(file, 'r').readlines()
    headers = ['Time', 'DNI [W/m2]', 'GHI [W/m2]', 'Solar Zenith [degrees]', 'Solar Azimuth [degrees]']

    data = zeros(shape=(len(lines) - 1, len(headers)))

    for i, file_line in enumerate(lines):

        if i == 0:
            continue
        else:
            data[i - 1][:] = [float(elem) for elem in file_line.split()]

    df = DataFrame(data, columns=headers)

    return df


def rmse(predictions: array, targets: array):
    return sqrt(((predictions - targets) ** 2).mean())


def dic2json(d: dict):

    dict_to_export = deepcopy(d)
    keys = d.keys()

    for k in keys:
        if isinstance(d[k], dict):
            dict_to_export[k] = dic2json(d=d[k])
        elif isinstance(d[k], Angle):
            dict_to_export[k] = d[k].deg
        elif isinstance(d[k], ndarray):
            dict_to_export[k] = d[k].tolist()
        else:
            dict_to_export[k] = d[k]

    return dict_to_export


def plot_line(a, b):
    return array([a[0], b[0]]), array([a[-1], b[-1]])
