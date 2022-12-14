"""
Set of functions created by André Santos (andrevitoras@gmail.com / avas@uevora.pt)
The functions implemented here are related to TracePro commands and scripts to run ray tracing simulations
They are related to write and append a script file (.scm) which configures, trace rays and export the results.
"""
from pathlib import Path
from numpy import array


class TraceproSun:

    def __init__(self, vector: array, shape: str, width: float):

        self.vector = vector
        self.width = width
        self.shape = shape


class TraceproGrid:

    def __init__(self, width: float, height: float, rays: int, flux_ray: float, origin: array, normal: array,
                 up: array, name='"Grid Source 1"'):

        self.y_half_height = height / 2
        self.x_half_width = width / 2

        self.rays = rays
        self.flux_per_ray = flux_ray

        self.origin = origin
        self.normal_vector = normal
        self.up_vector = up

        self.name = name


class TraceproSource:

    def __init__(self, beam: TraceproSun, grid: TraceproGrid):

        self.beam = beam
        self.grid = grid

    def as_script(self, script):

        script.write(';========================= Script to configure the Source of Rays ======================\n\n')
        script.write('(analysis: set-display-rays #f)\n')
        script.write(f'(raytrace:set-grid-origin {self.grid.origin}) ')


def create_script(file_path, file_name: str):

    full_file_path = Path(file_path, f"{file_name}.scm")
    script = open(full_file_path, 'w')

    return script


def close_script(script):
    script.close()
