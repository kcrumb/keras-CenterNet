from setuptools import setup
from Cython.Build import cythonize
import numpy

# run command:
#     python setup.py build_ext --inplace

setup(
    ext_modules = cythonize("utils/compute_overlap.pyx"),
    include_dirs = [numpy.get_include()]
)
