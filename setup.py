import numpy as np
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension(
        "pdbhelper",
        sources=["src/pdbhelper.pyx"],
        libraries=["c"],
        include_dirs=[np.get_include()])]
)
