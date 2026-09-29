from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("cython_proof.pyx", annotate=True),
    include_dirs=[np.get_include()] # Necesario para procesar los arreglos de NumPy
)
