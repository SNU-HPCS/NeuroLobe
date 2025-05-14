from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        'inference',
        ['inference.pyx'],
        extra_compile_args=['-O3', '-ffast-math','-march=native', '-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[np.get_include()],
        define_macros=[('CYTHON_TRACE', '1')],
    )
]

setup(
    name='correlate-parallel-world',
    ext_modules=cythonize(ext_modules,
                          compiler_directives={'language_level' : '3'}),
)
