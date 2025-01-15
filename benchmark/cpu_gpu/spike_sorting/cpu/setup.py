from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

#from Cython.Compiler.Options import get_directive_defaults

#directive_defaults['linetrace'] = True
#directive_defaults['binding'] = True

ext_modules = [
    Extension(
        'sorting',
        ['sorting.pyx'],
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
