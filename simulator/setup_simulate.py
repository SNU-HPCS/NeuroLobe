from setuptools import setup, Extension, Command
from Cython.Build import cythonize
import numpy

compiler_directives = {"language_level": 3}
compiler_args=['-O0', '-ggdb']

ext_modules=[
    Extension(name="Memory", sources=["Memory.pyx"], extra_compile_args=compiler_args),
    Extension(name="Router", sources=["Router.pyx"], extra_compile_args=compiler_args),
    Extension(name="Core", sources=["Core.pyx"], extra_compile_args=compiler_args),
    Extension(name="NoC", sources=["NoC.pyx"], extra_compile_args=compiler_args),
    Extension(name="External", sources=["External.pyx"], extra_compile_args=compiler_args),
    Extension(name="Debug", sources=["Debug.pyx"], extra_compile_args=compiler_args),
]

setup(
    name = 'bci_processor',
    ext_modules = cythonize(ext_modules, compiler_directives=compiler_directives),
    include_dirs=[numpy.get_include()]
)
