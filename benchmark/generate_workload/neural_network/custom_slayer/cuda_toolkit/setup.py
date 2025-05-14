from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ext_modules = [
    CUDAExtension('conv_cuda', ['conv.cu']),
    CUDAExtension('shift_cuda', ['shift.cu']),
    CUDAExtension('leaky_integrator_cuda', ['leaky_integrator.cu']),
    CUDAExtension('leaky_integrator_float_cuda', ['leaky_integrator_float.cu']),
]

setup(
    name='slayer_cuda',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    })
