from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rmsnorm_cluster',
    ext_modules=[
        CUDAExtension('rmsnorm_cluster', [
            'csrc/rmsnorm_cuda.cpp',
            'csrc/rmsnorm_kernel.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '--generate-code=arch=compute_100,code=sm_100',
            ],
        }),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
