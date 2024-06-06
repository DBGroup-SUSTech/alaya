from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import numpy as np
import os
import pybind11


__version__="0.0.1"

class CustomBuildExt(build_ext):
    def build_extensions(self):
        # self.compiler.set_executable('compiler_so', 'clang-18')
        # self.compiler.set_executable('compiler_cxx', 'clang-18')
        self.compiler.set_executable('compiler_so', 'g++-13')
        self.compiler.set_executable('compiler_cxx', 'g++-13')
        super().build_extensions()

include_dirs = [
    pybind11.get_include(),
    np.get_include(),
    '../include/',
    '../include/alaya',
    '../include/alaya/utils',
    '../thirdparty/faiss',
    '../thirdparty/fmt/include/',
    '../thirdparty/pybind/include',
]

bindings_dir = 'python'
source_files = ['bindings.cpp', '../src/utils/kmeans.cpp', '../src/utils/random_utils.cpp', '../src/utils/io_utils.cpp']
# source_files = ['bindings.cpp']

ext_modules = [
    Pybind11Extension(
        "alaya",
        source_files,
        include_dirs=include_dirs,
        libraries=["faiss", "fmt",],
        library_dirs=['../build/thirdparty/faiss/faiss/Release', '../build/thirdparty/fmt/Release'],
        extra_compile_args=['-std=c++20', '-fPIC', '-fopenmp', '-march=native', '-O3'],
        extra_link_args=['-std=c++20', '-fopenmp', '-lmkl_rt', '-liomp5'],
        define_macros=[('VERSION_INFO', __version__)],
        cxx_std=20,
    ),
]

setup(
    name='alaya',
    version=__version__,
    description='',
    author='DBGroup@SUSTech',
    ext_modules=ext_modules,
    # install_requires=['numpy'],
    cmdclass={'build_ext': CustomBuildExt},
    # cmdclass={'build_ext': build_ext},
)