from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import numpy
import os
from pathlib import Path

# Get the long description from the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __str__(self):
        return pybind11.get_cmake_dir() + "/../../../include"

# Define the extension module
ext_modules = [
    Extension(
        'topology_analysis',
        [
            'python_library.cpp',
            '../src_cpp/points.cpp',
            '../src_cpp/matrix.cpp',
            '../src_cpp/streamlines.cpp',
            '../src_cpp/magnetopause.cpp',
            '../src_cpp/read_file.cpp',
            # '../src_cpp/read_pvtr.cpp',
            # '../src_cpp/reader_writer.cpp',
            '../src_cpp/preprocessing.cpp',
            '../src_cpp/raycast.cpp',
            # '../src_cpp/fit_to_analytical.cpp',
            '../src_cpp/analysis.cpp',
        ],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            # Path to numpy headers
            numpy.get_include(),
            # Path to your headers
            '../headers_cpp',
        ],
        language='c++',
        # cxx_std=20,
    ),
]

# Custom build_ext command to add compiler-specific options
class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/O2'],
        'unix': ['-O3', '-march=native', '-std=c++20'],
    }
    
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if os.environ.get('DEBUG'):
        c_opts['unix'] = ['-O0', '-g', '-std=c++20']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        
        if ct == 'unix':
            opts.append(f'-DVERSION_INFO="{self.distribution.get_version()}"')
            if hasattr(self.compiler, 'compiler_so'):
                # Check for OpenMP support
                if any('openmp' in flag or 'fopenmp' in flag for flag in os.environ.get('CXXFLAGS', '').split()):
                    opts.append('-fopenmp')
                    link_opts.append('-fopenmp')
        elif ct == 'msvc':
            opts.append(f'/DVERSION_INFO=\\"{self.distribution.get_version()}\\"')
            
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
            
        build_ext.build_extensions(self)

setup(
    name='topology_analysis',
    version='1.0.0',
    author='Arnie',
    author_email='arnaud.rollandmail@gmail.com',
    description='High-performance Python module with C++ backend',
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    install_requires=[
        'numpy>=1.15.0',
    ],
    setup_requires=[
        'pybind11>=2.6.0',
        'numpy>=1.15.0',
    ],
    python_requires='>=3.7',
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
    ],
)