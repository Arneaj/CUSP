from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import numpy
import os
import subprocess
import glob
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
    

# def find_vtk():
#     """Find VTK installation"""
#     vtk_dirs = []
#     vtk_libs = []
    
#     # Try pkg-config first
#     try:
#         result = subprocess.run(['pkg-config', '--cflags', '--libs', 'vtk'], 
#                               capture_output=True, text=True)
#         if result.returncode == 0:
#             flags = result.stdout.strip().split()
#             includes = [f[2:] for f in flags if f.startswith('-I')]
#             libs = [f[2:] for f in flags if f.startswith('-l')]
#             return includes, libs
#     except FileNotFoundError:
#         pass
    
#     # Common VTK paths (adjust for your HPC system)
#     vtk_search_paths = [
#         '/usr/include/vtk*',
#         '/usr/local/include/vtk*',
#         '/opt/vtk*/include',
#         '/sw-eb/software/VTK/*/include/vtk*',  # HPC path
#         '/sw-eb/software/*/VTK/*/include/vtk*',
#     ]
    
#     for pattern in vtk_search_paths:
#         paths = glob.glob(pattern)
#         if paths:
#             vtk_dir = paths[-1]  # Use latest version
#             vtk_dirs.append(vtk_dir)
            
#             # Common VTK libraries
#             vtk_libs = [
#                 'vtkCommonCore', 'vtkCommonDataModel', 'vtkCommonExecutionModel',
#                 'vtkIOCore', 'vtkIOXML', 'vtkFiltersCore', 'vtkFiltersGeneral'
#             ]
#             break
    
#     return vtk_dirs, vtk_libs

# def find_eigen():
#     """Find Eigen installation"""
#     # Try pkg-config first
#     try:
#         result = subprocess.run(['pkg-config', '--cflags', 'eigen3'], 
#                               capture_output=True, text=True)
#         if result.returncode == 0:
#             flags = result.stdout.strip().split()
#             includes = [f[2:] for f in flags if f.startswith('-I')]
#             return includes
#     except FileNotFoundError:
#         pass
    
#     # Common Eigen paths
#     eigen_search_paths = [
#         '/usr/include/eigen3',
#         '/usr/local/include/eigen3',
#         '/opt/eigen*/include/eigen3',
#         '/sw-eb/software/Eigen/*/include/eigen3',  # HPC path
#         '/usr/include/Eigen',
#         '/usr/local/include/Eigen',
#         '/opt/eigen*/include/Eigen',
#         '/sw-eb/software/Eigen/*/include/Eigen',  # HPC path
#         '/sw-eb/software/*/Eigen/*/include/Eigen',  # HPC path
#     ]
    
#     for path in eigen_search_paths:
#         expanded_paths = glob.glob(path)
#         if expanded_paths:
#             return expanded_paths
    
#     return []

# def find_ceres():
#     """Find Ceres installation"""
#     ceres_dirs = []
#     ceres_libs = []
    
#     # Try pkg-config first
#     try:
#         result = subprocess.run(['pkg-config', '--cflags', '--libs', 'ceres'], 
#                               capture_output=True, text=True)
#         if result.returncode == 0:
#             flags = result.stdout.strip().split()
#             includes = [f[2:] for f in flags if f.startswith('-I')]
#             libs = [f[2:] for f in flags if f.startswith('-l')]
#             return includes, libs
#     except FileNotFoundError:
#         pass
    
#     # Common Ceres paths
#     ceres_search_paths = [
#         '/usr/include/ceres*',
#         '/usr/local/include/ceres*', 
#         '~/local/include/ceres*',
#         '/opt/ceres*/include/ceres*',
#         '/sw-eb/software/Ceres-Solver/*/include/ceres*',  # HPC path
#     ]
    
#     for path in ceres_search_paths:
#         expanded_paths = glob.glob(path)
#         for expanded_path in expanded_paths:
#             if expanded_path:
#                 ceres_dirs.append(expanded_path)
#                 ceres_libs = ['ceres', 'glog', 'gflags']
#                 break
#         if ceres_dirs:
#             break
    
#     return ceres_dirs, ceres_libs

# def find_abseil():
#     """Find Abseil installation"""
#     abseil_dirs = []
#     abseil_libs = []
    
#     # Try pkg-config first
#     try:
#         result = subprocess.run(['pkg-config', '--cflags', '--libs', 'absl_base'], 
#                               capture_output=True, text=True)
#         if result.returncode == 0:
#             flags = result.stdout.strip().split()
#             includes = [f[2:] for f in flags if f.startswith('-I')]
#             libs = [f[2:] for f in flags if f.startswith('-l') and f.startswith('-labsl')]
#             return includes, libs
#     except FileNotFoundError:
#         pass
    
#     # Common Abseil paths
#     abseil_search_paths = [
#         '/usr/include',
#         '/usr/local/include',
#         '/opt/abseil*/include', 
#         '/sw-eb/software/Abseil/*/include',  # HPC path
#         '/sw-eb/software/*/Abseil/*/include',
#     ]
    
#     for path in abseil_search_paths:
#         expanded_paths = glob.glob(path)
#         for expanded_path in expanded_paths:
#             if os.path.exists(os.path.join(expanded_path, 'absl')):
#                 abseil_dirs.append(expanded_path)
#                 # Common Abseil libraries
#                 abseil_libs = [
#                     'absl_base', 'absl_strings', 'absl_str_format', 'absl_time',
#                     'absl_synchronization', 'absl_stacktrace', 'absl_symbolize'
#                 ]
#                 break
#         if abseil_dirs:
#             break
    
#     return abseil_dirs, abseil_libs

# # Find all dependencies
# print("Searching for dependencies...")
# vtk_includes, vtk_libs = find_vtk()
# eigen_includes = find_eigen()
# ceres_includes, ceres_libs = find_ceres()
# abseil_includes, abseil_libs = find_abseil()

# # Print what was found
# print(f"VTK includes: {vtk_includes}")
# print(f"VTK libraries: {vtk_libs}")
# print(f"Eigen includes: {eigen_includes}")
# print(f"Ceres includes: {ceres_includes}")
# print(f"Ceres libraries: {ceres_libs}")
# print(f"Abseil includes: {abseil_includes}")
# print(f"Abseil libraries: {abseil_libs}")

# Collect all include directories
all_includes = [
    get_pybind_include(),
    numpy.get_include(),
    '../headers_cpp',
] #+ vtk_includes + eigen_includes + ceres_includes + abseil_includes

# # Collect all libraries
# all_libs = vtk_libs + ceres_libs + abseil_libs

# # Define compile-time macros for optional features
# compile_macros = []
# if vtk_includes:
#     compile_macros.append(('USE_VTK', '1'))
# if eigen_includes:
#     compile_macros.append(('USE_EIGEN', '1'))
# if ceres_includes:
#     compile_macros.append(('USE_CERES', '1'))
# if abseil_includes:
#     compile_macros.append(('USE_ABSEIL', '1'))    



# Define the extension module
ext_modules = [
    Extension(
        'topology_analysis',
        [
            'library.cpp',
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
        include_dirs=all_includes,
        # libraries=all_libs,
        language='c++',
        # define_macros=compile_macros,
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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
    ],
)