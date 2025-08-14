cmake_minimum_required(VERSION 3.31 FATAL_ERROR)

project(mag_cusp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_PREFIX_PATH "~/local" ${CMAKE_PREFIX_PATH})

include(FetchContent)


find_package(LAPACK QUIET)
if(NOT LAPACK_FOUND)
    message(STATUS "LAPACK not found. Fetching...")
    FetchContent_Declare(
        lapack
        GIT_REPOSITORY https://github.com/Reference-LAPACK/lapack.git
        GIT_TAG v3.12.1
    )

    FetchContent_MakeAvailable(lapack)
endif()

find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen not found. Fetching...")
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)
endif()

find_package(absl QUIET)
if(NOT absl_FOUND)
    message(STATUS "Abseil not found. Fetching...")
    FetchContent_Declare(
        abseil
        GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
        GIT_TAG 20240116.2
    )
    FetchContent_MakeAvailable(abseil)
endif()


# find_package(gflags QUIET)
# if(NOT gflags_FOUND)
#     FetchContent_Declare(
#         gflags
#         GIT_REPOSITORY https://github.com/gflags/gflags.git
#         GIT_TAG v2.2.2
#     )
#     FetchContent_MakeAvailable(gflags)
# endif()

# find_package(glog QUIET)
# if(NOT glog_FOUND)
#     FetchContent_Declare(
#         glog
#         GIT_REPOSITORY https://github.com/google/glog.git
#         GIT_TAG v0.7.1
#     )
#     FetchContent_MakeAvailable(glog)
#     set(glog_DIR "${glog_BINARY_DIR}" CACHE PATH "Path to glog build")
# endif()

find_package(Ceres QUIET)
if(NOT Ceres_FOUND)
    message(STATUS "Ceres not found. Fetching...")
    FetchContent_Declare(
        ceres
        GIT_REPOSITORY https://ceres-solver.googlesource.com/ceres-solver.git
        GIT_TAG 2.2.0
    )
    set(MINIGLOG ON CACHE BOOL "Use Ceres' minimal logging" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Disable Ceres tests" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "Disable Ceres examples" FORCE)
    FetchContent_MakeAvailable(ceres)
endif()

find_package(OpenMP QUIET)

find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found. Fetching...")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v3.0.0
    )
    FetchContent_MakeAvailable(pybind11)
endif()




set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")   # is this a good idea

# Prevent a "command line is too long" failure in Windows.
set(CMAKE_NINJA_FORCE_RESPONSE_FILE "ON" CACHE BOOL "Force Ninja to use response files.")


set(SOURCES
    ../src_cpp/points.cpp
    ../src_cpp/matrix.cpp
    # ../src_cpp/read_pvtr.cpp
    ../src_cpp/read_file.cpp
    # ../src_cpp/reader_writer.cpp
    ../src_cpp/raycast.cpp
    ../src_cpp/streamlines.cpp
    ../src_cpp/magnetopause.cpp
    ../src_cpp/preprocessing.cpp
    ../src_cpp/fit_to_analytical.cpp
    ../src_cpp/analysis.cpp
)

set(HEADERS
    ../headers_cpp/points.h
    ../headers_cpp/matrix.h   
    ../headers_cpp/raycast.h
    ../headers_cpp/streamlines.h
    ../headers_cpp/magnetopause.h
    ../headers_cpp/preprocessing.h  
    # ../headers_cpp/read_pvtr.h
    ../headers_cpp/read_file.h
    # ../headers_cpp/reader_writer.h
    ../headers_cpp/fit_to_analytical.h
    ../headers_cpp/analysis.h
)


# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find NumPy with error handling
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE NUMPY_ERROR
    RESULT_VARIABLE NUMPY_RESULT
)

if(NOT NUMPY_RESULT EQUAL 0)
    message(WARNING "Could not detect NumPy automatically: ${NUMPY_ERROR}")
    message(STATUS "Trying alternative NumPy detection...")
    
    # Try alternative method
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sys; sys.path.append('/sw-eb/software/SciPy-bundle/2023.07-gfbf-2023a/lib/python3.11/site-packages'); import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE NUMPY_ALT_RESULT
    )
    
    if(NOT NUMPY_ALT_RESULT EQUAL 0)
        # Fallback to manual path
        message(STATUS "Using fallback NumPy path...")
        set(NUMPY_INCLUDE_DIR "/sw-eb/software/SciPy-bundle/2023.07-gfbf-2023a/lib/python3.11/site-packages/numpy/core/include")
    endif()
endif()

message(STATUS "NumPy include directory: ${NUMPY_INCLUDE_DIR}")

# Create the module
pybind11_add_module(
    mag_cusp 
    library.cpp
    ${SOURCES}
)


target_link_libraries(mag_cusp PRIVATE 
    # ${VTK_LIBRARIES} 
    ${CERES_LIBRARIES} 
    OpenMP::OpenMP_CXX
)

# Include directories
target_include_directories(mag_cusp PRIVATE 
    ../src_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${NUMPY_INCLUDE_DIR}
    # ${VTK_INCLUDE_DIRS} 
    ${CERES_INCLUDE_DIRS}
)

# Compiler-specific options
target_compile_definitions(mag_cusp PRIVATE 
    # ${VTK_DEFINITIONS} 
    ${CERES_DEFINITIONS} VERSION_INFO=${EXAMPLE_VERSION_INFO}
)

# Set properties for the module
set_target_properties(mag_cusp PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN YES
)

if (UNIX)
  target_link_options(mag_cusp PUBLIC -pthread)
endif()

# vtk_module_autoinit is needed
# vtk_module_autoinit(
#   TARGETS mag_cusp
#   MODULES ${VTK_LIBRARIES}
# )

# execute_process(
#     COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('purelib'))"
#     OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )

install(TARGETS mag_cusp
    LIBRARY DESTINATION mag_cusp
)
