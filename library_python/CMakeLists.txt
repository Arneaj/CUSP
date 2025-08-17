cmake_minimum_required(VERSION 3.31 FATAL_ERROR)

project(mag_cusp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_PREFIX_PATH "~/local" ${CMAKE_PREFIX_PATH})

include(FetchContent)


set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Build with -fPIC" FORCE)


# Declare all dependencies first
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
endif()

find_package(absl QUIET)
if(NOT absl_FOUND)
FetchContent_Declare(
    abseil
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20240116.2
)
endif()

find_package(Ceres QUIET)
if(NOT Ceres_FOUND)
    FetchContent_Declare(
        ceres
        GIT_REPOSITORY https://ceres-solver.googlesource.com/ceres-solver.git
        GIT_TAG 2.2.0
    )
endif()

find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v3.0.0
    )
endif()




find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    # Configure Eigen to not build BLAS/LAPACK before making it available
    message(STATUS "Configuring Eigen...")
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Don't build pkg-config" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Disable all testing" FORCE)

    # Make Eigen available but try to limit what gets built
    FetchContent_MakeAvailable(eigen)
endif()



find_package(absl QUIET)
if(NOT absl_FOUND)
    # Configure and fetch Abseil
    message(STATUS "Fetching Abseil...")
    set(ABSL_ENABLE_INSTALL ON CACHE BOOL "Enable Abseil install" FORCE)
    set(ABSL_BUILD_TEST_HELPERS OFF CACHE BOOL "Disable Abseil tests" FORCE)
    FetchContent_MakeAvailable(abseil)
endif()



find_package(Ceres QUIET)
if(NOT Ceres_FOUND)
    # Configure Ceres options before making it available
    message(STATUS "Configuring Ceres...")
    set(MINIGLOG ON CACHE BOOL "Use Ceres' minimal logging" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Disable Ceres tests" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "Disable Ceres examples" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable Ceres benchmarks" FORCE)
    set(BUILD_DOCUMENTATION OFF CACHE BOOL "Disable Ceres documentation" FORCE)
    set(CERES_ENABLE_INSTALL OFF CACHE BOOL "Disable Ceres install" FORCE)

    # Tell Ceres where to find Eigen
    FetchContent_GetProperties(eigen SOURCE_DIR eigen_source_dir)
    set(EIGEN3_INCLUDE_DIR ${eigen_source_dir} CACHE PATH "Eigen include directory" FORCE)
    set(Eigen3_FOUND TRUE CACHE BOOL "Eigen found" FORCE)

    set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "Disable the ceres uninstall creation" FORCE)
    FetchContent_MakeAvailable(ceres)
endif()



find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # Fetch pybind11
    message(STATUS "Fetching pybind11...")
    FetchContent_MakeAvailable(pybind11)
endif()


find_package(OpenMP QUIET)





# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Prevent a "command line is too long" failure in Windows.
set(CMAKE_NINJA_FORCE_RESPONSE_FILE "ON" CACHE BOOL "Force Ninja to use response files.")





set(SOURCES
    ../src_cpp/points.cpp
    ../src_cpp/matrix.cpp
    ../src_cpp/read_file.cpp
    ../src_cpp/raycast.cpp
    ../src_cpp/streamlines.cpp
    ../src_cpp/magnetopause.cpp
    ../src_cpp/preprocessing.cpp
    ../src_cpp/fit_to_analytical.cpp
    ../src_cpp/analysis.cpp
)






# Find Python and NumPy
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

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
    
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sys; sys.path.append('/sw-eb/software/SciPy-bundle/2023.07-gfbf-2023a/lib/python3.11/site-packages'); import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE NUMPY_ALT_RESULT
    )
    
    if(NOT NUMPY_ALT_RESULT EQUAL 0)
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



# Link libraries
set(LINK_LIBRARIES "")

# Link Ceres
if(TARGET ceres)
    list(APPEND LINK_LIBRARIES ceres)
endif()

if(TARGET absl::log)
    list(APPEND LINK_LIBRARIES absl::log absl::log_initialize absl::log_globals)
endif()

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    list(APPEND LINK_LIBRARIES OpenMP::OpenMP_CXX)
endif()

target_link_libraries(mag_cusp PRIVATE ${LINK_LIBRARIES})

# Get source directories for include paths
FetchContent_GetProperties(eigen SOURCE_DIR eigen_source_dir)
FetchContent_GetProperties(absl SOURCE_DIR absl_source_dir)
FetchContent_GetProperties(ceres SOURCE_DIR ceres_source_dir BINARY_DIR ceres_binary_dir)

# Include directories
target_include_directories(mag_cusp PRIVATE 
    ../headers_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${NUMPY_INCLUDE_DIR}
    ${eigen_source_dir}
    ${absl_source_dir}
    ${ceres_source_dir}/include
    ${ceres_binary_dir}/include
)

# Compiler-specific options
target_compile_definitions(mag_cusp PRIVATE 
    VERSION_INFO=${EXAMPLE_VERSION_INFO}
)

# Add warning flags
target_compile_options(mag_cusp PRIVATE -Wall -Wextra)

# Set properties for the module
set_target_properties(mag_cusp PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN YES
)

if (UNIX)
    target_link_options(mag_cusp PUBLIC -pthread)
endif()

install(TARGETS mag_cusp
    LIBRARY DESTINATION mag_cusp
)
