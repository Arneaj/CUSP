cmake_minimum_required(VERSION 3.31 FATAL_ERROR)

project(MagCUSPS)

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_PREFIX_PATH "~/local" ${CMAKE_PREFIX_PATH})

include(FetchContent)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Build with -fPIC" FORCE)

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "Defaulting to Release build")
endif()


# ============================================================
# Eigen
# ============================================================
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen not found, fetching via FetchContent...")

    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )

    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(eigen)

    # Provide alias so we can always link Eigen3::Eigen
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})
endif()


# ============================================================
# Abseil
# ============================================================
find_package(absl QUIET)
if(NOT absl_FOUND)
    FetchContent_Declare(
        abseil
        GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
        GIT_TAG 20240116.2
        GIT_SHALLOW TRUE
    )
endif()


# ============================================================
# Ceres
# ============================================================
find_package(Ceres QUIET)
if(NOT Ceres_FOUND)
    message(STATUS "Ceres not found, fetching via FetchContent...")

    FetchContent_Declare(
        ceres
        GIT_REPOSITORY https://ceres-solver.googlesource.com/ceres-solver.git
        GIT_TAG 2.2.0
        GIT_SHALLOW TRUE
    )

    # Configure Ceres options (disable optional deps)
    set(MINIGLOG ON CACHE BOOL "Use Ceres' minimal logging" FORCE)
    set(GLOG OFF CACHE BOOL "No full glog" FORCE)
    set(GFLAGS OFF CACHE BOOL "No gflags" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCUMENTATION OFF CACHE BOOL "" FORCE)
    set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "" FORCE)
    set(ACCELERATESPARSE OFF CACHE BOOL "" FORCE)
    set(SCHUR_SPECIALIZATIONS OFF CACHE BOOL "" FORCE)
    set(CUDA OFF CACHE BOOL "" FORCE)
    set(LAPACK OFF CACHE BOOL "" FORCE)
    set(SUITESPARSE OFF CACHE BOOL "" FORCE)
    set(EIGENSPARSE OFF CACHE BOOL "" FORCE)

    # Make available
    FetchContent_MakeAvailable(ceres)

    # Exports a target called "ceres" — no need to set vars
endif()


# ============================================================
# Ceres
# ============================================================
find_package(VTK COMPONENTS 
    CommonCore
    CommonDataModel
    FiltersGeometry
    IOXML
    IOParallelXML
    InteractionStyle
    REQUIRED
)



# ============================================================
# OpenMP
# ============================================================
find_package(OpenMP QUIET)








# Prevent a "command line is too long" failure in Windows.
set(CMAKE_NINJA_FORCE_RESPONSE_FILE "ON" CACHE BOOL "Force Ninja to use response files.")




set(SOURCES
    src_cpp/points.cpp
    src_cpp/matrix.cpp
    src_cpp/read_pvtr.cpp
    src_cpp/read_file.cpp
    src_cpp/reader_writer.cpp
    src_cpp/raycast.cpp
    src_cpp/streamlines.cpp
    src_cpp/magnetopause.cpp
    src_cpp/preprocessing.cpp
    src_cpp/fit_to_analytical.cpp
    src_cpp/analysis.cpp
)




set(LINK_LIBRARIES "")

# Link Ceres
if(TARGET ceres)
    list(APPEND LINK_LIBRARIES ceres)
endif()

FetchContent_GetProperties(eigen SOURCE_DIR eigen_source_dir)
FetchContent_GetProperties(ceres SOURCE_DIR ceres_source_dir BINARY_DIR ceres_binary_dir)


message(STATUS "CERES_FOUND: ${Ceres_FOUND}")
message(STATUS "CERES_INCLUDE_DIRS: ${CERES_INCLUDE_DIRS}")
message(STATUS "CERES_LIBRARIES: ${CERES_LIBRARIES}")
message(STATUS "ceres_source_dir: ${ceres_source_dir}")
message(STATUS "ceres_binary_dir: ${ceres_binary_dir}")



# ************************************ read_pvtr: test of the reading of the file


add_executable(read_pvtr tests_cpp/read_pvtr_test.cpp
    src_cpp/read_pvtr.cpp
    src_cpp/read_file.cpp
    src_cpp/points.cpp
    src_cpp/matrix.cpp)


if (OpenMP_CXX_FOUND)
    target_link_libraries(read_pvtr PRIVATE ${VTK_LIBRARIES} OpenMP::OpenMP_CXX)
else()
    target_link_libraries(read_pvtr PRIVATE ${VTK_LIBRARIES})
endif()

target_include_directories(read_pvtr PRIVATE 
    ${VTK_INCLUDE_DIRS}
)

target_compile_definitions(read_pvtr PRIVATE ${VTK_DEFINITIONS})

if (UNIX)
  target_link_options(read_pvtr PUBLIC -pthread)
endif()


vtk_module_autoinit(
    TARGETS read_pvtr
    MODULES ${VTK_LIBRARIES}
)


# ************************************ raycasting: test of the reading of the file


add_executable(raycasting tests_cpp/raycast_test.cpp
    src_cpp/read_pvtr.cpp
    src_cpp/read_file.cpp
    src_cpp/reader_writer.cpp
    src_cpp/points.cpp
    src_cpp/matrix.cpp
    src_cpp/preprocessing.cpp
    src_cpp/streamlines.cpp
    src_cpp/magnetopause.cpp
    src_cpp/raycast.cpp)

  
if (OpenMP_CXX_FOUND)
    target_link_libraries(raycasting PRIVATE ${VTK_LIBRARIES} OpenMP::OpenMP_CXX)
else()
    target_link_libraries(raycasting PRIVATE ${VTK_LIBRARIES})
endif()



target_include_directories(raycasting PRIVATE 
    ${VTK_INCLUDE_DIRS}
)

target_compile_definitions(raycasting PRIVATE ${VTK_DEFINITIONS})

if (UNIX)
  target_link_options(raycasting PUBLIC -pthread)
endif()


vtk_module_autoinit(
    TARGETS raycasting
    MODULES ${VTK_LIBRARIES}
)


# *************************************** fit_to_analytical: test of the fitting to the interest points


add_executable(fit_to_analytical tests_cpp/fit_to_analytical_test.cpp
    src_cpp/points.cpp
    src_cpp/matrix.cpp
    src_cpp/read_pvtr.cpp
    src_cpp/read_file.cpp
    src_cpp/streamlines.cpp
    src_cpp/magnetopause.cpp
    src_cpp/preprocessing.cpp
    src_cpp/fit_to_analytical.cpp)


if (OpenMP_CXX_FOUND)
    target_link_libraries(fit_to_analytical PRIVATE ${VTK_LIBRARIES} ceres OpenMP::OpenMP_CXX)
else()
    target_link_libraries(fit_to_analytical PRIVATE ${VTK_LIBRARIES} ceres)
endif()

target_include_directories(fit_to_analytical PRIVATE 
    ${VTK_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ceres_source_dir}/include
    ${ceres_binary_dir}/include
)

target_compile_definitions(fit_to_analytical PRIVATE ${VTK_DEFINITIONS} ${CERES_DEFINITIONS})

if (UNIX)
  target_link_options(fit_to_analytical PUBLIC -pthread)
endif()


vtk_module_autoinit(
    TARGETS fit_to_analytical
    MODULES ${VTK_LIBRARIES}
)




# ********************************* read_and_write: read some data and write it in real dims


add_executable(read_and_write main_cpp/read_and_write.cpp
    src_cpp/points.cpp
    src_cpp/matrix.cpp
    src_cpp/read_pvtr.cpp
    src_cpp/read_file.cpp
    src_cpp/reader_writer.cpp
    src_cpp/streamlines.cpp
    src_cpp/magnetopause.cpp
    src_cpp/fit_to_analytical.cpp
    src_cpp/preprocessing.cpp
)


if (OpenMP_CXX_FOUND)
    target_link_libraries(read_and_write PRIVATE ${VTK_LIBRARIES} ceres OpenMP::OpenMP_CXX)
else()
    target_link_libraries(read_and_write PRIVATE ${VTK_LIBRARIES} ceres)
endif()


target_include_directories(read_and_write PRIVATE 
    ${VTK_INCLUDE_DIRS} 
    ${ceres_source_dir}/include
    ${ceres_binary_dir}/include
)

target_compile_definitions(read_and_write PRIVATE ${VTK_DEFINITIONS})

if (UNIX)
  target_link_options(read_and_write PUBLIC -pthread)
endif()

vtk_module_autoinit(
    TARGETS read_and_write
    MODULES ${VTK_LIBRARIES}
)




# ********************************* full_process: from read to fit


add_executable(full_process main_cpp/full_process.cpp
    src_cpp/points.cpp
    src_cpp/matrix.cpp
    src_cpp/read_pvtr.cpp
    src_cpp/read_file.cpp
    src_cpp/reader_writer.cpp
    src_cpp/raycast.cpp
    src_cpp/streamlines.cpp
    src_cpp/magnetopause.cpp
    src_cpp/preprocessing.cpp
    src_cpp/fit_to_analytical.cpp
    src_cpp/analysis.cpp)



if (OpenMP_CXX_FOUND)
    target_link_libraries(full_process PRIVATE ${VTK_LIBRARIES} ceres OpenMP::OpenMP_CXX)
else()
    target_link_libraries(full_process PRIVATE ${VTK_LIBRARIES} ceres)
endif()


target_include_directories(full_process PRIVATE 
    ${VTK_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ceres_source_dir}/include
    ${ceres_binary_dir}/include
)

target_compile_definitions(full_process PRIVATE ${VTK_DEFINITIONS} ${CERES_DEFINITIONS})

if (UNIX)
  target_link_options(full_process PUBLIC -pthread)
endif()

vtk_module_autoinit(
    TARGETS full_process
    MODULES ${VTK_LIBRARIES}
)



# ********************************* great_analysis: evaluation of functions


add_executable(great_analysis main_cpp/great_analysis.cpp
    src_cpp/points.cpp
    src_cpp/matrix.cpp
    src_cpp/read_pvtr.cpp
    src_cpp/read_file.cpp
    src_cpp/reader_writer.cpp
    src_cpp/raycast.cpp
    src_cpp/streamlines.cpp
    src_cpp/magnetopause.cpp
    src_cpp/preprocessing.cpp
    src_cpp/fit_to_analytical.cpp
    src_cpp/analysis.cpp)



if (OpenMP_CXX_FOUND)
    target_link_libraries(great_analysis PRIVATE ${VTK_LIBRARIES} ceres OpenMP::OpenMP_CXX)
else()
    target_link_libraries(great_analysis PRIVATE ${VTK_LIBRARIES} ceres)
endif()


target_include_directories(great_analysis PRIVATE 
    ${VTK_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ceres_source_dir}/include
    ${ceres_binary_dir}/include
)

target_compile_definitions(great_analysis PRIVATE ${VTK_DEFINITIONS} ${CERES_DEFINITIONS})

if (UNIX)
  target_link_options(great_analysis PUBLIC -pthread)
endif()

vtk_module_autoinit(
    TARGETS great_analysis
    MODULES ${VTK_LIBRARIES}
)
