cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(MagCUSPS)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_PREFIX_PATH "~/local" ${CMAKE_PREFIX_PATH})

include(FetchContent)




# Declare all dependencies first
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
endif()

find_package(absl QUIET)
if(NOT absl_FOUND)
    FetchContent_Declare(
        abseil
        GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
        GIT_TAG 20240116.2
        GIT_SHALLOW TRUE
    )
endif()

find_package(Ceres QUIET)
if(NOT Ceres_FOUND)
    FetchContent_Declare(
        ceres
        GIT_REPOSITORY https://ceres-solver.googlesource.com/ceres-solver.git
        GIT_TAG 2.2.0
        GIT_SHALLOW TRUE
    )
endif()


find_package(VTK COMPONENTS 
    CommonCore
    CommonDataModel
    FiltersGeometry
    IOXML
    IOParallelXML
    InteractionStyle
    QUIET
)
if (NOT VTK_FOUND)
    set(VTK_BUILD_TESTING OFF CACHE BOOL "Disable VTK testing" FORCE)
    set(VTK_BUILD_EXAMPLES OFF CACHE BOOL "Disable VTK examples" FORCE)
    set(VTK_BUILD_DOCUMENTATION OFF CACHE BOOL "Disable VTK documentation" FORCE)

    FetchContent_Declare(
        VTK
        GIT_REPOSITORY https://github.com/Kitware/VTK.git
        GIT_TAG v9.5.0
    )
    FetchContent_MakeAvailable(VTK)
endif()




find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    # Configure Eigen to not build BLAS/LAPACK before making it available
    message(STATUS "Configuring Eigen...")
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Don't build pkg-config" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Disable all testing" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "No Eigen tests" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "No docs" FORCE)
    set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "No demos" FORCE)

    # Make Eigen available but try to limit what gets built
    FetchContent_MakeAvailable(eigen)
endif()




find_package(Ceres QUIET)
if(NOT Ceres_FOUND)
    # Configure Ceres options before making it available
    message(STATUS "Configuring Ceres...")
    set(MINIGLOG ON CACHE BOOL "Use Ceres' minimal logging" FORCE)
    set(GLOG OFF CACHE BOOL "No full glog" FORCE)
    set(GFLAGS OFF CACHE BOOL "No gflags" FORCE)

    set(BUILD_TESTING OFF CACHE BOOL "Disable Ceres tests" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "Disable Ceres examples" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable Ceres benchmarks" FORCE)
    set(BUILD_DOCUMENTATION OFF CACHE BOOL "Disable Ceres documentation" FORCE)
    set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "Disable the ceres uninstall creation" FORCE)

    # set(CXSPARSE OFF CACHE BOOL "No CXSparse" FORCE)
    set(ACCELERATESPARSE OFF CACHE BOOL "No Accelerate" FORCE)
    # set(CUDSS OFF CACHE BOOL "No CUDSS" FORCE)
    set(SCHUR_SPECIALIZATIONS OFF CACHE BOOL "Remove Schur specializations" FORCE)
    set(CUDA OFF CACHE BOOL "No CUDA" FORCE)
    set(LAPACK OFF CACHE BOOL "No LAPACK" FORCE)
    set(SUITESPARSE OFF CACHE BOOL "No SuiteSparse" FORCE)
    set(EIGENSPARSE OFF CACHE BOOL "No Eigen sparse" FORCE)

    # Tell Ceres where to find Eigen
    FetchContent_GetProperties(eigen SOURCE_DIR eigen_source_dir)
    set(EIGEN3_INCLUDE_DIR ${eigen_source_dir} CACHE PATH "Eigen include directory" FORCE)
    set(Eigen3_FOUND TRUE CACHE BOOL "Eigen found" FORCE)

    FetchContent_MakeAvailable(ceres)
endif()



find_package(OpenMP QUIET)





# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Fix for assembler issues with DWARF-5 debug info
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --gdwarf-4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gdwarf-4")
endif()

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
    headers_cpp
)

target_compile_definitions(read_pvtr PRIVATE ${VTK_DEFINITIONS})

if (UNIX)
  target_link_options(read_pvtr PUBLIC -pthread)
endif()


set(target_name read_pvtr)

if(VTK_FOUND)
    # VTK was found via find_package
    target_include_directories(${target_name} PRIVATE ${VTK_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${VTK_LIBRARIES})
    target_compile_definitions(${target_name} PRIVATE ${VTK_DEFINITIONS})
    
    vtk_module_autoinit(
        TARGETS ${target_name}
        MODULES ${VTK_LIBRARIES}
    )
else()
    # VTK was built via FetchContent - link to specific targets
    set(VTK_MODULES
        VTK::CommonCore
        VTK::CommonDataModel
        VTK::FiltersGeometry
        VTK::IOXML
        VTK::IOParallelXML
        VTK::InteractionStyle
    )
    
    target_link_libraries(${target_name} PRIVATE ${VTK_MODULES})
    
    vtk_module_autoinit(
        TARGETS ${target_name}
        MODULES ${VTK_MODULES}
    )
endif()


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
    headers_cpp
)

target_compile_definitions(raycasting PRIVATE ${VTK_DEFINITIONS})

if (UNIX)
  target_link_options(raycasting PUBLIC -pthread)
endif()


set(target_name raycasting)

if(VTK_FOUND)
    # VTK was found via find_package
    target_include_directories(${target_name} PRIVATE ${VTK_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${VTK_LIBRARIES})
    target_compile_definitions(${target_name} PRIVATE ${VTK_DEFINITIONS})
    
    vtk_module_autoinit(
        TARGETS ${target_name}
        MODULES ${VTK_LIBRARIES}
    )
else()
    # VTK was built via FetchContent - link to specific targets
    set(VTK_MODULES
        VTK::CommonCore
        VTK::CommonDataModel
        VTK::FiltersGeometry
        VTK::IOXML
        VTK::IOParallelXML
        VTK::InteractionStyle
    )
    
    target_link_libraries(${target_name} PRIVATE ${VTK_MODULES})
    
    vtk_module_autoinit(
        TARGETS ${target_name}
        MODULES ${VTK_MODULES}
    )
endif()


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
    target_include_directories(fit_to_analytical PRIVATE 
        ${VTK_INCLUDE_DIRS} 
        ${CERES_INCLUDE_DIRS}
        OpenMP::OpenMP_CXX
        headers_cpp
    )
else()
    target_include_directories(fit_to_analytical PRIVATE 
        ${VTK_INCLUDE_DIRS} 
        ${CERES_INCLUDE_DIRS}
        headers_cpp
    )
endif()



target_compile_definitions(fit_to_analytical PRIVATE ${VTK_DEFINITIONS} ${CERES_DEFINITIONS})

if (UNIX)
  target_link_options(fit_to_analytical PUBLIC -pthread)
endif()


set(target_name fit_to_analytical)

if(VTK_FOUND)
    # VTK was found via find_package
    target_include_directories(${target_name} PRIVATE ${VTK_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${VTK_LIBRARIES})
    target_compile_definitions(${target_name} PRIVATE ${VTK_DEFINITIONS})
    
    vtk_module_autoinit(
        TARGETS ${target_name}
        MODULES ${VTK_LIBRARIES}
    )
else()
    # VTK was built via FetchContent - link to specific targets
    set(VTK_MODULES
        VTK::CommonCore
        VTK::CommonDataModel
        VTK::FiltersGeometry
        VTK::IOXML
        VTK::IOParallelXML
        VTK::InteractionStyle
    )
    
    target_link_libraries(${target_name} PRIVATE ${VTK_MODULES})
    
    vtk_module_autoinit(
        TARGETS ${target_name}
        MODULES ${VTK_MODULES}
    )
endif()




# ********************************* read_and_write: read some data and write it in real dims


add_executable(read_and_write main_cpp/read_and_write.cpp
    src_cpp/points.cpp
    src_cpp/matrix.cpp
    src_cpp/read_pvtr.cpp
    src_cpp/read_file.cpp
    src_cpp/reader_writer.cpp
    src_cpp/streamlines.cpp
    src_cpp/magnetopause.cpp
    src_cpp/preprocessing.cpp
)


if (OpenMP_CXX_FOUND)
    target_link_libraries(read_and_write PRIVATE headers_cpp ${VTK_LIBRARIES} ${CERES_LIBRARIES} OpenMP::OpenMP_CXX)
else()
    target_link_libraries(read_and_write PRIVATE headers_cpp ${VTK_LIBRARIES} ${CERES_LIBRARIES})
endif()


target_include_directories(read_and_write PRIVATE 
    ${VTK_INCLUDE_DIRS} 
    ${CERES_INCLUDE_DIRS}
    headers_cpp
)

target_compile_definitions(read_and_write PRIVATE ${VTK_DEFINITIONS} ${CERES_DEFINITIONS})

if (UNIX)
  target_link_options(read_and_write PUBLIC -pthread)
endif()


set(target_name read_and_write)

if(VTK_FOUND)
    # VTK was found via find_package
    target_include_directories(${target_name} PRIVATE ${VTK_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${VTK_LIBRARIES})
    target_compile_definitions(${target_name} PRIVATE ${VTK_DEFINITIONS})
    
    vtk_module_autoinit(
        TARGETS ${target_name}
        MODULES ${VTK_LIBRARIES}
    )
else()
    # VTK was built via FetchContent - link to specific targets
    set(VTK_MODULES
        VTK::CommonCore
        VTK::CommonDataModel
        VTK::FiltersGeometry
        VTK::IOXML
        VTK::IOParallelXML
        VTK::InteractionStyle
    )
    
    target_link_libraries(${target_name} PRIVATE ${VTK_MODULES})
    
    vtk_module_autoinit(
        TARGETS ${target_name}
        MODULES ${VTK_MODULES}
    )
endif()




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
    target_link_libraries(full_process PRIVATE headers_cpp ${VTK_LIBRARIES} ${CERES_LIBRARIES} OpenMP::OpenMP_CXX)
else()
    target_link_libraries(full_process PRIVATE headers_cpp ${VTK_LIBRARIES} ${CERES_LIBRARIES})
endif()


target_include_directories(full_process PRIVATE 
    ${VTK_INCLUDE_DIRS} 
    ${CERES_INCLUDE_DIRS}
    headers_cpp
)

target_compile_definitions(full_process PRIVATE ${VTK_DEFINITIONS} ${CERES_DEFINITIONS})

if (UNIX)
  target_link_options(full_process PUBLIC -pthread)
endif()


set(target_name full_process)

if(VTK_FOUND)
    # VTK was found via find_package
    target_include_directories(${target_name} PRIVATE ${VTK_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${VTK_LIBRARIES})
    target_compile_definitions(${target_name} PRIVATE ${VTK_DEFINITIONS})
    
    vtk_module_autoinit(
        TARGETS ${target_name}
        MODULES ${VTK_LIBRARIES}
    )
else()
    # VTK was built via FetchContent - link to specific targets
    set(VTK_MODULES
        VTK::CommonCore
        VTK::CommonDataModel
        VTK::FiltersGeometry
        VTK::IOXML
        VTK::IOParallelXML
        VTK::InteractionStyle
    )
    
    target_link_libraries(${target_name} PRIVATE ${VTK_MODULES})
    
    vtk_module_autoinit(
        TARGETS ${target_name}
        MODULES ${VTK_MODULES}
    )
endif()
