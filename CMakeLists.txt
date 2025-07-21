cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(full_process)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_PREFIX_PATH "~/local" ${CMAKE_PREFIX_PATH})
find_package(Ceres REQUIRED)

find_package(VTK COMPONENTS 
  CommonCore
  CommonDataModel
  FiltersGeometry
  IOXML
  IOParallelXML
  InteractionStyle
  REQUIRED
)

if (NOT VTK_FOUND)
  message(FATAL_ERROR "full_process: Unable to find the VTK build folder.")
endif()

find_package(OpenMP REQUIRED)


set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")   # is this a good idea

# Prevent a "command line is too long" failure in Windows.
set(CMAKE_NINJA_FORCE_RESPONSE_FILE "ON" CACHE BOOL "Force Ninja to use response files.")

# Add all your source files here
set(SOURCES
    src_cpp/points.cpp
    src_cpp/matrix.cpp
    src_cpp/read_pvtr.cpp
    src_cpp/read_file.cpp
    src_cpp/raycast.cpp
    src_cpp/streamlines.cpp
    src_cpp/magnetopause.cpp
    src_cpp/preprocessing.cpp
    src_cpp/fit_to_analytical.cpp
)

# If you have headers in a separate directory, add them too
set(HEADERS
    headers_cpp/points.h
    headers_cpp/matrix.h   
    headers_cpp/raycast.h
    headers_cpp/streamlines.h
    headers_cpp/magnetopause.h
    headers_cpp/preprocessing.h  
    headers_cpp/read_pvtr.h
    headers_cpp/read_file.h
    headers_cpp/fit_to_analytical.h
)

add_executable(full_process tests_cpp/full_process_test.cpp
    src_cpp/points.cpp
    src_cpp/matrix.cpp
    src_cpp/read_pvtr.cpp
    src_cpp/read_file.cpp
    src_cpp/raycast.cpp
    src_cpp/streamlines.cpp
    src_cpp/magnetopause.cpp
    src_cpp/preprocessing.cpp
    src_cpp/fit_to_analytical.cpp)


# Link VTK libraries
target_link_libraries(full_process PRIVATE ${VTK_LIBRARIES} ${CERES_LIBRARIES} OpenMP::OpenMP_CXX)

# Include VTK headers - this is crucial!
target_include_directories(full_process PRIVATE ${VTK_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

# Set VTK definitions
target_compile_definitions(full_process PRIVATE ${VTK_DEFINITIONS} ${CERES_DEFINITIONS})

if (UNIX)
  target_link_options(full_process PUBLIC -pthread)
endif()

# vtk_module_autoinit is needed
vtk_module_autoinit(
  TARGETS full_process
  MODULES ${VTK_LIBRARIES}
)




add_executable(fit_to_analytical tests_cpp/fit_to_analytical_test.cpp
    src_cpp/points.cpp
    src_cpp/matrix.cpp
    src_cpp/read_pvtr.cpp
    src_cpp/read_file.cpp
    src_cpp/streamlines.cpp
    src_cpp/magnetopause.cpp
    src_cpp/preprocessing.cpp
    src_cpp/fit_to_analytical.cpp)


# Link VTK libraries
target_link_libraries(fit_to_analytical PRIVATE ${VTK_LIBRARIES} ${CERES_LIBRARIES})

# Include VTK headers - this is crucial!
target_include_directories(fit_to_analytical PRIVATE ${VTK_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

# Set VTK definitions
target_compile_definitions(fit_to_analytical PRIVATE ${VTK_DEFINITIONS} ${CERES_DEFINITIONS})

if (UNIX)
  target_link_options(fit_to_analytical PUBLIC -pthread)
endif()

# vtk_module_autoinit is needed
vtk_module_autoinit(
  TARGETS fit_to_analytical
  MODULES ${VTK_LIBRARIES}
)

