cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(read_pvtr)

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
  message(FATAL_ERROR "read_pvtr: Unable to find the VTK build folder.")
endif()

# Prevent a "command line is too long" failure in Windows.
set(CMAKE_NINJA_FORCE_RESPONSE_FILE "ON" CACHE BOOL "Force Ninja to use response files.")

# Add all your source files here
set(SOURCES
    src_cpp/points.cpp
    src_cpp/matrix.cpp
    src_cpp/read_pvtr.cpp
    src_cpp/read_pvtr_tests.cpp
)

# If you have headers in a separate directory, add them too
set(HEADERS
    headers_cpp/points.h
    headers_cpp/matrix.h     
    headers_cpp/read_pvtr.h
)

add_executable(read_pvtr tests_cpp/read_pvtr_test.cpp
    src_cpp/read_pvtr.cpp
    src_cpp/points.cpp
    src_cpp/matrix.cpp)

# Link VTK libraries
target_link_libraries(read_pvtr PRIVATE ${VTK_LIBRARIES})

# Include VTK headers - this is crucial!
target_include_directories(read_pvtr PRIVATE ${VTK_INCLUDE_DIRS})

# Set VTK definitions
target_compile_definitions(read_pvtr PRIVATE ${VTK_DEFINITIONS})

if (UNIX)
  target_link_options(read_pvtr PUBLIC -pthread)
endif()

# vtk_module_autoinit is needed
vtk_module_autoinit(
  TARGETS read_pvtr
  MODULES ${VTK_LIBRARIES}
)