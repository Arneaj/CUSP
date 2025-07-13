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

add_executable(read_pvtr src_cpp/read_pvtr.cpp)

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