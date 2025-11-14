cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Voxelizer CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find PDAL
find_package(PDAL REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${PDAL_INCLUDE_DIRS})

# Main executable
add_executable(voxelizer
    src/main.cpp
    src/las_reader.cpp
    src/ply_writer.cpp
)

# Link PDAL
target_link_libraries(voxelizer PRIVATE ${PDAL_LIBRARIES})