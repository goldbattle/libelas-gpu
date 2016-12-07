cmake_minimum_required(VERSION 2.8.8)

# Project name
project(libelas-gpu)

# Include our cmake files
# SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

# Include libraries
find_package(CUDA)
# find_package(Eigen3 REQUIRED)
# find_package(Boost REQUIRED COMPONENTS system serialization system filesystem thread program_options date_time regex timer chrono)
# find_package(OpenCV 3 REQUIRED core imgcodecs videoio ximgproc)


# Try to compile with c++11
# http://stackoverflow.com/a/25836953
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Enable debug flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3  -Wall")

# Enable SSE3 cpu commands
# If we are on ARM we need to find an alternative
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3")

# Set our nvcc flags
# http://stackoverflow.com/a/13244930
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch sm_20 -gencode arch=compute_20,code=sm_20)


# Include our header files
include_directories(CPU GPU GPU_test)

# Include all source files in each sub-directory
# file(GLOB_RECURSE sources_cpu CPU/*.cpp)
# file(GLOB_RECURSE sources_gpu GPU/*.cu)
# file(GLOB_RECURSE sources_cpu GPU_test/*.cpp)


# Set files the build
set(sources_cpu
  CPU/descriptor.cpp
  CPU/elas.cpp
  CPU/filter.cpp
  CPU/matrix.cpp
  CPU/triangle.cpp
)

# Set files the build
set(sources_gpu
  GPU/elas_gpu.cu
)


# CPU binary
add_executable(libelas_cpu main_cpu.cpp ${sources_cpu})

# GPU binary, convert the gpu code to cpu code, then build
cuda_compile(sources_gpu_built main_gpu.cu ${sources_gpu})
cuda_add_executable(libelas_gpu ${sources_gpu_built} ${sources_cpu})

# Testing binary
add_executable(libelas_test main_test.cpp ${sources_cpu})
